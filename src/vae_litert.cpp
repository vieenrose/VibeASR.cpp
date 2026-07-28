// LiteRT/XNNPACK implementation of the audio front end, behind the same vae.h API
// as the ggml one in vae.cpp.
//
// WHY BOTH EXIST. Measured on a Boox Tab Mini C (Snapdragon 662, ARMv8.0), 10 s of
// audio, 45 s clip:
//
//                        ggml (I8_S)    LiteRT (int8)
//   encode / 10 s window    ~23.5 s        15.0 s      (and on half the threads)
//   feature cosine vs f32     0.958         0.992      (per-tensor vs per-channel scales)
//   peak anonymous RAM        ~200 MB       576 MB
//
// XNNPACK wins the convolutional encoder; ggml wins autoregressive decode by roughly
// an order of magnitude (see the MOSS-TD comparison in the VoxSum README). So the
// encoder can profitably move here while the Qwen2.5 decoder stays on llama.cpp —
// the split this file exists to enable. It costs anonymous memory, which is the
// trade the caller is choosing when it passes a .tflite.
//
// The export bundles BOTH tokenizer encoders and BOTH connectors and returns their
// SUM — exactly what prompt_builder.h computes from the two ggml planes. So this
// backend is "combined": vae_encode_acoustic returns the summed features and
// vae_encode_semantic returns zeros, which keeps the sum downstream correct without
// the caller needing to know which backend it has.
//
// The graph is FIXED-LENGTH (the export picks a window). Audio shorter than the
// window is zero-padded and the surplus frames dropped; longer audio is the caller's
// problem — vae_encode_impl in vae.cpp already slices into windows.

#include "vae.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <ctime>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {

bool ok(LiteRtStatus s, const char* what) {
    if (s == kLiteRtStatusOk) return true;
    fprintf(stderr, "[vae-litert] %s failed (%d)\n", what, (int)s);
    return false;
}

// XNNPACK settings ride in an opaque-options TOML payload (same trick as
// VoxSum's moss_lite_engine.cc — LrtCpuOptions is not exported by libLiteRt.so,
// but the payload is a trivial TOML string).
//
// weight_cache_file_path is not optional in practice: XNNPACK repacks every
// weight at load, and with no cache those repacked weights are ANONYMOUS memory.
// For this 700 MB encoder that measured 1228 MB of anon and 7.7 s of repacking
// per load; with a cache it is 576 MB and 0.1 s.
LiteRtOpaqueOptions make_cpu_options(int num_threads, const std::string& weight_cache) {
    char toml[1024];
    int off = 0;
    if (num_threads > 0)
        off += snprintf(toml + off, sizeof(toml) - off, "num_threads = %d\n", num_threads);
    if (!weight_cache.empty())
        off += snprintf(toml + off, sizeof(toml) - off,
                        "weight_cache_file_path = \"%s\"\n", weight_cache.c_str());
    if (off <= 0) return nullptr;
    char* payload = strdup(toml);
    LiteRtOpaqueOptions oo = nullptr;
    if (LiteRtCreateOpaqueOptions("xnnpack", payload, [](void* p) { free(p); }, &oo)
        != kLiteRtStatusOk) {
        free(payload);
        return nullptr;
    }
    return oo;
}

size_t buf_elems(LiteRtTensorBuffer b) {
    size_t bytes = 0;
    return LiteRtGetTensorBufferSize(b, &bytes) == kLiteRtStatusOk ? bytes / sizeof(float) : 0;
}

bool write_buf(LiteRtTensorBuffer b, const float* src, size_t n_src, size_t n_buf) {
    void* p = nullptr;
    if (!ok(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeWrite), "lock(w)")) return false;
    auto* dst = static_cast<float*>(p);
    const size_t n = std::min(n_src, n_buf);
    memcpy(dst, src, n * sizeof(float));
    if (n < n_buf) memset(dst + n, 0, (n_buf - n) * sizeof(float));  // zero-pad short windows
    LiteRtUnlockTensorBuffer(b);
    return true;
}

bool read_buf(LiteRtTensorBuffer b, float* dst, size_t n) {
    void* p = nullptr;
    if (!ok(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeRead), "lock(r)")) return false;
    memcpy(dst, p, n * sizeof(float));
    LiteRtUnlockTensorBuffer(b);
    return true;
}

}  // namespace

// Opaque to callers; vae.cpp's structs are separate and never mixed with these.
struct vae_litert_model {
    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    LiteRtCompiledModel cm = nullptr;
    LiteRtTensorBuffer in = nullptr, out = nullptr;
    size_t n_in = 0, n_out = 0;   // floats
    int32_t out_dim = 0;          // 1536
    int32_t window_samples = 0;

    ~vae_litert_model() {
        if (in) LiteRtDestroyTensorBuffer(in);
        if (out) LiteRtDestroyTensorBuffer(out);
        if (cm) LiteRtDestroyCompiledModel(cm);
        if (model) LiteRtDestroyModel(model);
        if (env) LiteRtDestroyEnvironment(env);
    }
};

extern "C" {

vae_litert_model* vae_litert_load(const char* path, int n_threads, int32_t out_dim) {
    auto* m = new vae_litert_model();
    m->out_dim = out_dim;

    const char* cache = getenv("VIBEASR_LITERT_CACHE");
    if (!ok(LiteRtCreateEnvironment(0, nullptr, &m->env), "CreateEnvironment")) { delete m; return nullptr; }
    if (!ok(LiteRtCreateModelFromFile(m->env, path, &m->model), "CreateModelFromFile")) { delete m; return nullptr; }

    LiteRtOptions opts = nullptr;
    if (!ok(LiteRtCreateOptions(&opts), "CreateOptions")) { delete m; return nullptr; }
    ok(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu), "SetAccelerators");
    if (LiteRtOpaqueOptions oo = make_cpu_options(n_threads, cache ? cache : ""))
        ok(LiteRtAddOpaqueOptions(opts, oo), "AddOpaqueOptions");
    if (!ok(LiteRtCreateCompiledModel(m->env, m->model, opts, &m->cm), "CreateCompiledModel")) {
        delete m; return nullptr;
    }

    // Single signature, one input (audio) and one output (summed features).
    // Buffers are sized from the compiled model's requirements, not from the
    // tensor type, because the runtime may demand padding/alignment beyond it.
    LiteRtSignature sig = nullptr;
    if (!ok(LiteRtGetModelSignature(m->model, 0, &sig), "GetModelSignature")) { delete m; return nullptr; }

    const auto make = [&](bool is_input, LiteRtTensorBuffer* out_buf) {
        LiteRtTensor tensor = nullptr;
        if (!ok(is_input ? LiteRtGetSignatureInputTensorByIndex(sig, 0, &tensor)
                         : LiteRtGetSignatureOutputTensorByIndex(sig, 0, &tensor), "SignatureTensor"))
            return false;
        LiteRtRankedTensorType tt;
        if (!ok(LiteRtGetRankedTensorType(tensor, &tt), "GetRankedTensorType")) return false;
        LiteRtTensorBufferRequirements reqs = nullptr;
        if (!ok(is_input ? LiteRtGetCompiledModelInputBufferRequirements(m->cm, 0, 0, &reqs)
                         : LiteRtGetCompiledModelOutputBufferRequirements(m->cm, 0, 0, &reqs),
                "BufferRequirements")) return false;
        size_t bytes = 0;
        if (!ok(LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes), "RequirementsSize")) return false;
        return ok(LiteRtCreateManagedTensorBuffer(m->env, kLiteRtTensorBufferTypeHostMemory,
                                                  &tt, bytes, out_buf), "CreateManagedTensorBuffer");
    };
    if (!make(true, &m->in) || !make(false, &m->out)) { delete m; return nullptr; }
    m->n_in = buf_elems(m->in);
    m->n_out = buf_elems(m->out);
    m->window_samples = (int32_t)m->n_in;
    fprintf(stderr, "[vae-litert] %s: window=%d samples (%.1fs), out=%zu floats (%zu frames x %d)%s\n",
            path, m->window_samples, m->window_samples / 24000.0, m->n_out,
            out_dim ? m->n_out / out_dim : 0, out_dim, cache ? ", weight cache on" : "");
    return m;
}

void vae_litert_free(vae_litert_model* m) { delete m; }

int32_t vae_litert_window_samples(const vae_litert_model* m) { return m ? m->window_samples : 0; }

// Returns frames written, or -1. `output` receives [n_frames * out_dim] floats.
int32_t vae_litert_encode(vae_litert_model* m, const float* audio, int32_t n_samples,
                          float* output, float* inference_time_ms) {
    if (!m || !audio || !output) return -1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (!write_buf(m->in, audio, (size_t)n_samples, m->n_in)) return -1;
    if (!ok(LiteRtRunCompiledModel(m->cm, 0, 1, &m->in, 1, &m->out), "RunCompiledModel")) return -1;

    // A short (zero-padded) window still produces a full graph's worth of frames;
    // only the ones actually covered by audio are meaningful.
    const int32_t frames_total = (int32_t)(m->n_out / m->out_dim);
    const int32_t frames_real =
        n_samples >= m->window_samples
            ? frames_total
            : std::min(frames_total, (int32_t)((int64_t)n_samples * frames_total / m->window_samples));
    if (frames_real <= 0) return -1;
    if (!read_buf(m->out, output, (size_t)frames_real * m->out_dim)) return -1;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (inference_time_ms)
        *inference_time_ms = (float)((t1.tv_sec - t0.tv_sec) * 1000.0 +
                                     (t1.tv_nsec - t0.tv_nsec) / 1e6);
    return frames_real;
}

}  // extern "C"
