// Times a graph with the ternary custom op registered, and reports what XNNPACK
// delegated.
//
// The question: custom ops are opaque to the XNNPACK delegate, so in a real decoder
// (~224 ternary projections between norms, RoPE, softmax) each one may split the
// graph and drop its neighbours out of delegation. If that happens the ternary win
// is handed back. A single-op graph cannot show it; three graphs can:
//
//   norms    L x (RMSNorm -> GELU)                fully delegable, no custom ops
//   ternary  L x ternary_matmul                   custom ops only
//   mixed    L x (ternary_matmul -> RMSNorm -> GELU)
//
// mixed ~= norms + ternary  => interleaving is free
// mixed >> norms + ternary  => the delegate is losing the ops in between
//
//   ./bench_partition model.tflite [iters]
//
// Run with LiteRT's INFO logging visible to see "Replacing N node(s)".

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "ternary_gemm.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

struct State { int calls = 0; };

LiteRtStatus Init(void*, const void*, size_t) { return kLiteRtStatusOk; }
LiteRtStatus Destroy(void*) { return kLiteRtStatusOk; }

LiteRtStatus GetOutputLayouts(void*, size_t n_in, const LiteRtLayout* in,
                              size_t n_out, LiteRtLayout* out) {
    if (n_in != 3 || n_out != 1) return kLiteRtStatusErrorInvalidArgument;
    memset(&out[0], 0, sizeof(out[0]));
    out[0].rank = 2;
    out[0].dimensions[0] = in[0].dimensions[0];
    out[0].dimensions[1] = in[1].dimensions[0];
    return kLiteRtStatusOk;
}

LiteRtStatus Run(void* ud, size_t n_in, const LiteRtTensorBuffer* inputs,
                 size_t n_out, LiteRtTensorBuffer* outputs) {
    if (n_in != 3 || n_out != 1) return kLiteRtStatusErrorInvalidArgument;
    static_cast<State*>(ud)->calls++;
    void *px = nullptr, *pw = nullptr, *ps = nullptr, *py = nullptr;
    size_t bx = 0, bw = 0, bs = 0;
    LiteRtGetTensorBufferSize(inputs[0], &bx);
    LiteRtGetTensorBufferSize(inputs[1], &bw);
    LiteRtGetTensorBufferSize(inputs[2], &bs);
    LiteRtLockTensorBuffer(inputs[0], &px, kLiteRtTensorBufferLockModeRead);
    LiteRtLockTensorBuffer(inputs[1], &pw, kLiteRtTensorBufferLockModeRead);
    LiteRtLockTensorBuffer(inputs[2], &ps, kLiteRtTensorBufferLockModeRead);
    LiteRtLockTensorBuffer(outputs[0], &py, kLiteRtTensorBufferLockModeWrite);

    const int n_rows = (int)(bs / sizeof(float));
    const int k = (int)(bw / n_rows) * 4;
    const int m = (int)(bx / sizeof(float) / k);
    static std::vector<int8_t> q;
    static std::vector<float> xs;
    q.resize((size_t)m * k);
    xs.resize(m);
    ternary_quantize_activations((const float*)px, m, k, q.data(), xs.data());
    ternary_gemm((const uint8_t*)pw, n_rows, k, q.data(), xs.data(), m,
                 (const float*)ps, 1, nullptr, (float*)py);

    LiteRtUnlockTensorBuffer(inputs[0]);
    LiteRtUnlockTensorBuffer(inputs[1]);
    LiteRtUnlockTensorBuffer(inputs[2]);
    LiteRtUnlockTensorBuffer(outputs[0]);
    return kLiteRtStatusOk;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argv[1];
    const int iters = argc > 2 ? atoi(argv[2]) : 20;

    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    if (LiteRtCreateEnvironment(0, nullptr, &env) != kLiteRtStatusOk ||
        LiteRtCreateModelFromFile(env, path, &model) != kLiteRtStatusOk) {
        fprintf(stderr, "load failed: %s\n", path);
        return 1;
    }
    LiteRtOptions opts = nullptr;
    LiteRtCreateOptions(&opts);
    LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu);
    State st;
    LiteRtCustomOpKernel kern = {Init, GetOutputLayouts, Run, Destroy};
    LiteRtAddCustomOpKernelOption(opts, "voxsum.ternary_matmul", 1, &kern, &st);

    LiteRtCompiledModel cm = nullptr;
    if (LiteRtCreateCompiledModel(env, model, opts, &cm) != kLiteRtStatusOk) {
        fprintf(stderr, "compile failed\n");
        return 1;
    }
    LiteRtSignature sig = nullptr;
    LiteRtGetModelSignature(model, 0, &sig);
    LiteRtParamIndex n_in = 0;
    LiteRtGetNumSignatureInputs(sig, &n_in);

    std::vector<LiteRtTensorBuffer> ins((size_t)n_in, nullptr);
    for (LiteRtParamIndex i = 0; i < n_in; i++) {
        LiteRtTensor t = nullptr;
        LiteRtRankedTensorType tt;
        LiteRtTensorBufferRequirements reqs = nullptr;
        size_t bytes = 0;
        LiteRtGetSignatureInputTensorByIndex(sig, i, &t);
        LiteRtGetRankedTensorType(t, &tt);
        LiteRtGetCompiledModelInputBufferRequirements(cm, 0, i, &reqs);
        LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes);
        LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt, bytes, &ins[i]);
        void* p = nullptr;
        LiteRtLockTensorBuffer(ins[i], &p, kLiteRtTensorBufferLockModeWrite);
        if (i == 0) for (size_t j = 0; j < bytes / sizeof(float); j++)
            ((float*)p)[j] = sinf((float)j * 0.1f) * 0.5f;
        else if (i == 1) for (size_t j = 0; j < bytes; j++)
            ((uint8_t*)p)[j] = (uint8_t)(0x1B + (j % 7));
        else for (size_t j = 0; j < bytes / sizeof(float); j++) ((float*)p)[j] = 0.02f;
        LiteRtUnlockTensorBuffer(ins[i]);
    }
    LiteRtTensorBuffer out = nullptr;
    {
        LiteRtTensor t = nullptr;
        LiteRtRankedTensorType tt;
        LiteRtTensorBufferRequirements reqs = nullptr;
        size_t bytes = 0;
        LiteRtGetSignatureOutputTensorByIndex(sig, 0, &t);
        LiteRtGetRankedTensorType(t, &tt);
        LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, 0, &reqs);
        LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes);
        LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt, bytes, &out);
    }

    if (LiteRtRunCompiledModel(cm, 0, (LiteRtParamIndex)ins.size(), ins.data(), 1, &out)
        != kLiteRtStatusOk) {
        fprintf(stderr, "invoke failed\n");
        return 1;
    }
    const int warm_calls = st.calls;
    const double t0 = now_s();
    for (int i = 0; i < iters; i++)
        LiteRtRunCompiledModel(cm, 0, (LiteRtParamIndex)ins.size(), ins.data(), 1, &out);
    const double each = (now_s() - t0) / iters;

    printf("RESULT %-24s %8.3f ms/run   custom_ops_per_run=%d\n",
           path, each * 1e3, warm_calls);
    return 0;
}
