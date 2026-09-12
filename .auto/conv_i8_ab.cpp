// Exp687: settle converter-vs-runtime for the blocked-int8 conv weights, on the host, with a real
// positive control.
//
// Both ggml_gemv/gemm_q4_0_4x4_q8_0 exist in the x86 libggml (nm: T ggml_gemm_q4_0_4x4_q8_0), so a
// 2-D Q4_0_4_4 matmul CAN be evaluated here - my earlier "the host cannot evaluate q4_0_4x4" came
// from the type traits' NULL vec_dot, but mul_mat dispatches to gemv/gemm for 2-D blocked matmuls.
//
// For each case the program computes the same product twice - once from the reference F16/F32
// weights, once from the blocked-int8 bytes - on identical activations, and reports the relative
// error. Cases:
//   CONTROL  an ffn tensor pair (f16 original vs llama-quantize's q4_0_4x4) -> must be ~1%%
//   CONV     a conv tensor from my converted file (2-D q4_0_4_4) vs the f16 original -> the verdict
// Data comes from .auto/dump_for_ab.py, which writes raw f32 weights, raw quantized bytes, and dims.
#include <ggml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static std::vector<float> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f) / 4; fseek(f, 0, SEEK_SET);
    std::vector<float> v(n);
    if (n && fread(v.data(), 4, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f); return v;
}
static std::vector<uint8_t> rdb(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (n && fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f); return v;
}

// one A/B: weight [k, m] (row-major, m rows of k, as ggml sees a [k, m] matrix), activations [k, L]
static double one(const char* tag, int64_t k, int64_t m, int64_t L,
                  const std::vector<float>& wref, const std::vector<uint8_t>& wq, bool have_q) {
    struct ggml_context* ctx = nullptr;
    ggml_init_params p{64u << 20, ctx, false};
    ggml_init(p);
    int64_t kq = k, mq = m;
    if (k % 32) { printf("  %-10s SKIP: k=%lld not a multiple of 32\n", tag, (long long)k); ggml_free(ctx); return -1; }
    if (mq % 4) { printf("  %-10s SKIP: m=%lld not a multiple of 4\n", tag, (long long)mq); ggml_free(ctx); return -1; }

    struct ggml_tensor* wr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    struct ggml_tensor* wq8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_4_4, kq, mq);
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, L);
    struct ggml_tensor* y_r = ggml_mul_mat(ctx, wr, x);
    // On x86 the blocked traits have from_float == NULL, so handing mul_mat an F32 src1 dereferences
    // a null pointer (that is the segfault this program gave before). Convert to Q8_0 explicitly -
    // which is exactly what the vae.cpp int8 branch does by default.
    struct ggml_tensor* xq = ggml_cast(ctx, x, GGML_TYPE_Q8_0);
    struct ggml_tensor* y_q = ggml_mul_mat(ctx, wq8, xq);

    memcpy(wr->data, wref.data(), wref.size() * 4);
    if (have_q) memcpy(wq8->data, wq.data(), wq.size());
    // deterministic activations
    std::vector<float> xv((size_t)k * L);
    for (size_t i = 0; i < xv.size(); i++) xv[i] = std::sin(0.001f * i) + 0.3f * std::cos(0.007f * i);
    memcpy(x->data, xv.data(), xv.size() * 4);

    struct ggml_cgraph* gr = ggml_new_graph(ctx); ggml_build_forward_expand(gr, y_r);
    struct ggml_cgraph* gq = ggml_new_graph(ctx); ggml_build_forward_expand(gq, y_q);
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (ggml_graph_compute_with_ctx(ctx, gr, 2) != 0) { printf("  %s ref FAILED\n", tag); ggml_free(ctx); return -1; }
    double worst = -1;
    if (!have_q) {
        printf("  %-10s reference product only (m=%lld k=%lld L=%lld)\n", tag, (long long)m, (long long)k, (long long)L);
        ggml_free(ctx); return 0;
    }
    if (ggml_graph_compute_with_ctx(ctx, gq, 2) != 0) { printf("  %s quant FAILED (blocked path rejected)\n", tag); ggml_free(ctx); return -1; }
    const float* a = (const float*)y_r->data; const float* b = (const float*)y_q->data;
    double scale = 0;
    for (int64_t i = 0; i < (int64_t)m * L; i++) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > worst) worst = e;
        scale = std::max(scale, std::fabs((double)a[i]));
    }
    printf("  %-10s max|diff|=%.6g  ref max|y|=%.6g  ratio=%.6f %s\n", tag, worst, scale,
           scale ? worst / scale : 0.0, worst / scale > 0.05 ? "<-- NOT equivalent" : "(equivalent)");
    ggml_free(ctx);
    return worst / (scale ? scale : 1.0);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: a crash must not swallow the diagnosis
    // args: (tag k m wref.bin wq.bin|-) repeated - five per case
    const int64_t L = 16;
    for (int i = 1; i + 4 < argc + 1; i += 5) {
        const char* tag = argv[i];
        int64_t k = atoll(argv[i + 1]), m = atoll(argv[i + 2]);
        std::string ref = argv[i + 3], q = argv[i + 4];
        std::vector<float> wr = rd(ref);
        if ((int64_t)wr.size() != k * m) { printf("  %s: ref has %zu floats, expected %lld\n", tag, wr.size(), (long long)(k * m)); continue; }
        if (q == "-") { one(tag, k, m, L, wr, {}, false); continue; }
        std::vector<uint8_t> wq = rdb(q);
        printf("  %s: quantized bytes=%zu expected=%zu\n", tag, wq.size(), (size_t)(72 * (m / 4) * (k / 32)));
        one(tag, k, m, L, wr, wq, true);
    }
    return 0;
}
