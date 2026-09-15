// ggml_mul_mat shape curve for the blocked-int8 path (Exp778).
//
// WHY THIS EXISTS. The ledger has carried the same unsolved line for dozens of runs: the VAE's
// q4_0_4x4 linears reach only ~25 GMAC/s in-graph while a microbench reported 40-45 GMAC/s at
// L >= 50. Every structural explanation tried (arena aliasing, thread counts, scheduling) came back
// negative, so the remaining question is a shape question: what does the kernel actually deliver at
// the SHAPE THE VAE USES (ne11 = 26 = one 26-frame window per piece), as opposed to the shapes the
// old microbench happened to try? If the curve at L=26 is ~40 GMAC/s, the in-graph gap is graph-side
// and worth chasing. If the curve at L=26 is ~25, the wall is the kernel's 4-column panel at short
// L and no in-graph work will move it.
//
// It mirrors the shipped path in the two ways that matter: the weight is quantized on THIS device
// with ggml_quantize_chunk(Q4_0_4_4) (host quantization writes zero scales - Exp688), and src1 is
// F32 so the blocked kernel does its own F32->Q8_0 conversion through from_float_to_mat, exactly as
// vae_conv_1d_i8 and the LM do.
//
// Build (NDK, link against the same libggml the app uses):
//   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang++ -O3 -std=c++17 \
//       -I 3rdparty/llama.cpp/ggml/include .auto/mm_shape_micro.cpp \
//       -L build-android/lib -lggml -lm -o .auto/mm_shape_micro
//   adb push .auto/mm_shape_micro build-android/lib/libggml.so /data/local/tmp/vibeasr/
// Run pinned to the two A78s:  taskset C0 ./mm_shape_micro
#include "ggml.h"
#include "ggml-backend.h"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char ** argv) {
    const int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    // (ne00, ne01) pairs taken from the real graph: the deepest VAE FFN linears are [8192,2048] and
    // [4096,1024] per the MMSHAPE census; 2048/1024 covers the early stages, whose C is smaller.
    const int shapes[][2] = {{8192, 2048}, {4096, 1024}, {2048, 512}, {1024, 256}};
    const int ls[] = {1, 2, 4, 8, 13, 26, 27, 28, 32, 52, 64, 128};
    fprintf(stderr, "threads=%d (GMAC/s = ne00*ne01*ne11 per call)\n", nthreads); fflush(stderr);
    for (auto & sh : shapes) {
        const int64_t ne00 = sh[0], ne01 = sh[1];
        fprintf(stderr, "weight [%lld x %lld] q4_0_4x4 (%.1f MB)\n", (long long)ne00, (long long)ne01,
            (double)ggml_row_size(GGML_TYPE_Q4_0_4_4, ne00) * ne01 / 1e6); fflush(stderr);
        for (int L : ls) {
            struct ggml_init_params p = { /*mem_size=*/ (size_t)384 << 20, /*mem_buffer=*/ nullptr,
                                          /*no_alloc=*/ false };
            struct ggml_context * ctx = ggml_init(p);
            if (!ctx) { printf("  ggml_init failed\n"); return 1; }
            struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_4_4, ne00, ne01);
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, L);
            // MUST be ne00*ne01 floats: ggml_quantize_chunk quantizes nrow rows of n_per_row each,
            // so a one-row buffer makes it read ne01 rows past the end (this was the tool's own segfault).
            std::vector<float> src((size_t)ne00 * ne01);
            for (size_t i = 0; i < src.size(); i++) src[i] = 0.01f * (float)(((long)i * 37) % 211 - 105);
            ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, src.data(), w->data, 0, ne01, ne00, nullptr);
            float * xd = (float *)x->data;
            for (int64_t j = 0; j < (int64_t)ne00 * L; j++) xd[j] = 0.01f * (float)((j * 17) % 199 - 99);
            struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            if (!gf) { printf("  L=%3d graph alloc FAILED (arena too small)\n", L); ggml_free(ctx); continue; }
            ggml_build_forward_expand(gf, y);

            // one warm call, then time enough repeats to beat the clock granularity
            if (ggml_graph_compute_with_ctx(ctx, gf, nthreads) != GGML_STATUS_SUCCESS) {
                printf("  L=%3d compute FAILED\n", L); ggml_free(ctx); continue;
            }
            const double macs = (double)ne00 * (double)ne01 * (double)L;
            int reps = 1;
            double t0 = now_s();
            ggml_graph_compute_with_ctx(ctx, gf, nthreads);
            double one = now_s() - t0;
            if (one > 0) reps = (int)(0.25 / one);
            if (reps < 1) reps = 1;
            t0 = now_s();
            for (int r = 0; r < reps; r++) ggml_graph_compute_with_ctx(ctx, gf, nthreads);
            double dt = (now_s() - t0) / reps;
            // bytes the weights alone imply, to say whether the call is traffic- or rate-bound
            const double wb = (double)ggml_nbytes(w);
            fflush(stdout);
            fprintf(stderr,"  L=%3d  %7.2f ms  %6.2f GMAC/s   %5.2f GB/s(weight-only)  %s\n", L, dt * 1e3,
                   macs / 1e9 / dt, wb / dt / 1e9,
                   wb / dt / 1e9 > 4.5 ? "TRAFFIC-bound (weight stream at DRAM rate)" : "rate/latency-bound"); fflush(stderr);
            ggml_free(ctx);
        }
    }
    return 0;
}
