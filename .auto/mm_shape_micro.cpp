// ggml_mul_mat shape curve for the blocked-int8 path (Exp778, extended by Exp812).
//
// WHY THIS EXISTS. The ledger carried "the VAE's q4_0_4x4 linears reach only ~25 GMAC/s in-graph while a
// microbench reported 40-45" as unsolved for dozens of runs. Exp778/785 showed the 40-45 figure was a
// TWO-THREAD reading and that per core the kernel is at its ceiling at the shape the deepest FFN uses
// (ne00=1536-8192, ne11=26). Use this tool to price ANY proposed shape change in one minute instead of a
// device ladder.
//
// Exp812 ADDED THE REGIME THAT ACTUALLY DOMINATES. The MAC census (GGML_MM_DEBUG_MACS, thread-invariant
// after the Exp811 fix) says 52% of a clip's blocked-int8 MACs are NOT in the shapes below - they are the
// VAE's EARLY stages, which run at tiny contraction and enormous column counts, e.g. (ne00,ne01,ne11) =
// (32,128,83200). The original curve only ever measured ne00>=1024 and ne11<=128, so the dominant regime
// was simply never measured. `--real` runs the census triples.
//
// It mirrors the shipped path in the two ways that matter: the weight is quantized on THIS device with
// ggml_quantize_chunk(Q4_0_4_4) (host quantization writes zero scales - Exp688), and src1 is F32 so the
// blocked kernel does its own F32->Q8_0 conversion through from_float_to_mat, exactly as the VAE and LM do.
//
// Build (NDK, link against the same libggml the app uses):
//   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang++ -O3 -std=c++17 \
//       -I 3rdparty/llama.cpp/ggml/include .auto/mm_shape_micro.cpp \
//       -L build-android/lib -lggml -lm -o .auto/mm_shape_micro
//   adb push .auto/mm_shape_micro build-android/lib/libggml.so /data/local/tmp/vibeasr/
// Run pinned to the two A78s:  taskset C0 ./mm_shape_micro        (classic grid)
//                              taskset C0 ./mm_shape_micro 1 real (census triples, 1 thread)
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

// One (ne00 x ne01) weight at L columns, timed. Returns GMAC/s (negative on failure, reason printed).
// src1_type: GGML_TYPE_F32 is what the FFN linears do (the op converts F32->Q8_0 internally, so the number
// includes that pass); GGML_TYPE_Q8_0 is what the CONV path does (im2col writes Q8_0 directly), so comparing
// the two says whether a shape's slowness is the GEMM or the activation quantization. Exp813: without this
// split the row/column "rate gap" could be read as a GEMM problem when it is a conversion problem, and the
// two need opposite fixes.
static double bench(const int nthreads, const int64_t ne00, const int64_t ne01, const int64_t L,
                    const char * note, const enum ggml_type src1_type = GGML_TYPE_F32) {
    struct ggml_init_params p { /*mem_size=*/ (size_t)512 << 20, /*mem_buffer=*/ nullptr,
                                /*no_alloc=*/ false };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) { fprintf(stderr, "  ggml_init failed (arena)\n"); return -1; }
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_4_4, ne00, ne01);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, src1_type, ne00, L);
    // MUST be ne00*ne01 floats: ggml_quantize_chunk quantizes nrow rows of n_per_row each, so a one-row
    // buffer makes it read ne01 rows past the end (this was the tool's own segfault, Exp778).
    std::vector<float> src((size_t)ne00 * ne01);
    for (size_t i = 0; i < src.size(); i++) src[i] = 0.01f * (float)(((long)i * 37) % 211 - 105);
    ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, src.data(), w->data, 0, ne01, ne00, nullptr);
    if (src1_type == GGML_TYPE_F32) {
        float * xd = (float *)x->data;
        for (int64_t j = 0; j < ne00 * L; j++) xd[j] = 0.01f * (float)((j * 17) % 199 - 99);
    } else {
        // quantize the activation ONCE, outside the timed region (as im2col does), so what we time is the
        // GEMM alone. 0.05f keeps values well inside the Q8_0 range without making the dot products trivial.
        std::vector<float> xf((size_t)ne00 * L);
        for (size_t j = 0; j < ne00 * L; j++) xf[j] = 0.05f * (float)(((long)j * 17) % 199 - 99);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, xf.data(), x->data, 0, L, ne00, nullptr);
    }

    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    if (!gf) { fprintf(stderr, "  [%lld x %lld] L=%lld graph alloc FAILED (arena too small)\n",
                       (long long)ne00, (long long)ne01, (long long)L); ggml_free(ctx); return -1; }
    ggml_build_forward_expand(gf, y);

    if (ggml_graph_compute_with_ctx(ctx, gf, nthreads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  [%lld x %lld] L=%lld compute FAILED\n",
                (long long)ne00, (long long)ne01, (long long)L); ggml_free(ctx); return -1;
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
    const double dt = (now_s() - t0) / reps;
    const double wb = (double)ggml_nbytes(w);
    const double gmac = macs / dt / 1e9;
    fprintf(stderr, "  [%5lld x %5lld] L=%6lld  %8.2f ms  %6.2f GMAC/s  %6.2f GB/s(w)  %.1f MB%s%s\n",
            (long long)ne00, (long long)ne01, (long long)L, dt * 1e3, gmac,
            wb / dt / 1e9, wb / 1e6, note && *note ? "  " : "", note ? note : "");
    fflush(stderr);
    ggml_free(ctx);
    return gmac;
}

int main(int argc, char ** argv) {
    const int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    const bool real = argc > 2 && !strcmp(argv[2], "real");

    if (!real) {
        // Classic grid: the deepest FFN shapes (the Exp778 curve the ledger quotes).
        static const int shapes[][2] = {{8192, 2048}, {4096, 1024}, {2048, 512}, {1024, 256}};
        static const int ls[] = {1, 2, 4, 8, 13, 26, 27, 28, 32, 52, 64, 128};
        fprintf(stderr, "threads=%d classic grid (GMAC/s = ne00*ne01*ne11 per call)\n", nthreads);
        for (auto & sh : shapes) {
            fprintf(stderr, "weight [%d x %d] q4_0_4x4 (%.1f MB)\n", sh[0], sh[1],
                (double)ggml_row_size(GGML_TYPE_Q4_0_4_4, sh[0]) * sh[1] / 1e6);
            for (int L : ls) bench(nthreads, sh[0], sh[1], L, "");
        }
        return 0;
    }

    // Exp812: the shapes the MAC census actually reports for a shipped 10 s clip at PIECES=1, biggest
    // first. 52% of the clip's blocked-int8 MACs live here. The VAE's early stages run one matmul per
    // linear over the STAGE's time axis, which at stage 0 is the 83,200-sample window itself.
    struct RealShape { int ne00, ne01, L; const char * why; };
    static const RealShape rs[] = {
        {32, 128, 83200, "stage0 fc1 (C=32 over the whole window)"},
        {128, 32, 83200, "stage0 fc2"},
        {64, 256, 41600, "stage1 fc1 (2x downsampled)"},
        {256, 64, 41600, "stage1 fc2"},
        {128, 512, 20800, "stage2 fc1"},
        {512, 128, 20800, "stage2 fc2"},
        {256, 1024, 5200, "stage3 fc1"},
        {1024, 256, 5200, "stage3 fc2"},
        {512, 2048, 1040, "stage4 fc1"},
        {2048, 512, 1040, "stage4 fc2"},
        {1024, 4096, 208, "stage5 fc1"},
        {4096, 1024, 208, "stage5 fc2"},
    };
    fprintf(stderr, "threads=%d  CENSUS SHAPES (the 52% bucket the classic grid never measured)\n", nthreads);
    double tot = 0;
    for (auto & s : rs) {
        const double g = bench(nthreads, s.ne00, s.ne01, s.L, s.why);
        if (g > 0) tot += (double)s.ne00 * s.ne01 * s.L / (g * 1e9);   // seconds at this rate
    }
    fprintf(stderr, "sum of these shapes at their measured rates = %.2f s of matmul time\n", tot);

    // Exp813: SAME shapes, but src1 pre-quantized to Q8_0 (what the conv path actually feeds). If the slow
    // few-rows regime speeds up dramatically, the cost was the F32->Q8_0 conversion inside mul_mat, not the
    // GEMM - and the lever is activation-quantization reuse, not column tiling.
    fprintf(stderr, "same census shapes with src1 = Q8_0 (GEMM alone, no conversion inside the op):\n");
    for (auto & s : rs) bench(nthreads, s.ne00, s.ne01, s.L, s.why, GGML_TYPE_Q8_0);

    // Exp812 ISOLATION: hold MACs CONSTANT at ~340 MMAC with ne00=128 and trade output rows against columns.
    // If time grows as ne01 falls (columns rise), the limit is re-reading the ACTIVATION tensor once per
    // output row - the GEMV-style access pattern - not the dot rate. That is the mechanism the census
    // anomaly above points at, and it would be fixable with column tiling in the vendored kernel.
    fprintf(stderr, "constant %g MMAC, ne00=128, rows x columns trade-off:\n", 128.0 * 32 * 83200 / 1e6);
    struct RC { int rows; int cols; };  
    static const RC rc[] = {{32, 83200}, {64, 41600}, {128, 20800}, {256, 10400},
                            {512, 5200}, {1024, 2600}, {2048, 1300}};
    for (auto & r : rc) {
        char note[64]; snprintf(note, sizeof note, "rows=%d cols=%d", r.rows, r.cols);
        bench(nthreads, 128, r.rows, r.cols, note);
    }
    fprintf(stderr, "constant %g MMAC with src1 = Q8_0 (does the gap survive without the conversion?):\n",
            128.0 * 32 * 83200 / 1e6);
    for (auto & r : rc) {
        char note[64]; snprintf(note, sizeof note, "rows=%d cols=%d", r.rows, r.cols);
        bench(nthreads, 128, r.rows, r.cols, note, GGML_TYPE_Q8_0);
    }
    return 0;
}
