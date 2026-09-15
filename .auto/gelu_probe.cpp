// gelu rate probe (Exp780). WHY: the shipped gelu is GGML_GELU_FP16 - a SCALAR loop that converts each
// f32 to f16 and gathers one entry from ggml's gelu table (65536 entries x 2 B = 128 KB). 128 KB does
// not fit an A78's 48 KB L1, so every element pays an L2 access. The shipped graph moves 980 MB per
// build through this loop (26 nodes, Exp778 profile) and the ablation prices it at 13.3% of VAE
// seconds - the largest single non-matmul item found since Exp662. Rate: 233 Melem/s/thread.
//
// Two candidate fixes are tested, both of which must be BIT-IDENTICAL because the table lookup *is*
// the function (a polynomial would be "more accurate" but is a different function - that is the
// gelu_quick lesson, Exp601/626):
//   A. batched issues: convert 4 f32->f16 with the FP16 vector unit, then do 4 independent table
//      gathers per iteration so their L2 latencies overlap (the scalar loop's gathers are limited by
//      the compiler's ability to keep loads in flight);
//   B. A + a table prefetch. The FFN's values cluster, so the hot table region is small, but the
//      access order follows the data, not the table.
//
// Reference is ggml's OWN unary gelu node, computed here through ggml so the comparison is against
// what inference actually produces (not against a re-derivation - Exp672's lesson about reconstructing
// a library routine at the call site).
#include "ggml.h"
#include <cmath>
#include <cstdint>
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

// The table, built exactly as ggml does at init: entry[t] = f16( gelu_f32( f16_to_f32(t) ) ).
// If this disagrees with ggml's own node anywhere, every conclusion below is void - so it is checked.
static std::vector<uint16_t> make_table() {
    std::vector<uint16_t> tab(65536);
    for (long t = 0; t < 65536; t++) {
        ggml_fp16_t h = (ggml_fp16_t)(uint16_t)t;
        float x = ggml_fp16_to_fp32(h);
        float g = 0.5f * x * (1.0f + erff(x * 0.7978845608f));
        tab[t] = (uint16_t) ggml_fp32_to_fp16(g);
    }
    return tab;
}

static void gelu_scalar(const int64_t n, float * y, const float * x, const uint16_t * tab) {
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) y[i] = 0.0f;
        else if (x[i] >= 10.0f) y[i] = x[i];
        else { ggml_fp16_t h = ggml_fp32_to_fp16(x[i]); uint16_t t; memcpy(&t, &h, 2); y[i] = ggml_fp16_to_fp32((ggml_fp16_t) tab[t]); }
    }
}

// Variant A: four elements in flight, same branches and same table entries.
static void gelu_batch4(const int64_t n, float * y, const float * x, const uint16_t * tab) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16_t t[4];
        float v[4];
        for (int k = 0; k < 4; k++) v[k] = x[i + k];
        for (int k = 0; k < 4; k++) { ggml_fp16_t h = ggml_fp32_to_fp16(v[k]); memcpy(&t[k], &h, 2); }
        float r[4];
        for (int k = 0; k < 4; k++) r[k] = ggml_fp16_to_fp32((ggml_fp16_t) tab[t[k]]);   // 4 gathers issued together
        for (int k = 0; k < 4; k++) y[i + k] = v[k] <= -10.0f ? 0.0f : (v[k] >= 10.0f ? v[k] : r[k]);
    }
    gelu_scalar(n - i, y + i, x + i, tab);
}

int main(int argc, char ** argv) {
    const int64_t n = argc > 1 ? atoll(argv[1]) : (1LL << 22);   // 4.19M elements = one real gelu node
    const int reps = argc > 2 ? atoi(argv[2]) : 20;
    std::vector<uint16_t> tab = make_table();

    // ground truth from ggml's own graph node
    std::vector<float> x(n), ref(n), y(n);
    for (int64_t i = 0; i < n; i++) x[i] = 3.0f * std::sin(0.0013f * i) + 0.25f * std::cos(0.37f * i);
    {
        struct ggml_init_params p = { (size_t)512 << 20, nullptr, false };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        memcpy(t->data, x.data(), (size_t) n * 4);
        struct ggml_tensor * o = ggml_gelu(ctx, t);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { printf("gelu node FAILED\n"); return 1; }
        memcpy(ref.data(), o->data, (size_t) n * 4);
        ggml_free(ctx);
    }

    gelu_scalar(n, y.data(), x.data(), tab.data());
    long diff_s = 0; for (int64_t i = 0; i < n; i++) if (memcmp(&y[i], &ref[i], 4)) diff_s++;
    printf("local table vs ggml node: %ld/%lld differ %s\n", diff_s, (long long) n, diff_s ? "<= TABLE MISMATCH, STOP" : "(table OK)");
    gelu_batch4(n, y.data(), x.data(), tab.data());
    long diff_a = 0; for (int64_t i = 0; i < n; i++) if (memcmp(&y[i], &ref[i], 4)) diff_a++;
    printf("batch4 vs ggml node:      %ld/%lld differ  %s\n", diff_a, (long long) n, diff_a ? "NOT IDENTICAL" : "BIT-IDENTICAL");

    struct { const char * name; void (*fn)(int64_t, float*, const float*, const uint16_t*); } arms[] =
        { { "scalar(ggml-shaped)", gelu_scalar }, { "batch4", gelu_batch4 } };
    for (auto & a : arms) {
        a.fn(n, y.data(), x.data(), tab.data());                      // warm
        double t0 = now_s();
        for (int r = 0; r < reps; r++) a.fn(n, y.data(), x.data(), tab.data());
        double dt = (now_s() - t0) / reps;
        printf("%-20s %7.2f ms  %7.0f Melem/s  %5.2f GB/s(touched)\n", a.name, dt * 1e3, n / dt / 1e6, 8.0 * n / dt / 1e9);
    }
    return 0;
}
