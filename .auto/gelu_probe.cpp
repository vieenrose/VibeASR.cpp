// gelu rate probe (Exp780b). WHY: the shipped graph moves 980 MB per build through ggml_gelu and the
// ablation prices that pass at 2.1 s = 13.3% of VAE seconds (Exp780), a rate of only 233 Melem/s per
// core (~8.6 cycles/element). The ledger's largest remaining in-scope item.
//
// The function is defined by a LOOKUP: under GGML_GELU_FP16, gelu(x) = table[f16bits(x)] with two
// boundary branches. A polynomial is NOT equivalent (that is the gelu_quick lesson, Exp601/626), so any
// speedup must keep the same table and the same branches - only the ISSUE ORDER may change.
//
// Variants, all claiming bit-identity:
//   A  ggml's loop verbatim            - the baseline; also validates that this file's replication is
//                                        faithful, which must be true before any variant means anything
//   B  4 elements in flight            - four independent gathers per iteration, so four L2 latencies
//                                        overlap instead of one at a time
//   C  B + explicit __builtin_prefetch on the next iteration's table entries
//   D  branch-light: clamp the index instead of branching (claiming equality is TESTED here, not
//      assumed - the two branches exist to avoid a table entry for saturated inputs)
#include "ggml.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void vA(int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) y[i] = 0.0f;
        else if (x[i] >= 10.0f) y[i] = x[i];
        else { uint16_t t; ggml_fp16_t h = GGML_FP32_TO_FP16(x[i]); memcpy(&t, &h, 2); y[i] = GGML_FP16_TO_FP32(ggml_table_gelu_f16[t]); }
    }
}
static void vB(int n, float * y, const float * x) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16_t t[4]; float v[4];
        v[0] = x[i]; v[1] = x[i+1]; v[2] = x[i+2]; v[3] = x[i+3];
        for (int k = 0; k < 4; k++) { ggml_fp16_t h = GGML_FP32_TO_FP16(v[k]); memcpy(&t[k], &h, 2); }
        float r[4];
        for (int k = 0; k < 4; k++) r[k] = GGML_FP16_TO_FP32(ggml_table_gelu_f16[t[k]]);
        for (int k = 0; k < 4; k++) y[i+k] = v[k] <= -10.0f ? 0.0f : (v[k] >= 10.0f ? v[k] : r[k]);
    }
    vA(n - i, y + i, x + i);
}
static void vC(int n, float * y, const float * x) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        if (i + 12 < n) for (int k = 4; k < 12; k++) __builtin_prefetch(x + i + k, 0, 0);
        uint16_t t[4]; float v[4];
        v[0] = x[i]; v[1] = x[i+1]; v[2] = x[i+2]; v[3] = x[i+3];
        for (int k = 0; k < 4; k++) { ggml_fp16_t h = GGML_FP32_TO_FP16(v[k]); memcpy(&t[k], &h, 2); }
        float r[4];
        for (int k = 0; k < 4; k++) r[k] = GGML_FP16_TO_FP32(ggml_table_gelu_f16[t[k]]);
        for (int k = 0; k < 4; k++) y[i+k] = v[k] <= -10.0f ? 0.0f : (v[k] >= 10.0f ? v[k] : r[k]);
    }
    vA(n - i, y + i, x + i);
}
static void vD(int n, float * y, const float * x) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16_t t[4]; float v[4]; int oob[4];
        v[0] = x[i]; v[1] = x[i+1]; v[2] = x[i+2]; v[3] = x[i+3];
        for (int k = 0; k < 4; k++) { ggml_fp16_t h = GGML_FP32_TO_FP16(v[k]); memcpy(&t[k], &h, 2); oob[k] = 0; }
        float r[4];
        for (int k = 0; k < 4; k++) r[k] = GGML_FP16_TO_FP32(ggml_table_gelu_f16[t[k]]);
        for (int k = 0; k < 4; k++) { if (v[k] <= -10.0f) r[k] = 0.0f; else if (v[k] >= 10.0f) { r[k] = v[k]; oob[k] = 1; } }
        (void) oob; y[i] = r[0]; y[i+1] = r[1]; y[i+2] = r[2]; y[i+3] = r[3];
    }
    vA(n - i, y + i, x + i);
}

int main(int argc, char ** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
    const int reps = argc > 2 ? atoi(argv[2]) : 12;
    std::vector<float> x(n), ref(n), y(n);
    for (int i = 0; i < n; i++) x[i] = 4.0f * std::sin(0.0013f * i) + 0.3f * std::cos(0.37f * i) + 0.02f * ((i * 2654435761u) % 1000) / 1000.0f;

    // ground truth = ggml's own unary gelu node, one thread (what inference runs)
    {
        struct ggml_init_params p = { (size_t)640 << 20, nullptr, false };
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
    printf("n=%d  (%.1f MB tensor, like one VAE gelu node)\n", n, 4.0 * n / 1e6);

    struct { const char * name; void (*fn)(int, float*, const float*); } arms[] =
        { {"A ggml loop", vA}, {"B batch4", vB}, {"C batch4+prefetch", vC}, {"D branch-light", vD} };
    for (auto & a : arms) {
        vA(n, y.data(), x.data());
        long dref = 0; for (int i = 0; i < n; i++) if (memcmp(&y[i], &ref[i], 4)) dref++;
        a.fn(n, y.data(), x.data());
        long dvar = 0; int64_t first = -1;
        for (int i = 0; i < n; i++) if (memcmp(&y[i], &ref[i], 4)) { if (first < 0) first = i; dvar++; }
        a.fn(n, y.data(), x.data());
        double t0 = now_s();
        for (int r = 0; r < reps; r++) a.fn(n, y.data(), x.data());
        double dt = (now_s() - t0) / reps;
        printf("%-18s %6.2f ms  %6.0f Melem/s  %5.2f GB/s  vs-ggml-node diffs=%ld", a.name, dt * 1e3, n / dt / 1e6, 8.0 * n / dt / 1e9, dvar);
        printf("%s\n", dvar ? (dref ? "  (BASELINE ITSELF DIFFERS - replication invalid)" : "  NOT identical") : "  BIT-IDENTICAL");
    }
    return 0;
}
