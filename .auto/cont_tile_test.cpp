// Exp863 oracle for the blocked transpose in ggml_compute_forward_dup_bytes.
//
// Checks cont(permute(x, 1, 0, 2, 3)) against the DEFINITION, not against another code path:
// x is [C, T] (channels innermost), the permuted view is [T, C] with nb0 = C*4, nb1 = 4, and the
// contiguous result must satisfy dst[t*C + c] == x[c*T + t] for every element. Shapes are chosen to
// exercise both remainder strips (ne00 % 4 = T % 4 and ne01 % 4 = C % 4) plus the exact geometries the
// VAE's stage boundary produces (see the GGML_CONT_TILE_DBG dump).
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

static int check(int64_t C, int64_t T) {
    const int64_t n = C * T;
    std::vector<float> in(n);
    for (int64_t i = 0; i < n; i++) in[i] = (float)std::sin(0.001 * i) * 1e3f + (float)(i % 7919) * 0.25f;

    // two tensors (x and the contiguous result) live in this pool, plus node overhead
    size_t sz = (size_t)(2 * n * (int64_t)sizeof(float)) + (1 << 20);
    std::vector<char> buf(sz);
    ggml_init_params p{ sz, buf.data(), false };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    memcpy(x->data, in.data(), n * sizeof(float));
    ggml_tensor * y = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // DEFINITION, stated once and independently of any kernel: x is [C, T] so x(a, b) = in[b*C + a];
    // the permuted view is v(i00, i01) = x(i01, i00) = in[i00*C + i01]; and dst is [T, C] contiguous,
    // so dst(i00, i01) sits at flat i01*T + i00. Hence d[j*T + i] must equal in[i*C + j].
    // (The first version of this test wrote in[c*T + t], i.e. it scored the result as if it were [C, T]
    // row-major - Exp654's class: the bug was in the scorer, not the system.)
    if (getenv("CTDP")) printf("  ptrs: x=%p  src_view=%p  y=%p  (view data == x data? %s)\n",
        x->data, y->src[0]->data, y->data, y->src[0]->data == x->data ? "yes" : "no");
    const float * d = (const float *) y->data;
    int64_t bad = 0, first_c = -1, first_t = -1;
    for (int64_t j = 0; j < C && bad < 4; j++)
        for (int64_t i = 0; i < T && bad < 4; i++)
            if (d[j*T + i] != in[i*C + j]) { if (first_c < 0) { first_c = j; first_t = i; } bad++; }
    int64_t total_bad = 0;
    for (int64_t j = 0; j < C; j++) for (int64_t i = 0; i < T; i++) if (d[j*T + i] != in[i*C + j]) total_bad++;
    bad = total_bad;

    printf("%s C=%lld T=%lld  ne0=%lld ne1=%lld nb0=%lld nb1=%lld  bad=%lld",
           bad ? "FAIL" : "ok  ", (long long)C, (long long)T,
           (long long)y->src[0]->ne[0], (long long)y->src[0]->ne[1],
           (long long)y->src[0]->nb[0], (long long)y->src[0]->nb[1], (long long)bad);
    if (bad) printf("  first mismatch at (i01=%lld,i00=%lld): got %.9g want %.9g",
                    (long long)first_c, (long long)first_t,
                    (double)d[first_c*T + first_t], (double)in[first_t*C + first_c]);
    printf("\n");
    if (getenv("CTD") && C <= 8 && T <= 8) {   // show the actual permutation, not an aggregate
        printf("  src (C=%lld x T=%lld), element (a=C idx, b=T idx) = in[b*C + a]:\n", (long long)C, (long long)T);
        for (int64_t b = 0; b < T; b++) { printf("    b=%lld:", (long long)b);
            for (int64_t a = 0; a < C; a++) printf(" %4.0f", in[b*C + a]); printf("\n"); }
        printf("  dst flat (row j of [T,C], lane i of i00):\n");
        for (int64_t j = 0; j < C; j++) { printf("    j=%lld:", (long long)j);
            for (int64_t i = 0; i < T; i++) printf(" %4.0f", d[j*T + i]); printf("\n"); }
    }
    ggml_free(ctx);
    return bad != 0;
}

int main(void) {
    int fails = 0;
    struct { int64_t C, T; } shapes[] = {
        {4, 4}, {32, 1000}, {64, 10400}, {512, 1040}, {2048, 26}, {1024, 208}, {128, 20800},
        {33, 1001}, {4, 8}, {1, 5}, {5, 1}, {64, 26}, {64, 27}, {64, 25}, {65, 64}, {66, 64}, {67, 64},
    };
    for (auto & s : shapes) fails += check(s.C, s.T);
    printf("%s\n", fails ? "SOME SHAPES FAILED" : "all shapes match the definition");
    return fails ? 1 : 0;
}
