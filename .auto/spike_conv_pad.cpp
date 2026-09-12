// Spike for conv-int8 quantization (Exp684). Three questions, all answerable on the host:
//  (1) Is BACK-zero kernel padding exact? ggml's blocked types need the matmul row length (ne0) to
//      be a multiple of 32, but conv kernels are K=4/8/16. Padding the kernel dim with TRAILING
//      zeros gives ne0 = 32*IC (legal) and the extra taps multiply zero weights, so the math should
//      be exact at any stride. im2col derives KW from the weight's ne[0] and writes zeros for
//      out-of-range taps, so the output is the same conv 24 columns shorter -> right-pad the input.
//  (2) Row order: im2col's row is ic*KW + kw, and a [K,IC,OC] ggml tensor's memory order is kw
//      fastest then ic - the same order - so padding the fastest dim stays consistent for any KW.
//  (3) Does the blocked matmul accept a Q8_0 im2col against a 2-D [32*IC, OC] Q4_0_4_4 weight, and
//      how big is the error? NOTE ggml_im2col ASSERTS src0->type == F16 even though it only ever
//      reads src0's ne[] - so an int8 weight needs either an F16 geometry carrier or a one-line
//      assert relaxation. Here the carrier is a real F16 padded tensor, which is equivalent.
// ARM A  reference : im2col(w[K])      + mul_mat(w[K]      [K*IC,OC]  F16)
// ARM B  padded    : im2col(wpad[32])  + mul_mat(wpad[32]  [32*IC,OC] F16), input right-padded by 24
// ARM C  int8      : im2col(wpad[32]) cast to Q8_0 + mul_mat(w_q4 [32*IC,OC] Q4_0_4_4)
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cmath>
#include <cstring>
#include <vector>

static void fill_rand(float * d, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) { seed = seed * 1664525u + 1013904223u;
        d[i] = ((int)(seed >> 8) % 2000 - 1000) / 1000.0f; }
}

struct Arm { std::vector<float> y; int OL; bool ok = false; };

// geomK = kernel dim the im2col sees; wgeom = geomK*IC*OC f32 (becomes F16); wq = int8 matrix rows
static Arm run_arm(int geomK, int IC, int OC, int Lx, int extra_right_pad, int stride,
                   const std::vector<float> & xin, const std::vector<float> & wgeom,
                   const std::vector<float> * wq_flat) {
    Arm out;
    const size_t cap = 256ull << 20;
    char * mem = new char[cap];
    struct ggml_init_params p = { cap, mem, false };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) { delete[] mem; return out; }

    struct ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lx, IC);  // the graph pads, not the caller
    struct ggml_tensor * car = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, geomK, IC, OC);  // geometry carrier
    struct ggml_tensor * xs  = x;
    if (extra_right_pad) xs = ggml_pad_ext(ctx, x, 0, extra_right_pad, 0, 0, 0, 0, 0, 0);

    struct ggml_tensor * y;
    if (wq_flat) {
        // 2-D [geomK*IC, OC]: the blocked 4x4 layout interleaves groups of 4 along ne1, so the
        // tensor must be DECLARED as the matrix the matmul uses. Declaring it 3-D [K,IC,OC] and
        // reshaping is NOT byte-preserving in layout terms (the interleave groups differ) - which
        // is exactly what produced NaNs in the first run of this spike.
        struct ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_4_4, (int64_t) geomK * IC, OC);
        struct ggml_tensor * col = ggml_cast(ctx, ggml_im2col(ctx, car, xs, stride, 0, 0, 0, 1, 0, false,
                                                             GGML_TYPE_F16), GGML_TYPE_Q8_0);
        struct ggml_tensor * c2 = col;
        y = ggml_mul_mat(ctx, wq, c2);
        std::vector<char> buf(ggml_nbytes(wq));
        ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, wq_flat->data(), buf.data(), 0, OC,
                            (int64_t) geomK * IC, nullptr);
        memcpy(wq->data, buf.data(), buf.size());
    } else {
        struct ggml_tensor * col = ggml_im2col(ctx, car, xs, stride, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        struct ggml_tensor * c2 = ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]);
        if (getenv("SPIKE_DBG")) fprintf(stderr, "  [f16 ] xs.ne=[%lld,%lld] car.ne0=%lld col.ne=[%lld,%lld,%lld]\n",
            (long long)xs->ne[0], (long long)xs->ne[1], (long long)car->ne[0],
            (long long)col->ne[0], (long long)col->ne[1], (long long)col->ne[2]);
        y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, car, (int64_t) geomK * IC, OC), c2);
    }

    std::vector<uint16_t> h(wgeom.size());
    for (size_t i = 0; i < wgeom.size(); i++) h[i] = ggml_fp32_to_fp16(wgeom[i]);
    memcpy(car->data, h.data(), h.size() * sizeof(uint16_t));
    memcpy(x->data, xin.data(), xin.size() * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 2) == 0) {
        out.OL = (int) y->ne[1];
        out.y.resize((size_t) OC * out.OL);
        memcpy(out.y.data(), y->data, out.y.size() * sizeof(float));
        out.ok = true;
    }
    ggml_free(ctx);
    delete[] mem;
    return out;
}

int main(void) {
    int fail = 0;
    for (int stride : {1, 2}) {
        const int IC = 64, OC = 32, K = 8, L = 300, KP = 32;
        const int P = K - stride, Lx = L + P;
        std::vector<float> xin((size_t) Lx * IC), w((size_t) K * IC * OC);
        fill_rand(xin.data(), xin.size(), 12345);
        fill_rand(w.data(), w.size(), 999);
        // padded weight, memory order oc-major then ic then kw (kw fastest), zeros for kw >= K
        std::vector<float> wpad((size_t) KP * IC * OC, 0.f);
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kw = 0; kw < K; kw++)
                    wpad[(size_t) oc * KP * IC + (size_t) ic * KP + kw] =
                        w[(size_t) oc * K * IC + (size_t) ic * K + kw];
        // arms B/C get the SAME input as A; the right pad is done by the graph (zeros)

        Arm A = run_arm(K,  IC, OC, Lx, 0,      stride, xin,  w, nullptr);
        Arm B = run_arm(KP, IC, OC, Lx, KP - K, stride, xin, wpad, nullptr);
        Arm C = run_arm(KP, IC, OC, Lx, KP - K, stride, xin, wpad, &wpad);
        printf("stride=%d  ran: A=%d B=%d C=%d   lengths A=%d B=%d C=%d\n", stride,
               A.ok, B.ok, C.ok, A.OL, B.OL, C.OL);
        if (!A.ok || !B.ok || !C.ok) { fail = 1; continue; }
        if (A.OL != B.OL || A.OL != C.OL) { printf("  FAIL: output length changed\n"); fail = 1; continue; }
        double mx = 0, rms = 0, mxi = 0, rsi = 0, sig = 0; int bad = 0;
        for (size_t i = 0; i < A.y.size(); i++) {
            if (!std::isfinite(C.y[i])) { bad++; continue; }
            double d = A.y[i] - B.y[i], e = A.y[i] - C.y[i];
            mx = std::fabs(d) > mx ? std::fabs(d) : mx; rms += d * d;
            mxi = std::fabs(e) > mxi ? std::fabs(e) : mxi; rsi += e * e; sig += (double) A.y[i] * A.y[i];
        }
        double sigrms = sqrt(sig / A.y.size());
        printf("  padding, F16 vs F16 : max|d|=%.3e   rms/signal=%.2e   <- want ~1e-3 or less\n",
               mx, sqrt(rms / A.y.size()) / sigrms);
        printf("  int8 padded vs ref  : %s max|d|=%.3e   rms/signal=%.4f  (%.2f%%)\n",
               bad ? "NON-FINITE!" : "", mxi, sqrt(rsi / A.y.size()) / sigrms, 100.0 * sqrt(rsi / A.y.size()) / sigrms);
        if (sqrt(rms / A.y.size()) / sigrms > 5e-3) { printf("  FAIL: kernel padding is NOT exact\n"); fail = 1; }
        // EXPECTED on x86: q4_0_4x4 has .to_float = .from_float = .vec_dot = NULL in ggml.c's
        // traits table (only the aarch64 gemv/gemm are populated), so a host matmul over it reads
        // garbage. Quantizing works on x86 (that is how the shipped gguf was made), EVALUATING does
        // not. So arm C's numbers are informational only - the int8 error must be measured on the
        // device. Do not "fix" this arm; it cannot pass here.
        if (bad > (int) A.y.size() / 100) {
            printf("  (expected) int8 arm non-finite on x86: %d of %zu - blocked 4x4 has no x86 "
                   "vec_dot\n", bad, A.y.size());
        }
    }
    printf("%s\n", fail ? "SPIKE: PROBLEM (padding or shapes)"
                        : "SPIKE OK: kernel padding exact at stride 1 and 2; int8 numerics need the device");
    return fail;
}
