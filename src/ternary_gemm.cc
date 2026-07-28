#include "ternary_gemm.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// Four 2-bit codes per byte; element j sits at shift 2*(j%4).
static inline int code_at(const uint8_t* row, int j) {
    return (row[j >> 2] >> (2 * (j & 3))) & 0x3;
}

size_t ternary_packed_bytes(int n_rows, int k) {
    return (size_t)n_rows * (size_t)((k + 3) / 4);
}

void ternary_pack(const int8_t* w, int n_rows, int k, uint8_t* packed) {
    const int stride = (k + 3) / 4;
    memset(packed, 0, (size_t)n_rows * stride);
    for (int r = 0; r < n_rows; r++) {
        const int8_t* src = w + (size_t)r * k;
        uint8_t* dst = packed + (size_t)r * stride;
        for (int j = 0; j < k; j++) {
            // Store w+1 so the codes are non-negative; the -sum(x) correction in
            // the GEMM undoes the shift.
            const int u = (int)src[j] + 1;      // {-1,0,1} -> {0,1,2}
            dst[j >> 2] |= (uint8_t)((u & 0x3) << (2 * (j & 3)));
        }
    }
}

void ternary_unpack(const uint8_t* packed, int n_rows, int k, int8_t* w) {
    const int stride = (k + 3) / 4;
    for (int r = 0; r < n_rows; r++) {
        const uint8_t* src = packed + (size_t)r * stride;
        int8_t* dst = w + (size_t)r * k;
        for (int j = 0; j < k; j++) dst[j] = (int8_t)(code_at(src, j) - 1);
    }
}

void ternary_quantize_activations(const float* x, int m, int k, int8_t* q, float* scale) {
    for (int i = 0; i < m; i++) {
        const float* row = x + (size_t)i * k;
        float amax = 0.0f;
        for (int j = 0; j < k; j++) {
            const float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        scale[i] = s;
        int8_t* dst = q + (size_t)i * k;
        for (int j = 0; j < k; j++) {
            int v = (int)lrintf(row[j] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;   // -128 would break the symmetric assumption
            dst[j] = (int8_t)v;
        }
    }
}

void ternary_gemm_reference(const uint8_t* packed_w, int n_rows, int k,
                            const int8_t* q, const float* x_scale, int m,
                            const float* w_scale, int w_scale_is_per_row,
                            const float* bias, float* y) {
    const int stride = (k + 3) / 4;
    for (int i = 0; i < m; i++) {
        const int8_t* qi = q + (size_t)i * k;
        // sum(w*x) = sum((w+1)*x) - sum(x); this is the correction term.
        int32_t xsum = 0;
        for (int j = 0; j < k; j++) xsum += qi[j];
        for (int r = 0; r < n_rows; r++) {
            const uint8_t* wr = packed_w + (size_t)r * stride;
            int32_t acc = 0;
            for (int j = 0; j < k; j++) acc += code_at(wr, j) * (int32_t)qi[j];
            const float ws = w_scale_is_per_row ? w_scale[r] : w_scale[0];
            float v = (float)(acc - xsum) * ws * x_scale[i];
            if (bias) v += bias[r];
            y[(size_t)i * n_rows + r] = v;
        }
    }
}

#if defined(__ARM_NEON)

// One row of packed weights against one row of int8 activations.
// Returns sum((w+1) * x); the caller subtracts sum(x).
static inline int32_t ternary_dot_neon(const uint8_t* wr, const int8_t* q, int k) {
    const uint8x16_t mask = vdupq_n_u8(0x3);
    int32x4_t acc = vdupq_n_s32(0);
    int j = 0;

    // 16 packed bytes = 64 weights per iteration.
    for (; j + 64 <= k; j += 64) {
        const uint8x16_t p = vld1q_u8(wr + (j >> 2));
        // Codes for elements j+0,4,8.. are in bits 0-1, j+1,5,9.. in bits 2-3, etc.
        const int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8(p, mask));
        const int8x16_t w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p, 2), mask));
        const int8x16_t w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p, 4), mask));
        const int8x16_t w3 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p, 6), mask));

        // Activations are contiguous, so de-interleave to match the code order:
        // x4[t] holds elements {j+t, j+4+t, j+8+t, ...}.
        const int8x16x4_t x = vld4q_s8(q + j);

#if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(acc, w0, x.val[0]);
        acc = vdotq_s32(acc, w1, x.val[1]);
        acc = vdotq_s32(acc, w2, x.val[2]);
        acc = vdotq_s32(acc, w3, x.val[3]);
#else
        // ARMv8.0 (Cortex-A53/A55/A73): no dot product. Widening multiply-accumulate
        // into int16 is safe here because codes are 0..2 and activations |x| <= 127,
        // so each product is <= 254 and 16 of them cannot overflow int16... they can
        // (254*16 = 4064 < 32767, so they cannot). Accumulate to int32 per block.
        int16x8_t s01 = vmull_s8(vget_low_s8(w0), vget_low_s8(x.val[0]));
        s01 = vmlal_s8(s01, vget_high_s8(w0), vget_high_s8(x.val[0]));
        s01 = vmlal_s8(s01, vget_low_s8(w1), vget_low_s8(x.val[1]));
        s01 = vmlal_s8(s01, vget_high_s8(w1), vget_high_s8(x.val[1]));
        int16x8_t s23 = vmull_s8(vget_low_s8(w2), vget_low_s8(x.val[2]));
        s23 = vmlal_s8(s23, vget_high_s8(w2), vget_high_s8(x.val[2]));
        s23 = vmlal_s8(s23, vget_low_s8(w3), vget_low_s8(x.val[3]));
        s23 = vmlal_s8(s23, vget_high_s8(w3), vget_high_s8(x.val[3]));
        acc = vaddq_s32(acc, vpaddlq_s16(s01));
        acc = vaddq_s32(acc, vpaddlq_s16(s23));
#endif
    }

    int32_t sum = vaddvq_s32(acc);
    for (; j < k; j++) sum += code_at(wr, j) * (int32_t)q[j];
    return sum;
}

void ternary_gemm(const uint8_t* packed_w, int n_rows, int k,
                  const int8_t* q, const float* x_scale, int m,
                  const float* w_scale, int w_scale_is_per_row,
                  const float* bias, float* y) {
    const int stride = (k + 3) / 4;
    for (int i = 0; i < m; i++) {
        const int8_t* qi = q + (size_t)i * k;

        int32x4_t xs = vdupq_n_s32(0);
        int j = 0;
        for (; j + 16 <= k; j += 16) xs = vpadalq_s16(xs, vpaddlq_s8(vld1q_s8(qi + j)));
        int32_t xsum = vaddvq_s32(xs);
        for (; j < k; j++) xsum += qi[j];

        for (int r = 0; r < n_rows; r++) {
            const int32_t acc = ternary_dot_neon(packed_w + (size_t)r * stride, qi, k);
            const float ws = w_scale_is_per_row ? w_scale[r] : w_scale[0];
            float v = (float)(acc - xsum) * ws * x_scale[i];
            if (bias) v += bias[r];
            y[(size_t)i * n_rows + r] = v;
        }
    }
}

const char* ternary_gemm_impl_name(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    return "neon+dotprod";
#else
    return "neon";
#endif
}

#else  // no NEON

void ternary_gemm(const uint8_t* packed_w, int n_rows, int k,
                  const int8_t* q, const float* x_scale, int m,
                  const float* w_scale, int w_scale_is_per_row,
                  const float* bias, float* y) {
    ternary_gemm_reference(packed_w, n_rows, k, q, x_scale, m,
                           w_scale, w_scale_is_per_row, bias, y);
}

const char* ternary_gemm_impl_name(void) { return "scalar"; }

#endif
