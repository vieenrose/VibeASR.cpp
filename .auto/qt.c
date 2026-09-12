// Exp688: is ggml_quantize_chunk(Q4_0_4_4) a no-op on x86? Quantize a known matrix and look.
#include <ggml.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    float src[128];
    for (int j = 0; j < 4; j++) for (int i = 0; i < 32; i++) src[j*32+i] = 0.1f*((i*5+j*3) % 21 - 10);
    unsigned char dst[4096];
    memset(dst, 0xAB, sizeof dst);
    size_t w = ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, src, dst, 0, 4, 32, NULL);
    int nz = 0;
    for (size_t i = 0; i < w; i++) if (dst[i] != 0xAB && dst[i] != 0) nz++;
    printf("wrote %zu bytes; bytes actually changed from the 0xAB marker and non-zero: %d\n", w, nz);
    printf("first block d = {%.6g, %.6g, %.6g, %.6g}  qs[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           ggml_fp16_to_fp32(((ggml_fp16_t *)dst)[0]), ggml_fp16_to_fp32(((ggml_fp16_t *)dst)[1]),
           ggml_fp16_to_fp32(((ggml_fp16_t *)dst)[2]), ggml_fp16_to_fp32(((ggml_fp16_t *)dst)[3]),
           dst[8], dst[9], dst[10], dst[11], dst[12], dst[13], dst[14], dst[15]);
    // and the ordinary type, as a sanity check that quantization works here at all
    size_t w2 = ggml_quantize_chunk(GGML_TYPE_Q4_0, src, dst, 0, 4, 32, NULL);
    memset(dst, 0xAB, sizeof dst);
    w2 = ggml_quantize_chunk(GGML_TYPE_Q4_0, src, dst, 0, 4, 32, NULL);
    nz = 0;
    for (size_t i = 0; i < w2; i++) if (dst[i] != 0xAB && dst[i] != 0) nz++;
    printf("Q4_0 (plain): wrote %zu bytes, %d informative bytes -> %s\n", w2, nz, nz ? "works" : "NOT WRITING");
    return 0;
}
