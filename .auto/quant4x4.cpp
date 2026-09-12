// quant4x4 <in.bin> <byte_offset> <ne> <nrows> <out.bin>
//
// Converter helper for conv-int8 quantization (Exp685). Reads ne*nrows ggml_half values in ggml
// tensor memory order (column j contiguous with n_per_row = ne), converts to f32, and writes the
// blocked Q4_0_4x4 blob for that [ne, nrows] matrix.
//
// ggml_quantize_chunk works on x86 for this type even though its to_float/from_float/vec_dot are
// NULL there (that is why the shipped ggufs could be built here at all) - quantizing is portable,
// EVALUATING is aarch64-only. So this tool is a host tool and the numbers it produces can only be
// judged on the device.
//
// Row-order note (the part that must not be gotten wrong): a 3-D [K,IC,OC] F16 conv tensor already
// has the memory layout of the 2-D [K*IC, OC] matrix that im2col expects, because both are ordered
// kw-fastest-then-ic. So no reordering happens here: the bytes are read straight through.
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: quant4x4 in.bin byte_offset ne nrows out.bin\n");
        return 2;
    }
    const long off = atol(argv[2]);
    const int64_t ne = atoll(argv[3]), nrows = atoll(argv[4]);

    FILE * in = fopen(argv[1], "rb");
    if (!in || fseek(in, off, SEEK_SET) != 0) { perror("open/seek"); return 1; }
    const size_t n = (size_t) ne * (size_t) nrows;
    std::vector<uint16_t> h(n);
    if (fread(h.data(), 2, n, in) != n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(in);

    std::vector<float> src(n);
    for (size_t i = 0; i < n; i++) src[i] = ggml_fp16_to_fp32(h[i]);

    const size_t rs = ggml_row_size(GGML_TYPE_Q4_0_4_4, ne);
    std::vector<char> dst(rs * (size_t) nrows);
    const size_t wrote = ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, src.data(), dst.data(), 0, nrows, ne,
                                             nullptr);
    if (wrote != dst.size()) { fprintf(stderr, "quantize wrote %zu, expected %zu\n", wrote, dst.size()); return 1; }

    FILE * out = fopen(argv[5], "wb");
    if (!out || fwrite(dst.data(), 1, dst.size(), out) != dst.size()) { perror("write"); return 1; }
    fclose(out);
    printf("%zu\n", dst.size());
    return 0;
}
