/*
 * Quantize a single F16 tensor file to Q4_0_4_4.
 * Usage: ./quantize_q4_0_4_4 <input_f16.bin> <output_q4_0_4_4.bin> <n_rows> <n_per_row>
 * Input: raw F16 data (float16) with layout [n_rows, n_per_row] (row-major)
 * Output: Q4_0_4_4 quantized data
 */
#include <ggml.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_f16.bin> <output_q4_0_4_4.bin> <n_rows> <n_per_row>\n", argv[0]);
        return 1;
    }

    const char* in_fname = argv[1];
    const char* out_fname = argv[2];
    int n_rows = atoi(argv[3]);
    int n_per_row = atoi(argv[4]);

    // Read F16 input
    FILE* f_in = fopen(in_fname, "rb");
    if (!f_in) { fprintf(stderr, "Failed to open %s\n", in_fname); return 1; }
    fseek(f_in, 0, SEEK_END);
    size_t fsize = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    size_t expected_f16 = n_rows * n_per_row * sizeof(ggml_fp16_t);
    if (fsize != expected_f16) {
        fprintf(stderr, "Input size mismatch: expected %zu, got %zu\n", expected_f16, fsize);
        return 1;
    }

    std::vector<ggml_fp16_t> f16_data(n_rows * n_per_row);
    fread(f16_data.data(), sizeof(ggml_fp16_t), n_rows * n_per_row, f_in);
    fclose(f_in);

    // Convert to float
    std::vector<float> f32_data(n_rows * n_per_row);
    for (size_t i = 0; i < f32_data.size(); ++i) {
        f32_data[i] = ggml_fp16_to_fp32(f16_data[i]);
    }

    // Quantize to Q4_0_4_4
    ggml_quantize_init(GGML_TYPE_Q4_0_4_4);
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_0_4_4, n_per_row);
    std::vector<uint8_t> q_data(n_rows * row_size);

    size_t result = ggml_quantize_chunk(
        GGML_TYPE_Q4_0_4_4,
        f32_data.data(),
        q_data.data(),
        0,  // start
        n_rows,
        n_per_row,
        nullptr  // no imatrix
    );

    fprintf(stderr, "Quantized %d rows x %d cols -> %zu bytes (row_size=%zu)\n", n_rows, n_per_row, result, row_size);

    // Write output
    FILE* f_out = fopen(out_fname, "wb");
    if (!f_out) { fprintf(stderr, "Failed to open %s\n", out_fname); return 1; }
    fwrite(q_data.data(), 1, result, f_out);
    fclose(f_out);

    return 0;
}