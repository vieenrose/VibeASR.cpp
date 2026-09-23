/*
 * Conv-int8 converter: Create a quantized VAE model with 14 non-dw conv weights
 * padded to K=32 and quantized to Q4_0_4_4. DW convs stay F16. FFN already Q4_0_4_4.
 *
 * Build: g++ -std=c++17 -I3rdparty/llama.cpp/ggml/include -I3rdparty/llama.cpp/include \
 *   -Lbuild/3rdparty/llama.cpp/ggml/src -Wl,-rpath,build/3rdparty/llama.cpp/ggml/src \
 *   -lggml -o convert_vae_convint8_full convert_vae_convint8_full.cpp
 *
 * Usage: ./convert_vae_convint8_full <input.gguf> <output.gguf>
 */
#include <ggml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <algorithm>

// True kernel sizes for the 14 non-dw downsample convs (acoustic 0-6, semantic 0-6)
static const int TRUE_K_SIZES[14] = {8, 4, 4, 8, 16, 16, 16, 8, 4, 4, 8, 16, 16, 16};

static const std::unordered_set<std::string> TARGET_CONV_NAMES = {
    "acoustic.downsample_layers.0.0.conv.conv.weight",
    "acoustic.downsample_layers.1.0.conv.conv.weight",
    "acoustic.downsample_layers.2.0.conv.conv.weight",
    "acoustic.downsample_layers.3.0.conv.conv.weight",
    "acoustic.downsample_layers.4.0.conv.conv.weight",
    "acoustic.downsample_layers.5.0.conv.conv.weight",
    "acoustic.downsample_layers.6.0.conv.conv.weight",
    "semantic.downsample_layers.0.0.conv.conv.weight",
    "semantic.downsample_layers.1.0.conv.conv.weight",
    "semantic.downsample_layers.2.0.conv.conv.weight",
    "semantic.downsample_layers.3.0.conv.conv.weight",
    "semantic.downsample_layers.4.0.conv.conv.weight",
    "semantic.downsample_layers.5.0.conv.conv.weight",
    "semantic.downsample_layers.6.0.conv.conv.weight",
};

// Get tensor data as float from F16 source
static std::vector<float> read_tensor_as_f32(struct gguf_context* ctx, int idx, FILE* f_in, size_t data_offset) {
    enum ggml_type type = gguf_get_tensor_type(ctx, idx);
    size_t n_elements = gguf_get_tensor_n_elements(ctx, idx);
    size_t offset = data_offset + gguf_get_tensor_offset(ctx, idx);

    std::vector<float> f32_data(n_elements);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n_elements);
        fseek(f_in, offset, SEEK_SET);
        fread(buf.data(), sizeof(ggml_fp16_t), n_elements, f_in);
        for (size_t i = 0; i < n_elements; ++i) {
            f32_data[i] = ggml_fp16_to_fp32(buf[i]);
        }
    } else if (type == GGML_TYPE_F32) {
        fseek(f_in, offset, SEEK_SET);
        fread(f32_data.data(), sizeof(float), n_elements, f_in);
    } else {
        fprintf(stderr, "  Warning: tensor type %d not F16/F32, skipping conversion\n", type);
    }
    return f32_data;
}

// Get raw tensor data as bytes (for copying non-target tensors)
static std::vector<uint8_t> read_tensor_raw(struct gguf_context* ctx, int idx, FILE* f_in, size_t data_offset) {
    size_t n_bytes = gguf_get_tensor_n_bytes(ctx, idx);
    size_t offset = data_offset + gguf_get_tensor_offset(ctx, idx);
    std::vector<uint8_t> data(n_bytes);
    fseek(f_in, offset, SEEK_SET);
    fread(data.data(), 1, n_bytes, f_in);
    return data;
}

// Pad K dimension from K_orig to 32 in [OC, IC, K] layout (K fastest)
static void pad_k_dimension_f32(std::vector<float>& data, int K_orig, int IC, int OC) {
    if (K_orig >= 32) return;
    int K_new = 32;
    std::vector<float> new_data(OC * IC * K_new, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int k = 0; k < K_orig; ++k) {
                size_t src_idx = (oc * IC + ic) * K_orig + k;
                size_t dst_idx = (oc * IC + ic) * K_new + k;
                new_data[dst_idx] = data[src_idx];
            }
        }
    }
    data.swap(new_data);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf>\n", argv[0]);
        return 1;
    }

    const char* in_fname = argv[1];
    const char* out_fname = argv[2];

    fprintf(stderr, "Reading %s...\n", in_fname);

    // Open input GGUF
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context* ctx_in = gguf_init_from_file(in_fname, params);
    if (!ctx_in) {
        fprintf(stderr, "Failed to open input file\n");
        return 1;
    }

    int n_tensors = gguf_get_n_tensors(ctx_in);
    fprintf(stderr, "Input has %d tensors\n", n_tensors);

    // Open input file for raw reading
    FILE* f_in = fopen(in_fname, "rb");
    if (!f_in) {
        fprintf(stderr, "Failed to open input file for reading\n");
        return 1;
    }
    size_t data_offset = gguf_get_data_offset(ctx_in);

    // Create output GGUF
    struct gguf_context* ctx_out = gguf_init_empty();
    if (!ctx_out) {
        fprintf(stderr, "Failed to create output GGUF\n");
        return 1;
    }

    // Copy all metadata KV pairs from input to output
    gguf_set_kv(ctx_out, gguf_get_kv(ctx_in));

    // Add/update vae.kernel_size metadata
    gguf_set_arr_data(ctx_out, "vae.kernel_size", GGML_TYPE_INT32, TRUE_K_SIZES, 14);

    // Set file_type to indicate mixed quantization
    gguf_set_val_u32(ctx_out, "general.file_type", 99); // custom

    fprintf(stderr, "Processing %d tensors...\n", n_tensors);

    // Process each tensor
    int target_idx = 0;
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        enum ggml_type type_in = gguf_get_tensor_type(ctx_in, i);
        size_t n_elements = gguf_get_tensor_n_elements(ctx_in, i);
        size_t n_dims = gguf_get_tensor_n_dims(ctx_in, i);

        // Get logical shape (GGUF stores reversed: [ne0, ne1, ne2, ne3])
        int64_t ne[4] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims; ++d) {
            ne[d] = gguf_get_tensor_dim(ctx_in, i, d);
        }

        // Logical shape: [K, IC, OC] for 3D conv (ne0=K, ne1=IC, ne2=OC)
        int K_logical = (int)ne[0];
        int IC = (int)ne[1];
        int OC = n_dims >= 3 ? (int)ne[2] : 1;

        bool is_target = TARGET_CONV_NAMES.count(name) > 0;

        if (is_target && type_in == GGML_TYPE_F16) {
            // Target conv weight: pad K to 32, quantize to Q4_0_4_4
            int true_K = TRUE_K_SIZES[target_idx++];

            fprintf(stderr, "  [%d/%d] %s: F16 [K=%d, IC=%d, OC=%d] -> pad K=%d->32 -> Q4_0_4_4\n",
                    i+1, n_tensors, name, K_logical, IC, OC, K_logical);

            // Read F16 data as float
            std::vector<float> f32_data = read_tensor_as_f32(ctx_in, i, f_in, data_offset);

            // Pad K dimension: data layout is [OC, IC, K] (K fastest)
            pad_k_dimension_f32(f32_data, K_logical, IC, OC);

            // Quantize to Q4_0_4_4
            // For Q4_0_4_4: n_rows = OC, n_per_row = K * IC (must be multiple of 32)
            // After padding: K=32, so n_per_row = 32 * IC (multiple of 32 ✓)
            int n_rows = OC;
            int n_per_row = 32 * IC;

            // Verify size matches
            size_t expected_elements = (size_t)n_rows * n_per_row;
            if (f32_data.size() != expected_elements) {
                fprintf(stderr, "  ERROR: size mismatch for %s: expected %zu, got %zu\n",
                        name, expected_elements, f32_data.size());
                return 1;
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
                nullptr
            );

            // Add tensor to output with new shape [32, IC, OC] and Q4_0_4_4 type
            int64_t new_ne[4] = {32, IC, OC, 1};
            struct ggml_tensor* meta = ggml_new_tensor_4d(nullptr, GGML_TYPE_Q4_0_4_4, 32, IC, OC, 1);
            // Note: ggml_new_tensor_4d allocates memory; we just need the metadata
            // Actually, we can't easily create a ggml_tensor without a context.
            // Instead, use gguf_add_tensor with a dummy tensor... 
            // The GGUF API expects a ggml_tensor* with proper metadata.
            // Let's use a different approach: write tensor info directly.

            // For simplicity, let's create a tensor in a temporary context
            struct ggml_init_params gparams = { /*mem_size*/ 0, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
            struct ggml_context* tmp_ctx = ggml_init(gparams);
            struct ggml_tensor* t = ggml_new_tensor_4d(tmp_ctx, GGML_TYPE_Q4_0_4_4, 32, IC, OC, 1);
            ggml_set_name(t, name);
            // The data will be set via gguf_set_tensor_data

            gguf_add_tensor(ctx_out, t);
            gguf_set_tensor_data(ctx_out, name, q_data.data(), result);

            ggml_free(tmp_ctx);

            fprintf(stderr, "    Quantized: %d rows x %d cols -> %zu bytes (row_size=%zu)\n",
                    n_rows, n_per_row, result, ggml_row_size(GGML_TYPE_Q4_0_4_4, n_per_row));

        } else {
            // Non-target tensor: copy as-is
            std::vector<uint8_t> raw_data = read_tensor_raw(ctx_in, i, f_in, data_offset);

            // Get original shape
            int64_t ne_orig[4] = {ne[0], ne[1], ne[2], n_dims >= 4 ? ne[3] : 1};

            // Create tensor metadata
            struct ggml_init_params gparams = { 0, nullptr, true };
            struct ggml_context* tmp_ctx = ggml_init(gparams);
            struct ggml_tensor* t = ggml_new_tensor_4d(tmp_ctx, type_in, ne_orig[0], ne_orig[1], ne_orig[2], ne_orig[3]);
            ggml_set_name(t, name);

            gguf_add_tensor(ctx_out, t);
            gguf_set_tensor_data(ctx_out, name, raw_data.data(), raw_data.size());

            ggml_free(tmp_ctx);
        }
    }

    fprintf(stderr, "Writing %s...\n", out_fname);

    // Write output file
    int ret = gguf_write_to_file(ctx_out, out_fname, false);
    if (ret != 0) {
        fprintf(stderr, "Failed to write output file: %d\n", ret);
        return 1;
    }

    fprintf(stderr, "Done: %s\n", out_fname);

    // Cleanup
    gguf_free(ctx_in);
    gguf_free(ctx_out);
    fclose(f_in);

    return 0;
}