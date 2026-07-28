#include <cstdlib>
#include "vae.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vae-i8_s-mad.h"

#include "time_compat.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif


static struct ggml_tensor* ggml_nn_rms_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma) {
    
    if (x->type == GGML_TYPE_I8_S) {
        x = ggml_rms_norm_scaled(ctx, x, gamma, 1e-5f);
    } else {
        x = ggml_rms_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, gamma);
    }
    
    return x;
}

static struct ggml_tensor* ggml_nn_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b) {
    
    int64_t IC = x->ne[0];
    int64_t N = x->ne[1];
    int64_t L = x->ne[2];
    int64_t OC = w->ne[1];
    
    x = ggml_reshape_2d(ctx, x, IC, L * N);
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        // I8_S fused path
        result = ggml_mul_mat_add(ctx, w, x, b);
    } else {
        result = ggml_mul_mat(ctx, w, x);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
    }
    
    result = ggml_reshape_3d(ctx, result, OC, L, N);

    return result;
}

static struct ggml_tensor* ggml_nn_linear_relu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b) {

    int64_t IC = x->ne[0];
    int64_t N = x->ne[1];
    int64_t L = x->ne[2];
    int64_t OC = w->ne[1];

    x = ggml_reshape_2d(ctx, x, IC, L * N);

    struct ggml_tensor* result;

    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        result = ggml_mul_mat_add_relu(ctx, w, x, b);
    } else {
        result = ggml_mul_mat(ctx, w, x);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
        result = ggml_relu(ctx, result);
    }

    result = ggml_reshape_3d(ctx, result, OC, L, N);

    return result;
}

static struct ggml_tensor* ggml_nn_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation) {
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        struct ggml_tensor* im2col = ggml_im2col_asym(ctx, w, x, stride, 0,
                                                       /*lp0=*/padding, /*rp0=*/0, /*p1=*/0,
                                                       dilation, 0, false, GGML_TYPE_I8_S);
        
        result = ggml_mul_mat_add(ctx,
                ggml_reshape_2d(ctx, w, (w->ne[0] * w->ne[1]), w->ne[2]),
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])),
                b);
    } else {
        if (padding > 0) {
            x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
            padding = 0;
        }
        result = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
    }
    
    return result;
}

static struct ggml_tensor* ggml_nn_conv_1d_dw(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation) {
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        struct ggml_tensor* new_x = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
        struct ggml_tensor* im2col = ggml_im2col_asym(ctx, w, new_x, stride, 0,
                                                       padding, 0, 0, dilation, 0, false, GGML_TYPE_I8_S);
        result = ggml_mul_mat_add(ctx, w, im2col, b);
        result = ggml_reshape_3d(ctx, result, result->ne[1], result->ne[2], 1);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
    } else {
        if (padding > 0) {
            x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
            padding = 0;
        }
        result = ggml_conv_1d_dw(ctx, w, x, stride, padding, dilation);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
    }
    
    return result;
}

static struct ggml_tensor* ggml_nn_layer_scale(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma) {
    return ggml_mul(ctx, x, gamma);
}

//
// ConvNeXt Block
//

struct ConvNeXtBlock {
    // Mixer (Depthwise Conv)
    struct ggml_tensor* mixer_norm_weight;
    struct ggml_tensor* mixer_conv_weight;
    struct ggml_tensor* mixer_conv_bias;
    struct ggml_tensor* mixer_layer_scale;
    
    // FFN
    struct ggml_tensor* ffn_norm_weight;
    struct ggml_tensor* ffn_fc1_weight;
    struct ggml_tensor* ffn_fc1_bias;
    struct ggml_tensor* ffn_fc2_weight;
    struct ggml_tensor* ffn_fc2_bias;
    struct ggml_tensor* ffn_layer_scale;
    
    int dim;
    int kernel_size;
    
    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x) {

        struct ggml_tensor* residual = x;
        bool is_i8s = (x->type == GGML_TYPE_I8_S);

        x = ggml_nn_rms_norm(ctx, x, mixer_norm_weight);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        x = ggml_nn_conv_1d_dw(ctx, x, mixer_conv_weight, mixer_conv_bias,
                                /*stride=*/1, /*padding=*/kernel_size-1, /*dilation=*/1);

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, mixer_layer_scale);
        } else {
            // F32 path: x = x * layer_scale + residual
            x = ggml_mul(ctx, x, mixer_layer_scale);
            x = ggml_add(ctx, x, residual);
        }
        
        residual = x;
        
        x = ggml_nn_rms_norm(ctx, x, ffn_norm_weight);
        
        if (is_i8s) {
            x = ggml_nn_linear_relu(ctx, x, ffn_fc1_weight, ffn_fc1_bias);
        } else {
            x = ggml_nn_linear(ctx, x, ffn_fc1_weight, ffn_fc1_bias);
            x = ggml_gelu(ctx, x);
        }
        
        x = ggml_nn_linear(ctx, x, ffn_fc2_weight, ffn_fc2_bias);

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, ffn_layer_scale);
        } else {
            x = ggml_mul(ctx, x, ffn_layer_scale);
            x = ggml_add(ctx, x, residual);
        }
        
        return x;
    }
};

//
// VAE Encoder
//

struct AudioVAEEncoder {
    // Architecture configuration
    static const int n_downsamples = 7;
    static const int downsample_strides[n_downsamples];
    static const int downsample_dims[n_downsamples];
    static const int n_stages = 7;
    static const int stage_depths[n_stages];
    
    // Kernel sizes will be read from actual weights during loading
    int downsample_kernel_sizes[n_downsamples];
    int stage_kernel_sizes[n_stages];
    
    int output_dim;
    int connector_output_dim;
    
    // Downsamples 0-6 (just conv, no norm)
    struct {
        struct ggml_tensor* conv_weight;
        struct ggml_tensor* conv_bias;
    } downsamples[n_downsamples];
    
    // Stages (ConvNeXt blocks)
    std::vector<ConvNeXtBlock> stages[n_stages];
    
    // Head (just conv)
    struct ggml_tensor* head_conv_weight;
    struct ggml_tensor* head_conv_bias;
    
    // Connector (fc1 -> norm -> fc2)
    struct ggml_tensor* connector_fc1_weight;
    struct ggml_tensor* connector_fc1_bias;
    struct ggml_tensor* connector_norm_weight;
    struct ggml_tensor* connector_fc2_weight;
    struct ggml_tensor* connector_fc2_bias;
    
    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x) {
        
        // Downsamples and stages
        for (int i = 0; i < n_stages; i++) {

            x = ggml_nn_conv_1d(ctx, x, downsamples[i].conv_weight,
                                 downsamples[i].conv_bias,
                                 downsample_strides[i], downsample_kernel_sizes[i]-downsample_strides[i], 1);
            
            for (int j = 0; j < stage_depths[i]; j++) {
                x = stages[i][j].forward(ctx, x);
            }

            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        }
        
        // Head
        x = ggml_nn_conv_1d(ctx, x, head_conv_weight, head_conv_bias, 1, 8-1, 1);
        
        // Connector: fc1 -> norm -> fc2
        x = ggml_nn_linear(ctx, x, connector_fc1_weight, connector_fc1_bias);
        x = ggml_nn_rms_norm(ctx, x, connector_norm_weight);
        x = ggml_nn_linear(ctx, x, connector_fc2_weight, connector_fc2_bias);

        return x;
    }
};

// Static configuration
const int AudioVAEEncoder::downsample_strides[n_downsamples] = {1, 2, 2, 4, 5, 5, 8};
const int AudioVAEEncoder::downsample_dims[n_downsamples] = {32, 64, 128, 256, 512, 1024, 2048};
const int AudioVAEEncoder::stage_depths[n_stages] = {3, 3, 3, 3, 3, 3, 8};

//
// VAE Model
//

struct vae_model {
    struct ggml_context* params_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t params_buffer = nullptr;

    // Read-only mapping of the GGUF backing the weights, when mmap succeeded.
    // Tensor data points into it, so it must outlive every tensor.
    void*  mapping = nullptr;
    size_t mapping_size = 0;

    AudioVAEEncoder acoustic_encoder;
    AudioVAEEncoder semantic_encoder;
    
    int acoustic_dim = 64;  // Final output dim after connector
    int semantic_dim = 128; // Final output dim after connector
    
    std::map<std::string, struct ggml_tensor*> tensors;
    
    ~vae_model() {
        if (params_buffer) {
            ggml_backend_buffer_free(params_buffer);
        }
        if (params_ctx) {
            ggml_free(params_ctx);
        }
#ifndef _WIN32
        // After the buffer, which only borrowed this memory when mapped.
        if (mapping) {
            munmap(mapping, mapping_size);
        }
#endif
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct vae_context {
    vae_model_t* model = nullptr;
    int n_threads = 4;

    struct ggml_context* compute_ctx = nullptr;
    // Graph allocator: sizes the compute buffer from the graph's actual
    // liveness and REUSES storage between tensors whose lifetimes do not
    // overlap. Kept on the context so consecutive encodes (acoustic then
    // semantic, window after window) reuse one buffer instead of each
    // allocating and freeing its own.
    ggml_gallocr_t galloc = nullptr;

    ~vae_context() {
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
    }
};

//
// Helper: Load encoder weights from GGUF
//

static bool load_encoder_weights(
    vae_model_t* model,
    AudioVAEEncoder& encoder,
    const std::string& prefix) {
    
    // Load all downsample layers (0-6)
    for (int i = 0; i < AudioVAEEncoder::n_downsamples; i++) {
        char buf[256];
        
        snprintf(buf, sizeof(buf), "%s.downsample_layers.%d.0.conv.conv.weight", prefix.c_str(), i);
        encoder.downsamples[i].conv_weight = model->tensors[buf];
        
        snprintf(buf, sizeof(buf), "%s.downsample_layers.%d.0.conv.conv.bias", prefix.c_str(), i);
        encoder.downsamples[i].conv_bias = model->tensors[buf];
        
        if (!encoder.downsamples[i].conv_weight || !encoder.downsamples[i].conv_bias) {
            fprintf(stderr, "[VAE] Error: Failed to load downsample %d weights\n", i);
            fprintf(stderr, "[VAE]   Looking for: %s.downsample_layers.%d.0.conv.conv.*\n", prefix.c_str(), i);
            return false;
        }
        
        // Read kernel size from weight tensor shape [out_channels, in_channels, kernel_size]
        // In GGUF, dimensions are reversed, so ne[0] is kernel_size
        encoder.downsample_kernel_sizes[i] = encoder.downsamples[i].conv_weight->ne[0];
        fprintf(stderr, "[VAE]   Downsample %d kernel size: %d\n", i, encoder.downsample_kernel_sizes[i]);
    }
    
    // Load stages
    for (int stage = 0; stage < AudioVAEEncoder::n_stages; stage++) {
        int depth = AudioVAEEncoder::stage_depths[stage];
        encoder.stages[stage].resize(depth);
        
        for (int block = 0; block < depth; block++) {
            ConvNeXtBlock& b = encoder.stages[stage][block];
            char buf[256];
            
            // Mixer
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.norm.weight", prefix.c_str(), stage, block);
            b.mixer_norm_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.mixer.conv.conv.conv.weight", prefix.c_str(), stage, block);
            b.mixer_conv_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.mixer.conv.conv.conv.bias", prefix.c_str(), stage, block);
            b.mixer_conv_bias = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.gamma", prefix.c_str(), stage, block);
            b.mixer_layer_scale = model->tensors[buf];
            
            // FFN
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn_norm.weight", prefix.c_str(), stage, block);
            b.ffn_norm_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear1.weight", prefix.c_str(), stage, block);
            b.ffn_fc1_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear1.bias", prefix.c_str(), stage, block);
            b.ffn_fc1_bias = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear2.weight", prefix.c_str(), stage, block);
            b.ffn_fc2_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear2.bias", prefix.c_str(), stage, block);
            b.ffn_fc2_bias = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn_gamma", prefix.c_str(), stage, block);
            b.ffn_layer_scale = model->tensors[buf];
            
            // Verify all loaded
            if (!b.mixer_norm_weight || !b.mixer_conv_weight || !b.mixer_conv_bias ||
                !b.mixer_layer_scale || !b.ffn_norm_weight || !b.ffn_fc1_weight ||
                !b.ffn_fc1_bias || !b.ffn_fc2_weight || !b.ffn_fc2_bias || !b.ffn_layer_scale) {
                fprintf(stderr, "[VAE] Error: Failed to load stage %d block %d\n", stage, block);
                return false;
            }
            
            // Get dim and kernel_size from mixer conv weight shape [kernel_size, 1, dim]
            // In GGUF, dimensions are reversed, so ne[0] is kernel_size, ne[2] is dim
            b.dim = b.mixer_conv_weight->ne[2];
            b.kernel_size = b.mixer_conv_weight->ne[0];
            
            // Store kernel size at stage level (use first block's kernel size)
            if (block == 0) {
                encoder.stage_kernel_sizes[stage] = b.kernel_size;
                fprintf(stderr, "[VAE]   Stage %d kernel size: %d\n", stage, encoder.stage_kernel_sizes[stage]);
            }
        }
    }
    
    // Load head (just conv)
    std::string head_conv_w = prefix + ".head.conv.conv.weight";
    std::string head_conv_b = prefix + ".head.conv.conv.bias";
    
    encoder.head_conv_weight = model->tensors[head_conv_w];
    encoder.head_conv_bias = model->tensors[head_conv_b];
    
    if (!encoder.head_conv_weight || !encoder.head_conv_bias) {
        fprintf(stderr, "[VAE] Error: Failed to load head weights\n");
        return false;
    }
    
    // Get output dim from head conv weight [kernel, in_dim, out_dim]
    encoder.output_dim = encoder.head_conv_weight->ne[2];
    
    // Load connector weights (fc1 -> norm -> fc2)
    std::string connector_fc1_w = prefix + "_connector.fc1.weight";
    std::string connector_fc1_b = prefix + "_connector.fc1.bias";
    std::string connector_norm_w = prefix + "_connector.norm.weight";
    std::string connector_fc2_w = prefix + "_connector.fc2.weight";
    std::string connector_fc2_b = prefix + "_connector.fc2.bias";
    
    encoder.connector_fc1_weight = model->tensors[connector_fc1_w];
    encoder.connector_fc1_bias = model->tensors[connector_fc1_b];
    encoder.connector_norm_weight = model->tensors[connector_norm_w];
    encoder.connector_fc2_weight = model->tensors[connector_fc2_w];
    encoder.connector_fc2_bias = model->tensors[connector_fc2_b];
    
    if (!encoder.connector_fc1_weight || !encoder.connector_fc1_bias ||
        !encoder.connector_norm_weight || !encoder.connector_fc2_weight || 
        !encoder.connector_fc2_bias) {
        fprintf(stderr, "[VAE] Error: Failed to load connector weights\n");
        return false;
    }
    
    // Get connector output dim from fc2 weight [input_dim, output_dim]
    encoder.connector_output_dim = encoder.connector_fc2_weight->ne[1];
    
    fprintf(stderr, "[VAE] Loaded encoder '%s': vae_output_dim=%d, connector_output_dim=%d\n", 
            prefix.c_str(), encoder.output_dim, encoder.connector_output_dim);
    
    return true;
}

//
// Public API Implementation
//

struct vae_model_params vae_model_default_params() {
    struct vae_model_params params;
    params.n_threads = 16;
    params.use_gpu = false;
    return params;
}

struct vae_context_params vae_context_default_params() {
    struct vae_context_params params;
    params.n_threads = 16;
    return params;
}

vae_model_t* vae_load_model_from_file(
    const char* model_path,
    struct vae_model_params params) {
    
    fprintf(stderr, "[VAE] Loading model from %s\n", model_path);
    
    auto model = new vae_model();
    
    // Initialize backend
    model->backend = ggml_backend_cpu_init();
    if (!model->backend) {
        fprintf(stderr, "[VAE] Error: Failed to initialize backend\n");
        delete model;
        return nullptr;
    }
    
    // Load GGUF file
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,  // Don't allocate memory for tensors yet
        /*.ctx      =*/ &model->params_ctx,
    };
    
    struct gguf_context* gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "[VAE] Error: Failed to load GGUF file\n");
        delete model;
        return nullptr;
    }
    
    // Read metadata
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    fprintf(stderr, "[VAE] Model contains %d tensors\n", n_tensors);
    
    // Map tensors by name
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model->params_ctx, name);
        model->tensors[name] = tensor;
    }
    
    const size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Point the tensors straight at a read-only MAPPING of the file rather than
    // copying every weight into a freshly allocated backend buffer.
    //
    // The copy cost 0.65 GB of ANONYMOUS memory — memory the kernel can neither
    // drop nor share, and which counts in full against an Android app's budget.
    // Mapped weights are clean, file-backed pages: evictable under pressure,
    // reloaded from storage on demand, and shared between processes opening the
    // same model. The bytes at each tensor's offset are exactly what the copy
    // used to move (ggml_nbytes of them), so the tensors see identical data.
    //
    // This is what llama.cpp already does for the LM half of this pipeline,
    // which is why the LM shows up as file-backed and the VAE did not.
    bool mapped = false;
#ifndef _WIN32
    const int fd = open(model_path, O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            void * base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (base != MAP_FAILED) {
                model->mapping      = base;
                model->mapping_size = (size_t)st.st_size;
                model->params_buffer = ggml_backend_cpu_buffer_from_ptr(base, (size_t)st.st_size);

                for (int i = 0; i < n_tensors; i++) {
                    const char* name = gguf_get_tensor_name(gguf_ctx, i);
                    struct ggml_tensor* tensor = model->tensors[name];
                    tensor->buffer = model->params_buffer;
                    tensor->data   = (char *)base + data_offset + gguf_get_tensor_offset(gguf_ctx, i);
                }
                mapped = true;
            }
        }
        close(fd);
    }
#endif

    if (!mapped) {
        // Fallback: copy into an owned buffer. Windows takes this path, as does
        // any filesystem that will not hand out a mapping.
        model->params_buffer = ggml_backend_alloc_ctx_tensors(model->params_ctx, model->backend);

        FILE* f = fopen(model_path, "rb");
        if (!f) {
            fprintf(stderr, "[VAE] Error: Failed to open file for reading\n");
            gguf_free(gguf_ctx);
            delete model;
            return nullptr;
        }

        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor* tensor = model->tensors[name];
            size_t offset = data_offset + gguf_get_tensor_offset(gguf_ctx, i);

            fseek(f, offset, SEEK_SET);

            size_t tensor_size = ggml_nbytes(tensor);
            std::vector<char> buf(tensor_size);
            fread(buf.data(), 1, tensor_size, f);

            ggml_backend_tensor_set(tensor, buf.data(), 0, tensor_size);
        }

        fclose(f);
    }

    gguf_free(gguf_ctx);
    
    // Load encoder weights
    if (!load_encoder_weights(model, model->acoustic_encoder, "acoustic")) {
        delete model;
        return nullptr;
    }
    
    if (!load_encoder_weights(model, model->semantic_encoder, "semantic")) {
        delete model;
        return nullptr;
    }
    
    model->acoustic_dim = model->acoustic_encoder.connector_output_dim;
    model->semantic_dim = model->semantic_encoder.connector_output_dim;
    
    fprintf(stderr, "[VAE] Model loaded successfully\n");
    fprintf(stderr, "[VAE]   Acoustic output dim (after connector): %d\n", model->acoustic_dim);
    fprintf(stderr, "[VAE]   Semantic output dim (after connector): %d\n", model->semantic_dim);
    
    return model;
}

void vae_free_model(vae_model_t* model) {
    delete model;
}

vae_context_t* vae_new_context_with_model(
    vae_model_t* model,
    struct vae_context_params params) {
    
    auto ctx = new vae_context();
    ctx->model = model;
    ctx->n_threads = params.n_threads;
    
    return ctx;
}

void vae_free(vae_context_t* ctx) {
    delete ctx;
}

int32_t vae_model_acoustic_dim(const vae_model_t* model) {
    return model->acoustic_dim;
}

int32_t vae_model_semantic_dim(const vae_model_t* model) {
    return model->semantic_dim;
}

static size_t vae_model_max_nodes(const vae_model_t* model) {
    size_t n_tensors = 552;  // VAE encoder has 552 tensors
    return std::max<size_t>(1024, n_tensors * 3);
}

// Encode ONE window in a single graph. Callers go through vae_encode_impl,
// which slices long audio into windows of this and stitches the results.
static int32_t vae_encode_window(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* audio,
    int32_t n_samples,
    float* output) {

    if (!ctx || !audio || !output) {
        return -1;
    }

    // Build the graph in a METADATA-ONLY context (no_alloc), then let
    // ggml_gallocr size and pack the compute buffer.
    //
    // This used to be one flat ggml context with no_alloc=false, sized by a
    // 128 GB reservation that leaned on Linux overcommit. That allocated every
    // intermediate tensor in the encoder simultaneously and never reused one,
    // so cost grew ~270 MB per second of audio: a 30 s window needed an 8 GB
    // context and peaked near 8 GB RSS. It also aborted outright wherever a
    // 128 GB request is refused — Windows, Android, and plain Linux under the
    // DEFAULT heuristic overcommit (vm.overcommit_memory=0).
    //
    // The graph allocator computes each tensor's live range and reuses storage
    // between tensors that never coexist. A ConvNeXt encoder is a near-linear
    // chain, so nearly everything folds into a couple of ping-pong buffers.
    const size_t max_nodes = vae_model_max_nodes(ctx->model);
    const size_t meta_size = ggml_tensor_overhead() * max_nodes
                           + ggml_graph_overhead_custom(max_nodes, false);
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ meta_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,   // tensor structs only; gallocr owns the data
    };

    if (ctx->compute_ctx) {
        ggml_free(ctx->compute_ctx);
    }
    ctx->compute_ctx = ggml_init(ctx_params);
    if (!ctx->compute_ctx) {
        fprintf(stderr, "[VAE] Error: could not create the graph context\n");
        return -1;
    }

    // Check if model weights are I8_S — if so, quantize input to I8_S for full INT8 pipeline
    bool use_i8_s = (encoder.downsamples[0].conv_weight->type == GGML_TYPE_I8_S);

    struct ggml_tensor* input = ggml_new_tensor_3d(
        ctx->compute_ctx, use_i8_s ? GGML_TYPE_I8_S : GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(input, use_i8_s ? "input_audio_i8s" : "input_audio");
    // Marks the buffer as caller-written so the allocator won't hand its
    // storage to some later tensor while it is still needed.
    ggml_set_input(input);

    // Build computation graph
    struct ggml_tensor* result = encoder.forward(ctx->compute_ctx, input);
    ggml_set_output(result);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx->compute_ctx, max_nodes, false);
    ggml_build_forward_expand(gf, result);

    if (!ctx->galloc) {
        ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model->backend));
    }
    if (!ctx->galloc || !ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "[VAE] Error: could not allocate the compute buffer\n");
        return -1;
    }
    if (getenv("VIBEASR_VAE_DEBUG_MEM")) {
        // The CPU backend allocates a per-compute "work" buffer OUTSIDE the graph
        // allocator, sized by the most demanding single op in the graph (im2col +
        // quantized mul_mat scratch). It does not show up in the gallocr total, so
        // report it separately or the accounting silently loses the larger half.
        struct ggml_cplan plan = ggml_graph_plan(gf, ctx->n_threads, nullptr);
        fprintf(stderr,
                "[VAE] %d samples (%.1fs) -> %d nodes, compute buffer %.1f MB, "
                "cpu work buffer %.1f MB\n",
                n_samples, n_samples / 24000.0, ggml_graph_n_nodes(gf),
                ggml_gallocr_get_buffer_size(ctx->galloc, 0) / (1024.0 * 1024.0),
                plan.work_size / (1024.0 * 1024.0));
        // work_size is a MAX over ops, so ONE worst node sets it. Find that node by
        // re-planning each node alone: exact by construction, no duplication of
        // ggml's per-op sizing rules (which differ per op and per quant type).
        // Planning the subgraph ENDING at node i covers nodes 0..i, so work_size is
        // monotonic in i and steps up exactly at the culprit. Bisect for the first
        // index that already reaches the full figure. (ggml_cgraph is opaque, so a
        // one-node view is not available; this needs only the public API.)
        auto work_through = [&](int i) -> size_t {
            const size_t sz = ggml_tensor_overhead() * max_nodes
                            + ggml_graph_overhead_custom(max_nodes, false);
            struct ggml_init_params ip = { sz, nullptr, true };
            struct ggml_context * tmp = ggml_init(ip);
            if (!tmp) return 0;
            struct ggml_cgraph * g = ggml_new_graph_custom(tmp, max_nodes, false);
            ggml_build_forward_expand(g, ggml_graph_node(gf, i));
            const size_t w = ggml_graph_plan(g, ctx->n_threads, nullptr).work_size;
            ggml_free(tmp);
            return w;
        };
        int lo = 0, hi = ggml_graph_n_nodes(gf) - 1;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (work_through(mid) >= plan.work_size) hi = mid; else lo = mid + 1;
        }
        const int worst_i = lo;
        const size_t worst = plan.work_size;
        {
            struct ggml_tensor * nd = ggml_graph_node(gf, worst_i);
            fprintf(stderr,
                    "[VAE]   work ceiling: node #%d %s out=[%lld,%lld,%lld] type=%s -> %.1f MB\n",
                    worst_i, ggml_op_name(nd->op), (long long)nd->ne[0],
                    (long long)nd->ne[1], (long long)nd->ne[2],
                    ggml_type_name(nd->type), worst / (1024.0 * 1024.0));
            for (int s = 0; s < 2; s++) {
                if (struct ggml_tensor * src = nd->src[s]) {
                    fprintf(stderr, "[VAE]     src%d [%lld,%lld,%lld] %s\n", s,
                            (long long)src->ne[0], (long long)src->ne[1],
                            (long long)src->ne[2], ggml_type_name(src->type));
                }
            }
        }
    }

    // Tensor data pointers only become valid once the graph is allocated, so
    // the input is filled HERE rather than at construction time.
    if (use_i8_s) {
        // Find max abs value
        float amax = 0.00001f;
        for (int32_t i = 0; i < n_samples; i++) {
            float abs_val = fabsf(audio[i]);
            if (abs_val > amax) amax = abs_val;
        }
        float scale = 127.0f / amax;

        // Quantize to int8
        int8_t * dst_i8 = (int8_t *) input->data;
        for (int32_t i = 0; i < n_samples; i++) {
            int v = (int)roundf(audio[i] * scale);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            dst_i8[i] = (int8_t)v;
        }

        // Store scale after int8 data
        float * scale_ptr = (float *)((char *) input->data + n_samples);
        *scale_ptr = scale;
    } else {
        memcpy(input->data, audio, n_samples * sizeof(float));
    }

    // Compute
    ggml_backend_cpu_set_n_threads(ctx->model->backend, ctx->n_threads);
    if (ggml_backend_graph_compute(ctx->model->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: Graph computation failed\n");
        return -1;
    }

    // Get output dimensions
    int64_t batch = result->ne[2];
    int64_t n_frames = result->ne[1];
    int64_t out_dim = result->ne[0];
    
    if (result->type == GGML_TYPE_I8_S) {
        // Dequantize I8_S output to F32
        const int64_t n_total = n_frames * out_dim * batch;
        const int8_t * src_i8 = (const int8_t *) result->data;
        const float * scale_ptr = (const float *)((const char *) result->data + n_total);
        const float dequant = 1.0f / (*scale_ptr);  // max_abs / 127.0
        for (int64_t i = 0; i < n_total; i++) {
            output[i] = (float)src_i8[i] * dequant;
        }
    } else {
        // Copy F32 output directly
        memcpy(output, result->data, n_frames * out_dim * batch * sizeof(float));
    }
    
    return (int32_t)n_frames;
}

// Total stride of the encoder: one output frame per this many input samples.
// Matches the demo's --compress-ratio default and the config's encoder ratios
// (1 * 8 * 5 * 5 * 4 * 2 * 2).
static const int32_t VAE_HOP = 3200;

// Window and left-context defaults, in whole frames so sample counts stay
// aligned to VAE_HOP. 10 s of window and 2 s of context at 24 kHz.
static const int32_t VAE_WINDOW_FRAMES_DEFAULT = 75;   // 240000 samples
static const int32_t VAE_CONTEXT_FRAMES_DEFAULT = 15;  //  48000 samples

static int32_t vae_env_frames(const char* name, int32_t fallback) {
    if (const char* v = getenv(name)) {
        const long long n = atoll(v);
        if (n >= 0) return (int32_t)n;
    }
    return fallback;
}

static int32_t vae_encode_impl(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms = nullptr) {

    if (!ctx || !audio || !output) {
        return -1;
    }

    struct timespec start_time, end_time;
    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }

    // Peak memory is set by the single largest op in the graph, and that op runs
    // at FULL input resolution (stage 0 is stride 1), so it scales with the length
    // of whatever we hand to one graph: ~31.7 MB per second of audio, of which
    // ~23.4 MB is the CPU backend's work buffer for one MUL_MAT_ADD_RELU.
    //
    // Slicing the audio into fixed windows caps that cost regardless of clip
    // length. The convolutions are CAUSAL (ggml_im2col_asym pads left only), so a
    // window needs preceding samples for its receptive field and NO lookahead:
    // encode [start - context, start + window), then keep only the frames from
    // `start` onward. Set VAE_WINDOW_FRAMES=0 to disable and encode in one pass.
    const int32_t win_frames = vae_env_frames("VAE_WINDOW_FRAMES", VAE_WINDOW_FRAMES_DEFAULT);
    const int32_t ctx_frames = vae_env_frames("VAE_CONTEXT_FRAMES", VAE_CONTEXT_FRAMES_DEFAULT);
    const int32_t total_frames = n_samples / VAE_HOP;

    int32_t n_frames = 0;
    if (win_frames <= 0 || total_frames <= win_frames) {
        n_frames = vae_encode_window(ctx, encoder, audio, n_samples, output);
    } else {
        // Dimension of one output frame — needed to place each window's frames in
        // the caller's buffer. Taken from the encoder rather than assumed.
        const int32_t out_dim = encoder.connector_output_dim;
        std::vector<float> win_out;

        for (int32_t f0 = 0; f0 < total_frames; f0 += win_frames) {
            const int32_t lead = std::min(ctx_frames, f0);          // frames of left context
            const int32_t take = std::min(win_frames, total_frames - f0);
            const int32_t fed  = lead + take;

            win_out.resize((size_t)fed * out_dim);
            const int32_t got = vae_encode_window(
                ctx, encoder, audio + (size_t)(f0 - lead) * VAE_HOP,
                fed * VAE_HOP, win_out.data());
            if (got < 0) return -1;

            // Drop the context frames; they only existed to prime the receptive
            // field. A short return means the encoder produced fewer frames than
            // the hop predicts, so trust `got` over the arithmetic.
            if (getenv("VIBEASR_VAE_DEBUG_MEM")) {
                fprintf(stderr,
                        "[VAE]   window f0=%d lead=%d take=%d fed=%d -> got=%d%s\n",
                        f0, lead, take, fed, got, got == fed ? "" : "  <-- MISALIGNED");
            }
            const int32_t keep = std::min(take, got - lead);
            if (keep <= 0) return -1;
            memcpy(output + (size_t)f0 * out_dim,
                   win_out.data() + (size_t)lead * out_dim,
                   (size_t)keep * out_dim * sizeof(float));
            n_frames = f0 + keep;
        }
    }

    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e6;
        *inference_time_ms = (float)elapsed;
    }

    return n_frames;
}

int32_t vae_encode_acoustic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output) {
    
    return vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples, output, nullptr);
}

int32_t vae_encode_acoustic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms) {
    
    return vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples, output, inference_time_ms);
}

int32_t vae_encode_semantic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output) {
    
    return vae_encode_impl(ctx, ctx->model->semantic_encoder, audio, n_samples, output, nullptr);
}

int32_t vae_encode_semantic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms) {
    
    return vae_encode_impl(ctx, ctx->model->semantic_encoder, audio, n_samples, output, inference_time_ms);
}
