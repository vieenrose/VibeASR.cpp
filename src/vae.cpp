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

// ============================================================================
// Streaming conv cache (VibeVoiceTokenizerStreamingCache port).
//
// Every causal SConv1d in the encoder today left-pads its input with P zeros
// (P = kernel - stride for downsamples, k - 1 for stride-1 convs) and runs the
// conv with no further padding. The cache replaces those zeros with the real
// history carried from previous pieces: y = conv(concat(hist[P], x), pad=0).
// History length == the old zero-pad length, so output shapes are unchanged
// and, with piece boundaries on multiples of 3200 samples (the total stride,
// hence grid-aligned at every layer), outputs equal the full-window cold run
// up to matmul blocking order. First piece of a window uses zero history =
// bit-path identical to the old behavior. Norms/linears/FFNs are pointwise
// (stateless) and need no cache. Only the F32 compute path is cached; the
// I8_S fused path is untouched.
// ============================================================================
struct vae_stream_slot {
    std::vector<float> hist;   // last P inputs, logical layout of a dense [P, n1, n2, n3]
    int64_t n1 = 0, n2 = 0, n3 = 0;
    bool warm = false;         // false until first piece fills hist (zeros used meanwhile)
};

struct vae_stream_cache {
    std::map<std::string, vae_stream_slot> slots;
    struct Tap { std::string key; struct ggml_tensor * xh = nullptr; struct ggml_tensor * x = nullptr; int64_t P = 0; };
    std::vector<Tap> taps;     // recorded per forward build, consumed post-compute
    int next_id = 0;
};

struct vae_cache_t {
    vae_stream_cache acoustic;
    vae_stream_cache semantic;
};

void vae_cache_reset(vae_cache_t * cache) {
    if (!cache) return;
    cache->acoustic.slots.clear();
    cache->acoustic.taps.clear();
    cache->acoustic.next_id = 0;
    cache->semantic.slots.clear();
    cache->semantic.taps.clear();
    cache->semantic.next_id = 0;
}

// Build concat(hist, x) along dim0 for one conv site, recording a tap.
// P = the site's left-pad length (== history length). Cold/empty slots use
// zeros, which reproduces the legacy pad_ext(zeros) path exactly.
static struct ggml_tensor * vae_cached_concat(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int64_t P,
    vae_stream_cache * cache) {
    std::string key = "s" + std::to_string(cache->next_id++);
    vae_stream_slot & slot = cache->slots[key];  // default-constructed if new
    if (!slot.warm) {
        slot.n1 = x->ne[1]; slot.n2 = x->ne[2]; slot.n3 = x->ne[3];
        slot.hist.assign((size_t)(P * slot.n1 * slot.n2 * slot.n3), 0.0f);
    }
    if (getenv("VAE_CACHE_TRACE") != nullptr) {
        fprintf(stderr, "[CACHE_X] %s xop=%d xne=[%lld,%lld,%lld,%lld] xnb=[%lld,%lld,%lld,%lld] contig=%d\n",
                key.c_str(), (int)x->op,
                (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)x->ne[3],
                (long long)x->nb[0], (long long)x->nb[1], (long long)x->nb[2], (long long)x->nb[3],
                ggml_is_contiguous(x) ? 1 : 0);
    }
    if (getenv("VAE_CACHE_TRACE") != nullptr) {
        double s = 0; for (size_t k = 0; k < slot.hist.size(); k++) s += slot.hist[k];
        int64_t T = x->ne[0];
        int64_t rest = x->ne[1] * x->ne[2] * x->ne[3];
        fprintf(stderr, "[CACHE_FEED] %s P=%lld T=%lld C_rest=%lld warm=%d hsum=%.6f xtype=%d contig=%d\n",
                key.c_str(), (long long)P, (long long)T, (long long)rest,
                slot.warm ? 1 : 0, s, (int)x->type, ggml_is_contiguous(x) ? 1 : 0);
    }
    // NOTE: pad + add, NOT ggml_concat: this ggml build's concat scrambles
    // non-trivial inputs (verified by unit test: alternating chunks), while
    // pad_ext (legacy zero-pad) and add (residuals everywhere) are proven.
    // xh = pad_ext(x) + [hist | zeros]: bit-exact vs legacy when hist=0
    // (x + 0 == x), correct carry when warm. conv below runs with pad=0.
    struct ggml_tensor * xp = ggml_pad_ext(ctx, x, (int)P, 0, 0, 0, 0, 0, 0, 0);
    // ggml dense tensors are Fortran-order (ne[0] fastest): time t of channel c
    // lives at t + c*N. slot.hist is stored channel-major: channel c occupies
    // [c*P, (c+1)*P). Scatter it into the first P time steps of hp (rest zeros).
    int64_t hne[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; d++) hne[d] = xp->ne[d];
    struct ggml_tensor * hp = ggml_new_tensor(ctx, x->type, GGML_MAX_DIMS, hne);
    int64_t N = xp->ne[0];
    int64_t nch = xp->ne[1] * xp->ne[2] * xp->ne[3];
    float * hp_data = (float *)hp->data;
    for (int64_t c = 0; c < nch; c++) {
        memcpy(hp_data + c * N, slot.hist.data() + c * P, (size_t)P * sizeof(float));
        memset(hp_data + c * N + P, 0, (size_t)(N - P) * sizeof(float));
    }
    struct ggml_tensor * xh = ggml_add(ctx, xp, hp);
    vae_stream_cache::Tap tap;
    tap.key = key; tap.xh = xh; tap.x = x; tap.P = P;
    cache->taps.push_back(tap);
    return xh;
}

// Post-compute: roll each site's history forward from its concat node.
static void vae_cache_update(vae_stream_cache * cache) {
    for (size_t i = 0; i < cache->taps.size(); i++) {
        const vae_stream_cache::Tap & tap = cache->taps[i];
        vae_stream_slot & slot = cache->slots[tap.key];
        struct ggml_tensor * xh = tap.xh;
        int64_t n0 = xh->ne[0];
        int64_t nch = xh->ne[1] * xh->ne[2] * xh->ne[3];
        // Fortran-order: channel c occupies [c*n0, (c+1)*n0); new history =
        // last P time steps of each channel.
        if (xh->data == nullptr) {
            fprintf(stderr, "[CACHE_UPDATE] %s NULL data!\n", tap.key.c_str());
        } else {
            float * xhd = (float *)xh->data;
            for (int64_t c = 0; c < nch; c++) {
                memcpy(slot.hist.data() + c * tap.P, xhd + c * n0 + (n0 - tap.P),
                       (size_t)tap.P * sizeof(float));
            }
        }
        slot.warm = true;
        if (getenv("VAE_CACHE_TRACE") != nullptr) {
            double s = 0; for (size_t k = 0; k < slot.hist.size(); k++) s += slot.hist[k];
            // full-buffer sum + op chain for structural verification
            double full = 0;
            size_t nelem = 1;
            for (int d = 0; d < GGML_MAX_DIMS; d++) nelem *= (size_t)xh->ne[d];
            float * fdata = (float *)xh->data;
            for (size_t k = 0; k < nelem; k++) full += fdata[k];
            // site-input tail sum (Fortran-order: channel c tail at c*xn0+xn0-P)
            double xtail = 0;
            struct ggml_tensor * xin = tap.x;
            if (xin && xin->data) {
                int64_t xch = xin->ne[1] * xin->ne[2] * xin->ne[3];
                int64_t xn0 = xin->ne[0];
                float * xd = (float *)xin->data;
                for (int64_t c = 0; c < xch; c++)
                    for (int64_t k = 0; k < tap.P; k++) xtail += fabsf(xd[c * xn0 + (xn0 - tap.P) + k]);
            }
            int src0op = xh->src[0] ? (int)xh->src[0]->op : -1;
            int src1op = xh->src[1] ? (int)xh->src[1]->op : -1;
            fprintf(stderr, "[CACHE_STORED] %s ptr=%p P=%lld n0=%lld stored_sum=%.6f fullsum=%.3f xtail=%.3f op=%d src=(%d,%d) ne=[%lld,%lld,%lld,%lld] nbxh=[%lld,%lld] nbx=[%lld,%lld]\n",
                    tap.key.c_str(), xh->data, (long long)tap.P, (long long)n0, s, full, xtail, (int)xh->op,
                    src0op, src1op,
                    (long long)xh->ne[0], (long long)xh->ne[1],
                    (long long)xh->ne[2], (long long)xh->ne[3],
                    (long long)xh->nb[0], (long long)xh->nb[1],
                    xin ? (long long)xin->nb[0] : -1LL, xin ? (long long)xin->nb[1] : -1LL);
        }
    }
    cache->taps.clear();
}


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
    int dilation,
    vae_stream_cache* cache = nullptr) {

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
        if (cache != nullptr && padding > 0 && x->type == GGML_TYPE_F32) {
            x = vae_cached_concat(ctx, x, padding, cache);
            padding = 0;
        } else if (padding > 0) {
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
    int dilation,
    vae_stream_cache* cache = nullptr) {
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        struct ggml_tensor* new_x = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
        struct ggml_tensor* im2col = ggml_im2col_asym(ctx, w, new_x, stride, 0,
                                                       padding, 0, 0, dilation, 0, false, GGML_TYPE_I8_S);
        result = ggml_mul_mat_add(ctx, w, im2col, b);
        result = ggml_reshape_3d(ctx, result, result->ne[1], result->ne[2], 1);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
    } else {
        if (cache != nullptr && padding > 0 && x->type == GGML_TYPE_F32) {
            x = vae_cached_concat(ctx, x, padding, cache);
            padding = 0;
        } else if (padding > 0) {
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
        struct ggml_tensor* x,
        vae_stream_cache* cache = nullptr) {

        struct ggml_tensor* residual = x;
        bool is_i8s = (x->type == GGML_TYPE_I8_S);

        x = ggml_nn_rms_norm(ctx, x, mixer_norm_weight);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        x = ggml_nn_conv_1d_dw(ctx, x, mixer_conv_weight, mixer_conv_bias,
                                /*stride=*/1, /*padding=*/kernel_size-1, /*dilation=*/1,
                                is_i8s ? nullptr : cache);

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
        struct ggml_tensor* x,
        vae_stream_cache* cache = nullptr) {

        if (cache != nullptr) {
            cache->next_id = 0;
            cache->taps.clear();
        }

        // Downsamples and stages
        for (int i = 0; i < n_stages; i++) {

            x = ggml_nn_conv_1d(ctx, x, downsamples[i].conv_weight,
                                 downsamples[i].conv_bias,
                                 downsample_strides[i], downsample_kernel_sizes[i]-downsample_strides[i], 1,
                                 cache);

            for (int j = 0; j < stage_depths[i]; j++) {
                x = stages[i][j].forward(ctx, x, cache);
            }

            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        }

        // Head
        x = ggml_nn_conv_1d(ctx, x, head_conv_weight, head_conv_bias, 1, 8-1, 1, cache);
        
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
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct vae_context {
    vae_model_t* model = nullptr;
    int n_threads = 4;

    struct ggml_context* compute_ctx = nullptr;

    // Compute arena, owned by us and reused across encode calls. A fresh
    // multi-GB allocation per call costs more in kernel page-fault handling than
    // the encode itself, and that cost grows with thread count.
    void*  compute_buf      = nullptr;
    size_t compute_buf_size = 0;

    ~vae_context() {
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
        free(compute_buf);
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

    // Map tensors by name
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model->params_ctx, name);
        model->tensors[name] = tensor;
    }
    
    // Allocate backend buffer
    model->params_buffer = ggml_backend_alloc_ctx_tensors(model->params_ctx, model->backend);
    
    // Load tensor data from file
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "[VAE] Error: Failed to open file for reading\n");
        gguf_free(gguf_ctx);
        delete model;
        return nullptr;
    }
    
    size_t data_offset = gguf_get_data_offset(gguf_ctx);
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
    // +128 headroom for streaming-cache concat/cont nodes (37 sites x ~2-3).
    return std::max<size_t>(1024, n_tensors * 3) + 128;
}

static int32_t vae_encode_impl(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms = nullptr,
    vae_stream_cache* cache = nullptr) {
    
    if (!ctx || !audio || !output || n_samples <= 0) {
        return -1;
    }
    
    // Start timing if requested
    struct timespec start_time, end_time;
    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }
    
    // Check if model weights are I8_S — if so, quantize input to I8_S for full INT8 pipeline
    const bool use_i8_s = (encoder.downsamples[0].conv_weight->type == GGML_TYPE_I8_S);

    // Create computation context with sufficient memory.
    // Arena use is linear in the input length, and depends on the weight type:
    // F32/F16 models need more memory than I8_S because their intermediates stay
    // in F32. Measured rates (ggml_used_mem / n_samples, constant across 4 s to
    // 267 s inputs): I8_S 8.9 KB per input sample, F32 54.6 KB, F16 55.1 KB.
    // F16 is not cheaper than F32 here -- only the weights are half precision.
    // A fixed reservation is wrong in both directions. 128 GB is refused
    // outright by Windows (no overcommit) and by Linux heuristic overcommit on
    // any host whose RAM + swap is smaller, aborting in ggml_aligned_malloc
    // before any audio is processed. A small fixed pool starts everywhere but
    // silently caps input length and then segfaults past it. Size the arena
    // from the actual sample count instead, with ~15% headroom.
    const size_t bytes_per_sample = use_i8_s ? 10240 : 65536;
    const size_t vae_ctx_mem_size =
        (size_t)n_samples * bytes_per_sample + (size_t)512 * 1024 * 1024;
    // Grow (never shrink) the reused arena. The pages are first-touched once, by
    // whichever request needs them; later requests find them already mapped.
    if (ctx->compute_buf_size < vae_ctx_mem_size) {
        void * grown = realloc(ctx->compute_buf, vae_ctx_mem_size);
        if (grown == NULL) {
            fprintf(stderr, "[VAE] Error: failed to allocate %.2f GB compute arena\n",
                    vae_ctx_mem_size / 1073741824.0);
            return -1;
        }
        ctx->compute_buf      = grown;
        ctx->compute_buf_size = vae_ctx_mem_size;
    }

    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx->compute_buf_size,
        /*.mem_buffer =*/ ctx->compute_buf,
        /*.no_alloc   =*/ false,  // Let ggml allocate tensors
    };

    if (ctx->compute_ctx) {
        ggml_free(ctx->compute_ctx);
    }
    ctx->compute_ctx = ggml_init(ctx_params);

    struct ggml_tensor* input;
    if (use_i8_s) {
        // Quantize F32 audio to I8_S
        input = ggml_new_tensor_3d(ctx->compute_ctx, GGML_TYPE_I8_S, n_samples, 1, 1);
        ggml_set_name(input, "input_audio_i8s");
        
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
        // Use F32 input directly
        input = ggml_new_tensor_3d(ctx->compute_ctx, GGML_TYPE_F32, n_samples, 1, 1);
        ggml_set_name(input, "input_audio");
        memcpy(input->data, audio, n_samples * sizeof(float));
    }
    
    // Build computation graph
    struct ggml_tensor* result = encoder.forward(ctx->compute_ctx, input, cache);
    
    // Build graph with pre-allocated nodes (similar to llama_ref.cpp)
    size_t max_nodes = vae_model_max_nodes(ctx->model);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx->compute_ctx, max_nodes, false);
    ggml_build_forward_expand(gf, result);
    // Pin streaming-cache concat nodes as graph outputs: otherwise ggml-alloc
    // reuses their buffers for later tensors and the post-compute history
    // readback sees overwritten memory (observed as exact-zero tails).
    if (cache != nullptr) {
        for (size_t ti = 0; ti < cache->taps.size(); ti++) {
            ggml_build_forward_expand(gf, cache->taps[ti].xh);
            if (cache->taps[ti].x) ggml_build_forward_expand(gf, cache->taps[ti].x);
        }
    }
    
    // Compute
    if (ggml_graph_compute_with_ctx(ctx->compute_ctx, gf, ctx->n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: Graph computation failed\n");
        return -1;
    }

    // Debug: dump one site input (ne0-major floats) if requested (VAE_DUMP_SITE=s7).
    // NOTE: runs BEFORE vae_cache_update reads nothing back, but taps are cleared
    // by update, so the dump must come first.
    if (getenv("VAE_DUMP_SITE") != nullptr && cache != nullptr) {
        for (size_t ti = 0; ti < cache->taps.size(); ti++) {
            const char * want = getenv("VAE_DUMP_SITE");
            bool match = (cache->taps[ti].key == want) ||
                (strcmp(want, "s567") == 0 &&
                 (cache->taps[ti].key == "s5" || cache->taps[ti].key == "s6" ||
                  cache->taps[ti].key == "s7"));
            if (match) {
                struct ggml_tensor * xin = cache->taps[ti].x;
                char path[512];
                static int site_seq = 0;
                snprintf(path, sizeof(path), "/tmp/vaedump_site_%s_%03d.bin", cache->taps[ti].key.c_str(), site_seq++);
                FILE * df = fopen(path, "wb");
                if (df) {
                    size_t n = 1;
                    for (int d = 0; d < GGML_MAX_DIMS; d++) n *= (size_t)xin->ne[d];
                    // NOTE: raw dump in memory order (valid only if dense; checked via trace).
                    fwrite(xin->data, ggml_type_size(xin->type), n, df);
                    fclose(df);
                    fprintf(stderr, "[VAE_DUMP] site %s: wrote %s (%lld elems)\n",
                            cache->taps[ti].key.c_str(), path, (long long)n);
                }
            }
        }
    }
    // Roll streaming histories forward from the recorded concat nodes.
    if (cache != nullptr) {
        vae_cache_update(cache);
    }

    // Debug: dump output frames (one row per frame) if requested.
    if (getenv("VAE_DUMP_FRAMES") != nullptr) {
        int64_t n_frames = result->ne[1];
        int64_t out_dim = result->ne[0];
        char path[512];
        static int dump_seq = 0;
        snprintf(path, sizeof(path), "%s-%03d.bin", getenv("VAE_DUMP_FRAMES"), dump_seq++);
        FILE * df = fopen(path, "wb");
        if (df) {
            // result layout after connector: check ne ordering via ggml contiguous copy
            fwrite(result->data, sizeof(float), (size_t)(n_frames * out_dim), df);
            fclose(df);
            fprintf(stderr, "[VAE_DUMP] wrote %s (%lld frames x %lld)\n",
                    path, (long long)n_frames, (long long)out_dim);
        }
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
    
    // Calculate inference time if requested
    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e6;
        *inference_time_ms = (float)elapsed;
    }
    
    return (int32_t)n_frames;
}

int32_t vae_encode_acoustic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output) {

    return vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples, output, nullptr);
}

vae_cache_t* vae_cache_new(void) {
    return new vae_cache_t();
}

void vae_cache_free(vae_cache_t* cache) {
    delete cache;
}

int32_t vae_encode_acoustic_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* output) {

    if (!cache) return -1;
    return vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples, output,
                            nullptr, &cache->acoustic);
}

int32_t vae_encode_semantic_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* output) {

    if (!cache) return -1;
    return vae_encode_impl(ctx, ctx->model->semantic_encoder, audio, n_samples, output,
                            nullptr, &cache->semantic);
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
