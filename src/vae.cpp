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
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define VAE_HAVE_MMAP 1
#else
#define VAE_HAVE_MMAP 0
#endif

// Measurement-only elementwise ablation knobs (Exp662): drop one group of elementwise ops to
// price its share of VAE time. ANY OF THESE MAKES THE OUTPUT INVALID - it is a timing ceiling,
// never a result, and it must never be used for an accuracy claim.
//   VAE_ABL_BIAS  conv/matmul bias adds
//   VAE_ABL_TAPS  depthwise tap mul+add (K taps x full [C,T] tensors)
//   VAE_ABL_SCALE layer-scale and gamma muls
//   VAE_ABL_RESID block residual adds
static bool vae_abl(const char* name) {
    static std::map<std::string, int> cache;
    auto it = cache.find(name);
    if (it == cache.end()) {
        const bool on = getenv(name) != nullptr;
        it = cache.emplace(name, on ? 1 : 0).first;
        if (on) fprintf(stderr, "[VAE_ABL] %s is ON - output is INVALID, timing ceiling only\n", name);
    }
    return it->second != 0;
}
static struct ggml_tensor* vae_abl_add(struct ggml_context* ctx, struct ggml_tensor* a,
                                       struct ggml_tensor* b, const char* knob) {
    if (vae_abl(knob)) return a;      // op removed entirely: the sum is never materialised
    return ggml_add(ctx, a, b);
}
static struct ggml_tensor* vae_abl_mul(struct ggml_context* ctx, struct ggml_tensor* a,
                                       struct ggml_tensor* b, const char* knob) {
    if (vae_abl(knob)) return a;
    return ggml_mul(ctx, a, b);
}

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
    struct Tap { std::string key; struct ggml_tensor * xh = nullptr; struct ggml_tensor * x = nullptr; int64_t P = 0; int dim = 0; };
    std::vector<Tap> taps;     // recorded per forward build, consumed post-compute
    // Host-fed tensors (the [hist|zeros] staging buffers of the concat sites).
    // Under the lifetime allocator their data does not exist during the graph
    // build, so the write is deferred to a Fill record consumed by the caller
    // right after ggml_gallocr_alloc_graph.
    struct Fill { struct ggml_tensor * t = nullptr; const std::vector<float> * hist = nullptr; int64_t P = 0; int dim = 0; };
    std::vector<Fill> fills;
    bool lifetime = false;
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
static void vae_fill_staging(struct ggml_tensor * t, const std::vector<float> * hist, int64_t P, int dim);

// Scatter a slot's history into a staging tensor created by vae_cached_concat.
// dim 0: time is ne[0], channels are ne[1..3] (element (t,c) at t + c*N).
// dim 1: time is ne[1], channels are ne[0] (element (c,t) at c + t*C).
static void vae_fill_staging(struct ggml_tensor * t, const std::vector<float> * hist, int64_t P, int dim) {
    float * d = (float *) t->data;
    if (dim == 0) {
        const int64_t N = t->ne[0];
        const int64_t nch = t->ne[1] * t->ne[2] * t->ne[3];
        for (int64_t c = 0; c < nch; c++) {
            memcpy(d + c * N, hist->data() + c * P, (size_t)P * sizeof(float));
            memset(d + c * N + P, 0, (size_t)(N - P) * sizeof(float));
        }
    } else {
        const int64_t C = t->ne[0];
        const int64_t total = C * t->ne[1] * t->ne[2] * t->ne[3];
        memset(d, 0, (size_t)total * sizeof(float));
        for (int64_t c = 0; c < C; c++) {
            for (int64_t k = 0; k < P; k++) d[c + k * C] = hist->data()[c * P + k];
        }
    }
}

static struct ggml_tensor * vae_cached_concat(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int64_t P,
    vae_stream_cache * cache,
    int time_dim = 0) {
    std::string key = "s" + std::to_string(cache->next_id++);
    vae_stream_slot & slot = cache->slots[key];  // default-constructed if new
    if (!slot.warm) {
        slot.n1 = x->ne[1]; slot.n2 = x->ne[2]; slot.n3 = x->ne[3];
        // History is one row per CHANNEL: on dim-0 time that is ne[1..3], but on
        // dim-1 time the channels are ne[0] (sizing by ne[1..3] would allocate
        // T/P times too much and mis-index later).
        const int64_t hist_n = (time_dim == 0) ? (slot.n1 * slot.n2 * slot.n3) : x->ne[0];
        slot.hist.assign((size_t)(P * hist_n), 0.0f);
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
    struct ggml_tensor * xp = (time_dim == 0)
        ? ggml_pad_ext(ctx, x, (int)P, 0, 0, 0, 0, 0, 0, 0)
        : ggml_pad_ext(ctx, x, 0, 0, (int)P, 0, 0, 0, 0, 0);
    // Cold window head: the history is all zeros, so the staging tensor and the
    // add are pure overhead (x + 0.0f == x exactly). Skipping them removes two
    // full passes over the padded tensor for the first piece of every window,
    // which is half the pieces at the shipped PIECES=2. Exact by construction,
    // so outputs are unchanged; the tap still reads the real tail from xp.
    if (!slot.warm && getenv("VAE_NO_COLD_SKIP") == nullptr) {
        if (getenv("VAE_CACHE_TRACE") != nullptr) {
            fprintf(stderr, "[CACHE_SKIP] %s cold head: returning pad_ext directly (P=%lld)\n",
                    key.c_str(), (long long)P);
        }
        vae_stream_cache::Tap tap;
        tap.key = key; tap.xh = xp; tap.x = x; tap.P = P; tap.dim = time_dim;
        cache->taps.push_back(tap);
        return xp;
    }
    // ggml dense tensors are Fortran-order (ne[0] fastest): time t of channel c
    // lives at t + c*N. slot.hist is stored channel-major: channel c occupies
    // [c*P, (c+1)*P). Scatter it into the first P time steps of hp (rest zeros).
    int64_t hne[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; d++) hne[d] = xp->ne[d];
    struct ggml_tensor * hp = ggml_new_tensor(ctx, x->type, GGML_MAX_DIMS, hne);
    if (cache->lifetime) {
        // Data placement happens after graph allocation: just pin and record.
        ggml_set_input(hp);
        vae_stream_cache::Fill fill;
        fill.t = hp; fill.hist = &slot.hist; fill.P = P; fill.dim = time_dim;
        cache->fills.push_back(fill);
    } else {
        vae_fill_staging(hp, &slot.hist, P, time_dim);
    }
    struct ggml_tensor * xh = ggml_add(ctx, xp, hp);
    vae_stream_cache::Tap tap;
    tap.key = key; tap.xh = xh; tap.x = x; tap.P = P; tap.dim = time_dim;
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
            if (tap.dim == 0) {
                for (int64_t c = 0; c < nch; c++) {
                    memcpy(slot.hist.data() + c * tap.P, xhd + c * n0 + (n0 - tap.P),
                           (size_t)tap.P * sizeof(float));
                }
            } else {
                // time on ne[1]: channels are ne[0] (nch above counts time here)
                const int64_t C = xh->ne[0];
                const int64_t nT = xh->ne[1];
                for (int64_t c = 0; c < C; c++) {
                    for (int64_t k = 0; k < tap.P; k++) {
                        slot.hist[c * tap.P + k] = xhd[c + (nT - tap.P + k) * C];
                    }
                }
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
        x = vae_abl_mul(ctx, x, gamma, "VAE_ABL_SCALE");
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
            result = vae_abl_add(ctx, result, b, "VAE_ABL_BIAS");
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
            result = vae_abl_add(ctx, result, b, "VAE_ABL_BIAS");
        }
        result = ggml_relu(ctx, result);
    }

    result = ggml_reshape_3d(ctx, result, OC, L, N);

    return result;
}

// F16-im2col variants of ggml_conv_1d / ggml_conv_1d_dw. The stock ops hardcode
// a F32 im2col: that doubles the im2col write+read traffic AND forces
// mul_mat's src1 conversion (vec_dot_type is F16 for F16 weights) to
// materialise another full F16 copy. Building the im2col directly in F16
// halves the traffic and removes that pass; the F32->F16 rounding is identical
// (same values, same rounding), so results are bit-identical.
static struct ggml_tensor* vae_conv_1d_f16(
    struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
    int s0, int p0, int d0) {
    // NOTE: use ggml_im2col (op IM2COL), NOT ggml_im2col_asym - the asym op is
    // hardwired to ggml_compute_forward_im2col_i8_s (BitNet I8_S output), so an
    // F16 dst_type there silently produces garbage.
    struct ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F16); // [IC*KW, OL, N, 1]
    return ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]),
            ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])));
}

static struct ggml_tensor* vae_conv_1d_dw_f16(
    struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
    int s0, int p0, int d0) {
    const int64_t C = a->ne[2];
    const int64_t L = b->ne[0];
    const int64_t N = b->ne[2];
    struct ggml_tensor* b4d = ggml_reshape_4d(ctx, b, L, 1, C, N);
    struct ggml_tensor* im2col = ggml_im2col(ctx, a, b4d, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F16);
    struct ggml_tensor* im2d = ggml_reshape_3d(ctx, im2col,
            im2col->ne[0], im2col->ne[1] * im2col->ne[3], im2col->ne[2]);
    struct ggml_tensor* a3d = ggml_reshape_3d(ctx, a, a->ne[0], 1, C);
    struct ggml_tensor* result = ggml_mul_mat(ctx, a3d, im2d);
    const int64_t OL = im2col->ne[1];
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, result, C, OL, N);
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
        result = vae_conv_1d_f16(ctx, w, x, stride, padding, dilation);
        if (b != NULL) {
            result = vae_abl_add(ctx, result, b, "VAE_ABL_BIAS");
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
        // Channels-first input [C, T] (the block's natural layout when it is not
        // transposed): time is ne[1], so cache concat and padding go along dim 1.
        const bool ct_in = (x->ne[2] == 1) && (w->ne[1] == 1) && (w->ne[3] == 1) &&
                           (x->ne[0] == w->ne[2]) && (x->ne[1] > 2 * w->ne[0]);
        if (cache != nullptr && padding > 0 && x->type == GGML_TYPE_F32) {
            x = vae_cached_concat(ctx, x, padding, cache, ct_in ? 1 : 0);
            padding = 0;
        } else if (padding > 0) {
            x = ct_in ? ggml_pad_ext(ctx, x, 0, 0, padding, 0, 0, 0, 0, 0)
                      : ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
            padding = 0;
        }
        // For the im2col fallback with a channels-first input: one cont/permute
        // back to [T, C] so the legacy path stays usable in every mode.
        const bool dw_taps = (getenv("VAE_DW_CT_OFF") == nullptr) &&
                             stride == 1 && dilation == 1 && b != NULL && ct_in;
        if (dw_taps) {
            // Depthwise conv as K dense-view multiply-adds in the [C, T] layout
            // (Exp580, default since it measured RTF 3.1523 -> 3.0829 on device).
            // The weight transposes ONCE to [C, K] so each tap is a contiguous
            // [C,1] slice, and each tap's input is a plain row offset of a dense
            // [C, T] tensor. Exp569's slower variant worked in [T, C], where the
            // weight broadcast along the innermost dimension and every tap view
            // was strided. VAE_DW_CT_OFF=1 restores the im2col + 3-D mul_mat path.
            const int64_t K = w->ne[0];
            const int64_t C = w->ne[2];
            struct ggml_tensor* xc = ct_in ? x : ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            // NOTE ggml's permute convention is result.ne[axis_i] = a.ne[i], so
            // going from [K, 1, C] to [C, K] needs (1, 2, 0), not (2, 0, 1).
            struct ggml_tensor* ww = ggml_permute(ctx, w, 1, 2, 0, 3);                   // [C, K]
            ww = ggml_cont(ctx, ww);
            if (ww->type != GGML_TYPE_F32) ww = ggml_cast(ctx, ww, GGML_TYPE_F32);
            const int64_t T_out = xc->ne[1] - (K - 1);
            if (getenv("VAE_DW_DEBUG") != nullptr) {
                static int dbg = 0;
                if (dbg++ < 4) {
                    fprintf(stderr, "[DWDBG] x=[%lld,%lld,%lld] w=[%lld,%lld,%lld] K=%lld C=%lld xc=[%lld,%lld] nbc=%lld ww=[%lld,%lld] nbc=%lld T_out=%lld pad_left=%d\n",
                            (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2],
                            (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2],
                            (long long)K, (long long)C,
                            (long long)xc->ne[0], (long long)xc->ne[1], (long long)xc->nb[1],
                            (long long)ww->ne[0], (long long)ww->ne[1], (long long)ww->nb[1],
                            (long long)T_out, padding);
                }
            }
            struct ggml_tensor* acc = nullptr;
            // ONE fused pass per tap instead of mul-then-add (Exp662 priced the tap chain at
            // 12.7% of VAE time; the fused form moves 3 tensors per tap instead of 5, worth
            // -4.6% VAE / -3.3% RTF with output BIT-IDENTICAL to the old chain - the kernel uses
            // vmul+vadd, not vfma, so the product rounds exactly as ggml_mul did. wk is a [C,1]
            // view of the weight, i.e. per-channel gamma. VAE_DW_AXPY_OFF=1 restores the chain.
            const bool axpy = getenv("VAE_DW_AXPY_OFF") == nullptr;
            for (int64_t k = 0; k < K; k++) {
                struct ggml_tensor* xk = ggml_view_2d(ctx, xc, C, T_out, xc->nb[1], (size_t)k * xc->nb[1]);
                struct ggml_tensor* wk = ggml_view_2d(ctx, ww, C, 1, ww->nb[1], (size_t)k * ww->nb[1]);
                if (vae_abl("VAE_ABL_TAPS")) {                      // measurement-only: drop the tap
                    if (acc == nullptr) {
                        acc = vae_abl_mul(ctx, xk, wk, "VAE_ABL_TAPS");
                    }
                } else if (axpy && acc != nullptr) {
                    acc = ggml_add_scaled(ctx, xk, acc, wk);        // acc + xk * wk
                } else {
                    struct ggml_tensor* term = vae_abl_mul(ctx, xk, wk, "VAE_ABL_TAPS");
                    acc = (acc == nullptr) ? term : vae_abl_add(ctx, acc, term, "VAE_ABL_TAPS");
                }
            }
            result = acc;   // [C, T_out]: the layout the block's residual uses
        } else {
            // The legacy helper wants time on ne[0]. With the taps disabled by
            // VAE_DW_CT_OFF the block still hands us a [C, T] tensor (ct_block
            // is independent of the knob), so transpose here - otherwise the
            // helper's [T,1,C,N] reshape asserts (same dim-0/dim-1 class of bug
            // as Exp586's guard mismatch, and the state Exp602's fix left the
            // knob working in; Exp586 re-broke it). The left-pad above was
            // applied on dim 1, which permutes to exactly dim-0 left-padding.
            if (ct_in) x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            result = vae_conv_1d_dw_f16(ctx, w, x, stride, padding, dilation);
        }
        if (b != NULL) {
            result = vae_abl_add(ctx, result, b, "VAE_ABL_BIAS");
        }
    }

    return result;
}

static struct ggml_tensor* ggml_nn_layer_scale(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma) {
    return vae_abl_mul(ctx, x, gamma, "VAE_ABL_SCALE");
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

        // Exp586: run the mixer channels-first so neither the transpose in nor
        // the one inside the depthwise helper is needed. F16 path only (the
        // I8_S fusion owns its own layout). VAE_CT_BLOCK_OFF=1 for the A/B.
        // Must match the depthwise helper's ct_in test EXACTLY: if the block skips
        // the transpose but the helper then rejects the channels-first input, the
        // helper's dim-0 cache/pad path is fed a [C, T] tensor. That mismatch bit
        // the piece-wise path (lean tier, and --xwin's 6400-sample pieces) where the
        // deep stages only have T = 1..8 samples per piece.
        const bool ct_block = (getenv("VAE_CT_BLOCK_OFF") == nullptr) && !is_i8s &&
                              mixer_conv_weight->ne[1] == 1 && x->ne[0] == mixer_conv_weight->ne[2] &&
                              x->ne[1] > 2 * mixer_conv_weight->ne[0];
        if (!ct_block) {
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        }

        x = ggml_nn_conv_1d_dw(ctx, x, mixer_conv_weight, mixer_conv_bias,
                                /*stride=*/1, /*padding=*/kernel_size-1, /*dilation=*/1,
                                is_i8s ? nullptr : cache);

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, mixer_layer_scale);
        } else if (getenv("VAE_LS_FUSE") != nullptr) {
            x = ggml_add_scaled(ctx, x, residual, mixer_layer_scale);   // x*scale + residual, 1 pass
        } else {
            // F32 path: x = x * layer_scale + residual
            x = vae_abl_mul(ctx, x, mixer_layer_scale, "VAE_ABL_SCALE");
            x = vae_abl_add(ctx, x, residual, "VAE_ABL_RESID");
        }
        
        residual = x;

        x = ggml_nn_rms_norm(ctx, x, ffn_norm_weight);
        
        if (is_i8s) {
            x = ggml_nn_linear_relu(ctx, x, ffn_fc1_weight, ffn_fc1_bias);
        } else {
            x = ggml_nn_linear(ctx, x, ffn_fc1_weight, ffn_fc1_bias);
            // Upstream trains exact erf GELU (ACT2FN["gelu"]); the tanh approx
            // differs by <= ~3e-4 elementwise. VAE_GELU_QUICK=1 selects it - a
            // model approximation, so it must pass the accuracy gate.
            if (getenv("VAE_GELU_QUICK") != nullptr) {
                x = ggml_gelu_quick(ctx, x);
            } else {
                x = ggml_gelu(ctx, x);
            }
        }
        
        x = ggml_nn_linear(ctx, x, ffn_fc2_weight, ffn_fc2_bias);

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, ffn_layer_scale);
        } else if (getenv("VAE_LS_FUSE") != nullptr) {
            x = ggml_add_scaled(ctx, x, residual, ffn_layer_scale);     // x*scale + residual, 1 pass
        } else {
            x = vae_abl_mul(ctx, x, ffn_layer_scale, "VAE_ABL_SCALE");
            x = vae_abl_add(ctx, x, residual, "VAE_ABL_RESID");
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
    int head_kernel_size = 0;  // read from weights (8 padded / 7 trimmed)
    
    // Connector (fc1 -> norm -> fc2)
    struct ggml_tensor* connector_fc1_weight;
    struct ggml_tensor* connector_fc1_bias;
    struct ggml_tensor* connector_norm_weight;
    struct ggml_tensor* connector_fc2_weight;
    struct ggml_tensor* connector_fc2_bias;
    
    // Stages [0, split) of the encoder with the streaming cache. The output is
    // the stage-boundary tensor in the conv layout: [T, C] (time fastest), the
    // exact tensor the next stage's downsample conv consumes. split == n_stages
    // leaves x untouched.
    struct ggml_tensor* forward_early(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        vae_stream_cache* cache,
        int split) {

        if (cache != nullptr) {
            cache->next_id = 0;
            cache->taps.clear();
        }

        for (int i = 0; i < split && i < n_stages; i++) {

            x = ggml_nn_conv_1d(ctx, x, downsamples[i].conv_weight,
                                 downsamples[i].conv_bias,
                                 downsample_strides[i], downsample_kernel_sizes[i]-downsample_strides[i], 1,
                                 cache);

            for (int j = 0; j < stage_depths[i]; j++) {
                x = stages[i][j].forward(ctx, x, cache);
            }

            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        }

        return x;
    }

    // Stages [split, n_stages) + head + connector. Two modes:
    //   cache != nullptr: piece-wise streaming (history from the cache).
    //   cache == nullptr: the input carries the whole window, so the convs
    //   only zero-pad at the window start - exactly the legacy cold-window
    //   semantics the cache reproduces piece-wise. Used by the deferred
    //   (window-level late stages) path.
    struct ggml_tensor* forward_late(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        vae_stream_cache* cache,
        int split) {

        for (int i = split; i < n_stages; i++) {

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
        x = ggml_nn_conv_1d(ctx, x, head_conv_weight, head_conv_bias, 1, head_kernel_size - 1, 1, cache);

        // Connector: fc1 -> norm -> fc2
        x = ggml_nn_linear(ctx, x, connector_fc1_weight, connector_fc1_bias);
        x = ggml_nn_rms_norm(ctx, x, connector_norm_weight);
        x = ggml_nn_linear(ctx, x, connector_fc2_weight, connector_fc2_bias);

        return x;
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        vae_stream_cache* cache = nullptr) {

        return forward_late(ctx, forward_early(ctx, x, cache, n_stages), cache, n_stages);
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
    // Zero-copy load: the tensors point straight into the read-only mapping and
    // it stays mapped for the model's lifetime. File-backed clean pages are
    // reclaimable under memory pressure, unlike the anonymous arena copy.
    void* mmap_base = nullptr;
    size_t mmap_size = 0;
    
    AudioVAEEncoder acoustic_encoder;
    AudioVAEEncoder semantic_encoder;
    
    int acoustic_dim = 64;  // Final output dim after connector
    int semantic_dim = 128; // Final output dim after connector
    
    std::map<std::string, struct ggml_tensor*> tensors;
    
    ~vae_model() {
        if (mmap_base) {
            munmap(mmap_base, mmap_size);
            mmap_base = nullptr;
        }
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

    // Second (context, arena) pair: the acoustic and semantic encoders are
    // independent (separate weights, separate streaming caches), so they can
    // run concurrently, one thread each. slot 0 = acoustic, slot 1 = semantic.
    struct ggml_context* compute_ctx2 = nullptr;
    void*  compute_buf2      = nullptr;
    size_t compute_buf_size2 = 0;

    ~vae_context() {
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
        if (compute_ctx2) {
            ggml_free(compute_ctx2);
        }
        free(compute_buf);
        free(compute_buf2);
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
    encoder.head_kernel_size = (int)encoder.head_conv_weight->ne[0];
    
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
    
    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    // Zero-copy plan: point the tensors straight into the read-only mapping and
    // keep it for the model's lifetime (file-backed clean pages are reclaimable
    // under memory pressure, unlike an anonymous arena copy, and there is no
    // 0.5 GB memcpy at startup). Requires every tensor's file offset to be
    // 32-byte aligned (gguf general.alignment; mmap itself is page-aligned),
    // which is checked from metadata before mapping. Enabled by default since
    // Exp575 measured it: RTF and transcripts identical, load 1.4 -> 1.2 s, and
    // the 536 MB of weights stop being a private anonymous copy (they become
    // clean, reclaimable file-backed pages). VAE_COPY_WEIGHTS=1 restores the
    // arena copy.
    const bool zerocopy_req = (getenv("VAE_COPY_WEIGHTS") == nullptr);
    bool zc_offsets_ok = zerocopy_req;
    for (int i = 0; zerocopy_req && i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = model->tensors[name];
        size_t offset = data_offset + gguf_get_tensor_offset(gguf_ctx, i);
        if ((offset % 32) != 0 || ggml_nbytes(tensor) == 0) { zc_offsets_ok = false; break; }
    }
    const bool zerocopy_req_ok = zerocopy_req && zc_offsets_ok;
    bool zerocopied = false;
    bool loaded = false;

#if VAE_HAVE_MMAP
    {
        int fd = open(model_path, O_RDONLY);
        struct stat st;
        if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0) {
            void* base = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (base != MAP_FAILED) {
                bool ok = true;
                madvise(base, (size_t) st.st_size, MADV_WILLNEED);
                struct CopyJob { struct ggml_tensor* t; size_t off; size_t sz; };
                std::vector<CopyJob> jobs;
                jobs.reserve(n_tensors);
                size_t total_bytes = 0;
                for (int i = 0; i < n_tensors; i++) {
                    const char* name = gguf_get_tensor_name(gguf_ctx, i);
                    struct ggml_tensor* tensor = model->tensors[name];
                    size_t offset = data_offset + gguf_get_tensor_offset(gguf_ctx, i);
                    size_t tensor_size = ggml_nbytes(tensor);
                    if (offset + tensor_size > (size_t) st.st_size) { ok = false; break; }
                    jobs.push_back({tensor, offset, tensor_size});
                    total_bytes += tensor_size;
                }
                if (ok && !jobs.empty()) {
                    if (zerocopy_req_ok && ((uintptr_t) base % 32) == 0) {
                        for (const CopyJob& j : jobs) j.t->data = (char*) base + j.off;
                        model->mmap_base = base;
                        model->mmap_size = (size_t) st.st_size;
                        zerocopied = true;
                        if (getenv("VAE_ZEROCOPY_STATS") != nullptr) {
                            fprintf(stderr, "[VAE_ZC] zero-copy weights: %.1f MB mapped, no arena copy\n",
                                    total_bytes / 1e6);
                        }
                    } else {
                        if (!model->params_buffer) {
                            model->params_buffer = ggml_backend_alloc_ctx_tensors(model->params_ctx, model->backend);
                        }
                        if (!model->params_buffer) { ok = false; }
                        // Parallel copy: the mmap page-cache -> arena memcpy is the
                        // whole cost of the load phase (0.4-1.4 GB) and is trivially
                        // parallel over tensors; a tensor straddling a worker's
                        // byte range may be copied twice, which is idempotent.
                        const int nworkers = 4;
                        std::vector<std::thread> th;
                        th.reserve(nworkers);
                        for (int w = 0; ok && w < nworkers; w++) {
                            th.emplace_back([&, w]() {
                                const size_t begin = total_bytes * (size_t) w / (size_t) nworkers;
                                const size_t end   = total_bytes * (size_t)(w + 1) / (size_t) nworkers;
                                size_t acc = 0;
                                for (const CopyJob& j : jobs) {
                                    if (acc >= end) break;
                                    if (acc + j.sz > begin) {
                                        ggml_backend_tensor_set(j.t, (const char*) base + j.off, 0, j.sz);
                                    }
                                    acc += j.sz;
                                }
                            });
                        }
                        for (auto& x : th) x.join();
                    }
                }
                if (!zerocopied) munmap(base, (size_t) st.st_size);
                loaded = ok;
            }
            close(fd);
        }
    }
#endif
    if (!loaded && !zerocopied) {
        if (!model->params_buffer) {
            model->params_buffer = ggml_backend_alloc_ctx_tensors(model->params_ctx, model->backend);
        }
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
            size_t got = fread(buf.data(), 1, tensor_size, f);
            (void) got;

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

// Static traffic + compute profile of a planned graph (VAE_GRAPH_STATS=1). Shared by the
// legacy piece-wise path and both split passes, so the SHIPPED configuration can be profiled
// (Exp631: the old inline-only version silently emitted nothing on forward_early/forward_late).
// Bytes are a device-independent bandwidth proxy (weights read once per mul_mat, src1 re-reads
// counted per use); MACs are output-elements x contraction length for MUL_MAT, element counts
// for elementwise ops, 0 for views.
static void vae_graph_stats_dump(struct ggml_cgraph* gf, const char* phase) {
    struct Row { size_t bytes = 0; size_t count = 0; double macs = 0.0; };
    std::map<std::string, Row> by_op;
    size_t total = 0;
    double total_macs = 0.0;
    const bool shapes = getenv("VAE_MMSHAPES") != nullptr;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        size_t b = ggml_nbytes(node);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s]) b += ggml_nbytes(node->src[s]);
        }
        double macs = 0.0;
        if (node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1]) {
            macs = (double) ggml_nelements(node) * (double) node->src[0]->ne[0];
        } else if (node->op != GGML_OP_RESHAPE && node->op != GGML_OP_PERMUTE &&
                   node->op != GGML_OP_VIEW && node->op != GGML_OP_TRANSPOSE) {
            macs = (double) ggml_nelements(node);
        }
        std::string key = ggml_op_name(node->op);
        if (node->op == GGML_OP_MUL_MAT && node->src[0]) {
            key += std::string("/") + ggml_type_name(node->src[0]->type);
            if (shapes) {
                const struct ggml_tensor* a = node->src[0];
                const struct ggml_tensor* b1 = node->src[1];
                fprintf(stderr, "[MMSHAPE:%s] %-8s a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                        phase, ggml_type_name(a->type),
                        (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
                        (long long)b1->ne[0], (long long)b1->ne[1], (long long)b1->ne[2], (long long)b1->ne[3],
                        (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);
            }
        }
        auto& e = by_op[key];
        e.bytes += b; e.count += 1; e.macs += macs;
        total += b; total_macs += macs;
    }
    fprintf(stderr, "[VAE_STATS:%s] nodes=%d total=%.1f MB macs=%.2f G\n", phase, n_nodes, total / 1e6, total_macs / 1e9);
    for (auto& kv : by_op) {
        fprintf(stderr, "[VAE_STATS:%s] %-16s n=%4zu bytes=%9.1f MB %5.1f%%  macs=%9.3f G\n",
                phase, kv.first.c_str(), kv.second.count, kv.second.bytes / 1e6,
                100.0 * kv.second.bytes / total, kv.second.macs / 1e9);
    }
}

static enum ggml_status vae_graph_compute_planned(struct ggml_cgraph * gf, int n_threads);

static int32_t vae_encode_impl(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms = nullptr,
    vae_stream_cache* cache = nullptr,
    int slot = 0,
    int n_threads_ovr = 0) {

    if (!ctx || !audio || !output || n_samples <= 0) {
        return -1;
    }
    // Per-slot (thread-local) context/arena references: slot 0 = acoustic,
    // slot 1 = semantic. With slot 0 this is exactly the historical behaviour.
    struct ggml_context*& compute_ctx_ref = (slot == 0) ? ctx->compute_ctx  : ctx->compute_ctx2;
    void*&                compute_buf_ref = (slot == 0) ? ctx->compute_buf  : ctx->compute_buf2;
    size_t&               compute_buf_size_ref = (slot == 0) ? ctx->compute_buf_size : ctx->compute_buf_size2;
    
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
    // Allocation: the same lifetime (ggml-alloc) scheme as the split early/late
    // passes - the context holds only metadata and the graph allocator places
    // the live set (a few tens of MB) instead of an arena scaled at
    // 10-64 KB per input sample (which made whole-file calls - the server and
    // the non-streaming demo - reserve gigabytes). VAE_LEGACY_ARENA=1 restores
    // the historical arena for A/B measurement.
    static ggml_gallocr_t gallocr_pw[2] = { nullptr, nullptr };
    static void * md_buf_pw[2] = { nullptr, nullptr };
    static size_t md_size_pw[2] = { 0, 0 };
    // No size limit: the work buffer is allocated outside the metadata arena
    // (see vae_graph_compute_planned), which is what made whole-file inputs
    // crash before Exp579.
    const bool legacy_arena = (getenv("VAE_LEGACY_ARENA") != nullptr);
    if (compute_ctx_ref) ggml_free(compute_ctx_ref);
    if (legacy_arena) {
        const size_t bytes_per_sample = use_i8_s ? 10240 : 65536;
        const size_t vae_ctx_mem_size =
            (size_t)n_samples * bytes_per_sample + (size_t)64 * 1024 * 1024;
        if (compute_buf_size_ref < vae_ctx_mem_size) {
            void * grown = realloc(compute_buf_ref, vae_ctx_mem_size);
            if (grown == NULL) {
                fprintf(stderr, "[VAE] Error: failed to allocate %.2f GB compute arena\n",
                        vae_ctx_mem_size / 1073741824.0);
                return -1;
            }
            compute_buf_ref      = grown;
            compute_buf_size_ref = vae_ctx_mem_size;
        }
        struct ggml_init_params ctx_params = {
            compute_buf_size_ref, compute_buf_ref, false,
        };
        compute_ctx_ref = ggml_init(ctx_params);
    } else {
        if (gallocr_pw[slot] == nullptr) {
            gallocr_pw[slot] = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            if (gallocr_pw[slot] == nullptr) { fprintf(stderr, "[VAE] Error: gallocr init failed\n"); return -1; }
        }
        const size_t md_need = (size_t)16 * 1024 * 1024;
        if (md_size_pw[slot] < md_need) {
            void * grown = realloc(md_buf_pw[slot], md_need);
            if (grown == NULL) { fprintf(stderr, "[VAE] Error: metadata arena alloc failed\n"); return -1; }
            md_buf_pw[slot] = grown; md_size_pw[slot] = md_need;
        }
        struct ggml_init_params ctx_params = { md_size_pw[slot], md_buf_pw[slot], true };
        compute_ctx_ref = ggml_init(ctx_params);
        if (compute_ctx_ref == nullptr) { fprintf(stderr, "[VAE] Error: ctx init failed\n"); return -1; }
    }

    // Input tensor (data placement happens after graph allocation in lifetime mode).
    float i8_scale = 1.0f;
    struct ggml_tensor* input;
    if (use_i8_s) {
        input = ggml_new_tensor_3d(compute_ctx_ref, GGML_TYPE_I8_S, n_samples, 1, 1);
        ggml_set_name(input, "input_audio_i8s");
        float amax = 0.00001f;
        for (int32_t i = 0; i < n_samples; i++) {
            float abs_val = fabsf(audio[i]);
            if (abs_val > amax) amax = abs_val;
        }
        i8_scale = 127.0f / amax;
        if (legacy_arena) {
            int8_t * dst_i8 = (int8_t *) input->data;
            for (int32_t i = 0; i < n_samples; i++) {
                int v = (int)roundf(audio[i] * i8_scale);
                if (v >  127) v =  127;
                if (v < -128) v = -128;
                dst_i8[i] = (int8_t)v;
            }
            float * scale_ptr = (float *)((char *) input->data + n_samples);
            *scale_ptr = i8_scale;
        }
    } else {
        input = ggml_new_tensor_3d(compute_ctx_ref, GGML_TYPE_F32, n_samples, 1, 1);
        ggml_set_name(input, "input_audio");
        if (legacy_arena) memcpy(input->data, audio, (size_t)n_samples * sizeof(float));
    }
    
    // Build computation graph
    if (cache != nullptr && !legacy_arena) cache->lifetime = true;
    struct ggml_tensor* result = encoder.forward(compute_ctx_ref, input, cache);
    
    // Build graph with pre-allocated nodes (similar to llama_ref.cpp)
    size_t max_nodes = vae_model_max_nodes(ctx->model);
    struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx_ref, max_nodes, false);
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
    
    if (!legacy_arena) {
        ggml_set_input(input);
        ggml_set_output(result);
        if (cache != nullptr) {
            for (size_t ti = 0; ti < cache->taps.size(); ti++) {
                ggml_set_output(cache->taps[ti].xh);
                if (cache->taps[ti].x) ggml_set_output(cache->taps[ti].x);
            }
        }
        if (getenv("VAE_ZC_STATS") != nullptr) fprintf(stderr, "[DBG] prealloc n_samples=%d nodes=%d\n", n_samples, ggml_graph_n_nodes(gf));
        if (!ggml_gallocr_alloc_graph(gallocr_pw[slot], gf)) {
            fprintf(stderr, "[VAE] Error: gallocr_alloc_graph failed (n_samples=%d)\n", n_samples);
            return -1;
        }
        if (getenv("VAE_ZC_STATS") != nullptr) fprintf(stderr, "[DBG] alloc ok buffer=%.1f MB\n", ggml_gallocr_get_buffer_size(gallocr_pw[slot], 0)/1e6);
        // Data placement after allocation: the audio input and the
        // streaming-cache staging buffers.
        if (use_i8_s) {
            int8_t * dst_i8 = (int8_t *) input->data;
            for (int32_t i = 0; i < n_samples; i++) {
                int v = (int)roundf(audio[i] * i8_scale);
                if (v >  127) v =  127;
                if (v < -128) v = -128;
                dst_i8[i] = (int8_t)v;
            }
            float * scale_ptr = (float *)((char *) input->data + n_samples);
            *scale_ptr = i8_scale;
        } else {
            memcpy(input->data, audio, (size_t)n_samples * sizeof(float));
        }
        if (cache != nullptr) {
            for (size_t fi = 0; fi < cache->fills.size(); fi++) {
                const vae_stream_cache::Fill & f = cache->fills[fi];
                vae_fill_staging(f.t, f.hist, f.P, f.dim);
            }
            cache->fills.clear();
        }
    }
    if (getenv("VAE_MEM_STATS") != nullptr) {
        static int mem_dumps = 0;
        if (mem_dumps++ < 4) {
            fprintf(stderr, "[VAE_MEM] slot=%d %s=%.1f MB used=%.1f MB nodes=%d\n",
                    slot, legacy_arena ? "arena" : "lifetime",
                    legacy_arena ? compute_buf_size_ref / 1e6 : ggml_gallocr_get_buffer_size(gallocr_pw[slot], 0) / 1e6,
                    ggml_used_mem(compute_ctx_ref) / 1e6, ggml_graph_n_nodes(gf));
        }
    }
    // Compute
    static bool graph_stats_dumped = false;
    if (getenv("VAE_GRAPH_STATS") != nullptr && !graph_stats_dumped) {
        graph_stats_dumped = true;
        vae_graph_stats_dump(gf, "piecewise");
    }
    if (getenv("VAE_ZC_STATS") != nullptr) fprintf(stderr, "[DBG] precompute\n");
    enum ggml_status st = legacy_arena
        ? ggml_graph_compute_with_ctx(compute_ctx_ref, gf, n_threads_ovr > 0 ? n_threads_ovr : ctx->n_threads)
        : vae_graph_compute_planned(gf, n_threads_ovr > 0 ? n_threads_ovr : ctx->n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: Graph computation failed\n");
        return -1;
    }
    if (getenv("VAE_ZC_STATS") != nullptr) fprintf(stderr, "[DBG] postcompute\n");

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

// ============================================================================
// Deferred late-stage encode (window-level batching of the deep stages).
//
// Motivation (measured): at 26 pieces per window the deepest ConvNeXt stage sees
// ONE latent frame per 3200-sample piece, so its ffn linears are GEMVs
// (a=[8192,2048] x b=[8192,1]) running at ~13.6 GMAC/s and re-streaming ~150 MB
// of weights per piece per encoder. The early stages must stay piece-wise (their
// activations are what the RAM budget allows), but the deep tail can be batched
// over the whole window: the streaming cache already makes the piece-wise early
// stages bit-equal to a full-window run, so the concatenated boundary tensors
// ARE the window-level boundary activations. Running [split, n_stages) once over
// them turns the deep GEMVs into L=26 GEMMs and reads each deep weight once per
// window instead of once per piece. The late convs then need no cache: with the
// whole sequence present they only zero-pad at the window start, which is
// exactly the legacy cold-window semantics the cache reproduces piece-wise.
// ============================================================================
// Lifetime-allocation graphs live in a small metadata-only context, so the
// generic ggml_graph_compute_with_ctx cannot be used: it allocates the op work
// buffer INSIDE the context arena (ggml_new_object(ctx, WORK_BUFFER, work_size)
// then work_data = ctx->mem_buffer + obj->offs) and silently overruns a small
// arena - on device that showed up as a segfault for inputs where work_size
// exceeded the metadata arena (between 96k and 144k samples, Exp579). Plan and
// allocate the work buffer ourselves instead.
static enum ggml_status vae_graph_compute_planned(struct ggml_cgraph * gf, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, nullptr);
    void * work = nullptr;
    if (cplan.work_size > 0) {
        work = malloc(cplan.work_size);
        if (work == nullptr) return GGML_STATUS_ALLOC_FAILED;
    }
    cplan.work_data = (uint8_t *) work;
    enum ggml_status st = ggml_graph_compute(gf, &cplan);
    free(work);
    return st;
}

static int vae_late_split(void) {
    // Shipped default: only the deepest stage is deferred. Split 5 was measured
    // (Exp565, 40-utt gate) at RTF -0.9% / VAE -1.4% but WER 4.41 -> 4.68%
    // (+0.27 pp, 34/40 transcripts identical), i.e. a net loss for the
    // recommended tier whose contract is "fastest acceptable accuracy"; it stays
    // an option (VAE_LATE_SPLIT=5). Splits <= 4 are additionally blocked by the
    // arena holding every tensor live (a 256 MiB single allocation at split 4):
    // they need the lifetime-based allocator project, not numerical work.
    int split = 6;
    if (const char* e = getenv("VAE_LATE_SPLIT")) {
        int v = atoi(e);
        if (v >= 0 && v <= 7) split = v;
    }
    return split;
}

// Encode stages [0, split) of one piece. Writes the dense boundary tensor
// (shape bshape = the graph result's ne[0..2]) to boundary_out and returns its
// element count, or -1. The streaming cache is carried exactly as in the
// un-split path.
static int32_t vae_encode_early_impl(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* audio,
    int32_t n_samples,
    float* boundary_out,
    int64_t bshape[4],
    vae_stream_cache* cache,
    int split,
    int slot,
    float* inference_time_ms,
    int n_threads_ovr = 0) {

    if (!ctx || !audio || !boundary_out || n_samples <= 0) return -1;
    struct ggml_context*& compute_ctx_ref = (slot == 0) ? ctx->compute_ctx  : ctx->compute_ctx2;
    void*&                compute_buf_ref = (slot == 0) ? ctx->compute_buf  : ctx->compute_buf2;
    size_t&               compute_buf_size_ref = (slot == 0) ? ctx->compute_buf_size : ctx->compute_buf_size2;

    struct timespec start_time, end_time;
    if (inference_time_ms) clock_gettime(CLOCK_MONOTONIC, &start_time);

    const size_t bytes_per_sample = 65536;  // F16-weight graph (see vae_encode_impl)
    // Allocation mode. Default: the graph allocator (ggml-alloc) reuses buffers
    // by tensor liveness, so the arena holds only the PEAK LIVE SET instead of
    // every tensor of the graph (the legacy fixed arena costs ~64 KB per input
    // sample: 274 MB at 3200 samples, 2.7 GB at 41600). VAE_LEGACY_ARENA=1
    // restores the historical arena for A/B measurement.
    static ggml_gallocr_t gallocr[2] = { nullptr, nullptr };
    // Metadata arena (tensor structs only, no data): allocated once per slot and
    // re-initialised per call, so pieces do not pay a fresh 64 MB mmap each.
    static void * md_buf[2] = { nullptr, nullptr };
    static size_t md_size[2] = { 0, 0 };
    const bool legacy_arena = (getenv("VAE_LEGACY_ARENA") != nullptr);

    size_t arena_off = 0;
    if (const char* e = getenv("VAE_ARENA_OFFSET")) arena_off = (size_t)strtoull(e, nullptr, 0);
    if (compute_ctx_ref) ggml_free(compute_ctx_ref);
    if (legacy_arena) {
        const size_t vae_ctx_mem_size = (size_t)n_samples * bytes_per_sample + (size_t)64 * 1024 * 1024;
        if (compute_buf_size_ref < vae_ctx_mem_size) {
            void* grown = realloc(compute_buf_ref, vae_ctx_mem_size);
            if (grown == NULL) { fprintf(stderr, "[VAE] Error: early arena alloc failed\n"); return -1; }
            compute_buf_ref = grown; compute_buf_size_ref = vae_ctx_mem_size;
        }
        struct ggml_init_params ctx_params = { compute_buf_size_ref, compute_buf_ref, false };
        // Diagnostic: the arena is reused across pieces, so a systematic
        // cache-set/aliasing interaction between the packed activation tensors
        // would show up as a base-address dependence. VAE_ARENA_OFFSET shifts
        // the ggml buffer base by N bytes (default 0 = shipped behaviour).
        if (arena_off > 0 && compute_buf_size_ref > arena_off + 1024 * 1024) {
            ctx_params.mem_buffer = (char*)compute_buf_ref + arena_off;
            ctx_params.mem_size = compute_buf_size_ref - arena_off;
        }
        compute_ctx_ref = ggml_init(ctx_params);
    } else {
        if (gallocr[slot] == nullptr) {
            gallocr[slot] = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            if (gallocr[slot] == nullptr) { fprintf(stderr, "[VAE] Error: gallocr init failed\n"); return -1; }
        }
        // no_alloc: this context only holds tensor metadata; the tensor data is
        // placed by the graph allocator below (and its buffers persist across
        // pieces because the gallocr object is cached per slot).
        const size_t md_need = (size_t)16 * 1024 * 1024;
        if (md_size[slot] < md_need) {
            void * grown = realloc(md_buf[slot], md_need);
            if (grown == NULL) { fprintf(stderr, "[VAE] Error: metadata arena alloc failed\n"); return -1; }
            md_buf[slot] = grown; md_size[slot] = md_need;
        }
        struct ggml_init_params ctx_params = { md_size[slot], md_buf[slot], true };
        compute_ctx_ref = ggml_init(ctx_params);
        if (compute_ctx_ref == nullptr) { fprintf(stderr, "[VAE] Error: ctx init failed\n"); return -1; }
    }

    struct ggml_tensor* input = ggml_new_tensor_3d(compute_ctx_ref, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(input, "input_audio");
    if (legacy_arena) memcpy(input->data, audio, (size_t)n_samples * sizeof(float));

    if (cache != nullptr && !legacy_arena) cache->lifetime = true;
    struct ggml_tensor* result = encoder.forward_early(compute_ctx_ref, input, cache, split);

    struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx_ref, vae_model_max_nodes(ctx->model), false);
    ggml_build_forward_expand(gf, result);
    if (cache != nullptr) {
        for (size_t ti = 0; ti < cache->taps.size(); ti++) {
            ggml_build_forward_expand(gf, cache->taps[ti].xh);
            if (cache->taps[ti].x) ggml_build_forward_expand(gf, cache->taps[ti].x);
        }
    }
    if (!legacy_arena) {
        // Lifetime allocation: input at the front, outputs (the boundary tensor
        // and every streaming-cache concat node read back after the compute)
        // pinned so their buffers survive to the end of the graph.
        ggml_set_input(input);
        ggml_set_output(result);
        if (cache != nullptr) {
            for (size_t ti = 0; ti < cache->taps.size(); ti++) {
                ggml_set_output(cache->taps[ti].xh);
                if (cache->taps[ti].x) ggml_set_output(cache->taps[ti].x);
            }
        }
        if (!ggml_gallocr_alloc_graph(gallocr[slot], gf)) {
            fprintf(stderr, "[VAE] Error: gallocr_alloc_graph failed (n_samples=%d)\n", n_samples);
            return -1;
        }
        if (getenv("VAE_MEM_STATS") != nullptr) {
            static int mem_dumps = 0;
            if (mem_dumps++ < 4) {
                fprintf(stderr, "[VAE_MEM] slot=%d lifetime buffer=%.1f MB nodes=%d\n",
                        slot, ggml_gallocr_get_buffer_size(gallocr[slot], 0) / 1e6,
                        ggml_graph_n_nodes(gf));
            }
        }
        // Data placement happens after the allocation, so write the input now
        // and materialise the streaming-cache staging buffers.
        memcpy(input->data, audio, (size_t)n_samples * sizeof(float));
        if (cache != nullptr) {
            for (size_t fi = 0; fi < cache->fills.size(); fi++) {
                const vae_stream_cache::Fill & f = cache->fills[fi];
                vae_fill_staging(f.t, f.hist, f.P, f.dim);
            }
            cache->fills.clear();
        }
    }
    struct timespec c0, c1;
    clock_gettime(CLOCK_MONOTONIC, &c0);
    if (getenv("VAE_GRAPH_STATS") != nullptr) {
        static bool dumped_early = false;
        if (!dumped_early) { dumped_early = true; vae_graph_stats_dump(gf, "early"); }
    }
    if (vae_graph_compute_planned(gf, n_threads_ovr > 0 ? n_threads_ovr : ctx->n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: early graph compute failed\n"); return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &c1);
    if (getenv("VAE_LATE_STATS") != nullptr) {
        static int dumps = 0;
        if (dumps++ < 6) {
            fprintf(stderr, "[VAE_EARLY] slot=%d nsamp=%d split=%d nthr=%d nodes=%d compute=%.1f ms used=%.1f MB\n",
                    slot, n_samples, split, n_threads_ovr > 0 ? n_threads_ovr : ctx->n_threads,
                    ggml_graph_n_nodes(gf),
                    (c1.tv_sec - c0.tv_sec) * 1000.0 + (c1.tv_nsec - c0.tv_nsec) / 1e6,
                    ggml_used_mem(compute_ctx_ref) / 1e6);
        }
    }
    if (cache != nullptr) vae_cache_update(cache);

    if (!ggml_is_contiguous(result)) {
        fprintf(stderr, "[VAE] Error: early boundary not contiguous\n"); return -1;
    }
    for (int d = 0; d < GGML_MAX_DIMS; d++) bshape[d] = result->ne[d];
    const int64_t n_elems = ggml_nelements(result);
    memcpy(boundary_out, result->data, (size_t)n_elems * sizeof(float));

    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        *inference_time_ms = (float)((end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                     (end_time.tv_nsec - start_time.tv_nsec) / 1e6);
    }
    return (int32_t)n_elems;
}

// Stages [split, n_stages) + head + connector over a whole window's boundary
// tensor (n_time_total x C, memory order matching forward_early's output).
// Returns the frame count, or -1.
static int32_t vae_encode_late_impl(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* boundary,
    const int64_t bshape[4],
    int64_t n_time_total,
    float* output,
    int split,
    int slot,
    float* inference_time_ms,
    int n_threads_ovr = 0) {

    if (!ctx || !boundary || !output || n_time_total <= 0) return -1;
    const int64_t C = bshape[1];
    if (bshape[0] <= 0 || C <= 0) return -1;
    struct ggml_context*& compute_ctx_ref = (slot == 0) ? ctx->compute_ctx  : ctx->compute_ctx2;
    void*&                compute_buf_ref = (slot == 0) ? ctx->compute_buf  : ctx->compute_buf2;
    size_t&               compute_buf_size_ref = (slot == 0) ? ctx->compute_buf_size : ctx->compute_buf_size2;

    struct timespec start_time, end_time;
    if (inference_time_ms) clock_gettime(CLOCK_MONOTONIC, &start_time);

    // The blocked-int8 GEMM kernel interleaves 4 activation columns; a column
    // count that is not a multiple of 4 drops off the fast path (measured on
    // device: VAE 314 s vs 26 s at 26 columns). Every deeper layer's column
    // count is the frame count times the stride product, so pad the FRAME count
    // to a multiple of 4 with zero columns. All late ops are causal (convs) or
    // pointwise (norms, linears), so the real frames are unaffected.
    int64_t cum = 1;
    for (int i = split; i < encoder.n_stages; i++) cum *= encoder.downsample_strides[i];
    int64_t t_pad = n_time_total;
    const int64_t n_frames_want = n_time_total / cum;
    if (n_time_total % cum == 0) {
        const int64_t frames_pad = (n_frames_want + 3) / 4 * 4;
        t_pad = frames_pad * cum;
    }

    // Allocation: same lifetime (ggml-alloc) scheme as the early pass, which
    // measured 2.5% faster and 160 MB leaner there by keeping only the live set
    // resident. VAE_LEGACY_ARENA=1 falls back to the fixed arena; with it,
    // VAE_LATE_ARENA_MB overrides the budget (default 128 MB). The late graph
    // has no streaming-cache sites, so there are no staging tensors to fill.
    static ggml_gallocr_t gallocr_late[2] = { nullptr, nullptr };
    static void * md_buf_late[2] = { nullptr, nullptr };
    static size_t md_size_late[2] = { 0, 0 };
    const bool legacy_late = (getenv("VAE_LEGACY_ARENA") != nullptr);
    if (compute_ctx_ref) ggml_free(compute_ctx_ref);
    if (legacy_late) {
        size_t late_arena = (size_t)128 * 1024 * 1024;
        if (const char* e = getenv("VAE_LATE_ARENA_MB")) {
            long v = atol(e);
            if (v > 0) late_arena = (size_t)v * 1024 * 1024;
        }
        const size_t vae_ctx_mem_size = late_arena;
        if (compute_buf_size_ref < vae_ctx_mem_size) {
            void* grown = realloc(compute_buf_ref, vae_ctx_mem_size);
            if (grown == NULL) { fprintf(stderr, "[VAE] Error: late arena alloc failed\n"); return -1; }
            compute_buf_ref = grown; compute_buf_size_ref = vae_ctx_mem_size;
        }
        struct ggml_init_params ctx_params = { compute_buf_size_ref, compute_buf_ref, false };
        compute_ctx_ref = ggml_init(ctx_params);
        if (compute_ctx_ref == nullptr) { fprintf(stderr, "[VAE] Error: late ctx init failed\n"); return -1; }
    } else {
        if (gallocr_late[slot] == nullptr) {
            gallocr_late[slot] = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            if (gallocr_late[slot] == nullptr) { fprintf(stderr, "[VAE] Error: late gallocr init failed\n"); return -1; }
        }
        const size_t md_need = (size_t)16 * 1024 * 1024;
        if (md_size_late[slot] < md_need) {
            void * grown = realloc(md_buf_late[slot], md_need);
            if (grown == NULL) { fprintf(stderr, "[VAE] Error: late metadata alloc failed\n"); return -1; }
            md_buf_late[slot] = grown; md_size_late[slot] = md_need;
        }
        struct ggml_init_params ctx_params = { md_size_late[slot], md_buf_late[slot], true };
        compute_ctx_ref = ggml_init(ctx_params);
        if (compute_ctx_ref == nullptr) { fprintf(stderr, "[VAE] Error: late ctx init failed\n"); return -1; }
    }

    struct ggml_tensor* input = ggml_new_tensor_3d(compute_ctx_ref, GGML_TYPE_F32, t_pad, C, 1);
    ggml_set_name(input, "input_boundary");
    // boundary is [n_time_total, C] with time fastest (element (t,c) at t + c*T),
    // so the pad columns are per-channel tails, not one flat block. Under the
    // lifetime allocator the write moves after ggml_gallocr_alloc_graph below.
    if (legacy_late) {
        const float* src = boundary;
        float* dst = (float*)input->data;
        for (int64_t c = 0; c < C; c++) {
            memcpy(dst + (size_t)c * t_pad, src + (size_t)c * n_time_total,
                   (size_t)n_time_total * sizeof(float));
            if (t_pad > n_time_total)
                memset(dst + (size_t)c * t_pad + n_time_total, 0,
                       (size_t)(t_pad - n_time_total) * sizeof(float));
        }
    }

    // cache == nullptr: the sequence provides its own history (cold window head).
    struct ggml_tensor* result = encoder.forward_late(compute_ctx_ref, input, nullptr, split);

    struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx_ref, vae_model_max_nodes(ctx->model), false);
    ggml_build_forward_expand(gf, result);
    if (!legacy_late) {
        ggml_set_input(input);
        ggml_set_output(result);
        if (!ggml_gallocr_alloc_graph(gallocr_late[slot], gf)) {
            fprintf(stderr, "[VAE] Error: late gallocr_alloc_graph failed\n"); return -1;
        }
        const float* src = boundary;
        float* dst = (float*)input->data;
        for (int64_t c = 0; c < C; c++) {
            memcpy(dst + (size_t)c * t_pad, src + (size_t)c * n_time_total,
                   (size_t)n_time_total * sizeof(float));
            if (t_pad > n_time_total)
                memset(dst + (size_t)c * t_pad + n_time_total, 0,
                       (size_t)(t_pad - n_time_total) * sizeof(float));
        }
        if (getenv("VAE_MEM_STATS") != nullptr) {
            static int dumps = 0;
            if (dumps++ < 4) {
                fprintf(stderr, "[VAE_MEM] late slot=%d buffer=%.1f MB nodes=%d\n",
                        slot, ggml_gallocr_get_buffer_size(gallocr_late[slot], 0) / 1e6,
                        ggml_graph_n_nodes(gf));
            }
        }
    }
    if (getenv("VAE_LATE_STATS") != nullptr) {
        static int dumps = 0;
        if (dumps++ < 3) {
            fprintf(stderr, "[VAE_LATE] slot=%d in=[%lld,%lld] arena=%.1f MB used=%.1f MB nodes=%d\n",
                    slot, (long long)n_time_total, (long long)C,
                    compute_buf_size_ref / 1e6, ggml_used_mem(compute_ctx_ref) / 1e6,
                    ggml_graph_n_nodes(gf));
        }
    }
    struct timespec b1, c0, c1;
    clock_gettime(CLOCK_MONOTONIC, &b1);
    clock_gettime(CLOCK_MONOTONIC, &c0);
    if (getenv("VAE_GRAPH_STATS") != nullptr) {
        static bool dumped_late = false;
        if (!dumped_late) { dumped_late = true; vae_graph_stats_dump(gf, "late"); }
    }
    if (vae_graph_compute_planned(gf, n_threads_ovr > 0 ? n_threads_ovr : ctx->n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: late graph compute failed\n"); return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &c1);
    if (getenv("VAE_LATE_STATS") != nullptr) {
        static int dumps = 0;
        if (dumps++ < 6) {
            fprintf(stderr, "[VAE_LATE] slot=%d t_pad=%lld C=%lld frames=%lld nthr=%d nodes=%d build=%.1f compute=%.1f used=%.1f MB\n",
                    slot, (long long)t_pad, (long long)C, (long long)n_frames_want,
                    n_threads_ovr > 0 ? n_threads_ovr : ctx->n_threads, ggml_graph_n_nodes(gf),
                    (b1.tv_sec - start_time.tv_sec) * 1000.0 + (b1.tv_nsec - start_time.tv_nsec) / 1e6,
                    (c1.tv_sec - c0.tv_sec) * 1000.0 + (c1.tv_nsec - c0.tv_nsec) / 1e6,
                    ggml_used_mem(compute_ctx_ref) / 1e6);
        }
    }

    const int64_t n_frames = result->ne[1];
    const int64_t out_dim  = result->ne[0];
    if (n_frames < n_frames_want) {
        fprintf(stderr, "[VAE] Error: late frames %lld < want %lld\n",
                (long long)n_frames, (long long)n_frames_want);
        return -1;
    }
    memcpy(output, result->data, (size_t)(n_frames_want * out_dim) * sizeof(float));

    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        *inference_time_ms = (float)((end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                     (end_time.tv_nsec - start_time.tv_nsec) / 1e6);
    }
    return (int32_t)n_frames_want;
}

int32_t vae_encode_early_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* boundary_out,
    int64_t bshape[4]) {

    if (!cache) return -1;
    return vae_encode_early_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples,
                                 boundary_out, bshape, &cache->acoustic, vae_late_split(), 0, nullptr);
}

int32_t vae_encode_late(
    vae_context_t* ctx,
    const float* boundary,
    const int64_t bshape[4],
    int64_t n_time_total,
    float* output) {

    return vae_encode_late_impl(ctx, ctx->model->acoustic_encoder, boundary, bshape,
                                n_time_total, output, vae_late_split(), 0, nullptr);
}

// Both encoders' early stages for one piece, concurrently (one thread each).
int32_t vae_encode_early_parallel_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* ab, int64_t ashape[4],
    float* sb, int64_t sshape[4],
    float* acoustic_ms, float* semantic_ms) {

    if (!ctx || !cache) return -1;
    const int split = vae_late_split();
    int32_t ra = -1, rs = -1;
    std::thread ta([&]() {
        ra = vae_encode_early_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples,
                                   ab, ashape, &cache->acoustic, split, 0, acoustic_ms, 1);
    });
    std::thread tb([&]() {
        rs = vae_encode_early_impl(ctx, ctx->model->semantic_encoder, audio, n_samples,
                                   sb, sshape, &cache->semantic, split, 1, semantic_ms, 1);
    });
    ta.join(); tb.join();
    if (ra < 0 || rs < 0) {
        fprintf(stderr, "[VAE] parallel early mismatch: acoustic=%d semantic=%d\n", ra, rs);
        return -1;
    }
    return 0;
}

// Both encoders' late stages for one window, concurrently (one thread each).
int32_t vae_encode_late_parallel(
    vae_context_t* ctx,
    const float* ab, const int64_t ashape[4],
    const float* sb, const int64_t sshape[4],
    int64_t n_time_total,
    float* aout, float* sout,
    float* acoustic_ms, float* semantic_ms) {

    if (!ctx) return -1;
    const int split = vae_late_split();
    int32_t ra = -1, rs = -1;
    std::thread ta([&]() {
        ra = vae_encode_late_impl(ctx, ctx->model->acoustic_encoder, ab, ashape,
                                  n_time_total, aout, split, 0, acoustic_ms, 1);
    });
    std::thread tb([&]() {
        rs = vae_encode_late_impl(ctx, ctx->model->semantic_encoder, sb, sshape,
                                  n_time_total, sout, split, 1, semantic_ms, 1);
    });
    ta.join(); tb.join();
    if (ra < 0 || rs < 0) {
        fprintf(stderr, "[VAE] parallel late mismatch: acoustic=%d semantic=%d\n", ra, rs);
        return -1;
    }
    return (ra == rs) ? ra : -1;
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

// Run the two encoders concurrently, one thread each. They are independent
// (separate weights, separate streaming caches, separate arenas), so this is
// semantically identical to the sequential pair - only the wall-clock overlaps.
int32_t vae_encode_parallel_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* output_acoustic,
    float* output_semantic,
    float* acoustic_ms,
    float* semantic_ms) {

    if (!ctx || !cache) return -1;
    // 1 thread per chain is the measured optimum (2 chains saturate the two
    // prime cores); VAE_PAR_THREADS lets an experiment try more per chain.
    int nthreads_per_chain = 1;
    if (const char* e = getenv("VAE_PAR_THREADS")) {
        int v = atoi(e);
        if (v > 0) nthreads_per_chain = v;
    }
    int32_t ra = -1, rs = -1;
    std::thread ta([&]() {
        ra = vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples,
                             output_acoustic, acoustic_ms, &cache->acoustic, 0, nthreads_per_chain);
    });
    std::thread tb([&]() {
        rs = vae_encode_impl(ctx, ctx->model->semantic_encoder, audio, n_samples,
                             output_semantic, semantic_ms, &cache->semantic, 1, nthreads_per_chain);
    });
    ta.join();
    tb.join();
    if (ra != rs || ra < 0) {
        fprintf(stderr, "[VAE] parallel encode mismatch: acoustic=%d semantic=%d\n", ra, rs);
        return -1;
    }
    return ra;
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
