#include "vae.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vae-i8_s-mad.h"

#include "time_compat.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>
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
//   VAE_ABL_CONV  the non-depthwise conv MATMULs (view+cast substitute; Exp683)
//   VAE_ABL_SCALE layer-scale and gamma muls
//   VAE_ABL_RESID block residual adds
static bool vae_abl(const char* name) {
    // thread_local ON PURPOSE (Exp669): with a plain function-local static, the two concurrent
    // encoder chains mutate this red-black tree (emplace of a new key) while the other thread is
    // searching it - undefined behaviour that shows up as a NULL child dereference inside
    // std::__tree_balance_after_insert. That race, not the depthwise conv kernel, was the
    // non-deterministic segfault of Exp666-668: the fault PC resolved to libc++ tree rebalancing
    // and the arms that avoided it were the ones where every key got inserted before the threads
    // started racing. Per-thread memo = no shared mutation, no lock on the hot path.
    thread_local std::map<std::string, int> cache;
    auto it = cache.find(name);
    if (it == cache.end()) {
        const bool on = getenv(name) != nullptr;
        it = cache.emplace(name, on ? 1 : 0).first;
        if (on) fprintf(stderr, "[VAE_ABL] %s is ON - output is INVALID, timing ceiling only\n", name);
    }
    return it->second != 0;
}

// Exp862 instrumentation: name a cont() node with its SITE number so GGML_OP_TIME can attribute that
// op per call site (parsed by ggml_cont_site in ggml.c). Diagnostic only - naming a node cannot
// change the graph, and the protocol transcript is byte-identical with and without it.
static struct ggml_tensor* vae_ct_site(struct ggml_tensor* t, int site) {
    char nm[16]; snprintf(nm, sizeof(nm), "cts%d", site);
    ggml_set_name(t, nm);
    return t;
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
    // padded=false: Exp821, the concat node was skipped (cold site, kernel does the left pad), so the tap must
    // derive the new history from x plus the implicit zeros instead of reading the padded tensor.
    struct Tap { std::string key; struct ggml_tensor * xh = nullptr; struct ggml_tensor * x = nullptr; int64_t P = 0; int dim = 0; bool padded = true; };
    std::vector<Tap> taps;     // recorded per forward build, consumed post-compute
    // Exp821: gate for the in-kernel causal left pad. It is output-neutral only where the history this build's
    // tap rolls is NEVER read back - true when the whole window is one piece and no deferred late pass rolls
    // the same slots (histories are cleared per window by vae_cache_reset). Measured otherwise: at p1 six runs
    // byte-identical, at p13+defer-ON the transcript changed (38 vs 39 tokens), so the fast path stays OFF
    // there until that interaction is understood. Config, so vae_cache_reset must not clear it.
    bool single_piece_window = false;
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
    int time_dim = 0,
    int64_t * lpad_out = nullptr) {
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
        // Exp821: a cold site's history is ALL ZEROS, so [hist | x] is just a left zero-pad - which the depthwise
        // conv1d kernel can do for free by starting its tap loop later (ggml_conv1d_dw_ct_lp). Returning x
        // unpadded deletes the pad node outright: 0.73 s per chain, and 75% of those bytes feed the dw kernel
        // (PAD-by-consumer: 123.1 of 164 MB). Only offered when the caller can consume the left pad, i.e. from
        // the conv1d-kernel branch, and the tap is told to roll the history from x plus zeros. At the shipped
        // p1 every site is cold (one piece per window), so this covers the whole dw share of the splice cost.
        // Exp823 diagnostic gate: VAE_DW_LPAD_FORCE=1 offers the fast path at ANY granularity, =2 offers it
        // only where P <= T (so the roll below never needs its leading-zeros branch, which is the case that
        // cannot occur at the shipped p1 and is the prime suspect for Exp821's one-token lean divergence).
        int64_t lpad_T = (time_dim == 0) ? x->ne[0] : x->ne[1];
        const char * lpad_force_env = getenv("VAE_DW_LPAD_FORCE");
        const bool   lpad_force_ok  = lpad_force_env != nullptr &&
                                      (atoi(lpad_force_env) >= 2 ? P <= lpad_T : true);
        if (lpad_out != nullptr && !vae_abl("VAE_DW_LPAD_OFF") &&
            (cache->single_piece_window || lpad_force_ok)) {
            if (getenv("VAE_LPAD_TRACE") != nullptr) {
                // thread_local IS required here: a plain function-local static is process-wide, and both
                // encoder chains call this -> concurrent std::map insert = the Exp669 race, reintroduced.
                static thread_local std::map<std::string, int> seen;
                if (seen[key]++ == 0) {
                    fprintf(stderr, "[LPAD] %s P=%lld T=%lld dim=%d warm_before=%d zeros_head=%lld\n", key.c_str(),
                            (long long)P, (long long)lpad_T, time_dim, slot.warm ? 1 : 0,
                            (long long)(P > lpad_T ? P - lpad_T : 0));
                }
            }
            vae_stream_cache::Tap tap;
            tap.key = key; tap.xh = x; tap.x = x; tap.P = P; tap.dim = time_dim; tap.padded = false;
            cache->taps.push_back(tap);
            *lpad_out = P;
            return x;
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
            if (!tap.padded) {
                // New history = the last P columns of the IMPLICIT [zeros(P) | x] (the pad node does not
                // exist). Column T+k of that tensor is zero while T+k < P, else x[T+k-P], so the first
                // max(0, P-T) history entries are zero and the rest is a contiguous run of x's tail.
                const int64_t T    = (tap.dim == 0) ? xh->ne[0] : xh->ne[1];
                const int64_t nz   = tap.P > T ? tap.P - T : 0;              // leading zeros
                const int64_t copy = tap.P - nz;                             // entries coming from x
                const int64_t src  = T - tap.P + nz;                         // == max(0, T - P)
                if (tap.dim == 0) {
                    const int64_t N = xh->ne[0];
                    const int64_t nch = xh->ne[1] * xh->ne[2] * xh->ne[3];
                    for (int64_t c = 0; c < nch; c++) {
                        std::fill(slot.hist.data() + c * tap.P, slot.hist.data() + c * tap.P + nz, 0.0f);
                        memcpy(slot.hist.data() + c * tap.P + nz, xhd + c * N + src, (size_t)copy * sizeof(float));
                    }
                } else {
                    const int64_t C = xh->ne[0];
                    for (int64_t c = 0; c < C; c++) {
                        std::fill(slot.hist.data() + c * tap.P, slot.hist.data() + c * tap.P + nz, 0.0f);
                        for (int64_t k = 0; k < copy; k++) {
                            slot.hist[c * tap.P + nz + k] = xhd[src + k * C];
                        }
                    }
                }
            } else if (tap.dim == 0) {
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
        // VAE_ABL_NORM (measurement-only, Exp779): RMS_NORM is 53 nodes / 490 MB per graph build.
        // Skipping it keeps every shape, so the timing is a true ceiling; output INVALID by design.
        //
        // VAE_NORM_FUSE_OFF=1 restores the norm-then-mul form. The f32 path used to run rms_norm and a
        // SEPARATE ggml_mul for gamma (4 tensor passes); ggml_rms_norm_gamma folds the gain in, which is
        // bit-identical because it is the same two f32 multiplies in the same order.
        // NOTE for future ablation runs: with the fold active there IS no separate scale node, so
        // VAE_ABL_SCALE legitimately removes nothing - its Exp783 0.30 s is already inside the norm's
        // cost here. Reading either knob as "this op is free" after the fold is the stale-knob error
        // that has misled this loop five times.
        if (!vae_abl("VAE_ABL_NORM")) {
            bool fuse = gamma && gamma->type == GGML_TYPE_F32 && ggml_is_contiguous(gamma)
                     && !getenv("VAE_NORM_FUSE_OFF");
            x = fuse ? ggml_rms_norm_gamma(ctx, x, gamma, 1e-5f)
                     : ggml_rms_norm(ctx, x, 1e-5f);
            if (!fuse) x = vae_abl_mul(ctx, x, gamma, "VAE_ABL_SCALE");
        }
    }
    
    return x;
}

static struct ggml_tensor* ggml_nn_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    bool apply_bias = true) {   // false = return the pre-bias tensor so a consumer can fuse it (Exp673)
    
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
        if (b != NULL && apply_bias) {
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
    struct ggml_tensor* im2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1]));
    if (vae_abl("VAE_ABL_CONV")) {
        // Measurement-only (Exp683): drop the conv MATMUL and keep everything else - shapes, the
        // im2col, the bias add, the launch count. A subset view of the im2col rows has the same
        // [OC, cols] shape whenever K*IC >= OC, so unlike shrinking the contraction (Exp516, which
        // silently switched to a slower tiny-dot path) this removes the F16 dot work and nothing
        // else. The cast back to F32 is what mul_mat would have written anyway and is ~two orders
        // cheaper than the dots. Output is INVALID by design: never gate it, never ship it, never
        // quote accuracy from it. Read the result off vae_s, not rtf (vae_s is LM-independent).
        const int64_t OC = a->ne[2];
        if (OC <= im2d->ne[0]) {
            return ggml_cast(ctx, ggml_view_2d(ctx, im2d, OC, im2d->ne[1], im2d->nb[1], 0),
                             GGML_TYPE_F32);
        }
        static thread_local bool warned = false;   // thread_local: a shared static here is the Exp669 race
        if (!warned) { warned = true;
            fprintf(stderr, "[VAE_ABL_CONV] a shape with OC > K*IC fell through; ablation under-reports\n");
        }
    }
    return ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]),
            im2d);
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
    result = vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3)), 1);
    return ggml_reshape_3d(ctx, result, C, OL, N);
}

// --- blocked-int8 conv weights (Exp685) -------------------------------------------------------
// A non-depthwise conv weight can be stored two ways:
//   F16        3-D [K, IC, OC]   the original layout; im2col reads its shape for the geometry
//   Q4_0_4x4   2-D [K*IC, OC]    blocked int8, so the conv dots run the fast 4x4 gemm instead of
//                                F16 dots. The VAE is kernel-RATE-bound (Exp681) and these dots are
//                                12.3% of VAE seconds (Exp683), which is the whole case for this.
// The blocked layout needs the matmul row (= ne[0]) to be a multiple of the 32-element block. For
// every non-depthwise conv in this model K*IC already is (128..16384, verified in Exp684), so NO
// kernel padding is required - and no data reordering either, because a 3-D [K,IC,OC] tensor's
// memory order (kw fastest, then ic) is exactly the row-major [K*IC, OC] matrix that im2col's row
// order (ic*KW + kw) expects.
// What the 2-D form costs: im2col can no longer tell K from IC, and ggml_im2col asserts an F16 src0
// even though its loops provably read only src1->data. So the geometry rides in a metadata-only
// carrier - an F16-shaped tensor that is never read. Its data points at a static dummy on purpose:
// a leaf with data == NULL gets handed to the pool allocator, and [K,IC,OC] in F16 is ~700 MB.
static struct ggml_context* vae_geom_ctx = nullptr;
static float g_vae_geom_dummy[16];

// Called from the loader only. Deliberately not lazily initialized inside the graph path: two
// concurrent encoder threads racing on a shared lazy init is the Exp669 segfault.
static void vae_geom_ctx_init() {
    if (!vae_geom_ctx) {
        struct ggml_init_params params = { /*mem_size=*/ 1 << 16, /*mem_buffer=*/ nullptr,
                                           /*no_alloc=*/ true };
        vae_geom_ctx = ggml_init(params);
    }
}

// Kernel size of a conv weight. 3-D: ne[0]. Blocked int8 2-D [K*IC, OC]: row / IC, with IC from the
// architecture table because the tensor no longer carries it. Returns -1 for an unsupported layout.
static int vae_conv_kernel_size(const struct ggml_tensor* w, int64_t ic) {
    if (ggml_n_dims(w) == 2 && ggml_is_quantized(w->type)) {
        if (ic <= 0 || w->ne[0] % ic != 0) return -1;
        return (int) (w->ne[0] / ic);
    }
    return (int) w->ne[0];
}

// Geometry carrier for a blocked-int8 conv weight; nullptr when the weight needs none (F16 path).
static struct ggml_tensor* vae_conv_geom(struct ggml_tensor* w, int K) {
    if (!w || ggml_n_dims(w) != 2 || !ggml_is_quantized(w->type)) return nullptr;
    const int64_t row = w->ne[0], OC = w->ne[1];
    if (K <= 0 || row % K != 0 || row % 32 != 0 || OC % 4 != 0) return nullptr;
    struct ggml_tensor* g = ggml_new_tensor_3d(vae_geom_ctx, GGML_TYPE_F16, K, row / K, OC);
    g->data = g_vae_geom_dummy;   // never dereferenced: im2col uses ne[] only
    return g;
}

static struct ggml_tensor* vae_conv_1d_i8(
        struct ggml_context* ctx, struct ggml_tensor* geom, struct ggml_tensor* w,
        struct ggml_tensor* x, int s0, int p0, int d0) {
    // This fork's blocked mul_mat supports an F32 src1 only: it converts to Q8_0 internally (the
    // from_float_to_mat route, which is what the LM uses every run). Handing it a PRE-converted Q8_0
    // tensor computes something else - silently, since device builds compile the asserts out - and the
    // symptom is a 1024-token runaway instead of the reference 39 with the same file (int8 file, 2 reps
    // each: F32 src1 -> 2.4649/2.4632 with the reference transcript; Q8_0 src1 -> 11.41/11.50, garbage).
    // The wrong arm is ~2% faster on vae_s, so the switch stays for experiments. The obvious way to get
    // that 2% legitimately was ggml_im2col with dst_type=Q8_0 (ggml's own quantized staging, no cast
    // node): it ABORTS in this build (exit 134, Exp692), so the F32 route is the only supported one.
    // This fork's blocked conv matmul REQUIRES an F32 src1: it converts to Q8_0 internally (the
    // from_float_to_mat route, same as the LM). Two alternatives were tried and both fail, one loudly:
    //   * ggml_im2col(dst=Q8_0)            -> SIGABRT, no quantized im2col path in this build (Exp692)
    //   * ggml_im2col(dst=F16) -> mul_mat  -> GGML_ASSERT(src1->type == GGML_TYPE_F32) (Exp693)
    //   * ggml_cast(F16 -> Q8_0) -> mul_mat-> runs but computes garbage: 1024 tokens vs the reference
    //     39, silently, since that path's asserts are compiled out on device. It is ~2% faster on
    //     vae_s, which is exactly why the hatch below is kept and labeled broken.
    //   * F16 im2col + cast(F16->F32) -> mul_mat (VAE_CONV_I8_F16COL=1) -> CORRECT (byte-identical)
    //     but EXACTLY PARITY (Exp715: 2.3900/2.3942 vs base 2.3831/2.3959, vae_s 15.9 both): the
    //     halved staging bytes do not pay because the cast node adds back a write+read. Staging is
    //     NOT the ~2% - what the broken Q8CAST arm was saving is the in-kernel quantize_mat_q8_0,
    //     which needs the interleaved 4x4 layout no im2col/cast form produces (a ggml.c forward for
    //     that layout + a correct pre-converted read path would be the only route; the current
    //     pre-converted path misreads, silently). The 2% is priced at that surgery and no cheaper.
    const bool q8cast = vae_abl("VAE_CONV_I8_Q8CAST");
    // Shipped route IS the direct-F32 staging (im2col writes F32, the blocked gemm converts it to
    // interleaved Q8_0 in-kernel via quantize_mat_q8_0). The f16col arm halves the staging BYTES
    // but adds an F32-cast node in front of the matmul (the blocked gemm asserts an F32 src1),
    // i.e. F16 write + F16 read + F32 write + F32 read vs F32 write + F32 read: strictly more
    // traffic UNLESS the memory system is latency-bound rather than bandwidth-bound. Untested form.
    const bool f16col = vae_abl("VAE_CONV_I8_F16COL");
    // Exp720 arm: stage the im2col DIRECTLY as interleaved Q8_0 panels (new ggml forward), i.e.
    // exactly the bytes the blocked gemm's in-kernel conversion would have produced. Removes the
    // per-matmul conversion AND 3/4 of the staging bytes. Bit-identity is the acceptance test.
    const bool q8col = vae_abl("VAE_CONV_I8_Q8COL");
    struct ggml_tensor* im2col = ggml_im2col(ctx, geom, x, s0, 0, p0, 0, d0, 0, false,
                                             q8col ? GGML_TYPE_Q8_0 : (f16col ? GGML_TYPE_F16 : GGML_TYPE_F32));
    struct ggml_tensor* col = ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                                              im2col->ne[2] * im2col->ne[1]);
    if (f16col && !q8cast) {
        col = ggml_cast(ctx, col, GGML_TYPE_F32);           // blocked gemm requires an F32 src1
    }
    if (vae_abl("VAE_ABL_CONV")) {   // same measurement-only substitution as the F16 path (Exp683)
        struct ggml_tensor* f = (q8cast || f16col) ? ggml_cast(ctx, col, GGML_TYPE_F32) : col;
        return ggml_view_2d(ctx, f, geom->ne[2], f->ne[1], f->nb[1], 0);
    }
    return ggml_mul_mat(ctx, w, q8cast ? ggml_cast(ctx, col, GGML_TYPE_Q8_0) : col);
}

static struct ggml_tensor* ggml_nn_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation,
    vae_stream_cache* cache = nullptr,
    struct ggml_tensor* geom = nullptr) {

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
        result = ggml_is_quantized(w->type) && geom
                 ? vae_conv_1d_i8(ctx, geom, w, x, stride, padding, dilation)
                 : vae_conv_1d_f16(ctx, w, x, stride, padding, dilation);
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
        result = vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3)), 2);
    } else {
        // Channels-first input [C, T] (the block's natural layout when it is not
        // transposed): time is ne[1], so cache concat and padding go along dim 1.
        const bool ct_in = (x->ne[2] == 1) && (w->ne[1] == 1) && (w->ne[3] == 1) &&
                           (x->ne[0] == w->ne[2]) && (x->ne[1] > 2 * w->ne[0]);
        // Exp821: enumerate the route to ggml_conv1d_dw_ct BEFORE the splice, because only that route can
        // consume an implicit left pad. If lp survives to any other branch (tap chain, im2col fallback) the
        // result would be silently wrong - those branches need the padded tensor to exist - so this predicate
        // must stay the exact complement of the kernel branch's condition below.
        const bool kernel_route = ct_in && stride == 1 && dilation == 1 && b != NULL &&
                                  !vae_abl("VAE_DW_CT_OFF") && !vae_abl("VAE_DW_CONV1D_OFF") &&
                                  !vae_abl("VAE_ABL_TAPS") && !vae_abl("VAE_DW_LPAD_OFF");
        if (cache != nullptr && padding > 0 && x->type == GGML_TYPE_F32) {
            int64_t lp = 0;
            x = vae_cached_concat(ctx, x, padding, cache, ct_in ? 1 : 0, kernel_route ? &lp : nullptr);
            padding = (int)lp;   // 0 unless the pad node was deleted; then the kernel left-pads by lp
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
            struct ggml_tensor* xc = ct_in ? x : vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), 3);
            // NOTE ggml's permute convention is result.ne[axis_i] = a.ne[i], so
            // going from [K, 1, C] to [C, K] needs (1, 2, 0), not (2, 0, 1).
            struct ggml_tensor* ww = ggml_permute(ctx, w, 1, 2, 0, 3);                   // [C, K]
            ww = vae_ct_site(ggml_cont(ctx, ww), 4);
            if (ww->type != GGML_TYPE_F32) ww = ggml_cast(ctx, ww, GGML_TYPE_F32);
            // +padding: with the pad node deleted, padding carries the implicit left pad (Exp821) and the
            // output length must stay what it was with the materialised [hist | x] tensor.
            const int64_t T_out = xc->ne[1] + padding - (K - 1);
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
            // DEFAULT since Exp670: -3.7% RTF with a byte-identical transcript. It looked unsafe
            // in Exp666 because it coincided with a pre-existing data race in vae_abl()'s shared
            // knob memo, which Exp669 fixed (thread_local); see the crash post-mortem there.
            // VAE_DW_CONV1D_OFF=1 restores the tap-chain fallback.
            if (vae_abl("VAE_ABL_TAPS")) {
                // Measurement-only (Exp671): drop the whole depthwise conv but keep the graph's
                // shapes. Needed because this branch now PRECEDES the tap chain - without it the
                // knob would silently ablate nothing, which is the third time this bug class
                // appeared (Exp663/665). Output is INVALID with this knob on.
                result = ggml_view_2d(ctx, xc, C, T_out, xc->nb[1], 0);   // [C, T_out]
            } else if (getenv("VAE_DW_CONV1D_OFF") == nullptr && stride == 1 && dilation == 1) {
                // ONE depthwise conv op instead of a K-long elementwise chain: (K+1) tensor
                // passes instead of ~3K, i.e. Exp665's 9.0% tap cost -> ~1.5%. Bit-identical to
                // the tap chain by construction (ascending k, product rounded before each add,
                // no fma) - Exp666.
                result = ggml_conv1d_dw_ct_lp(ctx, ww, xc, padding);   // [C, T_out]
            } else {
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
            }
        } else {
            // The legacy helper wants time on ne[0]. With the taps disabled by
            // VAE_DW_CT_OFF the block still hands us a [C, T] tensor (ct_block
            // is independent of the knob), so transpose here - otherwise the
            // helper's [T,1,C,N] reshape asserts (same dim-0/dim-1 class of bug
            // as Exp586's guard mismatch, and the state Exp602's fix left the
            // knob working in; Exp586 re-broke it). The left-pad above was
            // applied on dim 1, which permutes to exactly dim-0 left-padding.
            if (ct_in) x = vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), 5);
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
            x = vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), 6);
        }

        x = ggml_nn_conv_1d_dw(ctx, x, mixer_conv_weight, mixer_conv_bias,
                                /*stride=*/1, /*padding=*/kernel_size-1, /*dilation=*/1,
                                is_i8s ? nullptr : cache);

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, mixer_layer_scale);
        } else if (vae_abl("VAE_ABL_SCALE")) {
            // measurement-only (Exp674): the fused op applies the scale in-register, so price the
            // MULTIPLY alone with an add of the same pass count -> separates ALU from bandwidth cost.
            x = ggml_add(ctx, x, residual);
        } else if (vae_abl("VAE_ABL_RESID")) {
            // measurement-only: drop the residual operand (one fewer read pass) -> x*scale only.
            x = ggml_mul(ctx, x, mixer_layer_scale);
        } else if (getenv("VAE_LS_FUSE_OFF") == nullptr) {   // default ON since Exp671: -1.1% at byte-identical output
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
        } else if (getenv("VAE_GELU_BIAS_OFF") == nullptr && getenv("VAE_GELU_QUICK") == nullptr &&
                   ffn_fc1_bias && ffn_fc1_bias->type == GGML_TYPE_F32 && ggml_is_contiguous(ffn_fc1_bias) &&
                   ffn_fc1_bias->ne[0] == ffn_fc1_weight->ne[1] &&
                   ffn_fc1_bias->ne[1] == 1 && ffn_fc1_bias->ne[2] == 1 && ffn_fc1_bias->ne[3] == 1) {
            // Exp673: mul_mat -> gelu(x+b) in ONE node, removing the bias ADD and its write+read
            // pass (bias adds measured 5.9% of VAE time). Bit-identical by construction: the fused
            // kernel reuses the same f32 add and the same ggml_vec_gelu_f32. Built by calling
            // ggml_nn_linear's OWN body with apply_bias=false - Exp672 proved that re-deriving the
            // reshape/mul_mat sequence at the call site is NOT equivalent. The gelu-approximation
            // knob is deliberately excluded above: VAE_GELU_QUICK must keep its own path.
            x = ggml_nn_linear(ctx, x, ffn_fc1_weight, ffn_fc1_bias, /*apply_bias=*/false);
            // VAE_ABL_GELU (measurement-only): gelu is the largest single byte-mover in the shipped
            // graph (26 nodes / 980 MB = 17% of bytes per build, Exp778 profile). Fusing it into the
            // fc1 matmul's epilogue is the only way to remove that pass, so price it before asking
            // about touching mul_mat. Output INVALID by design.
            // The substitute is the linear output ITSELF (no new node): an earlier version wrapped it
            // in ggml_view_2d, and the view's non-default strides then tripped
            // GGML_ASSERT(ggml_are_same_shape) inside the block's add_scaled - i.e. the ablation arm,
            // not the shipped path, was invalid (a repeat of the Exp763/764 knob-bug class).
            if (!vae_abl("VAE_ABL_GELU")) x = ggml_gelu_bias(ctx, x, ffn_fc1_bias);
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
        } else if (vae_abl("VAE_ABL_SCALE")) {
            x = ggml_add(ctx, x, residual);                    // scale multiply priced out (Exp674)
        } else if (vae_abl("VAE_ABL_RESID")) {
            x = ggml_mul(ctx, x, ffn_layer_scale);             // residual operand dropped (Exp674)
        } else if (getenv("VAE_LS_FUSE_OFF") == nullptr) {   // default ON since Exp671: -1.1% at byte-identical output
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
        struct ggml_tensor* conv_geom = nullptr;   // geometry carrier, blocked-int8 weights only
    } downsamples[n_downsamples];
    
    // Stages (ConvNeXt blocks)
    std::vector<ConvNeXtBlock> stages[n_stages];
    
    // Head (just conv)
    struct ggml_tensor* head_conv_weight;
    struct ggml_tensor* head_conv_bias;
    struct ggml_tensor* head_conv_geom = nullptr;  // geometry carrier, blocked-int8 weights only
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
                                 cache, downsamples[i].conv_geom);

            for (int j = 0; j < stage_depths[i]; j++) {
                x = stages[i][j].forward(ctx, x, cache);
            }

            x = vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), 7);

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
                                 cache, downsamples[i].conv_geom);

            for (int j = 0; j < stage_depths[i]; j++) {
                x = stages[i][j].forward(ctx, x, cache);
            }

            x = vae_ct_site(ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), 8);

        }

        // Head
        x = ggml_nn_conv_1d(ctx, x, head_conv_weight, head_conv_bias, 1, head_kernel_size - 1, 1, cache,
                            head_conv_geom);

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
        // In GGUF, dimensions are reversed, so ne[0] is kernel_size. A blocked-int8 weight is stored
        // 2-D [K*IC, OC], so its kernel size comes from row / IC (see vae_conv_1d_i8).
        vae_geom_ctx_init();
        encoder.downsample_kernel_sizes[i] = vae_conv_kernel_size(
                encoder.downsamples[i].conv_weight,
                i ? AudioVAEEncoder::downsample_dims[i-1] : 0);
        if (encoder.downsample_kernel_sizes[i] <= 0) {
            fprintf(stderr, "%s: downsample %d conv has an unsupported weight layout\n", __func__, i);
            return false;
        }
        encoder.downsamples[i].conv_geom = vae_conv_geom(encoder.downsamples[i].conv_weight,
                                                        encoder.downsample_kernel_sizes[i]);
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
    
    // Get output dim from head conv weight [kernel, in_dim, out_dim] - but a blocked-int8 weight is 2-D
    // [K*IC, OC], where OC is ne[1] and ne[2] is the implicit trailing 1. Reading ne[2] there silently
    // set output_dim = 1 (a latent dim of 1 passes every shape check and produces a 1024-token runaway),
    // which is why the int8 file looked broken even after its bytes were proven correct.
    encoder.output_dim = ggml_n_dims(encoder.head_conv_weight) == 2
                         ? encoder.head_conv_weight->ne[1] : encoder.head_conv_weight->ne[2];
    encoder.head_kernel_size = vae_conv_kernel_size(encoder.head_conv_weight,
                                                    AudioVAEEncoder::downsample_dims[6]);
    if (encoder.head_kernel_size <= 0) {
        fprintf(stderr, "%s: head conv unsupported layout: ne=[%lld,%lld,%lld] type=%d ic=%lld\n", __func__,
                (long long) encoder.head_conv_weight->ne[0], (long long) encoder.head_conv_weight->ne[1],
                (long long) encoder.head_conv_weight->ne[2], (int) encoder.head_conv_weight->type,
                (long long) AudioVAEEncoder::downsample_dims[6]);
        return false;
    }
    encoder.head_conv_geom = vae_conv_geom(encoder.head_conv_weight, encoder.head_kernel_size);
    
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

// VAE_CONV_I8_CMP=1 - load-time A/B for the blocked-int8 conv weights (Exp687's probe).
//
// It has to happen here rather than inside the model graph because a node nothing consumes is pruned
// and would silently never run, and it has to happen on device because Q4_0_4_4 cannot be evaluated
// on x86 at all (the ARM gemv/gemm report "unsupported" and the generic path needs a NULL vec_dot).
//
// Mechanism: multiplying the weight by a Q8_0 IDENTITY returns the dequantized weight itself -
// sum_k W[k,j] * I[k,i] = W[i,j], and Q8_0 represents 1.0 exactly (scale = 1/127, entry 127) - so no
// knowledge of the block layout is needed: this is ggml's own reader, the same one inference uses.
//
// The self-test is the load-bearing part: a hand-built matrix is quantized with the runtime's own
// ggml_quantize_chunk (the same function the offline converter calls) and read back through this
// mechanism. Until that control says "mechanism OK", the conv verdict below means nothing (Exp660's
// rule: a checker is untrustworthy until a known-good case makes it pass or fail correctly).
static void vae_probe_int8_weights(struct vae_model* model) {
    if (!getenv("VAE_CONV_I8_CMP")) return;
    static bool done = false;                      // load runs once, on the loading thread
    if (done) return;
    done = true;

    const int64_t CK = 32, CM = 4;                 // control: one 32-wide block x 4 rows
    std::vector<float> csrc((size_t)CK * CM);
    for (int64_t j = 0; j < CM; j++)
        for (int64_t i = 0; i < CK; i++)
            csrc[(size_t)j * CK + i] = 0.25f * (float)(((i * 7 + j * 13) % 41) - 20);
    std::vector<uint8_t> cq(4096);                 // the blob for [CK, CM] is 72 bytes
    size_t wrote = ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, csrc.data(), cq.data(), 0, CM, CK, nullptr);
    fprintf(stderr, "[I8CMP] control: quantized %lld x %lld -> %zu bytes\n", (long long)CM, (long long)CK, wrote);
    // Derived scalars the graph depends on. A blocked 2-D weight loses the channel/kernel axes, so
    // every ne[0]/ne[1]/ne[2] read has to be replaced by a derivation - this prints them to catch the
    // ones that were missed (output_dim silently became 1 that way).
    fprintf(stderr, "[I8CMP] dims: acoustic output_dim=%d head_K=%d | semantic output_dim=%d head_K=%d\n",
            model->acoustic_encoder.output_dim, model->acoustic_encoder.head_kernel_size,
            model->semantic_encoder.output_dim, model->semantic_encoder.head_kernel_size);

    const char* pick = nullptr;                    // first converted conv that fits an identity
    int64_t pk = 0, pm = 0;
    for (auto& kv : model->tensors) {
        struct ggml_tensor* t = kv.second;
        if (!t || t->type != GGML_TYPE_Q4_0_4_4 || ggml_n_dims(t) != 2) continue;
        if (t->ne[0] > 4096 || t->ne[1] % 4) continue;
        if (strstr(kv.first.c_str(), "conv.weight")) { pick = kv.first.c_str(); pk = t->ne[0]; pm = t->ne[1]; break; }
    }
    if (!pick) { fprintf(stderr, "[I8CMP] no converted conv tensor found (is this a conv-int8 file?)\n"); return; }
    struct ggml_tensor* w = model->tensors[pick];
    // Reference values come from the SOURCE FILE (VAE_CONV_I8_REF), read as raw f16 by name - not from
    // a tensor copied into the converted file. An earlier version added a <name>_ref tensor and the
    // loader never registered it (562 keys for a 563-tensor file), which is a gguf-counting puzzle
    // this avoids entirely: the untouched source is a better reference anyway.
    std::string rname = pick;
    struct ggml_tensor* ref = nullptr;
    std::vector<ggml_fp16_t> rbuf;
    const char* ref_path = getenv("VAE_CONV_I8_REF");
    if (!ref_path) {
        fprintf(stderr, "[I8CMP] set VAE_CONV_I8_REF=/path/to/source.gguf for the verdict (control still runs)\n");
    } else {
        struct gguf_init_params gp{ /*no_alloc=*/ true, /*ctx=*/ nullptr };
        struct gguf_context* g2 = gguf_init_from_file(ref_path, gp);
        if (!g2) {
            fprintf(stderr, "[I8CMP] could not open ref file %s\n", ref_path);
        } else {
            int idx = -1;
            // Resolve by name via the same iteration the loader uses: gguf_find_tensor's return
            // convention (tensor index vs data offset) differs across ggml versions, and guessing it
            // wrong would read some other tensor and produce a confident, wrong verdict.
            for (uint32_t i = 0; i < gguf_get_n_tensors(g2); i++)
                if (strcmp(gguf_get_tensor_name(g2, i), rname.c_str()) == 0) { idx = (int)i; break; }
            if (idx < 0) {
                fprintf(stderr, "[I8CMP] %s not found in %s\n", rname.c_str(), ref_path);
            } else if (gguf_get_tensor_type(g2, idx) != GGML_TYPE_F16) {
                fprintf(stderr, "[I8CMP] ref tensor in %s is not f16 - wrong source file?\n", ref_path);
            } else {
                const size_t n = (size_t)pk * pm;                     // [k, m] in tensor memory order
                rbuf.resize(n);
                FILE* f = fopen(ref_path, "rb");
                if (!f || fseek(f, (long)(gguf_get_data_offset(g2) + gguf_get_tensor_offset(g2, idx)), SEEK_SET) != 0 ||
                    fread(rbuf.data(), sizeof(ggml_fp16_t), n, f) != n) {
                    fprintf(stderr, "[I8CMP] short read of the ref tensor\n");
                    rbuf.clear();
                } else {
                    ref = (struct ggml_tensor*)1;                     // only used as a "we have data" flag
                }
                if (f) fclose(f);
            }
            gguf_free(g2);
        }
    }
    fprintf(stderr, "[I8CMP] target %s  weight=[%lld, %lld]  tensors=%zu  ref=%s\n", pick, (long long)pk, (long long)pm,
            model->tensors.size(), ref ? getenv("VAE_CONV_I8_REF") : "unavailable");

    struct ggml_context* ctx = nullptr;
    // Small arena on purpose: the probe's tensors are tens of KB, and a big anonymous allocation at
    // load time (when ~2.2 GB is already resident) can fail - ggml_init then returns NULL and the very
    // next ggml_new_tensor_* segfaults. Never call into a context without checking it.
    ggml_init_params p{ (size_t)(16 << 20), ctx, false };
    ctx = ggml_init(p);        // ggml_init RETURNS the context; discarding it leaves NULL and makes
                               // the next ggml_new_tensor_* segfault (this was the probe's own bug)
    if (!ctx) { fprintf(stderr, "[I8CMP] ggml_init FAILED (arena allocation)\n"); return; }
    struct ggml_tensor* wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_4_4, pk, pm);
    struct ggml_tensor* id = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pk, 1);
    struct ggml_tensor* y = ggml_mul_mat(ctx, wq, ggml_cast(ctx, id, GGML_TYPE_Q8_0));
    memcpy(wq->data, w->data, ggml_nbytes(w));
    float* idd = (float*)id->data;

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    // Read the weight ONE COLUMN AT A TIME. Feeding a full identity was the first attempt and gave
    // zeros plus a NaN: with a wide src1 only column 0 of y got written and the rest stayed as
    // uninitialized arena - the symptom was exactly "first sample has a value, every later sample is
    // 0". A [k, 1] selector is the shape the gemv path is written for, and y is then exactly column i
    // of the dequantized weight.
    std::vector<float> got((size_t)pk * pm);
    bool ok = true;
    for (int64_t i = 0; i < pk && ok; i++) {
        for (int64_t r = 0; r < pk; r++) idd[r] = (r == i) ? 1.0f : 0.0f;
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { ok = false; break; }
        const float* yr = (const float*)y->data;
        for (int64_t j = 0; j < pm; j++) got[(size_t)i * pm + j] = yr[j];
    }
    if (!ok) {
        fprintf(stderr, "[I8CMP] selector product FAILED to compute - the probe itself is broken\n");
        ggml_free(ctx);
        return;
    }
    const float* yd = got.data();                  // element (k=i, m=j) at i*pm + j
    if (ref) {
        const ggml_fp16_t* rd16 = rbuf.data();
        double worst = 0, scale = 0;
        for (int64_t i = 0; i < pk; i++)
            for (int64_t j = 0; j < pm; j++) {
                double got = yd[(size_t)i * pm + j];
                double want = ggml_fp16_to_fp32(rd16[(size_t)j * pk + i]);
                worst = fmax(worst, fabs(got - want));
                scale = fmax(scale, fabs(want));
            }
        fprintf(stderr, "[I8CMP] conv samples: ");
        for (int s = 0; s < 4; s++) {
            int64_t i = (int64_t)s * 7 % pk, j = (int64_t)s * 5 % pm;
            fprintf(stderr, "(%g vs %g) ", yd[(size_t)i * pm + j], ggml_fp16_to_fp32(rd16[(size_t)j * pk + i]));
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "[I8CMP] conv vs source: max|diff|=%.6g  max|w|=%.6g  bound amax/15=%.6g  -> %s\n", worst,
                scale, scale / 15.0,
                worst <= scale / 15.0 * 1.15 ? "WITHIN Q4 ROUNDING (bytes correct)" : "EXCEEDS Q4 BOUND (converter)");
    }
    // Same-activation convolution A/B (Exp768): identity readback only proves the quantized
    // elements are plausible; this tests the actual im2col row pairing. The picked tensor is
    // acoustic downsample layer 1: IC=32, K=row/IC=4. Both products use the SAME F32 im2col;
    // only the weight representation differs. This avoids any Python layout model and runs on
    // the ARM reader that inference uses.
    if (ref && pk == 128 && pm == 64) {
        const int64_t ic = 32, k = pk / ic, tlen = 9;
        struct ggml_tensor* geom2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, ic, pm);
        geom2->data = g_vae_geom_dummy;
        struct ggml_tensor* xa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tlen, ic);
        float* xd = (float*) xa->data;
        for (int64_t c = 0; c < ic; c++) for (int64_t t = 0; t < tlen; t++)
            xd[c * tlen + t] = 0.03f * (float)(((c * 11 + t * 7) % 29) - 14);
        struct ggml_tensor* ci = ggml_im2col(ctx, geom2, xa, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
        struct ggml_tensor* cc = ggml_reshape_2d(ctx, ci, ci->ne[0], ci->ne[2] * ci->ne[1]);
        struct ggml_tensor* wf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, pk, pm);
        memcpy(wf->data, rbuf.data(), (size_t) pk * pm * sizeof(ggml_fp16_t));
        struct ggml_tensor* yq = ggml_mul_mat(ctx, wq, cc);
        struct ggml_tensor* yf = ggml_mul_mat(ctx, wf, cc);
        struct ggml_cgraph* gab = ggml_new_graph(ctx);
        ggml_build_forward_expand(gab, yq); ggml_build_forward_expand(gab, yf);
        if (ggml_graph_compute_with_ctx(ctx, gab, 1) == GGML_STATUS_SUCCESS) {
            const float* aq = (const float*) yq->data; const float* af = (const float*) yf->data;
            double worst = 0, norm = 0;
            for (int64_t i = 0; i < yq->ne[0] * yq->ne[1]; i++) {
                worst = fmax(worst, fabs((double) aq[i] - af[i])); norm = fmax(norm, fabs((double) af[i]));
            }
            // VERDICT CALIBRATION (Exp769 follow-up): a Q4_0 weight is NOT expected to reproduce an
            // F16 product. Per-element bound is amax/15 (0.0217 for this tensor); a 128-term dot with
            // |x|<=0.42 gives a random-walk estimate sqrt(128)*0.42*0.0217/sqrt(3) ~ 0.06 and a
            // worst case of 1.17. The 0.033 observed was therefore NEVER evidence of mispairing -
            // an absolute threshold on this ratio is meaningless at 4 bits.
            // The decisive test is instead PREDICTED vs ACTUAL: rebuild the product from the
            // identity-readback bytes (got[], which are the kernel's own dequantization) and compare
            // to what the kernel produced. ~0 difference PROVES the kernel pairs contracting index r
            // with im2col row r exactly as the identity readback does - i.e. converter and kernel
            // agree, and any end-to-end garbage must come from the GRAPH (geometry/padding/cache).
            const float* cd = (const float*) cc->data;
            const int64_t cols = yq->ne[1];
            double pmax = 0, anorm = 0;
            for (int64_t t = 0; t < cols; t++)
                for (int64_t j = 0; j < pm; j++) {
                    double acc = 0;
                    for (int64_t r = 0; r < pk; r++) acc += (double) cd[t * pk + r] * got[(size_t) r * pm + j];
                    pmax = fmax(pmax, fabs(acc - (double) aq[j + t * pm]));
                    anorm = fmax(anorm, fabs((double) aq[j + t * pm]));
                }
            fprintf(stderr, "[I8CMP] predicted-vs-kernel product: max|diff|=%.6g max|q4|=%.6g ratio=%.4g -> %s\n",
                    pmax, anorm, anorm ? pmax / anorm : 0.0,
                    (anorm && pmax / anorm < 1e-3) ? "PAIRING PROVEN CORRECT (garbage is in the graph)"
                                                   : "PAIRING DIFFERS FROM READBACK");
            fprintf(stderr, "[I8CMP] same-im2col conv A/B: max|diff|=%.6g max|f16|=%.6g ratio=%.4f"
                            " (Q4 rounding-scale est ~0.06: NOT a pairing test)\n",
                    worst, norm, norm ? worst / norm : 0.0);
        } else fprintf(stderr, "[I8CMP] same-im2col conv A/B compute FAILED\n");
    }
    // control read-back: same mechanism, bytes made by the same quantizer call the converter uses
    {
        struct ggml_tensor* cw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_4_4, CK, CM);
        memcpy(cw->data, cq.data(), wrote);
        struct ggml_tensor* cid = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CK, 1);
        float* cd = (float*)cid->data;
        struct ggml_tensor* cy = ggml_mul_mat(ctx, cw, ggml_cast(ctx, cid, GGML_TYPE_Q8_0));
        struct ggml_cgraph* g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, cy);
        bool cok = true;
        std::vector<float> cg((size_t)CK * CM);
        for (int64_t i = 0; i < CK && cok; i++) {
            for (int64_t r = 0; r < CK; r++) cd[r] = (r == i) ? 1.0f : 0.0f;
            if (ggml_graph_compute_with_ctx(ctx, g2, 1) != GGML_STATUS_SUCCESS) { cok = false; break; }
            const float* cyr = (const float*)cy->data;
            for (int64_t j = 0; j < CM; j++) cg[(size_t)i * CM + j] = cyr[j];
        }
        if (cok) {
            const float* cd2 = cg.data();
            double worst = 0, scale = 0;
            for (int64_t i = 0; i < CK; i++)
                for (int64_t j = 0; j < CM; j++) {
                    double got = cd2[(size_t)i * CM + j], want = csrc[(size_t)j * CK + i];
                    worst = fmax(worst, fabs(got - want));
                    scale = fmax(scale, fabs(want));
                }
            fprintf(stderr, "[I8CMP] control samples: ");
            for (int s = 0; s < 4; s++) {
                int64_t i = s * 5 % CK, j = s % CM;
                fprintf(stderr, "(%g vs %g) ", cd2[(size_t)i * CM + j], csrc[(size_t)j * CK + i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[I8CMP] control: max|diff|=%.6g max|w|=%.6g ratio=%.4f -> %s\n", worst, scale,
                    scale ? worst / scale : 0.0,
                    worst / (scale ? scale : 1) < 0.25 ? "mechanism OK" : "MECHANISM BROKEN - verdict above means nothing");
        } else {
            fprintf(stderr, "[I8CMP] control FAILED to compute\n");
        }
        // (the control above is what makes the verdict below meaningful - Exp660's rule)
    }
    ggml_free(ctx);
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
            // A short read used to be DISCARDED, and buf is zero-initialised - so a truncated or partially
            // written gguf loaded "successfully" and fed ZEROS as weights: fluent, confident, wrong transcripts
            // with exit code 0 and normal timing (Exp849 reproduced it with truncate -s -8MB / -64MB). The mmap
            // path's bounds check only set loaded=false, which routes here, so this is where the error must land.
            if (got != tensor_size) {
                fprintf(stderr, "[VAE] Error: %s is truncated or corrupt: tensor '%s' needs %zu bytes at offset %zu, read %zu\n",
                        model_path, name, tensor_size, offset, got);
                fclose(f);
                gguf_free(gguf_ctx);
                delete model;
                return nullptr;
            }

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

    vae_probe_int8_weights(model);   // no-op unless VAE_CONV_I8_CMP=1

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
    size_t pad_fed[2] = {0, 0}, pad_fed_n[2] = {0, 0};   // [0]=via im2col (general convs) [1]=via CONV1D (dw)
    size_t total = 0;
    size_t view_bytes = 0;      // stride-only nodes, reported for reference and NOT added to total
    double total_macs = 0.0;
    const bool shapes = getenv("VAE_MMSHAPES") != nullptr;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        // Views cost NO traffic: they only rewrite strides. Counting their nbytes (dst + both srcs,
        // often a 2.6 GB activation relayed twice) made RESHAPE look like 31% of the VAE's bytes and
        // hid what the graph actually moves. Excluded here and reported separately as 'views (free)'.
        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_TRANSPOSE) {
            view_bytes += ggml_nbytes(node);
            continue;
        }
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
        // Exp821: attribute the streaming-cache PAD nodes to their CONSUMER without touching any call site.
        // A conv's src[1] IS the tensor the splice produced, so summing it over IM2COL gives the pad bytes
        // feeding the general convs and over CONV1D the ones feeding the depthwise kernel. That split decides
        // where the ~3.2% lives: the dw half is a parameter in a kernel I own, the im2col half died in Exp819.
        // If the two buckets sum to the PAD row's bytes, the attribution is confirmed (1:1 producer:consumer).
        if ((node->op == GGML_OP_IM2COL || node->op == GGML_OP_CONV1D) && node->src[1]) {
            pad_fed[node->op == GGML_OP_CONV1D ? 1 : 0] += ggml_nbytes(node->src[1]);
            pad_fed_n[node->op == GGML_OP_CONV1D ? 1 : 0] += 1;
        }
        auto& e = by_op[key];
        e.bytes += b; e.count += 1; e.macs += macs;
        total += b; total_macs += macs;
    }
    fprintf(stderr, "[VAE_STATS:%s] nodes=%d total=%.1f MB macs=%.2f G\n", phase, n_nodes, total / 1e6, total_macs / 1e9);
    fprintf(stderr, "[VAE_STATS:%s] PAD-by-consumer  ->im2col %9.1f MB (n=%zu)   ->dwconv %9.1f MB (n=%zu)\n",
            phase, pad_fed[0] / 1e6, pad_fed_n[0], pad_fed[1] / 1e6, pad_fed_n[1]);
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

void vae_cache_set_whole_window(vae_cache_t* cache, int on) {
    // Exp821: see vae_stream_cache::single_piece_window. Set by the streaming demo, the only caller that knows
    // both the piece count and whether the deferred late pass will run.
    if (cache == nullptr) return;
    cache->acoustic.single_piece_window = on != 0;
    cache->semantic.single_piece_window = on != 0;
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
