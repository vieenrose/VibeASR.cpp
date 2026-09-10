#ifndef VAE_H
#define VAE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for VAE model
typedef struct vae_model vae_model_t;
typedef struct vae_context vae_context_t;

// Model parameters
struct vae_model_params {
    int32_t n_threads;
    bool use_gpu;
};

// Context parameters
struct vae_context_params {
    int32_t n_threads;
};

// Get default model parameters
struct vae_model_params vae_model_default_params(void);

// Get default context parameters
struct vae_context_params vae_context_default_params(void);

// Load VAE model from GGUF file
// Returns NULL on failure
vae_model_t* vae_load_model_from_file(
    const char* model_path,
    struct vae_model_params params);

// Free model
void vae_free_model(vae_model_t* model);

// Create context from model
vae_context_t* vae_new_context_with_model(
    vae_model_t* model,
    struct vae_context_params params);

// Free context
void vae_free(vae_context_t* ctx);

// Opaque streaming conv cache (per-window state for piece-wise encode).
// Carries each causal conv's left-history across pieces so that encoding a
// window in 3200-sample-multiple pieces reproduces the full-window cold run.
// Histories are tiny (~0.65 MB per encoder). Reset per window for upstream
// parity (upstream encodes every window cold).
typedef struct vae_cache_t vae_cache_t;

vae_cache_t* vae_cache_new(void);
void vae_cache_free(vae_cache_t* cache);
void vae_cache_reset(vae_cache_t* cache);  // clear all histories (call per window)

// Piece-wise encode with carried conv state. n_samples should be a multiple
// of 3200 (grid alignment at every strided layer). First piece after reset
// behaves exactly like the uncached call. Returns frames, or -1 on error.
int32_t vae_encode_acoustic_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* output);

// Concurrent acoustic+semantic encode (one thread each; independent weights and
// caches). Returns the frame count, or -1 on failure.
int32_t vae_encode_parallel_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* output_acoustic,
    float* output_semantic,
    float* acoustic_ms,
    float* semantic_ms);

// ---------------------------------------------------------------------------
// Deferred late-stage encode (window-level batching of the deep stages).
//
// The early stages stay piece-wise with the carried cache (boundary tensor per
// piece); the deep stages then run ONCE per window over the concatenated
// boundary tensors (bshape = the per-piece boundary's ne[0..3], with ne[0] the
// time axis). Stage split defaults to 6 (deepest stage only); VAE_LATE_SPLIT
// overrides. Equivalent to the piece-wise path by the cache invariant (outputs
// equal a full-window cold run up to matmul blocking order).
int32_t vae_encode_early_parallel_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* boundary_acoustic, int64_t ashape[4],
    float* boundary_semantic, int64_t sshape[4],
    float* acoustic_ms, float* semantic_ms);

int32_t vae_encode_late_parallel(
    vae_context_t* ctx,
    const float* boundary_acoustic, const int64_t ashape[4],
    const float* boundary_semantic, const int64_t sshape[4],
    int64_t n_time_total,
    float* output_acoustic, float* output_semantic,
    float* acoustic_ms, float* semantic_ms);

// Single-encoder variants (debug / RAM-lean experiments).
int32_t vae_encode_early_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* boundary_out,
    int64_t bshape[4]);

int32_t vae_encode_late(
    vae_context_t* ctx,
    const float* boundary,
    const int64_t bshape[4],
    int64_t n_time_total,
    float* output);

int32_t vae_encode_semantic_cached(
    vae_context_t* ctx,
    vae_cache_t* cache,
    const float* audio,
    int32_t n_samples,
    float* output);

// Get model info
int32_t vae_model_acoustic_dim(const vae_model_t* model);
int32_t vae_model_semantic_dim(const vae_model_t* model);

// Encode audio to acoustic features
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * acoustic_dim], caller must allocate
// Returns number of frames on success, -1 on error
int32_t vae_encode_acoustic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output);

// Encode audio to semantic features
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * semantic_dim], caller must allocate
// Returns number of frames on success, -1 on error
int32_t vae_encode_semantic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output);

// Encode audio to acoustic features with timing
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * acoustic_dim], caller must allocate
// inference_time_ms: output parameter for inference time in milliseconds
// Returns number of frames on success, -1 on error
int32_t vae_encode_acoustic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms);

// Encode audio to semantic features with timing
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * semantic_dim], caller must allocate
// inference_time_ms: output parameter for inference time in milliseconds
// Returns number of frames on success, -1 on error
int32_t vae_encode_semantic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms);

#ifdef __cplusplus
}
#endif

#endif // VAE_H
