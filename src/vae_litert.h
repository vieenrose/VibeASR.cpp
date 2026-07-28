// LiteRT/XNNPACK audio front end — an alternative to the ggml one in vae.cpp.
// See vae_litert.cpp for why both exist and what the trade is.
//
// "Combined": the export runs both tokenizer encoders and both connectors and
// returns their SUM, which is what prompt_builder.h derives from the two ggml
// planes. Callers therefore use the summed result directly and supply zeros for
// the semantic plane.
//
// Only compiled when VIBEASR_LITERT=ON.

#ifndef VAE_LITERT_H
#define VAE_LITERT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vae_litert_model vae_litert_model;

/** Load a .tflite export. `out_dim` is the feature width (1536 for VibeVoice).
 *  Set VIBEASR_LITERT_CACHE to an XNNPACK weight-cache FILE path — without one,
 *  repacked weights are anonymous RAM (1228 MB vs 576 MB for the 700 MB encoder)
 *  and every load repays ~7.7 s of repacking. Returns NULL on failure. */
vae_litert_model* vae_litert_load(const char* path, int n_threads, int32_t out_dim);

void vae_litert_free(vae_litert_model* m);

/** Fixed window the graph was exported for, in samples @ 24 kHz. */
int32_t vae_litert_window_samples(const vae_litert_model* m);

/** Encode up to one window. Shorter input is zero-padded and only the frames it
 *  really covers are returned. Writes [frames * out_dim] floats. -1 on error. */
int32_t vae_litert_encode(vae_litert_model* m, const float* audio, int32_t n_samples,
                          float* output, float* inference_time_ms);

#ifdef __cplusplus
}
#endif

#endif  // VAE_LITERT_H
