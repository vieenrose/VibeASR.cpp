# Streaming ASR (VibeVoice-ASR-Streaming-1.5B) on CPU

This fork extends VibeASR.cpp with chunked streaming inference for
[`microsoft/VibeVoice-ASR-Streaming-1.5B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B)
(Qwen2.5-1.5B decoder, per-2.93 s chunks with `Speaker N:` labels), complementing
the built-in offline BitNet path (`VibeVoice-ASR-BitNet`).

## Accuracy (validated 2026-09-09)

40-utterance LibriSpeech `test-clean` subset, jiwer, same normalization for all:

| system | vs ground truth | vs official PyTorch |
|---|---|---|
| `asr_streaming` (VAE F16 + LM Q4_K_M, `--vae-pieces 13`) | **WER 4.13%** (S=25 D=1 I=4) | **WER 2.2%** |
| official PyTorch `streaming_generate` (bf16) | WER 4.82% (S=28 D=3 I=4) | — |

Delta on ground truth: **-0.7 pp** (within subset noise) — accuracy is preserved.
69 s in-domain clip: parity WER 3.7% (11 substitutions, 0 del/ins).
5/5 reruns bit-identical (`-t 4`).

## Build

```bash
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target asr_streaming -j
```

## Convert weights (source: `microsoft/VibeVoice-ASR-Streaming-1.5B`)

The streaming checkpoint shares the VAE architecture and Qwen2.5-1.5B decoder
with BitNet, but its LM weights are normal BF16 (not ternary), so the BitNet
LM converter is bypassed:

```bash
# VAE encoder (works as-is, same tensor names)
python utils/convert_vae_to_gguf.py <streaming-checkpoint-dir> --outtype f16 \
    -o streaming-vae-encoder-f16.gguf          # ~1.4 GB
# LM (strips model.language_model.* -> model.*, flattens decoder_config, no ternary step)
python utils/convert_streaming_lm_stage.py <streaming-checkpoint-dir>
# -> models-streaming/streaming-lm-f16.gguf (~4.2 GB)
./build/bin/llama-quantize --token-embedding-type Q6_K \
    streaming-lm-f16.gguf streaming-lm-q4_k_m.gguf Q4_K_M   # ~1.1 GB
```

## Run

```bash
./build/bin/asr_streaming --vae-model streaming-vae-encoder-f16.gguf \
    --lm-model streaming-lm-q4_k_m.gguf --audio input_24k.wav -t 4 \
    [--vae-pieces 13] [--context "VibeVoice,diarization"] [--max-tokens 256]
```

`--vae-pieces` must divide 26 (1, 2, 13, 26; default 13). Each 83200-sample
window is encoded in 6400-sample pieces with carried conv state (reset per
window = upstream cold-window parity); frames are bit-exact vs full-window
encode. Peak RSS is flat in file length: 3.26 GB (17 s) -> 3.27 GB (69 s),
vs 12.5 GB for PyTorch CPU fp32 and 5.5 -> 15.5 GB for offline BitNet on long files.

## Implementation notes

- `demo/asr_streaming.cpp`: faithful port of upstream `streaming_generate`
  (plain-text prompt, `[speech_start]+26f+[speech_end]`, greedy to
  `<|text_chunk_end|>` 151665, feed-it-back invariant, per-chunk prints).
- `src/vae.{h,cpp}`: `vae_cache_t` streaming conv cache (histories ~0.65 MB/encoder).
  History splicing uses pad + add, deliberately NOT `ggml_concat`: this build's
  concat scrambles non-trivial inputs (unit-tested), while pad/add are proven ops.
  ggml dense tensors are Fortran-order (ne[0] fastest) — history gather/scatter
  must be per-channel strided copies, not one linear memcpy.
- RF of the VAE encoder exceeds 72 frames (> 26-frame window), so overlap-split
  cannot work (verified: prepending 83k zeros changes all 26 frames); the
  carried-state cache is required. The stock BitNet I8_S VAE path is untouched
  (`asr_infer` behavior unchanged); the streaming VAE stays F16 because naive
  PTQ I8_S collapses output (repetition loops) — that VAE was never QAT-trained.
- Env-gated diagnostics: `VAE_CACHE_TRACE=1` (per-site checksums),
  `VAE_DUMP_FRAMES=<prefix>` (output frames), `VAE_DUMP_SITE=<sN|all>` (site inputs).\n
## Phone evaluation (OPPO CPH2371, Dimensity 1300, 8 GB RAM, Android 13)

Cross-built with NDK r26d (`arm64-v8a`, `android-33`, `GGML_ARM_DOTPROD=ON`;
baseline kernels — the ggml dotprod probe can't run under cross-compilation).
Binaries + 2.5 GB GGUFs pushed to `/data/local/tmp/vibeasr`
(`libomp.so` from the NDK must sit beside them for `LD_LIBRARY_PATH=.`).

| clip | phone RTF | desktop RTF | phone peak RSS |
|---|---|---|---|
| 10 s, `-t 4` | 13.33 (VAE 109 s + LM 24 s) | 1.13 | 3.09 GB |
| 10 s, `-t 4` + dotprod build | 13.21 (no gain: LM decode is bandwidth-bound) | — | — |
| 10 s, `-t 4` pinned to big cores (`taskset F0`) | 12.37 | — | — |
| 10 s, `-t 8` | 12.43 | — | 3.32 GB |
| 17 s, `-t 4` | 11.60, transcript correct | 0.94 | 3.12 GB |
| 69 s, `-t 4` | 12.50, cross-device WER 2.65% vs desktop | 1.03 | 3.21 GB |

Correctness and flat memory hold on-device. Speed does not: consistently ~12x
off real-time (VAE convs 12.8x slower than desktop: NEON-128 vs AVX2+clocks;
Q4 LM 9.4x: bandwidth-bound, SDOT can't help GEMV decode; `-t 8` gains 7% —
little cores add nothing, the load is bandwidth-bound).

Lever tally toward RTF < 1 (needs ~12x): dotprod ~0%, pinning 7%, `-t 8` 7%,
selective FFN-Q8 unbuilt (~1.2x est.). Even adding the broken I8_S VAE (3.4x,
wrong transcripts without QAT) the ceiling is ~RTF 3-4. **RTF < 1 on this phone
class requires retraining** (QAT INT8 VAE and/or a smaller encoder+LM), not
more porting. Default Android build stays on the portable baseline; dotprod is
opt-in (`-DGGML_ARM_DOTPROD=ON`) for known-dotprod fleets.
