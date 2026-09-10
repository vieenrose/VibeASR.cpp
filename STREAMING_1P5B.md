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

65-experiment optimization loop on-device (CPU-only, accuracy-guarded;
baseline 12.24 → best 6.52, −47%). Key finding: this SoC is **6× Cortex-A55
+ 2× Cortex-A78** — mask `F0` was 2 little + 2 big, and every `-t 4` run
straggled on little cores. The optimum is **`-t 2` on the two big cores**
(plain `-t 2` suffices; EAS places them correctly, no `taskset` needed to
ship). A78s sustain ~1.3 GHz under load (mobile sustained equilibrium, not
throttling); no i8mm/SVE exists, so NEON-F32 + DOTPROD is the full ISA story.

### Second wave: A78 codegen + loader (Exp457-473, 12.24 → 5.96/6.02)

Three findings re-opened the loop after it had converged at 6.52:

1. **Harness bug**: `measure.sh` pushed only `bin/asr_streaming`, never the
   `libggml.so`/`libllama.so` it links against (the binary contains no ggml
   code — `llvm-objdump` shows 0 sdot there, all 646 in `libggml.so`). Every
   *ggml-side* experiment had silently measured the old kernels, which
   invalidated the early `-mcpu=cortex-a78` and ThinLTO discards. Fixed with an
   md5-diff push (same guard added to `eval40.sh`).
2. **`-mcpu=cortex-a78` for the whole build** is worth **−6.5% (10 s), −7.7%
   (69 s)**, all of it VAE-side and A/B/A-bracketed. `GGML_ARM_DOTPROD=ON`
   already sets `-march=armv8.2-a+dotprod`, so this only adds the arch features
   + tuning the default `armv8-a` tune lacks: `sdot` 598→646, `fmla`
   1284→1542 in the shipped `libggml.so`. Tune-only (`-mtune`) captures 41% of
   it; portable in-tree defaults stay armv8.0-safe (device build only).
   The F16 path benefits most (−51% VAE): it was **instruction-bound** on
   software f16 handling, not bandwidth-bound. Consequence: the F16 VAE now
   matches Q8-mixed (43.0 s vs 43.9 s on the 10 s clip) despite 2× the weight
   bytes — **activation traffic dominates weight traffic** in the encoder.
3. **mmap VAE loader** (`src/vae.cpp`): the old loader value-initialised a
   fresh `std::vector<char>` per tensor (a full 1.4 GB zero-fill for F16) and
   copied twice (file→buf→tensor). mmap + one memcpy: startup load
   5.4 s → 2.5-4.1 s. RTF excludes load by construction (RTF = VAE+LM), so this
   is wall-clock, not RTF, but it is free.

PGO was re-tested properly after the push fix (Exp471: protocol-trained,
instrumented train/use cycle) and gives **0%** — the codegen axis is saturated.

| tier | files | phone RTF (10 s / 17 s / 69 s) | phone peak RSS | WER |
|---|---|---|---|---|
| **leader (VAE F16 + LM Q4_K_M, A78 build)** | **2.5 GB** | **6.02 / 5.23 / 5.62** | **2.99 GB** | **4.55% (40-utt on-device), 2.06% vs PT-ref** |
| recommended-old (VAE Q8-mixed, A78 build) | 1.9 GB | 6.14 / — / 5.76 | 2.44 GB | 4.55% (40-utt on-device) |
| min-size (VAE Q4-FFN, A78 build) | 1.6 GB | 6.48 / — / — | 2.16 GB | 5.23% (40-utt, pre-A78) |
| ultra-lean (Q4-FFN + 26 pieces, A78 build) | 1.6 GB | 6.61 / — / — | 1.97 GB | 5.23% (40-utt, pre-A78) |
| _pre-A78 F16 (for reference)_ | 2.5 GB | _10.5 / — / 9.73_ | 2.99 GB | 4.13% (desktop 40-utt), 3.3% (69 s) |

40-utt gates above are **on-device** (`.auto/eval40.sh` + `.auto/score_hyp.py`,
hyp sets in `eval-librispeech/hyp-{a78,f16a78}/`) because a codegen gate must run
on the target ISA; the desktop numbers (4.13%/4.41%) remain as history, with a
cross-arch offset of the same class (<1 pp). F16 vs Q8 on-device differ by one
substitution (4.55% both) while against the PyTorch reference F16 is closer
(2.06% vs 2.61%).

Correctness and flat memory hold on-device at all lengths (cross-device WER
< 1% same-config). `--xwin` (cross-window VAE carry) is ~8% faster on short
clips but drifts (+11.5% WER) on 69 s — shorts-only opt-in, legacy windows
default. The VAE runs at ~50% of DRAM roofline (Exp75: 72% of time in GEMM
kernels, fusion ceiling ≈1.2×); remaining kernel upside needs fused NEON
intrinsics, i.e. a 3rdparty change. **RTF < 1 on this phone class requires
retraining** (QAT INT8 VAE and/or a smaller encoder+LM), not more porting.

## Deeper quantization (measured)

40-utt LibriSpeech `test-clean` subset, same normalization (LM candidates are
requants from Q4_K_M, i.e. slightly pessimistic; VAE-Q8 also checked on the 69 s
multi-window clip to exercise cache carries):

| LM \ VAE | F16 (1.4 GB) | Q8-mixed (0.8 GB) | I8_S (0.67 GB) |
|---|---|---|---|
| Q4_K_M (1.1 GB) | **4.13%** ✅ shipped | **4.41%** ✅ (+0.3pp) | collapsed (loops) |
| Q4_0 (1.0 GB) | 5.6% on 5-file screen | — | — |
| Q3_K_M (0.9 GB) | 6.20% (+2.1pp) | — | — |
| Q2_K (0.7 GB) | 7.58% (+3.5pp) | — | — |

Recommended max-quant combo: **VAE Q8-mixed + LM Q4_K_M** (files 1.9 GB,
desktop RSS 2.72 GB, RTF ~1.1; phone: RTF 6.5/5.7/6.3 on 10/17/69 s,
RSS 2.44 GB flat, `-t 2`). 69 s WER 3.67%, identical parity class.
`--outtype q8_0_mixed` in `convert_vae_to_gguf.py` quantizes large weights
(last dim % 32 == 0) to Q8_0, keeps conv kernels/bias/norms in F16/F32
(Q8_0 blocks need 32-wide rows; depthwise kernels can't quantize).
Below Q4 the LM falls off a cliff (+2pp at Q3, +3.5pp at Q2) — rejected.
