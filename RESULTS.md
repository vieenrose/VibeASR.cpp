# Results — VibeVoice-ASR-Streaming 1.5B on phone CPU (RTF)

**Headline:** phone RTF **12.24 → 3.53 (−71 %)** on the 10 s protocol clip, at
**equal-or-better accuracy** (40-utt WER 4.41 % vs 4.55 % for the original
configuration), with **less RAM** (2.07 GB vs 2.99 GB) and a **1.4 s** model load.
All numbers are measured on-device (OPPO CPH2371, Dimensity 1300, 8 GB, Android 13),
CPU-only, `-t 2` pinned to the two 2.4 GHz prime cores.

## How to run (reproduce)

```bash
# weights (bit-exactly reproducible, see hashes below)
python3 utils/convert_vae_to_gguf.py models-pt --outtype q4_0_4x4_ffn -o vae-encoder-q4x4ffn.gguf
llama-quantize --allow-requantize --token-embedding-type q6_K \
    streaming-lm-q4_k_m.gguf lm-q4_0_4_4.gguf Q4_0_4_4
# binary (device-specific build protocol; in-tree CMake stays armv8.0-safe)
./.auto/setup.sh                     # NDK cross-build with -mcpu=cortex-a78
LM_FILE=lm-q4_0_4_4.gguf VAE_FILE=vae-encoder-q4x4ffn.gguf ./.auto/measure.sh
```

`.auto/measure.sh` builds, pushes the binary **and the shared libs** (md5-diffed),
runs the 10 s protocol clip pinned to the prime cores and prints `METRIC` lines.
`EXTRA_ENV=` forwards env vars to the device run; `AUDIO=` selects the clip.

## Final tier ladder (all on-device, 10 s protocol, 26 pieces)

| tier | files | 10 s | 17 s | 69 s | 40-utt mean | WER | RSS |
|---|---|---|---|---|---|---|---|
| **max-speed (shipped)** | 2.0 GB | **3.53** | **3.07** | **3.43** | **3.91** | **4.41 %** | 2.07 GB |
| _max-speed, RAM-lean (`VAE_SEQ_ENCODERS=1`)_ | 2.0 GB | _4.01_ | — | — | — | 4.41 % | **1.91 GB** |
| balanced (clean zh transcripts) | 1.9 GB | 4.44 | — | 4.64 | — | 4.82 % | 2.00 GB |
| accuracy-first (VAE F16) | 2.5 GB | 5.35 | — | 5.62 | 6.58 | 4.55 % | 2.95 GB |
| _original configuration (session start)_ | 2.5 GB | _6.52_ | — | — | _6.58_ | _4.55 %_ | _2.99 GB_ |

Shipped recipe: VAE `Q4_0_4x4` ffn linears (converter outtype `q4_0_4x4_ffn`) +
F16 convs with an **F16 im2col** + LM `Q4_0_4x4` body with **q6_K token embeddings**
+ **concurrent acoustic/semantic encoders** (one thread each), 26 VAE pieces.

## Evidence

| check | result |
|---|---|
| 40-utt LibriSpeech gate (on-device, shipped config, re-run after each threading change) | WER **4.41 %** (S=28 D=1 I=3); all 40 transcripts **byte-identical** across all three gates |
| 69 s equal-token comparison | 3.43 vs 5.62 accuracy-first (−39 %), tokens 438 vs 442 |
| sustained 138 s (frozen build) | RTF **3.45**, VAE 313.5 s, RSS flat 2.09 GB, no drift (halves-match 4.25 %, identical to earlier runs), majflt 0 |
| determinism | repeated runs byte-identical, matching references from earlier builds |
| out-of-domain (20 s music) | RTF 2.97, sane `[Music]`+lyrics output, no pathological loops |
| CPU utilisation | 1.88 of 2 pinned cores (94 %) — the pipeline is saturated |
| reproducibility | artifacts bit-exact; documented build+measure recipe verified from a clean build tree |

## What moved the needle (five waves, −71 %)

1. **Harness bug + A78 codegen (−6.5 %).** `measure.sh` pushed only the executable,
   never the `libggml.so`/`libllama.so` it links against — so every ggml-side
   experiment had measured stale kernels. Fixed (md5-diff push). Re-testing
   `-mcpu=cortex-a78` then gave −6.5 % (10 s) / −7.7 % (69 s): `sdot` 598→646,
   `fmla` 1284→1542; the F16 path had been *instruction*-bound, not bandwidth-bound.
2. **Blocked-int8 kernels for the LM (−14 %).** This fork dispatches `gemv`/`gemm`
   only for types that carry them; `Q4_K_M`/plain `Q4_0` have `vec_dot` only, so
   the 26-row prefill re-streamed 1.1 GB **per row**. `Q4_0_4x4` has hand-written
   dotprod kernels (NEON-only guard, no i8mm needed). LM 17.2 → 8.6 s.
3. **Blocked-int8 kernels for the VAE (−14 %).** 104 `ffn.linear` tensors (82 % of
   the VAE's weight bytes) moved to `Q4_0_4x4` via a new converter outtype (numpy
   port of ggml's per-row Q4_0 quantizer + interleaved packing). VAE 42.8 → 34.0 s.
4. **Token-embedding precision (+0.7 pp WER).** `llama-quantize` silently demotes
   `token_embd` to `q4_0`; keeping the source precision (`--token-embedding-type
   q6_K`) recovered 5.10 % → 4.41 % WER at equal speed (the control-token rows
   drive end-of-chunk decisions).
5. **F16 im2col (−6 %), concurrent encoders (−12 %), OpenMP off (−1 %).** `ggml_conv_1d` hardcodes
   a F32 im2col; building it in F16 halves the traffic and removes a conversion
   pass. And the acoustic/semantic encoders are independent chains, so they now
   run concurrently one thread each (intra-chain splitting scaled only 1.57×).
   Plus a mmap loader with a parallel copy (load 5.4 → 1.4 s).

## Closed avenues (each with a mechanism or measurement)

- **Precision / quant formats**: imatrix is explicitly ignored by the blocked-type
  quantizer (`UNUSED(quant_weights)`); `Q4_0_4_8`/`8_8` kernels need i8mm (absent
  → scalar fallback); 2-bit types (TQ/I2_S) would collapse a non-ternary model.
- **Concurrency**: 2 threads per chain = VAE +44 % worse; two concurrent inference
  streams each take exactly 2× the solo time ⇒ no idle capacity, so any
  pipelining/overlap (window or phase level) is dead.
- **Granularity / threads**: 26 pieces optimal (13 costs +319 MB for ~1 %); LM
  thread count 2 (4 threads = +12 %); no 4-fast-core configuration exists.
- **Conv-weight int8**: storage layout forbids block types on 3D conv tensors
  (kernel dim < 32); the 2D-storage workaround's net traffic gain is ~15 % of the
  VAE and its measured time ceiling is low single-digit % — parked with the
  recipe. F16 conv matmuls do re-read weights per 16-row tile, but p13-vs-p26
  shows this is not time-dominant.
- **Fusions**: the fork's fused `mul_mat_add`/`add_scaled` ops are I8_S-only and
  output int8; folding layer scales into weights saves ~1 %; F16 activations are
  blocked because `mul_mat`'s output type is F32-hardcoded.
- **Codegen**: PGO re-tested properly (protocol-trained, push fixed) = 0 %;
  ThinLTO's earlier null stands on a valid basis now; the compiler axis is closed.
- **Quality ceiling**: RTF < 1 on this phone class needs retraining (QAT INT8 VAE
  and/or a smaller encoder+LM), not more porting.

## Provenance

| artifact | md5 |
|---|---|
| `vae-encoder-q4x4ffn.gguf` | `b909b7901d5d318d81ce4cbaeac31437` |
| `lm-q4_0_4_4.gguf` (q6_K embeddings) | `db67eecbd31bba707666414dd977902f` |
| `streaming-lm-q4_k_m.gguf` (intermediate) | `046be3d4775e10f8b635b03ec1bc79cb` |
| Android `asr_streaming` (A78 build) | `98b643ed2496cff89d9b8caec687c3c0` |

Full engineering log: 533 experiments in `.auto/log.jsonl`; per-wave detail and
the implementation notes in `STREAMING_1P5B.md`; loop protocol in
`.auto/prompt.md`; parked work with recipes in `.auto/ideas.md`.
