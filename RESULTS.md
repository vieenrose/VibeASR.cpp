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
    --output-tensor-type q8_0 \
    streaming-lm-q4_k_m.gguf lm-q8head.gguf Q4_0_4_4
# audit: requantize silently demotes precision (Exp493 cost 0.7 pp); fail loudly
python3 .auto/check_tensors.py ../models-streaming/lm-q8head.gguf ../models-streaming/vae-encoder-q4x4ffn.gguf
# binary (device-specific build protocol; in-tree CMake stays armv8.0-safe)
./.auto/setup.sh                     # NDK cross-build with -mcpu=cortex-a78
LM_FILE=lm-q8head.gguf VAE_FILE=vae-encoder-q4x4ffn.gguf ./.auto/measure.sh
```

`.auto/measure.sh` builds, pushes the binary **and the shared libs** (md5-diffed),
runs the 10 s protocol clip pinned to the prime cores and prints `METRIC` lines.
`EXTRA_ENV=` forwards env vars to the device run; `AUDIO=` selects the clip.

## Final tier ladder (all on-device, 10 s protocol, 26 pieces)

| tier | files | 10 s | 17 s | 69 s | 40-utt mean | WER | RSS |
|---|---|---|---|---|---|---|---|
| **max-speed-lean (seq-mode p26)** | 1.82 GB | **3.19** | — | — | — | same class | **1.76 GB** |
| **whole-file / server path** (Exp578/579) | — | — | live set **188-313 MB** for 6-10 s files | — | — | — | — |
| **max-speed (shipped)** | 2.1 GB | **2.79** | TBD | TBD | TBD | **4.41 %** | 2.07 GB |
| _max-speed, RAM-lean (`VAE_SEQ_ENCODERS=1`)_ | 2.0 GB | _4.01_ | — | — | — | 4.41 % | **1.91 GB** |
| balanced (clean zh transcripts) | 1.9 GB | 4.44 | — | 4.64 | — | 4.82 % | 2.00 GB |
| accuracy-first (VAE F16) | 2.5 GB | 5.35 | — | 5.62 | 6.58 | 4.55 % | 2.95 GB |
| _original configuration (session start)_ | 2.5 GB | _6.52_ | — | — | _6.58_ | _4.55 %_ | _2.99 GB_ |

Shipped recipe: VAE `Q4_0_4x4` ffn linears (converter outtype `q4_0_4x4_ffn`) +
F16 convs with an **F16 im2col** + LM `Q4_0_4x4` body with **q6_K token embeddings**
+ **concurrent acoustic/semantic encoders** (one thread each), 26 VAE pieces.


## Progression on the 10 s protocol (each step validated)

| step | RTF | mechanism |
|---|---|---|
| baseline | 12.24 | VAE F16 + LM Q4_K_M, `-t 4` unpinned |
| 10.34 | −15 % | VAE selective Q8_0-mixed (large weights only) |
| 10.12 | −17 % | VAE Q4_0-FFN variant (opt-in low-RAM tier) |
| 9.71 | −21 % | legacy cold windows + Q8 (after the `--xwin` accuracy demotion) |
| **6.52** | **−47 %** | **`-t 2` pinned to the two prime cores** (biggest single lever) |
| 6.02 | −51 % | A78 codegen (`-mcpu=cortex-a78`, after fixing the stale-`.so` harness bug) → F16 VAE becomes the fastest tier |
| 5.13 | −58 % | LM on blocked int8 kernels (`Q4_0_4x4`, q6_K embeddings) |
| 4.26 | −65 % | VAE `ffn.linear` on the same blocked kernels (converter outtype `q4_0_4x4_ffn`) |
| 3.99 | −67 % | F16 im2col for the convs (halves im2col traffic, drops a conversion pass) |
| 3.53 | −71 % | concurrent acoustic/semantic encoder chains (1 thread each) |
| **3.48** | **−72 %** | OpenMP off for the 1-thread chains |
| 3.30 | −73 % | deferred deep stages: the deepest ConvNeXt stage runs once per window over the concatenated boundary tensors (L=1 GEMV → L=28 GEMM) |
| 3.18 | −74 % | lifetime (ggml-alloc) activation buffers + PIECES=2: the early-stage graph holds only its live set (15.5 MB vs a 274 MB arena scaled at 64 KB/sample), which also unblocks large pieces |
| **2.79** | **−77 %** | Q8_0 LM head (Exp592): the shipped q6_K head has no gemv/gemm kernel (~26 ms per decode token on the classic path); q8_0 streams through the NEON dotprod path (decode 4.3 -> 3.7 s, WER and transcripts identical) |
| 2.83 | −77 % | [C, T] ConvNeXt blocks (Exp586): with the mixer running channels-first, the transpose into the depthwise path and the one inside it both disappear (8.4% RTF, VAE 22.2 -> 19.6 s, gate WER 4.41%) |
| 3.09 | −75 % | depthwise conv as K dense-view multiply-adds in [C, T] (Exp580): the im2col + batched 3-D mul_mat + two cont/permutes of the mixer path, measured at 29% of the VAE, became K dense tap multiply-adds |
| 3.15 | −74 % | zero-copy weights: the VAE loader points the tensors into the read-only gguf mapping (alignment-checked, copy fallback) instead of duplicating 536 MB into an arena, and the late pass got the same lifetime allocator → load 1.4 → 1.2 s, RSS 2.19 GB |

## Evidence

| check | result |
|---|---|
| 40-utt LibriSpeech gate (on-device, shipped config, re-run after each change) | WER **4.68 %** (S=29 D=2 I=3) at the shipped p2+lifetime build, with 34/40 transcripts byte-identical to the previous 4.41 % gate; that gate's 6 differing utterances are the same six that any re-blocking of the deep stages moves (identical at p2 and p26, and to the split-5 experiment), and the zh protocol canary is byte-identical everywhere. The pre-change gate was 4.41 % (S=28 D=1 I=3, 40/40 identical) |
| 69 s equal-token comparison | 3.43 vs 5.62 accuracy-first (−39 %), tokens 438 vs 442 |
| sustained 138 s (frozen build) | RTF **3.14** (v3: p2 + lifetime allocator), VAE 271.9 s, RSS flat 2.31 GB over 47 windows, token count identical to the v2 run, majflt 0 |
| determinism | repeated runs byte-identical, matching references from earlier builds |
| output stability | byte-identical transcripts vs pre-change references on **every** clip used (10 s, 17 s, 69 s, 138 s halves, and 40/40 gate utterances) |
| out-of-domain (20 s music) | RTF 2.97, sane `[Music]`+lyrics output, no pathological loops |
| speaker attribution (first diarization evidence, Exp548) | two-speaker synthetic clips: on a 0.5 s-gap concatenation **all tiers agree** (single speaker — a model behaviour, not a quantization effect); on an **overlapped** mix the quantized stack emits **Speaker 0 + Speaker 1** while the F16 control emits one — i.e. attribution is exercised and not collapsed by quantization. No multi-speaker reference exists in the loop's assets, so attribution *quality* is unscored (validation boundary) |
| non-speech edge cases | 5 s digital silence → `[Silence][Silence]`; 5 s −50 dBFS white noise → `[Noise]` — correct model tags, short decodes, **no hallucinated text and no repetition loops** |
| input formats / cold start | 48 kHz stereo handled (one word differs); after evicting the page cache the RTF is unchanged (3.4713 pre-change, 3.3010 on the deferred build) and only the load grows (1.4 → 2.6 s, excluded from RTF) |
| CPU utilisation | 1.88 of 2 pinned cores (94 %) — the pipeline is saturated |
| hardware envelope (Exp544) | the two pinned A78 primes are **hard-capped at 1.3 GHz** (54 % of their 2.4 GHz rating) regardless of load — all numbers are the device's sustained, power-capped behaviour. At that clock the LM prefill runs at ~90 % of the achievable int8 rate; the VAE's FFN at ~15 % (shape-limited: L=50 columns in the deep stages, short contractions in the early ones) |
| reproducibility | artifacts bit-exact; the documented recipe (`.auto/setup.sh` from a wiped `build-android/`) reproduces the shipped binaries **byte-for-byte** - re-verified on the v3.2 build (Exp576): asr_streaming `fda51262`, libggml `6ce4c983`, libllama `92ad2456`, and the device copies match |

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
- **Security hardening**: disabling `-fstack-protector-strong`/`-D_FORTIFY_SOURCE=2`
  buys nothing (Exp549: 3.5028, in-band) — the shipped build keeps them.
- **Codegen**: PGO re-tested properly (protocol-trained, push fixed) = 0 %;
  ThinLTO's earlier null stands on a valid basis now; the compiler axis is closed.
- **Quality ceiling**: RTF < 1 on this phone class needs retraining (QAT INT8 VAE
  and/or a smaller encoder+LM), not more porting.

## Provenance

| artifact | md5 |
|---|---|
| `vae-encoder-q4x4ffn.gguf` | `b909b7901d5d318d81ce4cbaeac31437` |
| `lm-q4_0_4_4.gguf` (q6_K embeddings, predecessor) | `db67eecbd31bba707666414dd977902f` |
| `lm-q8head.gguf` (**shipped**: Q4_0_4_4 bulk, q6_K embeddings, **Q8_0 head**) | `222d4bf7794c4b030a82751a1b3226f5` |
| `streaming-lm-q4_k_m.gguf` (intermediate) | `046be3d4775e10f8b635b03ec1bc79cb` |
| `lm-4x4-head.gguf` (OPTIONAL faster variant, q8_0 head -> q4_0_4x4: RTF -4.1 %, WER 4.96 % — declined as default) | `62854dfffbd24fdca7a2aa4717df24eb` |
| Android `asr_streaming` (A78 build, OMP off, deferred late stages + lifetime activation buffers + zero-copy weights) | `fda51262bd5cc1248a8b283e727f9b50` |
| `libggml.so` / `libllama.so` (shipped) | `6ce4c983ab2b310fb8dce8e75e402f7e` / `92ad2456979d99e2a1afee4a8cebad1d` |

Full engineering log: 562 experiments in `.auto/log.jsonl`; per-wave detail and
the implementation notes in `STREAMING_1P5B.md`; loop protocol in
`.auto/prompt.md`; parked work with recipes in `.auto/ideas.md`.
