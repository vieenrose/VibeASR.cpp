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
- Env-gated diagnostics: `VAE_GRAPH_STATS=1` (per-op byte-traffic profile of the
  first built graph - MUL_MAT ~32%, IM2COL ~11%, ADD ~10%, MUL/gelu ~7% of a
  ~700 MB/piece materialising-op total; RESHAPE/PERMUTE are views and cost
  nothing), `VAE_CACHE_TRACE=1` (per-site checksums),
  `VAE_DUMP_FRAMES=<prefix>` (output frames), `VAE_DUMP_SITE=<sN|all>` (site inputs).\n
**Quote WER with its tool.** Two scorers disagree slightly on the 40-utt gate because their error accounting differs on
1-2 of 731 token alignments: `score_hyp.py` implies a 726-token reference (4.68 %) where `.auto/compare_arms.py`
uses 731 (4.65 %), with identical S+D+I. Convention: **absolute** accuracy numbers come from `score_hyp.py`,
**between-system** claims come from `compare_arms.py` (paired, McNemar + bootstrap), and the two are never mixed in
one sentence. Re-deriving a scorer would re-quote every historical number, so this stays a quoting rule (Exp847).

## Reproduce & verify (Exp513)

Both deployed artifacts are **bit-exactly reproducible** from `models-pt` with
the frozen recipes, and the copies on the phone match the host byte-for-byte:

| artifact | recipe | md5 |
|---|---|---|
| `vae-encoder-convint8.gguf` (**default since Exp690**) | `python3 .auto/conv_int8.py vae-encoder-q4x4ffn.gguf out.gguf --device` - needs `.auto/quant4x4_arm` (NDK clang++ against build-android/libggml.so). `--device` is mandatory, not cosmetic: x86 `ggml_quantize_chunk(Q4_0_4_4)` writes valid nibbles with **zero scales**. Verify: `VAE_CONV_I8_CMP=1 VAE_CONV_I8_REF=<source> ` (prints the dequant-vs-source ratio plus its own control) | `52884a747af5aa1d…` (sha256) |
| `vae-encoder-q4x4ffn.gguf` | `python3 utils/convert_vae_to_gguf.py models-pt --outtype q4_0_4x4_ffn -o out.gguf` (33 s) - the F16-conv reference build | `b909b7901d5d318d81ce4cbaeac31437` |
| `lm-q4_0_4_4.gguf` | `llama-quantize --allow-requantize --token-embedding-type q6_K streaming-lm-q4_k_m.gguf out.gguf Q4_0_4_4` (1.2 s) | `db67eecbd31bba707666414dd977902f` |
| `streaming-lm-q4_k_m.gguf` (intermediate) | `utils/convert_streaming_lm_stage.py` + quantize | `046be3d4775e10f8b635b03ec1bc79cb` |
| Android build (`asr_streaming`) | `cmake -B build-android ... -DCMAKE_C_FLAGS="-mcpu=cortex-a78"` (`.auto/setup.sh`) | `98b643ed2496cff89d9b8caec687c3c0` |

Build fingerprint on the phone: `libggml.so 40a1f284e75dbc3c180215267068c12e`,
`libllama.so 92ad2456979d99e2a1afee4a8cebad1d`. To re-run the full tier
measurement: `./.auto/measure.sh` (VAE_FILE now defaults to `vae-encoder-convint8.gguf` and
LM_FILE to `lm-q8head.gguf`; the script hash-checks both against the device and pushes on
difference - before Exp689 it pushed neither, so an experiment could measure stale device bytes and
report them as a result). 1 piece is the harness default (PIECES=1 since v4.0, Exp706; p2 is the documented fallback); the binary `--vae-pieces` flag itself
defaults to 13. WER gate: `./.auto/eval40.sh <tag> 0 40` +
`venv-vibe/bin/python .auto/score_hyp.py <tag>`; for paired output-equivalence use
`.auto/compare_arms.py --gate hyp-A hyp-B eval-librispeech/refs.json`.

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
   5.4 s → 2.5-4.1 s, and since Exp531 the copy itself runs on four workers
   (load_s 1.6 → 1.4 s on the shipped tier, transcript byte-identical). RTF
   excludes load by construction (RTF = VAE+LM), so this is wall-clock, not RTF,
   but it is free. The rest of the load is the LM side (llama.cpp, off-limits).

PGO was re-tested properly after the push fix (Exp471: protocol-trained,
instrumented train/use cycle) and gives **0%** — the codegen axis is saturated.

### Third wave: LM blocked-int8 kernels (Exp476-477, 6.01 → 5.13)

The LM used 29% of the time and was **not** compute-bound: this fork's
`ggml.c` dispatches to blocked `gemv`/`gemm` kernels only when the *weight
type* carries them, and `Q4_K_M`/plain `Q4_0` carry `vec_dot` only — so every
prefill re-streamed the 1.1 GB weight set **once per input row** (a 26-frame
window ≈ 29 GB of traffic). `GGML_TYPE_Q4_0_4_4` ships hand-written asm
gemv/gemm kernels (160 `sdot`/`udot`, guarded by `__aarch64__`+`__ARM_NEON`,
so they run **without** i8mm — unlike the 8x8 variant which needs
`__ARM_FEATURE_MATMUL_INT8` and is unusable on this SoC).

Re-quantising the LM to `Q4_0_4_4` (`llama-quantize --allow-requantize …
Q4_0_4_4`, 1.3 s on the host) gives, at **equal output length** (69 s clip,
443 vs 442 tokens):

* RTF 4.8179 vs 5.6089 (**−14.1%**), LM 76.6 s vs 130.6 s (**−41%**):
  prefill 29.2 vs 68.8 (−58%), decode 47.4 vs 61.7 (−23%).
* Control that isolates the mechanism: plain `Q4_0` (same 4.5 bpw, no blocked
  kernels) = LM 17.2 s, i.e. identical to `Q4_K_M` — the win is the kernel
  path, not the bit width.
* 40-utt on-device gate: **WER 5.10%** (S=31 D=2 I=4) vs 4.55% (S=27 D=1 I=4)
  for `Q4_K_M` = **+0.55 pp**; paired text diff 2.34%, 15/40 utts differ.
  The `Q4_0_4x4` quantizer uses symmetric absmax (`d = amax/8`) instead of
  plain `Q4_0`'s asymmetric min/max, which is where the extra error comes from
  (`ggml-aarch64.c`, 3rdparty).

**Follow-up (Exp494-495): almost all of that +0.55 pp was the embedding table,
not the transformer body.** llama-quantize's default demotes `token_embd.weight`
q6_K → q4_0; keeping it at the source precision with
`--token-embedding-type q6_K` (no new rounding at all) costs +60 MB and recovers
the accuracy:

| LM file | 40-utt WER | protocol RTF | zh-clip tokens |
|---|---|---|---|
| Q4_K_M (accuracy-first) | 4.55% | 6.00 | 45 |
| Q4_0_4x4, token_embd q4_0 | 5.10% | 5.13 | 37 |
| **Q4_0_4x4, token_embd q6_K** | **4.41%** | **5.21** | **42** |

Mechanism: the embedding rows for the control tokens (`[speech_start]`,
`[speech_end]`, `<|text_chunk_end|>`) are what the LM uses to decide chunk
boundaries, so their quantization error shifts end-of-chunk decisions. The
recipe for this artifact is therefore
`llama-quantize --allow-requantize --token-embedding-type q6_K \
 <q4_k_m> <out> Q4_0_4_4`.

An **imatrix** was also tested (Exp492) and is inert for this type:
`quantize_q4_0_4x4()` does `UNUSED(quant_weights)`.
* Caveat: on the 10 s zh protocol clip the fast LM emitted 37 tokens vs 45
  (content truncated) — the 10 s RTF ratio (5.13 vs 6.01) therefore flatters
  itself by generating fewer tokens; quote the 69 s equal-length number.

### Seventh wave: OpenMP off for the 1-thread chains (Exp535, 3.53 → 3.47)

The concurrent-encoder design runs each chain with **one** thread, but a
OpenMP-enabled ggml still entered a parallel region per op (860 ops × 208
launches per clip) for zero parallelism. Building ggml without OpenMP
(`-DGGML_OPENMP=OFF`, now part of `.auto/setup.sh`) removes that bookkeeping;
llama.cpp falls back to its own threadpool for the LM (unchanged at 8.7 s).

A/B/A in one session at equal temperature (34.6-34.9 °C): OMP-off 3.4885 /
3.4660 (VAE 26.0 s both) vs OMP-on 3.5070 (VAE 26.4 s) → **−1.1 % RTF, −1.5 %
VAE**. New shipped band **3.47-3.49**.

### Sixth wave: concurrent encoders (Exp521, 3.99 → 3.53)

The acoustic and semantic encoders are independent — separate weights, separate
streaming caches, separate compute arenas. Thread-scaling data (Exp519: the VAE
scales only 1.57× from 1→2 threads; Exp678 bounds that at ≤1.85× and measures the mode choice
itself at only ~5 % — concurrent 17.2 s vs sequential 18.1 s wall for the same two chains, so
this is a small, already-banked win rather than a throughput lever) showed the dependent chains do not saturate
two cores, so the two encoders now run **concurrently, one thread each**
(`vae_encode_parallel_cached`, a second `(ggml_context, arena)` pair; default
on, `VAE_SEQ_ENCODERS=1` opts back out). Opting out is now a **last resort**:
at fine granularity the per-encoder arenas are small, so concurrency costs only
~30 MB and is worth −11 % (Exp641–643, see the RESULTS.md ladder).

* VAE 31.4 → **26.7 s (−14.7 %)**, RTF 3.99 → **3.53 (−11.6 %)** on the 10 s
  protocol; 69 s sustained 3.83 → **3.43 (−10.4 %)**; LM unchanged.
* Cost: **+165 MB RSS** (2.07 GB) — exactly the second arena's *used* footprint
  (`ggml_used_mem` = 169.3 MB). The former `+512 MB` arena slack constant was
  dead weight and is now 64 MB; unused arena pages were never resident, so the
  sequential path's RSS is unchanged. The `ac`/`sem` phase timers now *overlap*
  and no longer sum to the VAE total.
* Parity is byte-level, verified four ways: desktop transcript, phone 10 s
  transcript, the first three LibriSpeech gate utterances vs the previous
  sequential gate files, and the 69 s sustained transcript — all identical.
  (Per-element work is split-invariant; the only reductions are per-row or
  per-4-row-group, so 1-thread and 2-thread results agree exactly.)

### Fifth wave: F16 im2col for the convs (Exp503-508, 4.24 → 3.99)

`ggml_conv_1d`/`ggml_conv_1d_dw` hardcode a **F32 im2col**: that doubles the
im2col write+read traffic and forces `mul_mat` to materialise a second full
F16 copy of its src1 (`vec_dot_type` is F16 for F16 weights). Our conv weights
are F16, so `src/vae.cpp` now builds the convs itself with
`ggml_im2col(..., GGML_TYPE_F16)` + `mul_mat` (the stock ops' structure copied
verbatim, only the im2col dtype changed):

* VAE 33.8 → **31.4 s (−7.5%)**, RTF 4.24 → **3.99 (−6.3%)**, RSS 2069 MB.
* Not bit-identical (the F32→F16 rounding happens in the copy instead of in
  `mul_mat`), so the 40-utt gate was re-run: **WER 4.41% (S=28 D=1 I=3, H=697),
  identical error counts** to the pre-change gate. Mean 40-utt RTF 4.47.
* All four clips agree: 17 s 3.50 (−6.7%), 69 s 3.83 (−5.9%), slice-B 4.15
  (−5.9%), 40-utt mean 4.47 (−5.7%), every token count identical.
* Other tiers gain too (their convs are F16): balanced 5.14 → 4.91,
  accuracy-first 6.01 → 5.80, lean (p26) 4.39 → **4.00 @ 1.91 GB**.

**Gotcha for anyone repeating this:** do *not* use `ggml_im2col_asym` — its op
(`GGML_OP_IM2COL_ASYM`) is hardwired to `ggml_compute_forward_im2col_i8_s`, so
an F16 `dst_type` silently writes I8_S data into an F16 tensor (caught on the
x86 desktop build: output collapsed to 7 tokens). Use `ggml_im2col`.

### Fourth wave: VAE blocked-int8 kernels (Exp480-483, 6.01 → 4.26)

The same trick applies to the VAE, and the VAE is the bigger prize (71% of
time). Its 104 `ffn.linear` weight tensors are ~82% of the VAE file bytes and
their matmuls also went through the `vec_dot` path (weights re-read per
activation row). `utils/convert_vae_to_gguf.py` gained a `q4_0_4x4_ffn`
outtype: the quantizer is ggml's own per-row Q4_0 (`quantize_row_q4_0_ref`),
only the **byte packing** is the interleaved `block_q4_0x4` layout
(`make_block_q4_0x4`, xor 0x88) ported to numpy (`quantize_q4_0_4x4()`);
everything else stays F16.

* VAE 42.7 s → **34.0 s (−20.4%)**, RTF 6.00 → 5.14, RSS 2.16 GB (from 2.99),
  load 1.6 s. Versus the earlier plain-`Q4_0` VAE tier (VAE 47.4 s) this is
  −28%: again the kernel path, not the bit width.
* 40-utt on-device gate: **WER 4.82%** (S=29 D=2 I=4) vs 4.55% (S=27 D=2 I=4)
  = **+0.27 pp** — the best accuracy-per-speed trade of the loop.
* Combined with the fast LM (`Q4_0_4x4` on both) → the max-speed tier:
  10 s **4.26**, 69 s equal-token **4.05** (−27.9% vs accuracy-first), 40-utt
  mean **4.77**, WER 5.10% (S=31 D=3 I=3), RSS **2.05 GB**, 69 s transcript
  diff vs accuracy-first 3.93%.

### Final tier ladder (Exp525–542-era snapshot — SUPERSEDED)

> **Superseded in v4.0** (whole-window pieces: PIECES=1 is −1.2 % with ZERO gate discordants, Exp706): the shipped
default is PIECES=1 at RTF **~2.43** (v3.6 fused f32 depthwise-conv kernel −3.5 %, v3.7 layer-scale
epilogue −1.1 %, v3.8 gelu+bias −0.8 %, **v3.9 blocked-int8 conv weights −3.0 % with the gate paired
equivalent at McNemar p=1.0 - Exp690**)
> (69 s 2.41, 138 s 2.48, RSS 2.46 GB, gate 4.51 % with 0/731 tokens differing from the p2 gate); the current ladder
> lives in RESULTS.md. The table below (all on-device gated, 10 s protocol
> `-t 2`/C0, 26-piece default) is preserved as history.

| tier | files | 10 s | 17 s | 40-utt mean | 69 s (equal tokens) | WER (40-utt) | RSS |
|---|---|---|---|---|---|---|---|
| **max-speed (shipped default as of Exp542): 4x4 VAE + 4x4 LM (q6_K emb) + F16 im2col + concurrent encoders + OMP off, 26 pieces** | 2.0 GB | **3.48** (3.47-3.50, n=7, sd 0.010) | **3.05** | **3.86** | **3.38** | **4.41%**\*\*\* | 2.07 GB |
| _max-speed, RAM-lean (`VAE_SEQ_ENCODERS=1`)_ | 2.0 GB | _4.01_ | — | — | — | 4.41% | **1.91 GB** |
| balanced: VAE Q4_0_4x4-FFN + LM Q4_K_M (final build) | 1.9 GB | **4.40** | — | — | — | 4.82% | 2.07 GB |
| fast-LM: VAE F16 + LM Q4_0_4x4 (q6_K emb) _(pre-im2col)_ | 2.5 GB | 5.21 | — | 5.73 | — | 4.41% | 2.94 GB |
| balanced-lean: VAE Q4_0_4x4-FFN + 26 pieces (Q4_K_M LM) _(pre-im2col)_ | 1.9 GB | 5.21 | — | — | 4.94 | — | 1.98 GB |
| _max-speed @13 pieces, pre-concurrency (history)_ | 2.0 GB | _3.99_ | — | — | _3.83_ | 4.41% | 2.07 GB |
| accuracy-first: VAE F16 + LM Q4_K_M (final build) | 2.5 GB | **5.34** | — | — | 5.62 | 4.55% | 2.95 GB |
| ultra-lean (plain Q4-FFN + 26 pieces, history) | 1.6 GB | 6.61 | — | — | — | 5.23% | 1.97 GB |
| Q8-mixed VAE (history) | 1.9 GB | 6.14 | — | 6.71 | 5.76 | 4.55% | 2.44 GB |
| _pre-A78 F16 (history)_ | 2.5 GB | _10.5_ | — | — | _9.73_ | 4.13% (desktop) | 2.99 GB |

\*\* Gate re-run at the shipped 26-piece default (Exp514): WER 4.41%
(S=28 D=1 I=3, H=697) with **byte-identical transcripts** to both the 13-piece
gate and the pre-F16-im2col gate - every protocol and kernel change since then
is numerically inert on this gate.
\*\*\* Full 40-utt gates re-run twice more - after the concurrent-encoder change
(Exp525) and after GGML_OPENMP=OFF (Exp537): WER 4.41% each time with the same
error counts and **all 40 transcripts byte-identical** across all three gates -
numerical identity proven over the whole gate set for every threading change.
Mean 40-utt phone RTF: 4.4745 (sequential) -> 3.9113 (concurrent) -> **3.8597**
(+OMP off).

Note (Exp542): with concurrent encoders every tier carries the same two-arena
footprint, so the balanced tier no longer has a RAM advantage over the shipped
tier (both ~2.07 GB); its niche is now purely the clean zh transcript (the 4x4
LM garbles rare/proper tokens, Exp499). RAM-constrained devices use the
RAM-lean tier instead (`--vae-pieces 13 VAE_DEFER_LATE=1`, encoders still
concurrent): 2.54 on the protocol clip at 1.75 GB (Exp708; defer is REQUIRED at
fine granularity - without it p13 pays 2.67 in the deep-layer GEMV regime). The
old default (PIECES=13, no flags) predates the v4.0 defer rework and now runs
piece-wise by default, so the flag must be set explicitly.

With the corrected embedding the **max-speed tier Pareto-dominates the
accuracy-first tier**: 28% faster on the 40-utt mean and 27.5% on the 69 s clip,
equal-or-better WER (4.41 vs 4.55, one substitution apart), and 0.9 GB less RAM.
The 69 s token count is 438 vs 442 (-0.9%), i.e. the old short-clip truncation
(37/45) is mostly gone. Remaining zh-clip caveat, measured on both short clips:
the 10 s protocol clip ends 3 tokens early (42/45) and the 17 s clip is
content-complete but garbles the brand name ("YyY … YSR" vs "Y-voice … Y-voice
ASR") — i.e. the 4-bit body's residual cost is rare/proper-token fidelity, not
missing content. A controlled test (Exp499) localises it: a plain-`Q4_0`
requant of the same source - same tensor demotions, but the `vec_dot` path -
transcribes "Y-voice" cleanly, so the garble comes from the blocked-int8
kernel/layout path itself, not from which tensors are 4-bit. Use the balanced
tier (Q4_K_M LM, clean zh transcripts) when proper nouns matter; the 69 s
English WER story is unaffected (4.41%).

\* The `Q4_0_4x4` LM shifts greedy end-of-chunk decisions on some short clips,
so its 10 s / 17 s token counts are below baseline (40/45 and 64/68) and those
RTF ratios are partly fewer-decode-tokens effects; the 69 s clip is the only
one where its output length matches (444 vs 442-443), so **4.05 (−28%) is the
headline speed claim** for the max-speed tier. The balanced tier (4x4 VAE +
Q4_K_M LM) does not truncate (46/45, 69/68) and is the best all-round tier:
−14% RTF for +0.27 pp WER.

The **balanced-lean** variant (`--vae-pieces 26` on the same files) traded
+1% (10 s) / +2.4% (69 s) RTF for −180 MB RSS at *identical* transcripts in the
Exp542-era concurrent regime. The current RAM-lean tier is **p13, concurrent
encoders, defer explicitly ON** (`--vae-pieces 13` + `VAE_DEFER_LATE=1` — required
since v4.1, whose default is defer-OFF): 2.54 on the protocol clip (n=3, warm-session read;
same-session m2-OFF control 2.57), **2.47 on 138 s** and **2.70 on the 40-utt mean** (both re-measured
at v4.4, Exp786/789), at 1.75 GB and WER 4.79 % (2 tokens of 731 from the default tier, p = 0.5) — Exp643/708/711/717/767/786/789.

That also corrected the earlier "finer splits lose" conclusion, which held only
on the time axis: the deferred late pass captures the deep-layer batching
benefit window-wide at ANY fine piece count, so per-piece overhead is all that
is left — and the RAM driver is the *early-stage activation arena*, which scales
with piece size. The measured map over the runnable set {1, 2, 13, 26}
(non-divisors need a window-loop rework, Exp605), re-swept at v4.2 (conv-int8 + m2) with 3-4
interleaved reps per arm (Exp772), is **defer-OFF**: p1 **2.378** @ 2.37 GB (default,
time-optimal), p2 2.454 @ 2.07 (+3.2 %, fallback), p26 2.707 @ 1.71 (+13.8 %); **defer-ON**:
p13 2.531 @ 1.75 (the RAM-lean tier), p26 2.554 @ 1.72 — so p1 is the speed default while p13 is
the RAM sweet spot. **Those absolutes predate v4.5** (Exp821 moved p1 to 2.18 and its RSS to 2.19 GB), and
the shift is *asymmetric by construction*: the in-kernel conv left pad only fires when a window is a single
piece with no deferred late pass, so p1 gained ~3 % while p2/p13/p26 did not — which widens the p1-vs-p2 gap
from ~3 % to ~6 % and re-arms this axis under its own documented trigger (a per-piece cost-structure change).
Re-sweep only if a p1-vs-p2 RAM decision ever matters; the ratios among the fine counts (p13/p26) are
unaffected because none of them can take the fast path. `VAE_LATE_SPLIT=5` on this tier is **parity, not a win** (3 interleaved reps 2.425 vs 2.424 at
+19 MB, byte-identical output, Exp794): the −0.9 % recorded for it in Exp565 predates conv-int8, and
at p13 a piece is already 8 frames, so the deferred stage sat in the GEMM regime before the boundary
moved — there is nothing left for it to batch. split=6 stays the default and no third rung exists
between the shipping tiers. The p1-vs-p2 gap is ~3 % on the shipping config; a same-session 2×2 (Exp773)
puts it at +3.0 % with int8 convs and +1.9 % with F16 convs, so roughly 1 pp of it is the int8 conv
path's per-piece cost (paid twice per window when pieces are finer), and the blocked-tail kernel is
NOT a factor (`GGML_MM_M2_OFF=1` costs +0.7 % at p1 and +0.8 % at p2 — equal, so it is the LM's
prefill that benefits, not the VAE's piece shape). The v3.9 map's −1.2 % datum came from an
aborted/partly-confounded interleave, so treat the apparent "widening" as mostly a correction of
that number rather than a regression. Mechanism for the remaining ~2 %: finer pieces re-stream stage
weights and re-run graph build/launches per piece, i.e. twice per window at p2. Re-swept at v3.9 and
v4.2 only — do not re-sweep without another cost-structure change. Output relations changed with the fusions:
p1/p2 are byte-identical everywhere measured (protocol transcript hash-equal in 6+ reps;
40/40 gate transcripts identical, b=0/c=0); p13 differs from p2 by 2 gate tokens (p=0.5). The old
"p2 differs (40 vs 39 tokens)" sentence was the taps-vs-im2col era (Exp640) and no longer holds.
Deferred-vs-piece-wise has a granularity boundary inside the coarse regime too (Exp746):
at p1 defer-OFF wins −1.2 % (batching nothing, pure staging cost), at p2 it ties
(2.4621 vs 2.46, both eras - Exp639's verdict is the one pre-int8 closure that
survived), at p13 defer-ON wins −5 % (deep layers GEMV-shaped per piece). Rule of
thumb: defer pays iff pieces are GEMV-shaped; at GEMM-shaped pieces it ties (p2)
or costs (p1).

Against the original baseline (12.24) the max-speed tier is **−65%**; against
the pre-A78 loop best (6.52) it is −35%. The balanced tier dominates the fast
tier (same 10 s RTF, better WER, 0.7 GB less RAM) and the old ultra-lean tier
(much faster, better WER, +0.19 GB).

40-utt gates above are **on-device** (`.auto/eval40.sh` + `.auto/score_hyp.py`,
hyp sets in `eval-librispeech/hyp-{a78,f16a78}/`) because a codegen gate must run
on the target ISA; the desktop numbers (4.13%/4.41%) remain as history, with a
cross-arch offset of the same class (<1 pp). F16 vs Q8 on-device differ by one
substitution (4.55% both) while against the PyTorch reference F16 is closer
(2.06% vs 2.61%).

Correctness and flat memory hold on-device at all lengths (cross-device WER
< 1% same-config). **Determinism (Exp518):** two consecutive runs of the shipped config produce
byte-identical transcripts, matching a reference saved several builds earlier
(greedy decoding + fixed kernels => reproducible outputs).

**Gate-transfer verification (Exp520):** the F16-VAE tiers' 40-utt gate was
measured before the F16-im2col change. Re-running the accuracy-first tier on
the 69 s clip reproduced the pre-change transcript byte-for-byte (442 tokens),
so that gate transfers unchanged; together with the shipped tier's 40/40
byte-identical gate transcripts, every tier's accuracy number is valid for the
current kernels without a fresh 40-minute gate per tier.

**Sustained long-form check (Exp515 pre-, Exp526 post-concurrency, Exp538 at
the frozen build):** a 138 s clip (the 69 s chat concatenated with itself) runs
at RTF **3.45** (3.90 sequential -> 3.52 concurrent -> 3.45 with OMP off) with RSS flat at 2.09 GB (+165 MB for the concurrent encoders'
second arena, no growth over 8 minutes, majflt 0) and the two halves of the
transcript matching at 4.25% WER - identical to the pre-concurrency run, i.e.
no thermal cliff, no memory growth, no context drift, and no threading
instability over 36 chunks / 72 parallel encoder launches. `--xwin` (cross-window VAE carry) is now ~6 % SLOWER
on the protocol clip (re-measured at v4.1, Exp714: 2.55 vs 2.39 interleaved - it was -3.2 % pre-defer-flip;
reference now 2.38 at v4.2, sign unchanged;
the carry's per-chunk splice cost scales with piece size, and p1 doubles it). It remains a useful DIAGNOSTIC
(it moves diarization attribution on the multi-speaker probe) and a server-path option, not a speed lever.
It drifts (+11.5 % WER) on 69 s as well, so auto-selection by length was declined twice - a product
decision, not a loop protocol. The VAE runs at ~50% of DRAM roofline (Exp75: 72% of time in GEMM
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

## 9. Final-window flush — v4.6 protocol change (Exp829/Exp830)

The fixed-window protocol always zero-pads the **last** window to 26 frames, so the encoder spends most of
that window's work on silence: `vae_s = 0.21 + 3.45 x windows` with `windows = ceil(len/70400)`, i.e. a 10 s
clip encodes 332,800 samples for 240,000 of audio (1.387x). The last window now encodes only the real frames
(rounded up to a 2-frame piece) and the LM is fed the frames that exist instead of 26.

Measured (paired, interleaved reps):

| length | padded | flushed | delta | padded fraction of last window |
|---|---|---|---|---|
| 10 s protocol | 2.1866 | **1.9395** | −11.8 % | 0.653 |
| 17 s | 2.1760 | 2.0890 | −4.0 % | 0.193 |
| 69 s | 2.2091 | 2.1759 | −1.5 % | 0.080 |
| 138 s | 2.2542 | ~2.25 | ~0 % | 0.003 |
| 40-utt gate mean | 2.4456 | **1.9811** | −19.0 % | ~0.65 per clip |

**This is a protocol change, so it was gated, not just benchmarked.** 40-utterance gate: hybrid WER
4.38 % vs 4.51 % (jiwer-path scorer 4.68 % vs 4.82 %), paired token test **0 discordant of 731**
(McNemar p = 1.0), and one FEWER insertion. The direction is the point: the padded tail frames had been
generating tokens from silence (17 s 108 → 106 tokens, 138 s 877 → 876), so removing them removed an
insertion source rather than trading accuracy for speed. The gate improved MORE than the protocol clip,
which is the opposite of an overfit signature.

Hatch: `FLUSH_TAIL_OFF=1` restores the fixed-26-frame tail and reproduces the pre-Exp829 protocol exactly
(2.1866, transcript `55ac39b635cb`), so the old benchmark remains reachable.

The RAM-lean tier (`--vae-pieces 13` + `VAE_DEFER_LATE=1`) now flushes too (Exp830): its boundary buffers
are allocated at full-window spacing and packed down to the frames actually present, so a short final window
is self-consistent. Lean protocol cell 2.40 → **2.12** (−11.5 %) at ~1.75 GB, and its transcript is now
byte-identical to the shipped tier's — the previously documented 1-token lean/shipped difference was the
padded tail interacting with the deferred late path, not a real divergence.
