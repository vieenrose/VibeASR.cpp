# Results — VibeVoice-ASR-Streaming 1.5B on phone CPU (RTF)

**Headline:** phone RTF **12.24 → 2.70 (−78 %)** on the 10 s protocol clip, at
**equal-or-better accuracy** (40-utt WER 4.41 % vs 4.55 % for the original
configuration), with **less RAM** (2.23 GB vs 2.99 GB) and a **1.2 s** model load.
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
`EXTRA_ENV=` forwards env vars to the device run; `AUDIO=` selects the clip (or use
`measure.sh --clip PATH`, which validates and pushes the file — since Exp648 unknown
flags are a hard error, because `--clip` used to be silently ignored and measured the
default 10 s clip instead). Behavioural probes and their scorer:
`.auto/build_gate_multispk.py` (deterministic, CC0 Common Voice 17.0 sources) and
`.auto/score_stream.py --selftest` (WER by language + diarization attribution with the
standard optimal tag↔voice mapping; 5 self-tests). The probe is also published as a
public dataset mirror for reuse elsewhere:
**hf.co/datasets/Luigi/bilingual-zh-en-multispk-probe** (CC0, wav + `gold.json`;
regenerated audio must hash to `984e60b14cfe…` to be the same probe).

## Final tier ladder (all on-device, 10 s protocol, 2 pieces)

| tier | files | 10 s | 17 s | 69 s | 40-utt mean | WER | RSS |
|---|---|---|---|---|---|---|---|
| **max-speed-lean (`--vae-pieces 13 VAE_DEFER_LATE=1`, Exp643/708/711/732/766/781/784/785)** | 1.75 GB | **2.41** | **2.39** | **2.41** | **2.69**° | **4.79 %**† | **1.75 GB** |
| _same, 26 pieces (−32 MB, +1.3 % time, identical output)_ | 1.82 GB | _2.88_ ‡ | — | _2.83_ ‡ | _3.20_ ‡ | _4.55 % (tag `leanp26c`)_ | _1.84 GB_ |
| **whole-file / server path** (Exp578/579) | — | — | live set **188-313 MB** for 6-10 s files | — | — | — | — |
| **max-speed (shipped, v4.4 — v4.2 + fused gelu+bias + fused rms_norm·gamma, Exp708/709/765/766/781/784/785)** | 2.37 GB | **2.26** | **2.25** | **2.28** | **2.54** | **4.51 %** | 2.37 GB |
| _138 s sustained (v4.4, Exp709/701/766/781/785)_ | | _2.35_ | | | _RSS 2388.5 MB flat, majflt 0_ | | |
| _same tier at PIECES=2, defer ON (previous default)_ | 2.11 GB | _2.46_ | _2.41_ | _2.43_ | _2.75_ | _4.51 %_ | _2.12 GB_ |
| _same tier with F16 conv weights (`VAE_FILE=vae-encoder-q4x4ffn.gguf`, the reference build)_ | 2.2 GB | _2.54_ | _2.47_ | _2.50_ | _2.82_ | _4.41 %_ | _2.23 GB_ |

Gate cells re-measured at v3.9 with corrected harness defaults (Exp694): shipped tier 40-utt mean
**2.7524**, WER 4.51 %, differing from the frozen reference `hyp-gate645` by **1 token of 731**
(discordants 0 vs 1, McNemar p=1.0 → output-equivalent); lean p13 mean **2.8187** (re-measured Exp711, 2 tokens from the
shipped tier (p=0.5). The `°` estimate cells are gone. NOTE on provenance: two gates run earlier on
Exp690 used `eval40.sh`'s then-stale defaults (Q4_K_M LM, 13 pieces) and are superseded by these — the
stamp in each `hyp-*/run-info.log` is what proved that, and every pre-existing gate row above checks out
correctly. `.auto/tier.env` plus the audit now make that class of mistake fail loudly.
| _last resort: p26 + `VAE_SEQ_ENCODERS=1` (sequential encoders)_ | 1.82 GB | _3.19_ | — | _3.13_ | — | _4.55 %_ | _1.82 GB_ |
| balanced (clean zh transcripts) | 1.9 GB | 4.44 | — | 4.64 | — | 4.82 % | 2.00 GB |
| accuracy-first (VAE F16, 10 s + RSS re-measured Exp702, 69 s Exp703) | 2.5 GB | 4.34 | — | 3.98 | 6.58 | 4.55 % (EN gate only — the Q4_K_M LM emits repetition-class artifacts on zh: "Y Y boys"/letter-soup on the 10 s clip with EITHER VAE, while the shipped 4x4+q8head LM is clean; same class as Exp499's Y-voice note, so this tier's accuracy claim covers English read speech, not zh) | 3.07 GB |
| _original configuration (session start)_ | 2.5 GB | _6.52_ | — | — | _6.58_ | _4.55 %_ | _2.99 GB_ |

> ° Measured before the Exp670 depthwise-conv1d path became the default; scale by ~0.965.
> The **17 s** column was re-measured in Exp675 on `chat17.wav` = a deterministic 17.0 s excerpt of `chat69.wav` (first 408,000 frames). The clip the old 17 s cells were taken on no longer exists: `chat.wav` and `chat69.wav` became byte-identical (both 3,311,576 B) when the 69 s clip was pushed, so the pre-Exp675 17 s numbers are not comparable to the new ones and were replaced, not rescaled. Length ladder is monotone and the lean/shipped gap is now consistent across all four lengths (+5–6 % at v4.2, was +3.8–4.4 % at v3.8), which is the useful cross-check: per-piece overhead does not grow with clip length.
> Re-measured on the current default (**v4.4**, five fusions + m2 kernel): shipped tier 10 s **2.26**, 17 s **2.25**, 69 s **2.28**, 138 s **2.35**, 40-utt mean **2.5415** (tag `norm784`: **40/40 byte-identical** to both `gelufuse781` and the frozen `gatem2` - b=0/c=0 of 731 tokens, McNemar p=1.0 in both pairs, WER 4.51 %); lean tier 10 s **2.41**, 17 s **2.39**, 69 s **2.41**, 138 s **2.47** (2 reps each, RSS 1752-1766 MB, all majflt 0). Token counts are unchanged at every length (39/108/446/877 shipped), so the two fusions are output-neutral end to end, not just on the protocol clip. ° = lean 40-utt mean carried from the pre-fusion gate and corrected by the measured tier delta; re-run to confirm.

> **The WER column is read-speech-only (Exp651, Exp653, Exp654).** Every number above is
> LibriSpeech *test-clean*. On held-out consumer audio (Common Voice 17.0, CC0, never used
> by this loop) the shipped tier scores **29.6 %** (English, 220 tokens) and **16.0 %**
> (zh-TW, 468 tokens), so a WER from this ladder must carry the qualifier “read speech”.
> Whether the tiers *separate* on hard audio is **unresolved, and now bounded**
> (Exp652 → 653 → 655 oscillated because it was argued with unpaired CIs). Tested
> properly as a *paired* comparison (identical audio, identical reference; exact McNemar
> on the discordant tokens — `.auto/compare_arms.py`): pooled over 688 hard-audio tokens
> the discordants are 27 (shipped worse) vs 15 (accuracy-first worse), **p = 0.088** —
> suggestive, not significant. zh +1.28 pp [−1.07, +3.63], en +3.18 pp [0.00, +6.36].
> In zh only **30 of 468 tokens differ at all** between the arms while both fail on 56,
> so the error mass is shared (model-side) and *any* precision-only change is capped at
> ~1–3 pp by construction. Settling it needs ≥1400 tokens (power 0.71), ideally ~2750
> (power 0.95) — below that, do not re-run: it cannot conclude.

Shipped recipe: VAE `Q4_0_4x4` ffn linears (converter outtype `q4_0_4x4_ffn`) +
F16 convs with an **F16 im2col** + LM `Q4_0_4x4` body with **q6_K token embeddings**
+ **Q8_0 output head** + **concurrent acoustic/semantic encoders** (one thread each), 2 VAE pieces (PIECES=2 harness default).

**RAM-lean tier (v4.2 recipe, Exp708/711/717/767).** Same files, `--vae-pieces 13
VAE_DEFER_LATE=1` — the defer flag is now REQUIRED (defer is off by default since
v4.1; without it p13 pays the L≈1 deep-GEMV cost: 2.67) — 2.54 on the protocol clip
(n=3; same-session m2-OFF control 2.57, so the m2 delta on lean is −1.0 % and the
absolute carries the session's warm offset — the known ±1 % lean wobble, Exp739-744),
2.82 on the 40-utt mean (pre-m2 tag `lean711b`) and 2.58 on 138 s (pre-m2) at 1.75 GB RSS
(uniform +5–6 % vs the default across every length at this stack, unlike the +0.9 %-on-shorts era of Exp646).
Long-form is clean at this granularity: the 138 s soak holds RSS flat
(peak 1766 MB, series ~1756–1808 MB, `majflt 0`, Exp717) and its repetition structure matches the
default's (188 vs ~205 repeated 10-grams — the repeats are in the clip, not a
loop), with 19 sentence-level wording diffs of the usual marginal class. Its
accuracy is the gated 4.55 % **by byte-identity, not by inference**: a full
40-utt gate at p13 (tag `leanp13c`) scored WER 4.55 % (S=29 D=1 I=3, H=696) with
**40/40 transcripts byte-identical** to the `leanp26c` set, so the two fine
granularities are the same model output at different cost. 26 pieces is the
−32 MB / +1.3 % variant of the same tier. The `VAE_SEQ_ENCODERS=1` config that used to define this tier is now a
last resort: at fine granularity the early-stage activations are tiny, so
concurrency costs ~30 MB and buys −11 % RTF (3.13 → 2.78 on 69 s). The deferred
late stages are what make this cheap — at p26/p13 they turn the deep GEMVs into
window GEMMs (−5.7 %); at p2 the pieces are already 13 frames, so defer
measures a no-op there.

**Column definition (corrected in Exp645).** The `40-utt mean` column is the
arithmetic mean of the per-utterance phone RTF over the 40-utt gate set
(`.auto/eval40.sh`; short utterances pay window amortization), **not** a clip
cell — it sits above the 10 s number by construction. The shipped tier measures
**2.66** (tag `gatem2`, v4.2; the historical `gate645` read 3.1235 on the pre-v4
stack); the cell once carried the 10 s value by mistake, and the era table's 3.91 was
the same quantity pre-optimization. On this axis the lean tier costs **+5.3 %** over
the default (2.8187, tag `lean711b`) — matching its +5.4 % on the 10 s protocol clip.
(Retired note: the Exp646-era +0.9 %-on-shorts reading held under the p2/defer stack;
at v4.1 the per-piece overhead scales with window count again and the lean gap is
uniform ~+5–6 % at every length — the ~350 MB saving is no longer nearly free.)

† Verified, not inferred: `leanp13c` = WER 4.55 % with 40/40 transcripts
byte-identical to the `leanp26c` gate set. The invariance is expected because
the early stages are exact under the streaming-cache invariant; the dw
taps-vs-F16-im2col branch (Exp640) is selected by piece size, which is why
p13/p26 agree with each other and differ from p2 (40 vs 39 tokens on the
protocol clip). **Paired note (Exp656):** that p2-vs-p13 difference amounts to **one
reference token out of 731** on the 40-utt gate (McNemar p = 1.0), so the two tiers
should be read as output-equivalent; the WER column's 4.41 vs 4.55 exaggerates a
single-word difference into an apparent systematic cost.


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
| 40-utt LibriSpeech gate (on-device, shipped config, re-run after each change) | WER **4.68 %** (S=29 D=2 I=3) at the shipped p2+lifetime build, with 34/40 transcripts byte-identical to the previous 4.41 % gate; that gate's 6 differing utterances are the same six that any re-blocking of the deep stages moves (identical at p2 and p26, and to the split-5 experiment), and the zh protocol canary is byte-identical everywhere. The pre-change gate was 4.41 % (S=28 D=1 I=3, 40/40 identical). Re-validated at HEAD (Exp613, after the gelu-knob and dw-taps-guard commits): WER **4.41 %** (S=28 D=1 I=3 H=697), 40/40 transcripts byte-identical to the hyp-q8head reference set |
| 69 s equal-token comparison | 3.43 vs 5.62 accuracy-first (−39 %), tokens 438 vs 442 |
| sustained 138 s | RTF **2.51** (v3.9 stack, Exp701; v3.8 value 2.565, v3.5-era 2.80), **877** tokens (marginal class, identical count to Exp690), majflt 0. Memory sampled during the run rather than only at exit: 17 samples over 5.4 min, RSS **2073–2077 MB flat**, trend **−3.6 MB/min** (noise), peak (VmHWM) 2129.5 MB — the conv-int8 weights cut ~124 MB off the soak and the fusion-era graph nodes still cost nothing on long runs; repetition screen 199 vs ~205 baseline (no loop pathology) |
| co-runner guard (Exp679) | While building that sampler, two `asr_streaming` processes were found **running concurrently** on the device (2.25 GB resident each, 16 and 15 min elapsed). Cause: a host-side `timeout` kills the `adb` client but not the device-side process. Two pinned 2-thread runs share two A78s, so every timing taken while a survivor is live is ~2× inflated — numbers, not errors. `measure.sh` now warns when a co-runner exists and `audit_harness.py` **fails** if one has a model resident. Sleeping sub-MB survivors are clutter, not bias (there were five, all idle). Two `set -e` traps in the first version of that guard (`grep -c` exits 1 on the idle case; a bare `[ ] && echo` returns 1 when false) silently aborted `measure.sh` for ~25 min of wall time — caught because the sampler refused to report on zero samples |
| determinism | repeated runs byte-identical, matching references from earlier builds |
| output stability | byte-identical transcripts vs pre-change references on **every** clip used (10 s, 17 s, 69 s, 138 s halves, and 40/40 gate utterances) |
| out-of-domain (20 s music) | RTF 2.97, sane `[Music]`+lyrics output, no pathological loops |
| speaker attribution | scored since **Exp648** on a real multi-speaker reference (`eval-bilingual/gate_ms.wav`, 4 gold voices): the shipped tier uses **one** `Speaker` tag for all four voices — attribution error **0.567** (exactly the one-bucket optimum), per-voice consistency 0.743, tags used 1/4. Earlier synthetic probes (Exp548/609) *did* split on an overlapped two-speaker mix, so tags are reachable; the collapse is behaviour-dependent (sequential turns with 300 ms gaps here), not a quantization artifact. Quantization itself is exonerated: tiers agree on the synthetic clips. **Root cause found (user-requested debug): the collapse is the windowed protocol, not the model.** The pipeline carries no speaker state across windows — `vae_cache_reset` per window in `demo/asr_streaming.cpp`, KV history is text only, and no speaker-embedding/clustering/voiceprint code exists in `src/` or `demo/` — so chunk labels are numbered window-locally and a second tag needs two voices' contrast *inside one window's* acoustic embeddings. Proven on the same file, current stack: `--xwin` (only change: encoder carry across windows) moves attribution **0.567 → 0.294**, tags 1/4 → 2/4 (WER 14.4 % → 27.8 %, so carry is a diagnostic, not a fix); 100/1000 ms gaps use 3 tags. The HF probe card's 'collapsed to a single tag' line has been corrected to this mechanism, re-uploaded live (commit 6ef33879, verified by re-download). **v2 overlap probe closes the loop** (same user request: `gate_ms_v2.wav` 45 s, 3 grid-aligned overlap pairs + 4 sequential controls, same 4 voices, builder `.auto/build_gate_ms_v2.py` seed 650, manifest `eval-bilingual/manifest_ms_v2.json`, blessed into `device_assets.json`, published to the HF dataset as `bilingual_multispk_v2_45s.wav` + `gold_v2.json`/`turns_v2.tsv`/`transcript_v2.txt`): baseline tags **4/4**, attribution **0.424** vs v1's 0.567 — the separation v1 structurally cannot show. v2 WER 25.9 % vs v1 14.4 % is the harder (overlap) test, not a regression. Four-arm characterization (autoresearch, not optimization - 85 tokens): xwin WER 20.0 %/attr 0.429/tags 5-4 (phantom 5th speaker - carry helps overlap words, not identity); accfirst WER 18.8 %/attr 0.427; lean-p13 tag-identical to shipped (WER 25.9 %/attr 0.424). Attribution invariant 0.42-0.43 across carry/precision/granularity while numbering renumbers locally - separation robust, no knob moves it |
| bilingual + code-switch probe (Exp648, 57 s, 12 turns; **re-scored Exp654**) | WER **14.4 %** overall — **English 10.7 %**, **zh-TW 15.9 %** — with **zero** deletions and insertions: content, turn order and language identification are all correct; the remaining Chinese errors are **phonetic/homophone confusions** (三→山, 盐→缘, 联→莲, 合→和, 镜→尽). Intra-utterance code-switching is handled (one turn mixes both languages and is transcribed). Neither gap is visible to the 40-utt gate or the protocol RTF. Probe clip ran at RTF **2.40** / RSS 2.23 GB. *Superseded:* the original row said 30.9 % / zh 39.1 % — the scorer had not been folding Traditional↔Simplified, so ~26 pp of that was script mismatch, not misrecognition |
| paired significance testing (Exp655) | between-system claims in this loop were previously made with unpaired CIs, which caused the Exp652/653 flip-flop. Now tested with exact McNemar + a paired bootstrap over tokens (`.auto/compare_arms.py`; self-tests include analytic cases — (0,3) → p = 0.25, (5,1) → 0.21875, (10,2) → 0.0386, identical systems → p = 1, Δ = 0 — and one implementation bug in the first revision was caught by them). Worked example: shipped vs RAM-lean p13 on held-out English are statistically **indistinguishable** (Δ −0.45 pp, 0 vs 1 discordant, p = 1.0), so the lean tier's ~350 MB saving is free on hard audio as well as on the clean gate |
| paired testing of archived gate runs (Exp656, zero device time) | the archived per-utterance transcripts (`eval-librispeech/hyp-<tag>/`) let historical claims be re-examined *paired*: default p2 vs RAM-lean p13 differ on **exactly 1 token out of 731** (b=0, c=1, McNemar p = 1.0) — so the "4.41 vs 4.55 %" gap is a single word, not an accuracy difference, and with Exp655 (identical on hard audio, p = 1.0) the lean tier is **output-equivalent to the default in both domains**. Three controls that must be zero, all were: same config re-run (0/731), same config 24 h and ~30 runs apart (0/731), p13 vs p26 (0/731, reproducing the documented byte-identity independently) |
| fused depthwise-tap kernel (Exp664) | `GGML_OP_ADD_SCALED` gained an **f32 NEON path** (`a*gamma + b`), so the tap accumulation is one op per tap instead of `mul`+`add`: **protocol 2.764 → 2.697 (−2.4 %), VAE 19.6 → 18.9 s, 40-utt gate mean 3.124 → 3.012, lean p13 2.84 → 2.77**, with **40/40 gate transcripts byte-identical** (paired: 0 of 731 tokens differ). Two lessons worth keeping: (1) written in plain C the compiler emits **vfma**, which shifted 4 gate tokens (+0.55 pp, 4/0 discordants, p = 0.125) for the same speed — bit-identity required explicit `vmul`+`vadd` with **no fma contraction**, because the old chain rounded the product first; (2) the op requested a work buffer sized for the I8_S path **regardless of type**, which armed the Exp578 `ggml_graph_compute_with_ctx` trap and segfaulted a piece-wise path mid-gate — work buffers must be requested by type. A `% g_ne0` inside the inner loop also cost +9 % (blocks vectorisation): hoist per-channel indexing out of hot loops. Escape hatch `VAE_DW_AXPY_OFF=1`; `VAE_LS_FUSE=1` (layer-scale+residual) is byte-identical but speed-neutral and stays off |
| fused depthwise-conv kernel (Exp666 → 669 → 670) | the tap CHAIN became ONE f32 depthwise conv1d kernel filling the reserved `GGML_OP_CONV1D` slot: **(K+1) tensor passes instead of ~3K** → **protocol 2.66 → 2.57 (−3.5 %), VAE 18.5 → 17.6 s, gate mean 3.012 → 2.881, lean p13 2.77 → 2.68, RSS −9 MB**, with **40/40 gate transcripts byte-identical** (plus a 0/8 lean spot-check). Bit-compatible by construction: ascending k, `vmul`+`vadd`, never `vfma`. It was reverted at Exp666 for a non-reproducible segfault whose real cause was **a data race in this repo's own ablation harness** — `vae_abl()` memoized env knobs in a function-local `static std::map` shared by both concurrent encoder threads (`emplace` vs `find` = UB), fixed `thread_local` in Exp669 after a `VAE_FAULT_TRACE=1` dladdr handler resolved the fault PC to `std::__ndk1::__tree_balance_after_insert`. Escape hatch `VAE_DW_CONV1D_OFF=1` |
| blocked-int8 conv weights (Exp685 built, Exp688-690 fixed and shipped) | the 14 non-depthwise VAE convs move from F16 dots to the Q4_0_4_4 blocked kernels: **paired 3+3 reps 2.541 → 2.464 (−3.0 %)**, every int8 rep below every ref rep; VAE 17.2 → 16.5 s; **RSS 2239 → 2116 MB**; load 2.5 → 1.4 s (132 MB smaller); 69 s 2.4316, 138 s 2.4958, lean p13 2.5501 @ 1752 MB. **Gate paired vs the frozen reference: WER 4.38 % both, 2-vs-1 discordant tokens, McNemar p=1.0** (output-equivalent). Conv dots were 12.3 % of VAE seconds (Exp683) and the VAE is kernel-RATE- not traffic-bound (Exp681), which is why shrinking the weights is not the point - changing the dot rate is. Three bugs had to die first: (1) `ggml_quantize_chunk(Q4_0_4_4)` on **x86 writes valid nibbles with ZERO scales** → quantize on device (`conv_int8.py --device`), refuse zero-scale files; (2) `encoder.output_dim` read `head_conv_weight->ne[2]`, which for a 2-D blocked weight is the implicit **1** → latent dim collapsed silently → `n_dims`-aware; (3) this fork computes a **pre-converted Q8_0 src1** for blocked mul_mat INCORRECTLY and silently (asserts off): 1024 garbage tokens vs the reference 39 → feed F32 and let ggml convert, the LM's route (`VAE_CONV_I8_Q8CAST=1` reproduces the wrong, 2 % faster arm). Verify any rebuilt file with `VAE_CONV_I8_CMP=1` (dequantizes through the runtime's own reader; control included) |
| layer-scale + residual fused (Exp664 knob, shipped Exp671) | the f32 block epilogue is ONE `GGML_OP_ADD_SCALED` (`x*scale + residual`) instead of `mul` + `add`. Re-priced on the post-conv-kernel graph, where it is no longer neutral (Exp664 measured it while the tap chain dominated — a stale-profile verdict, the third time that error appeared): **3 reps 2.5714 → 2.5429 (−1.1 %)**, every fused rep below every baseline rep; **gate 2.8806 → 2.842** with **40/40 byte-identical** transcripts (WER 4.41 %, S=28 D=1 I=3); 69 s 2.5199. Escape hatch `VAE_LS_FUSE_OFF=1` |
| fc1 bias fused into gelu (Exp672 attempt, shipped Exp673) | the ffn block's `mul_mat -> add(bias) -> gelu` became `mul_mat -> gelu(x+b)`: one node instead of two, removing a write+read pass over the ffn tensor (bias adds were 5.9 % of VAE). **3 reps 2.5375 → 2.5169 (−0.8 %)**, VAE 17.3 → 17.1 s, gate 2.842 → 2.8197 with **40/40 byte-identical** transcripts (WER 4.41 %), 69 s 2.4974, lean p13 2.6155. Bit-identity is structural: the fused kernel performs the same f32 add and then the UNCHANGED `ggml_vec_gelu_f32` in place (this build's gelu is the `GGML_GELU_FP16` table lookup, so reusing the routine is what preserves the values). The winning detail vs Exp672: the fused form is built by calling `ggml_nn_linear(…, apply_bias=false)` — i.e. the shipped matmul/reshape body — rather than re-deriving it at the call site. `VAE_GELU_BIAS_OFF=1` restores add+gelu; `VAE_GELU_QUICK` keeps its own path |
| one-pass fused gelu+bias (Exp780 pricing → Exp781 shipped) | The Exp673 fusion still looped the FFN row **twice** - an f32 add into dst, then `ggml_vec_gelu_f32` in place - so the row was touched **four times where twice suffice**. Folding the 16-bit gelu table lookup into the bias loop halves it. Priced first by ablation (`VAE_ABL_GELU`): gelu cost **13.3 % of VAE seconds** (2.1 s of 15.8), the largest non-matmul item since the v3.6-v3.8 fusions and 4x the next elementwise item. **Paired 3+3 reps 2.3855/2.3797/2.3809 → 2.3194/2.2995/2.3298 (−2.8 %)**, VAE 15.87 → 15.0 s (−5.3 %), 40-utt mean 2.6641 → **2.5742** with **40/40 byte-identical** transcripts (b=0/c=0 of 731, McNemar p=1.0), lean p13 2.54 → 2.44, RSS unchanged. Bit-identical by construction: same f32 add, then the same per-element gelu body (table + the two boundary branches) without the round trip through memory. **Negative control:** batching four table gathers in flight is *parity* (vae_s 15.9 both), which is what proved the cost is memory traffic and not the 128 KB table's gather latency - the ledger's mechanism guess was wrong even though its rate math (233 Melem/s/core) was right. Escape hatch `GGML_GELU_BATCH_OFF=1` (restores both the two-pass form and the scalar loop) |
| fused rms_norm-gamma (Exp784) | The f32 path ran `ggml_rms_norm` and then a **separate** `ggml_mul` for the gain - 4 tensor passes where 2 suffice. `ggml_compute_forward_rms_norm_f32` already read `src[1]` as an optional gamma and only a **builder** that set it was missing, so the change is 4 lines in `ggml.c` plus the call site (`ggml_rms_norm_gamma`). Paired 3+3 reps 2.2935/2.2938/2.2992 -> **2.2651/2.2652/2.2638 (-1.3 %)**, VAE 15.0 -> 14.7 s, **transcript byte-identical in all six reps** (same-build A/B); 40-utt mean 2.5742 -> **2.5415** with 40/40 byte-identity (b=0/c=0 of 731). Hatch `VAE_NORM_FUSE_OFF=1`. Note: with the fold active `VAE_ABL_SCALE` removes nothing (the mul node is gone), so its 0.30 s now lives inside `VAE_ABL_NORM`'s number and the two must not be added |
| two-column blocked-int8 GEMV tail kernel (Exp763 census → 764 → 765, shipped Exp766) | the blocked-int8 mul_mat ran gemm over `ne11−ne11%4` columns then ONE gemv call per leftover column, each re-streaming the whole weight tensor (92 ms/column = one decode token's cost). Census (`GGML_MM_DEBUG_SHAPES=1`): LM window prefills ne11=26 (2 cols), the LM initial pass ne11=31 (3 cols), VAE FFN ne11=26 — total tail waste 1237 ms = 5.2 % of wall. `ggml_gemv_q4_0_4x4_q8_0_m2` computes 2 columns per single weight pass by replaying the identical sdot ladder (scale product moved v16→v13 so the derived nibbles survive; all `.inst` encodings reused verbatim) → **byte-identical by construction**: protocol 2.396 → 2.376 (−0.8 %), prefill −143 ms, gate **40/40 byte-identical, b=0/c=0 of 731 tokens** (tag `gatem2`, mean 2.66), lean −1.0 % same-session (2.54 vs m2-OFF 2.57). Below the 2 % stop rule but shippable under the byte-identity rule (q8head precedent). Ceiling autopsy: the m2 pass costs ~146 ms/window vs 91 for one gemv pass (1.6×) — the doubled sdot ladder is issue-bound, so 2× work per pass yields only 1.25×; padding the columns was parity (Exp763b). `GGML_MM_M2_OFF=1` restores per-column calls (+1.0 %, identical transcript). **No further tail work exists (Exp774):** a 4-column-panel kernel requires `ceil(ne11/4)` weight passes, so the tail pass is one of the 7 mandatory passes for ne11 = 26 rather than an extra one — fusing it into the gemm could recover only the 2 unused column slots (ALU), and the measured cost of skipping the whole tail (0.2 s of the ~1.13 s a pass-bound model predicts, ratio 0.18) shows the VAE is not pass-bound. `Q4_0_4x8` does not help either: its gemm also uses `ncols_interleaved = 4` |
| rollback-path audit, v4.4 (Exp677 original, re-run Exp787) | All SIX escape hatches exercised in **one binary**, individually and combined (2 reps each, interleaved): vs the shipped default 2.2856 - `VAE_DW_CONV1D_OFF` **+5.0 %**, `VAE_GELU_BIAS_OFF` **+4.3 %**, `GGML_GELU_BATCH_OFF` **+2.9 %**, `VAE_LS_FUSE_OFF` **+0.6 %**, `VAE_NORM_FUSE_OFF` **+0.6 %**, `GGML_MM_M2_OFF` **+0.1 %**, ALL SIX OFF **+13.8 %** (2.6013). **Every arm produced the same transcript hash (25b53ca2fc20) and 39 tokens**, including all-off, so the whole fusion+kernel stack is output-equivalent to the pre-change path measured *within one build* - the strongest form of the claim, with no thermal or provenance assumption. Two readings changed since Exp710: the gelu pair is now the second-largest rollback cost (it was one fusion, now it is two), and `GGML_MM_M2_OFF` reads +0.1 % here against +1.0 % at v4.2, i.e. its benefit has been overtaken by the fusions (kept because it is free and byte-identical). `VAE_DW_CT_OFF` remains **debug-only, not a rollback** - it also sets `dw_taps=false` and changes the transcript (Exp710) |
| CPU affinity / big.LITTLE topology (Exp659) | the loop pins with `taskset C0`, where **`C0` is a hex mask** = `0xC0` = cpu6-7, the two A78 primes (`/proc/<pid>/status` → `Cpus_allowed_list: 6-7`; capacity 1024 vs 273 on cpu0-5). Six A55 little cores have therefore been idle in all 658 runs — so this was measured: shipped `C0`/−t2 **2.768**, all 8 cpus −t8 2.934 (+6.0 %), all 8 −t4 3.132 (+13.1 %), little-cores-only 12.78 (**+362 %**, i.e. an A55 thread is 4.6× slower). Output and RSS were identical in every arm. Conclusion: the shipped pinning is optimal and the little cores are not usable capacity for a barrier-bound loop; `−t`/core-count experiments are closed |
| hallucination / runaway audit (Exp658, existing artifacts) | insertions = words never spoken, and WER hides them among substitutions. Rate per 1000 reference tokens: clean read speech **2.1**, held-out zh-TW **2.1**, bilingual probe 0–2.1, **held-out consumer English 50.5** (59.1 before correcting my own tokenizer's contraction splitting) — a ~25× rate rise. They are **diffuse** (8 of 24 turns, longest spurious run 3 words) and there is **no runaway repetition in any set** (max repeated 8-gram ×1). Composition is 46 % function words / 54 % content words (`back`, `seat`, `stone`, `support`) = LM-prior dominance under acoustic uncertainty. Tier-independent (shipped 13 / accuracy-first 12 / lean 13), so precision is not the lever. Product line: no hallucination loops anywhere; the one hot spot is consumer-quality English at ~5 % of emitted words |
| the gate's **resolution limit** (Exp657) | paired token distances between archived configurations and the shipped 40-utt gate output (731 tokens): ctblock-era **0**, lean p13 **1**, gelu_quick **3** (1/2), defer5 **2** (2/0), dw-taps **2** (2/0), f16-VAE **6** (3/3) — every one McNemar p ≥ 0.5. So a 0.1–0.3 pp WER difference *is* one or two words in 40 utterances and carries no evidence either way. Consequences: state output-equivalence as “N tokens differ (b/c), p=…”, keep WER for absolute accuracy only, and prefer the transcript/byte-identity check (which this loop has always used) over WER parity when judging accuracy-neutral changes. Caveat: sets predating the Exp617 `run-info.log` stamp may include code drift, so those numbers are distances between outputs, not causal effects of one flag |
| scorer integrity (Exp654) | two bugs found by auditing the substitution list rather than the headline number: (a) `score_stream` skipped `fold_script`, inflating all zh WERs ~2.6×; (b) the speaker-mapping fast path for >8 voices ignored the one-to-one constraint, so diarization error was understated (zh held-out 0.233→**0.667**, en held-out 0.184→**0.507**, en control 0.215→**0.734**; ≤8-voice sets used the exact search and are unchanged). Self-tests now run across all three manifest shapes (script-varying, many-speaker, bilingual) — 6/6 each, selected via `GATE_MANIFEST` |
| held-out generalization (Exp651, **corpus bound**) | Common Voice 17.0 `en` test split (CC0, 24 clips / 24 voices, never used by this loop): WER **29.6 %** (220 tokens; jiwer 30.7 %) vs **4.41 %** on the LibriSpeech gate. Control: LibriSpeech clips in the *same* concatenated 23-voice protocol score **5.49 %**, so the protocol costs ~1 pp and the remaining ~24 pp is **domain** (consumer mics, accents, conditions). Quoted WER must therefore carry the qualifier "read speech (LibriSpeech test-clean)"; on in-the-wild English the operating point is ~30 %. zh-TW held-out added in Exp653/654 (`holdout_zh.wav`, sha `682021e58034`, 48 clips / 48 voices, 468 tokens): **16.0 %** — far better than English, because most of the apparent error there was script handling, not acoustics. Assets: `eval-bilingual/holdout_en.wav` (sha `8f29749cd35b`), `control_ls.wav` (sha `290b65b4d8bf`), `holdout_zh.wav` (sha `682021e58034`). **Watchdog re-run at v3.9 after the conv-int8 weight change (Exp695): no regression** — English 29.55 % → **26.36 %**, zh-TW 16.03 % → **14.74 %**, both paired against the archived transcripts of the same tier (`.auto/hyp-holdout_en-v39.txt`, `hyp-holdout_zh-v39.txt`): net 5 tokens better in each domain, McNemar p=0.18 / 0.23. Read that as parity, not as a gain — there is no mechanism by which 4-bit conv weights improve recognition, and 688 paired tokens only give 0.42 power. Diarization unchanged (en attribution 0.495, zh 0.668) |
| non-speech edge cases | 5 s digital silence → `[Silence][Silence]`; 5 s −50 dBFS white noise → `[Noise]` — correct model tags, short decodes, **no hallucinated text and no repetition loops** |
| input formats / cold start | 48 kHz stereo handled (one word differs); after evicting the page cache the RTF is unchanged (3.4713 pre-change, 3.3010 on the deferred build) and only the load grows (1.4 → 2.6 s, excluded from RTF) |
| CPU utilisation | 1.88 of 2 pinned cores (94 %) — the pipeline is saturated |
| hardware envelope (Exp544) | the two pinned A78 primes are **hard-capped at 1.3 GHz** (54 % of their 2.4 GHz rating) regardless of load — all numbers are the device's sustained, power-capped behaviour. At that clock the LM prefill runs at ~90 % of the achievable int8 rate; the VAE's FFN at ~15 % (shape-limited: L=50 columns in the deep stages, short contractions in the early ones) |
| reproducibility | artifacts bit-exact; the documented recipe (`.auto/setup.sh` from a wiped `build-android/`) reproduces the shipped binaries **byte-for-byte** - re-verified on the v3.5 build (Exp610): libggml `6ce4c983`, libllama `92ad2456`. The *executable* hash moves with any `src/`/`demo/` commit: it read `4bb34abc` at Exp610 but is `84efcee2` now, because the Exp639 `vae.cpp` guard fix landed after the table row was written (Exp649). RULE: refresh the executable hash in the commit that changes `src/`/`demo/`, and verify host-vs-device copies match rather than trusting this table |

## What moved the needle (five waves, −77 %)

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
   run concurrently one thread each (intra-chain splitting scaled only 1.57×; Exp678 bounds that
   at ≤1.85× and shows the choice is not a throughput lever - 2 chains at 1 thread each = 17.0 s
   each vs one chain alone on both cores = 9.2 s, i.e. concurrent and sequential do the same work
   rate within 5 %).
   Plus a mmap loader with a parallel copy (load 5.4 → 1.4 s).

## Closed avenues (each with a mechanism or measurement)

- **Precision / quant formats**: imatrix is explicitly ignored by the blocked-type
  quantizer (`UNUSED(quant_weights)`); `Q4_0_4_8`/`8_8` kernels need i8mm (absent
  → scalar fallback); 2-bit types (TQ/I2_S) would collapse a non-ternary model.
- **Concurrency**: 2 threads per chain = VAE +44 % worse; two concurrent inference
  streams each take exactly 2× the solo time ⇒ no idle capacity, so any
  pipelining/overlap (window or phase level) is dead.
- **Granularity / threads**: 2 pieces optimal (Exp620 re-sweep on the shipped tier: p13 +3.2 % slower, p1 ties at +335 MB); LM
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
- **Codegen**: PGO re-tested properly (protocol-trained, push fixed) = 0 %; ThinLTO (`-flto=thin` on top of the shipped A78 flags, wiped build, libs verified pushed) = 0 % null (2.7820, in-band), so the invalid-basis revisit is closed; `-funroll-loops` is BYTE-IDENTICAL to the shipped binary on a wiped build (Release/-O3 already makes the same unroll decisions everywhere in this tree); the compiler axis is closed.
  ThinLTO's earlier null stands on a valid basis now; the compiler axis is closed.
- **Quality ceiling**: RTF < 1 on this phone class needs retraining (QAT INT8 VAE
  and/or a smaller encoder+LM), not more porting.

## Upstream issues (drafted from our numbers, not yet filed - neither author has upstream access)

Drafts below were written from this project's gate data with a research agent (DeerFlow) and
reviewed against the logs. File to llama.cpp / ggml when someone has an identity.

### B1: [ARM CPU] No gemv/gemm kernel for q6_K (output head): 26ms/tok fallback on A78-dotprod; request q6_K_4x4 tile or canonical fast head type

Hi, reporting a measured ARM-CPU gap with repro numbers, asking for either (a) a q6_K GEMV/GEMM
tile for aarch64-dotprod, or (b) maintainer guidance canonicalizing large-vocab heads to a type
that has one. Setup: 2x pinned Cortex-A78 @ measured 1.3GHz (Dimensity 1300, ARMv8.2+DOTPROD, no
i8mm), single-stream single-token decode, Qwen2.5-family LM n_embd=1536, vocab 151936 ->
output.weight 233.4M params. Sizes: q6_K 191MB (0.82 B/p), q8_0 248MB (1.06), q4_0_4x4 141MB (0.60),
q5_K 158MB (0.68). Kernels: ggml-cpu-aarch64.c has NEON/dotprod tiles for q4_0*/q8_0 only; q6_K
(and q5_K) fall back to scalar vec_dot_q6_K_q8_K. Measured: q6_K head 191MB@~7.3GB/s eff -> ~26ms
of 105ms/token (24.8%). q8_0 derived: 248MB@~13GB/s dotprod ~= ~19ms; end-to-end decode delta
measured 17ms/token with ~5ms run variance (decode 4.3->3.6s, -16%, WER-neutral). q4_0_4x4 ~15ms
via interleaved GEMV BUT gate 4.41->4.96% (+0.55pp) + zh proper-noun loss; q5_K -14% decode but
4.41->5.10% (+0.42pp). All four variants ran the same graph/protocol on 2xA78@1.3GHz with the
40-utt on-device LibriSpeech gate + bilingual zh canary as the decision pair. So no type today is
both fast and accuracy-neutral. Ask: q6_K_4x4 (or q8_0_4x4) GEMV tile, or docs blessing q8_0 as the
ARM head type. I can attach hashes/gate counts. Thanks!

### B4: [quantize/docs] imatrix + quant-weights silently no-op for Q4_0_4x4/4x8/8x8 family; request docs + CLI warning (+q4_K output-type footgun, unconfirmed)

Hi, requesting a docs + CLI guard, with evidence. Fact: quantize_q4_0_4x4() marks quant_weights
UNUSED and ggml_quantize_chunk never consults an imatrix for the 4x4/8x8 family (quants.c;
verified still true upstream). Evidence: built a 220-chunk imatrix from public-domain bilingual
text, applied during Q4_0_4x4 requant (llama-quantize --allow-requantize ... Q4_0_4_4): only
token_embd/plain-q4_0 tensors could be weighted and both reported 'did not find weights'; 40-utt
gate + protocol RTF bit-identical with/without imatrix. Ask (durable part): one-line docs note the
blocked 4x4/8x8 family ignores imatrix/quant-weights (only K-quants/IQ consume them). Nice-to-have:
CLI warn when --imatrix meets a 4x4/8x8 target. Companion, please-confirm: --output-tensor-type
q4_K wrote a broken ~198MB partial (of a ~1GB model) in my fork - please confirm on current HEAD
before merging that half; if confirmed, the same note should list supported output types loudly.
Hashes/gate counts available. Thanks!

## Provenance

| artifact | md5 |
|---|---|
| `vae-encoder-q4x4ffn.gguf` | `b909b7901d5d318d81ce4cbaeac31437` |
| `lm-q4_0_4_4.gguf` (q6_K embeddings, predecessor) | `db67eecbd31bba707666414dd977902f` |
| `lm-q8head.gguf` (**shipped**: Q4_0_4_4 bulk, q6_K embeddings, **Q8_0 head**) | `222d4bf7794c4b030a82751a1b3226f5` |
| `streaming-lm-q4_k_m.gguf` (intermediate) | `046be3d4775e10f8b635b03ec1bc79cb` |
| `lm-4x4-head.gguf` (OPTIONAL faster variant, q8_0 head -> q4_0_4x4: RTF -4.1 %, WER 4.96 % — declined as default) | `62854dfffbd24fdca7a2aa4717df24eb` |
| Android `asr_streaming` (A78 build, OMP off, deferred late stages + lifetime activation buffers + zero-copy weights + [C,T] blocks, Q8_0 head; incl. the Exp639 dw-guard) | `84efcee2f5475ce4c5a462ea82500fc1` |
| `libggml.so` / `libllama.so` (shipped) | `6ce4c983ab2b310fb8dce8e75e402f7e` / `92ad2456979d99e2a1afee4a8cebad1d` |

Full engineering log in `.auto/log.jsonl`; per-wave detail and
the implementation notes in `STREAMING_1P5B.md`; loop protocol in
`.auto/prompt.md`; parked work with recipes in `.auto/ideas.md`.

_‡ Cells in italics or marked ‡ predate the Exp664 fused-tap build and are expected ≈−3 % faster; only the protocol and gate-mean cells of the two shipped tiers have been re-measured so far._
