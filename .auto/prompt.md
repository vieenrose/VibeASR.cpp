# Autoresearch: phone RTF for streaming-1.5B (CPU-only, accuracy-guarded)

## Objective
Minimize inference RTF of `./asr_streaming` (VibeVoice-ASR-Streaming-1.5B:
VAE-F16 + LM-Q4_K_M, `--vae-pieces 13`) on the connected OPPO phone
(Dimensity 1300, 8 GB RAM, Android 13, arm64) via `adb`, CPU only.
Current baseline (this protocol): RTF ~12.4-13.3 on the 10 s slice.
Goal direction: as far below that as honest engineering goes (RTF < 1 is
believed unreachable without retraining; do NOT chase it by cheating).

## Metrics
- **Primary**: `rtf` (unitless, lower is better) — wall generation time / audio
  duration on the phone, 10 s slice, `-t 4`, big-core pinned, pieces=13.
- **Secondary**: `vae_s`, `lm_s` (phase split — attack the bigger), `peak_rss_mb`
  (must stay phone-safe; alarm if > 3500), `tokens` (guard band), `load_s`.

## How to Run
`./.auto/measure.sh` — builds Android target, pushes, runs phone bench,
emits `METRIC name=value` lines. Runtime ~3-6 min (dominated by the phone run).
`./.auto/checks.sh` — post-run transcript sanity (auto-runs on passing runs).

## Files in Scope
- `demo/asr_streaming.cpp` — chunk loop, threading, batch sizes, piece control.
- `src/vae.cpp`, `src/vae.h` — VAE encode path (70-80% of time): arena mgmt,
  graph build, cache, batching. KEEP behavior bit-parity unless deliberately
  trading precision (then WER-gate it).
- `CMakeLists.txt`, `src/CMakeLists.txt` — compiler flags for the Android target
  (must stay portable arm64-v8a baseline unless guarded + justified).
- `utils/convert_*.py` — quantization changes are allowed but challenger must
  re-validate WER on the 40-utt set (slow; prefer runtime ideas first).

## Off Limits
- `3rdparty/llama.cpp` (forked upstream — do not touch; if a ggml change is the
  only way, write it up in ideas.md instead).
- Model weights themselves (`models-streaming/` lives outside this repo).
- Anything that changes the task: fewer windows, truncated decode
  (`--max-tokens` fixed 256), different clips, hotwords tuned to the clip,
  benchmark-specific shapes (pieces must divide 26 — general by construction).
- Phone-unfriendly requirements (CUDA/Mali/NPU backends, root-only tricks).

## Constraints
- CPU-only inference on the Oppo via adb. No new dependencies.
- Correctness: `.auto/checks.sh` must pass (non-degenerate transcript, token
  band). Full re-validation (17 s + 69 s + 40-utt LibriSpeech WER) is required
  before any result is trusted for the final report — loop keeps are provisional.
- Do NOT overfit to the 10 s slice: ideas must be justified by mechanism
  (profiling data in secondary metrics), not by the clip's content. When in
  doubt, cross-check on the 17 s clip before keeping.
- THERMAL DISCIPLINE (learned Exp15/16): back-to-back runs drift up to +60%
  (heat soak + platform throttling; NOT code). Protocol: 5-min idle cooldown
  before any keep-decision run; bracket keeps as A/B/A (candidate between two
  baseline-config runs in one session); distrust single-run deltas < ~10%.
  Confidence score < 1.0x = noise. Majflt telemetry distinguishes swap pain
  (has been 0 throughout).

## What's Been Tried
- Exp4 (KEEP, recommended tier): VAE selective Q8_0-mixed (large weights only,
  convs stay F16). Loop RTF 10.34 (-15%), RSS 2.71GB, 40-utt WER 4.41% (+0.28pp),
  69s WER 3.67%. Phone Q8 FASTER though desktop Q8 slower (ARM dotprod).
- Exp5 (discard): LM Q3_K_M (6.20%, +2.1pp) / Q2_K (7.58%, +3.5pp) full-40.
  Accuracy cost unacceptable; LM stays Q4_K_M. Revisit only imatrix-guided.
- Exp6 (KEEP, min-size tier): VAE Q4_0-FFN (104 tensors). Loop RTF 10.12 (-17%),
  RSS 2.42GB, 40-utt WER 5.23% (+1.1pp), 69s WER 3.67%. Opt-in; Q8 stays default.
- Exp7 (KEEP): default `n_ctx` 16384->4096. RSS -336MB (2982 vs 3318MB),
  RTF equal. Caveat: >15min files need `-c 16384`. Stacks with VAE-Q8.
- Probe (groundwork for Exp8): upstream `encode_then_split` (whole-file encode =
  full-context features, proxy for cross-window carry) on 40-utt: WER 3.72%
  vs 4.82% cold windows. Full context is BETTER and removes 1.18x overlap
  recompute. Exp8 implements this via never-reset cache + delayed emission.
- Exp8 (DEMOTED to opt-in --xwin flag): cross-window carry is FAITHFUL (matches
  upstream full-context incl. its weakness) but full-context DEGRADES long files:
  69s WER 12% both precisions (S~30 spread, elaborative drift via KV history),
  while cold windows hold 3.7%. Upstream PyTorch fullctx on 69s also 12.67% -
  inherent, not a port bug. Shorts: xwin 3.99% vs cold 4.13% (fine).
  Bounded-reset hybrid (reset every 8 hops) was WORSE (14%, discontinuity
  deletions). Loop protocol REVERTED to legacy cold windows; --xwin kept as
  flag for short-file use. Lesson: shorts-only WER gates miss length effects;
  69s gate mandatory for protocol changes.
- Exp15/16 (discarded, fully reverted): split thread pools (VAE 4 big + LM 6 all)
  + in-code phase affinity. LM 23.7->17.7 real, but 6-thread llama pool
  spins/steals during VAE phases: VAE 65->91s. Net negative. Recovery run after
  revert: VAE 74 and falling. Revisit only with idle-sleeping pools.
- Exp12 (KEEP, infra): swap-tax probe (VmHWM + majflt telemetry). majflt=0:
  NOT swap-bound, gap is pure compute. Q8+xwin re-run 9.01, reproducible.
- Exp13 (killed at design): FP16 VAE compute. mul_mat hardcodes F32 output;
  only precisions are DEFAULT/F32. Per-GEMM casts would cost MORE traffic
  (2.5A+W vs 2A+W) than F16 saves. Dead without 3rdparty changes (off-limits).
- Exp14 (discard): proper ThinLTO (global IPO + clean rebuild; first attempt only
  set ggml-scope flag so link lacked -flto — invalid, redone correctly).
  RTF 8.94 vs 8.90 (noise). ggml.c already single-TU. Compiler avenue closed.
- Exp17/log#16 (KEEP, honest best tier): Q8 + legacy-cold on loop protocol.
  RTF 9.71 (VAE 73.5), RSS 2.44GB. Fully validated: 40-utt 4.41%, 69s 3.67%.
  xwin numbers (8.90) stand for shorts only; xwin demoted to opt-in flag.
- Exp18 (discard): kernel-trim (true-size conv weights, no zero taps). Parity held
  (max 4e-3) but 15% SLOWER on desktop (power-of-2 kernels beat fewer MACs).
  Reverted converter to always-pad; kept dynamic head_kernel_size (robust to
  both variants). Lesson: MACs != speed; authors padded for a reason.
- Exp19 (discard): fused residual mul+add via custom map_custom3 op (strided-safe,
  broadcast-mirrored). Bit-exact frames (0.00e+00) but 7-9% SLOWER both hosts:
  scalar loop loses to ggml SIMD vec kernels. Fully reverted. Lesson: fusion
  needs NEON+AVX2 intrinsics to even tie; parked with fused-blocks project.
- Exp20 (discard): own-VAE I8_S PTQ (was BitNet's foreign weights the cause?).
  Collapsed identically (7 vs 44 tokens). Converter verified correct (F32
  intermediate). Cause = per-tensor global scale precision. Dead twice over.
- Exp21 (killed pre-implementation): direct depthwise-conv kernel (avoid Kx im2col
  expansion, zero channel reuse). Killed on semantic evidence: this fork's ggml
  breaks documented semantics (concat scrambles, cont verbatim-copies, dw output
  transposed — all unit-proven), so custom ops are quicksand; unused helper
  removed with zero residue. Parity-validated paths only from here on.
- Exp22 (KEEP, tier re-validation): Q8+legacy on current tree after all churn
  (LTO round-trip, affinity add/revert, dynamic head): 40-utt 4.41%, 69s 3.67%,
  phone 9.87 — all identical to pre-churn. Shippable tier confirmed current.
- Exp23 (discard): single-pool -t6 pinned (4big+2little) to capture LM scaling
  without split-pool contention. LM -29% real, but VAE +8% little-core
  stragglers cancel it exactly (9.68 vs 9.71, noise). Default stays -t4/F0.
  Lesson: thread scaling is zero-sum here.
- Exp24 (discard): OMP passive-wait (OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0).
  VAE -3% but LM +28% wake latency on single-token decodes; net +5.6% worse.
  Spinning is correct here. No code changes.
- Exp25 (discard): explicit OMP placement (CLOSE/cores, no-spin). +80% worse
  (119s VAE!) — fights taskset. Threading space fully mapped: -t4/F0 default
  stands; -t6/8, split pools, OMP all neutral-or-worse. Do not revisit.
- Exp26 (discard): PGO train/use cycle (fixed shapes => no content-overfit risk).
  10.04 vs 9.81 A/B back-to-back (noise against). No gain; profile-skew and/or
  nothing to layout. Session creep 8.90->9.8 across the day => heat soak is now
  quantified: ALL keep decisions require A/B/A bracketing in one session.
- Exp27 (KEEP, anchor correction): cooled re-anchor of Q8+xwin = 9.75. Full-day
  band for identical config: 8.90 (cool morning) .. 9.81 (hot) .. 9.75 (cooled).
  Report the tier as a BAND (~9-10), never a point. Morning singles were lucky.
- Exp2 (killed pre-implementation): persistent VAE graph to skip 104 rebuilds/run.
  Measured graph build at 0.2-1ms (<1% of VAE time) via temp VAE_PROFILE
  instrumentation (since reverted). Would have saved ~100ms of 12,000ms.
- Exp1 (discarded): `-mcpu=cortex-a78` + `GGML_ARM_DOTPROD` in CMakeLists —
  RTF 12.33 vs 12.24 (+0.7%, noise). Compiler tuning exhausted. NOTE: tool
  auto-revert/auto-commit cannot reach this nested repo (operates in parent
  cwd) — ALL reverts and keep-commits must be done manually in VibeASR.cpp.
- ggml CPU-backend executor vs legacy reference path: 1.06x desktop / 0.98x
  phone, bit-exact. REMOVED (dead code). Do not revisit without new evidence.
- `GGML_ARM_DOTPROD=ON` rebuild: 0% (LM decode is bandwidth-bound). Kept out of
  defaults; revisit only with dotprod-friendly (INT8) compute to feed it.
- big-core pinning (`taskset F0`): +7%, baked into measure.sh protocol.
- `-t 8`: +7% over `-t 4` unpinned; redundant with pinning. Try `-t 6/8` pinned
  (uses 2 little cores) — likely noise-level, low priority.
- VAE pieces 26 vs 13: -0.1 GB RSS but +17% time. 13 is the sweet spot.
- VAE I8_S PTQ: collapses to repetition loops (needs QAT). Dead end.
- LM Q3_K_M (+2.1pp WER), Q2_K (+3.5pp): rejected on accuracy. LM stays Q4_K_M.
- Baseline numbers (phone, -t 4, pinned, pieces 13): 10 s RTF ~12.4-13.3
  (VAE ~109 s of it), RSS ~3.1-3.3 GB; 17 s RTF 11.6; 69 s RTF 12.5.
  Accuracy anchor: 40-utt WER 4.13% (official PyTorch 4.82%), parity 2.2%.
