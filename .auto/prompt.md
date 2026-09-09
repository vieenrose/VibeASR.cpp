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
- Thermal noise is real on phones: distrust single-run deltas < ~5%; re-run
  suspicious improvements. Confidence score < 1.0x = noise.

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
- Exp8 (KEEP, new loop protocol): `--xwin` cross-window carry + immediate emission.
  Desktop 69s VAE 41.3 vs 49.6s (-17%, theory-matched); phone 10s RTF 12.03
  (edge-heavy short file, within noise of legacy 12.24); 40-utt WER 3.99% vs
  4.13% cold (proxy predicted 3.72%). Fixed own double-emission bug + cold-tail
  rule along the way. Loop protocol switched to --xwin; re-baselining below.
- Exp12 (KEEP, infra): swap-tax probe (VmHWM + majflt telemetry). majflt=0:
  NOT swap-bound, gap is pure compute. Q8+xwin re-run 9.01, reproducible.
- Exp13 (killed at design): FP16 VAE compute. mul_mat hardcodes F32 output;
  only precisions are DEFAULT/F32. Per-GEMM casts would cost MORE traffic
  (2.5A+W vs 2A+W) than F16 saves. Dead without 3rdparty changes (off-limits).
- Exp14 (discard): proper ThinLTO (global IPO + clean rebuild; first attempt only
  set ggml-scope flag so link lacked -flto — invalid, redone correctly).
  RTF 8.94 vs 8.90 (noise). ggml.c already single-TU. Compiler avenue closed.
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
