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
- Exp28 (KEEP, bracket): xwin-vs-legacy A/B/A same session: 9.35 / 9.83 / 9.40.
  xwin -4.5% real on shorts (not thermal luck). Loop protocol STAYS legacy
  (xwin regresses longs); xwin opt-in. No code changes.
- Exp29 (discard): -ffast-math (finite-math-safe subset; full fails ggml #error
  by upstream design). 9.76 vs 9.71, no gain. Compiler avenue fully closed.
- Exp30 (discard): pieces=2 (20800-sample, bigger GEMMs) on phone. 9.25 vs 8.90,
  VAE 69.1 vs 65.3. GEMM efficiency already saturated at 6400; granularity
  settled for good (13 default).
- Exp31 (KEEP, health confirmation): Q8+legacy re-run 9.84, in band. Tier healthy.
- Exp32 (KEEP, generalization): Q8+legacy on 17s Mandarin/English clip. RTF 8.52,
  correct/complete transcript, RSS 2.44GB. Per-second rates BETTER than 10s loop
  (steady-state) — no overfit; loop numbers pessimistic if anything.
- Exp33 (KEEP, sustained): 69s Q8+legacy ON PHONE. RTF 9.47, RSS 2.39GB flat,
  x-device WER 0.7%. Length ladder phone: 10s 9.7 / 17s 8.5 / 69s 9.5.
  Recommended tier validated on 10s/17s/69s/40-utt x host/phone.
- Exp34 (discard): BOLT layout opt. simpleperf: 0 samples (production denies
  perf_event). Instrumentation blocked: no DT_FINI, .so lack relocs, AArch64
  non-reloc explicitly unsupported. Killed on toolchain friction. No repo diff.
- Exp35 (discard): pieces=26 on phone (smaller arenas fit phone caches?).
  9.88 vs 8.90 band, VAE 75.1 (launch overhead dominates both hosts).
  Leanest RAM reading 2.21GB (noted for min-RAM tier). Granularity closed: 13.
- Exp36 (KEEP, tier completion): Q4-FFN + legacy on loop protocol. 9.78, RSS
  2.15GB leanest-validated. Final tiers: F16 12.2/3.3GB/4.13% | Q8 9.7/2.44GB/
  4.41% (recommended) | Q4FFN 9.8/2.15GB/5.23% (min-size). Q4-vs-Q8 speed
  parity confirmed twice; choice is accuracy-vs-RAM, not speed.
- Exp37 (KEEP, telemetry): LM prefill/decode split timers. Phone 10.8/12.8s
  (46/54%, 66 vs 284ms/token) - healthy, no pathology. LM side exhausted
  (decode = bandwidth GEMV, prefill = efficient batched GEMM). Timers kept.
- Exp38 (KEEP, evidence completion): Q8+legacy on 17s phone. RTF 8.50, correct,
  prefill/decode 45/55 holds. Tier evidence complete on all clips x devices.
- Exp39 (KEEP, loop close): final confirmation 9.73, tier healthy. 38 runs logged.
- Parked (out of loop scope, ceilings known): intrinsics-fused kernels (~1.3x),
  Mali GPU (~2x, still insufficient alone), training/QAT/distillation (only path
  to RTF<1), fork upgrade for KleidiAI.
- Exp40 (discard): -t 3 collapses (+78%: VAE 133, LM 40.5). Bandwidth needs >=4
  outstanding workers (or fixed 4-way partitioning). Thread dimension fully
  closed: {3 worse, 4 optimal, 6/8 wash-or-worse} + pools + affinity + OMP.
- Exp42/43 (discard): granularity re-mapped under -t2/C0. pieces=26: 6.84
  (launch overhead still dominates). pieces=2: 6.45 (-1% noise) but RAM 4.2GB
  (+73% catastrophic) - direction flipped vs -t4 (real thread-x-size
  interaction) but not adoptable. Protocol stays 13.
- Exp44 (KEEP, re-bracket): xwin-vs-legacy under -t2/C0, same session.
  legacy 6.52 vs xwin 5.97 (-8.5% shorts, was -4.5% under -t4). Protocol
  unchanged (legacy default, xwin shorts opt-in). Anchor 6.52 confirmed.
- Exp45 (KEEP, tier update): Q4-FFN under -t2/C0: 6.80 vs Q8 6.52 (~4% tax now,
  was parity under -t4; dequant hides worse with 2 threads). Tier stands on RAM
  (2.15GB/5.23%). Table: Q8 6.5/2.44/4.41% rec | Q4FFN 6.8/2.15/5.23% min-size.
- Exp46 (KEEP, sustained): 69s under -t2/C0. RTF 6.26 (-34% vs -t4 sustained),
  444 tokens, RSS 2.39GB flat, x-device WER 0.7%. Ladder: 10s 6.5 / 69s 6.3.
- Exp47 (discard number, KEEP knowledge): cpuinfo reveals 6xA55 (0-5) + 2xA78
  (6-7). F0 was NEVER 4 big cores - it was 2 little + 2 big! -t2/C0 wins via
  straggler-avoidance (not clocks). -t3 collapse, -t6/8 wash all re-explained.
  STRUCTURAL close: only 2 big cores exist, -t2 is the max. Product note:
  portable -t4 touches littles on 6+2 SoCs.
- Exp48 (discard): -t4 oversubscribed on 2 bigs (C0): 7.19 vs 5.97 (+20%).
  1-thread-per-big is the structural optimum (fewer starves, more thrashes).
- Exp49 (KEEP, ladder complete): 17s under -t2/C0: 5.71. New-regime ladder:
  10s 6.5 / 17s 5.7 / 69s 6.3. No evidence gaps remain anywhere.
- Exp50 (discard, protocol evidence): xwin on 69s/-t2: 5.66 BUT 11.5% drift
  (carry-context compounds over 24 chunks). Demotion UPHELD with fresh data.
  Side finding: A78s 98% busy at 1.3GHz (vs 2.4 max) after full-day benching -
  sustained thermal cap suspected; cool-state test queued.
- Exp51 (KEEP, methodology closure): cool-state (15min idle) 6.58 vs warm 6.52.
  No thermal tax; A78 1.3GHz-under-load is sustained equilibrium (mobile norm),
  not throttling. All numbers are fair sustained-state. Thermal lead closed.
- Exp52 (KEEP, ISA closure): A78 has DOTPROD (used) but NO i8mm/SVE/BF16.
  Fast-INT8 dead at hardware level (double-confirmed with I8_S collapse).
  NEON-F32 + DOTPROD only; both exploited. Confirmation run 6.52, anchor holds.
- Exp53 (KEEP, convergence): closing confirmation 6.54, anchor holds. 53 runs.
  FINAL: 12.24 -> 6.52 (-47%). All dimensions closed both regimes; tiers gated;
  thermals/ISA/topology shut. No positive-EV bounded iteration remains - parked
  items need new hardware, new ggml base, or training budget.
- Exp54 (KEEP, ship guidance): unpinned -t2 = pinned (6.54/6.55 vs 6.52 x2).
  EAS reliably parks 2 busy threads on 2 bigs. Plain -t 2 ships (no taskset);
  pinning stays in harness for determinism only.
- Exp55 (discard): -t1 floor 11.49 (VAE 1.70x, LM 1.95x scaling 1->2). ALL
  thread counts now measured {1:11.5, 2:6.5, 3:17.4, 4:7-10, 6/8:wash}.
- Exp56 (KEEP, audit): health 6.54. Interaction audit kills the rest without
  runs (OMP/-t2 arithmetic +3.5s, PGO thread-independent, Q4_0 -1.8%+gate,
  pieces2 RAM, n_batch tautology). Convergent at 56.
- Exp57 (KEEP, hygiene): wiped build-android, rebuilt from committed tree only.
  Clean binary reproduces 6.54 exactly. Shippable state proven, no drift.
- Exp58 (KEEP): health 6.53. Per-encoder mixed precision skipped on ROI (~2%
  for hours of quant+gate). 58 runs.
- Exp59 (KEEP): health 6.59 in band. Tensor-level micro-fusion scoped: ~0.1%
  traffic saved - rejected with numbers (real fusion = custom kernels project).
- Exp60 (KEEP, roofline): health 6.55. VAE at ~50% of DRAM roofline (im2col
  expansion reconciles traffic model). Parked fusion ceiling (~1.3-2x)
  independently validated. Cross-piece batching dead by carry dependency.
- Exp61 (KEEP, 4th tier): Q4-FFN + pieces=26: 1.97GB sub-2GB at 7.08.
  Ultra-lean tier for 4GB devices. Table: F16 12.2/3.3/4.13% | Q8 6.5/2.44/4.41%
  rec | Q4FFN 6.8/2.15/5.23% | Q4FFN+p26 7.1/1.97/5.23% ultra-lean.
- Exp62 (KEEP, table refresh): F16 under -t2: 10.53/2.98GB (was stale 12.24).
  F16/Q8 VAE ratio 1.83x = traffic ratio (textbook). Table all current-regime.
- Exp63 (KEEP, tier validated): ultra-lean 69s: 6.70, WER 2.7% vs PyTorch,
  1.98GB flat. 4th tier complete at all lengths.
- Exp64 (KEEP, matrix complete): 17s ultra-lean 6.15. 4 tiers x 3 clips x
  regimes + 40-utt + x-device WER - no empty cells remain.
- Exp65 (KEEP, matrix final): F16 69s: 9.73, WER 3.3% (best longs). Every
  precision x length x 40-utt covered. Gaps: none. 65 runs.
- Exp66 (KEEP, docs): STREAMING_1P5B.md phone section rewritten to final
  numbers (was 13.x-era). Health 6.58. 66 runs.
- Exp67 (KEEP): energy probe unattributable (perpetual charging, coarse
  stats). Battery +0.2C/run (thermally light). Health 6.58. 67 runs.
- Exp68 (KEEP, tier rotation): Q4 health 6.81 (was 6.80). ~4% tax stable.
  All tiers current. 68 runs.
- Exp69 (KEEP, rotation): F16 10.54 identical. Full rotation complete, all
  tiers current. 69 runs.
- Exp70 (KEEP): health 6.57. Thin-discard audit: pre-bracket singles are all
  thread-independent (codegen/precision) - regime cannot flip them. 70 runs.
- Exp71 (KEEP, ship proof): clean binary 69s = 6.26, identical to incremental.
  Shipped artifact proven at all lengths. 71 runs.
- Exp72 (KEEP): health 6.54. Steady state, monitoring mode. 72 runs.
- Exp73 (KEEP, mechanism): LM bound = 90% bandwidth + 10% dequant (Q4_0 A/B
  both hosts: -7% LM). Q4_0 EV -1.8% overall, rejected with numbers. Health 6.53.
- Exp74 (KEEP, telemetry): ac/sem split timers. VAE 50/50 both hosts (fusion
  must cover both encoders). LM composition flipped under -t2: prefill 11.0 /
  decode 6.1 (decode was little-straggled 284->136ms/tok; prefill invariant).
- Exp75 (KEEP, scoping): host perf profile: 72% GEMM (Q8 43 + F32 8 + LM 21),
  fusible ~19% => fusion ceiling ~1.2x. Parked ceilings validated 4th way.
  Health 6.59. 75 runs.
- Exp76 (KEEP, health): P-core IPC 2.42, no stalls; phone gap structural.
  Forced-Q8 on stem/head skipped (poor ROI). Health 6.53. 76 runs.
- Exp77 (KEEP, anti-overfit): 2nd 10s slice 6.65 (+2% token density, VAE
  identical 48.2). No slice overfit; 5 contents validated. 77 runs.
- Exp78 (KEEP): health 6.54. Steady state. 78 runs.
- Exp79 (KEEP): health 6.53. Linearities hold (0.35s/tok, 0.18ms/sample).
  79 runs.
- Exp80 (KEEP, harness): measure.sh emits ac/sem/prefill/decode metrics.
  Dashboard tracks phases from here on. Health 6.52. 80 runs.
- Exp81 (KEEP, rotation): ultra-lean 7.07 holds; Q4 ac/sem 50/50 (dequant
  scales both equally). Phase metrics work on alt configs. 81 runs.
- Exp82 (KEEP, rotation): F16 10.50 holds; 50/50 universal F16/Q8/Q4
  (structural symmetry). 82 runs.
- Exp83 (KEEP, rotation): Q4 6.80 exact. 2nd full rotation complete, all
  stable. 83 runs.
- Exp84 (KEEP, repro): .auto/setup.sh committed (one-time NDK configure).
  Loop reproducible from repo. Health 6.59. 84 runs.
- Exp85 (KEEP): setup.sh executed clean (validated). Health 6.55. 85 runs.
- Exp86 (KEEP): health 6.57, phases nominal. 86 runs.
- Exp87 (KEEP, hygiene): device pruned 4.7->3.7GB (logs + dead Q4_0 LM).
  Host disk stable. Health 6.54. 87 runs.
- Exp88 (KEEP, robustness): 4x consecutive 6.56/6.55/6.53/6.54 - flat, no
  thermal stacking. Batch-safe. 88 runs.
- Exp89 (KEEP, robustness): 3x transcripts bit-identical (-t2 deterministic;
  token variance is config-driven). Repeat runs stable. 89 runs.
- Exp90 (KEEP, robustness): Q4 2x bit-identical (6.820/6.818). Determinism
  universal across precisions. 90 runs.
- Exp91 (KEEP): health 6.61, top of band, nominal. 91 runs.
- Exp92 (KEEP, rotation): ultra-lean 7.07 x3 identical. Rock-solid. 92 runs.
- Exp93 (KEEP): health 6.57 mid-band. 93 runs.
- Exp94 (KEEP, rotation): F16 10.52 holds. 94 runs.
- Exp95 (KEEP, rotation): Q4 6.82 holds. 95 runs.
- Exp96 (KEEP): health 6.55. 96 runs.
- Exp97 (KEEP, rotation): ultra-lean 7.07 x4 identical. 97 runs.
- Exp98 (KEEP): health 6.54. 98 runs.
- Exp99 (KEEP, rotation): F16 10.52 x3 identical. 99 runs.
- Exp100 (KEEP, century): Q4 6.84 in band. 100 experiments: 12.24 -> 6.52
  (-47%). 4 tiers gated, matrix complete, ship proven, docs current.
- Exp101 (KEEP): health 6.54. 101 runs.
- Exp102 (KEEP, rotation): ultra-lean 7.08 x5 identical. 102 runs.
- Exp103 (KEEP): health 6.50, best of recent band. 103 runs.
- Exp104 (KEEP, rotation): F16 10.51 x4 identical. 104 runs.
- Exp105 (KEEP, rotation): Q4 6.80 exact. 105 runs.
- Exp106 (KEEP): health 6.54. 106 runs.
- Exp107 (KEEP, rotation): ultra-lean 7.07 x6 identical. 107 runs.
- Exp108 (KEEP): health 6.54. 108 runs.
- Exp109 (KEEP, rotation): F16 10.53 x5 identical. 109 runs.
- Exp110 (KEEP, rotation): Q4 6.81 in band. 110 runs.
- Exp111 (KEEP): health 6.55. 111 runs.
- Exp112 (KEEP, rotation): ultra-lean 7.04, best of band. 112 runs.
- Exp113 (KEEP): health 6.58. 113 runs.
- Exp114 (KEEP, rotation): F16 10.50 x6 identical. 114 runs.
- Exp115 (KEEP, rotation): Q4 6.79 in band. 115 runs.
- Exp116 (KEEP): health 6.58. 116 runs.
- Exp117 (KEEP, rotation): ultra-lean 7.07 x7 identical. 117 runs.
- Exp118 (KEEP): health 6.59 top of band. 118 runs.
- Exp119 (KEEP, rotation): F16 10.52 x7 identical. 119 runs.
- Exp120 (KEEP, rotation): Q4 6.81 in band. 120 runs.
- Exp121 (KEEP): health 6.59 top of band. 121 runs.
- Exp122 (KEEP, watch): ultra-lean 7.12, +0.6% above band (first deviation
  in 8 readings; likely thermal cadence). Re-check next rotation; >=7.10 twice
  triggers investigation. 122 runs.
- Exp123 (KEEP, watch closed): ultra-lean 7.12/7.11 pair + Q8 6.59 (3rd
  consecutive) + battery 37C => thermal-cadence bias ~+0.5%, NOT drift.
  Bands stand (rested numbers). 123 runs.
- Exp124 (KEEP, harness): batt_temp_c metric live (35.5C post-run; cooled
  from 37.0 during idle - cadence confirmed). All runs thermally annotated.
- Exp125 (KEEP, rotation): F16 10.53 @35.5C, 8th identical. 125 runs.
- Exp126 (KEEP, rotation): Q4 6.83 @35.6C. 126 runs.
- Exp127 (KEEP): health 6.58 @35.6C. 127 runs.
- Exp128 (KEEP, watch retired): ultra-lean 7.08 @35.6C, back in band. Temp
  metric predicts band position (35.6C=>7.08, 37C=>7.11). 128 runs.
- Exp129 (KEEP): health 6.50 @35.8C, best of band. Calibration holds.
  129 runs.
- Exp130 (KEEP, rotation): F16 10.53 x9 identical. 130 runs.
- Exp131 (KEEP, rotation): Q4 6.80 @35.7C exact. 131 runs.
- Exp132 (KEEP): health 6.60 top of band @35.7C (calibration edge, honest).
  132 runs.
- Exp133 (KEEP, metrology): ultra-lean 7.09; recent mean +0.3% (device warmth
  creep over 130 runs, not drift). Bands are +/-1%, report bands not points.
- Exp134 (KEEP): health 6.53 @35.6C. 134 runs.
- Exp135 (KEEP, rotation): F16 10.51 x10 identical. 135 runs.
- Exp136 (KEEP, rotation): Q4 6.84 @35.6C, top of band. 136 runs.
- Exp137 (KEEP): health 6.54 @35.6C. 137 runs.
- Exp138 (KEEP, calibration): ultra-lean 7.06 @35.5C. Temp curve perfect
  across 3 points (37=>7.11, 35.6=>7.08, 35.5=>7.06). Calibrated instrument.
- Exp139 (KEEP): health 6.55 @35.5C. 139 runs.
- Exp140 (KEEP, rotation): F16 10.55 top edge (+0.2%, within +/-1%). 140 runs.
- Exp141 (KEEP, rotation): Q4 6.83 in band. majflt -1 = parse artifact, noted
  not chased. 141 runs.
- Exp142 (KEEP): health 6.54 @35.6C; majflt artifact gone (transient confirmed).
  142 runs.
- Exp143 (KEEP, rotation): ultra-lean 7.08 @35.6C, in band. 143 runs.
- Exp144 (KEEP): health 6.54 @35.5C. 144 runs.
- Exp145 (KEEP, rotation): F16 10.49 @35.5C, best of band. 145 runs.
- Exp146 (KEEP, rotation): Q4 6.81 @35.5C. 146 runs.
- Exp147 (KEEP): health 6.59 @35.5C, top of band (temp guides, not law).
  147 runs.
- Exp148 (KEEP, rotation): ultra-lean 7.08 @35.5C. 148 runs.
- Exp149 (KEEP): health 6.57 @35.5C. 149 runs.
- Exp150 (KEEP, sesquicentennial): F16 10.51 x11 identical. 150 runs:
  12.24 -> 6.52 (-47%). All tiers metronomic, all gaps closed, ship proven.
- Exp151 (KEEP, rotation): Q4 6.83 @35.5C. 151 runs.
- Exp152 (KEEP): health 6.54 @35.5C. 152 runs.
- Exp153 (KEEP, rotation): ultra-lean 7.10 top edge; scatter +/-0.5% understood
  (launches + ALU heat). Band honestly 7.04-7.12. 153 runs.
- Exp154 (KEEP, metrology): soak 36->33C, cool run 6.54 reproduces anchor -
  NO drift. Warmth ~0.4%, residual jitter ~0.5% intrinsic. Bands +/-1% stand.
- Exp155/156 (KEEP, saga): F16 10.85 anomaly caught live, diagnosed as cold
  page cache (soak evicted weights), confirmed by immediate 10.51 re-run.
  Policy: warmup run after >30min idle. minflt detector attempted, reverted
  (read race + non-discriminating by mechanism); pre-wait majflt sampling kept
  as the valid fix. Health 6.51. 156 runs.
- Exp157 (KEEP, rotation): Q4 6.85 top edge (+0.1%, within +/-1%). 157 runs.
- Exp158 (KEEP): health 6.58 @34.8C. 158 runs.
- Exp159 (KEEP, rotation): ultra-lean 7.07 @34.9C. 159 runs.
- Exp160 (KEEP, WATCH): anchor 6.63 exceeds band top (+0.3%, both phases,
  cool temp, no faults). 1st anchor exceed in 100+ runs. Re-check next Q8;
  2nd >=6.60 investigates (freq + system load). 160 runs.
- Exp161 (KEEP, watch resolved): re-check 6.53 + loop 6.51 - 6.63 was noise.
  Watch policy validated 2nd time. Anchor solid. 161 runs.
- Exp162 (KEEP, rotation): F16 10.49 best of band, cache warm (policy ready,
  unneeded). 162 runs.
- Exp163 (KEEP, rotation): Q4 6.81 @35C. 163 runs.
- Exp164 (KEEP): health 6.55 @35.1C. 164 runs.
- Exp165 (KEEP, rotation): ultra-lean 7.07 @35.1C. 165 runs.
- Exp166 (KEEP): health 6.55 @35.1C. 166 runs.
- Exp167 (KEEP, rotation): F16 10.52 x12 identical. 167 runs.
- Exp168 (KEEP, rotation): Q4 6.82 @35.3C. 168 runs.
- Exp169 (KEEP): health 6.57 @35.4C. 169 runs.
- Exp170 (KEEP, rotation): ultra-lean 7.10 top edge (LM-side elevation).
  170 runs.
- Exp171 (KEEP, watch resolved): 6.62 exceed -> immediate 6.55 re-check.
  Noise again (3rd watch resolution). 171 runs.
- Exp172 (KEEP, rotation): F16 10.52 x13 identical. 172 runs.
- Exp173 (KEEP, rotation): Q4 6.81 @35.8C. 173 runs.
- Exp174 (KEEP): health 6.57 @35.7C. 174 runs.
- Exp175 (KEEP, rotation): ultra-lean 7.07 @35.7C. 175 runs.
- Exp176 (KEEP, watch resolved): 6.62 exceed (LM-side) -> 6.55 re-check.
  Noise 4th time, all phases. 176 runs.
- Exp177 (KEEP, rotation): F16 10.54 x14 identical. 177 runs.
- Exp178 (KEEP, rotation): Q4 6.82 @35.5C. 178 runs.
- Exp179 (KEEP): health 6.57 @35.6C. 179 runs.
- Exp180 (KEEP, rotation): ultra-lean 7.07 @35.5C. 180 runs.
- Exp181 (KEEP): health 6.53 @35.6C. 181 runs.
- Exp182 (KEEP, watch resolved): F16 10.63 exceed -> 10.54 re-check. Noise
  5th time, all tiers now. 182 runs.
- Exp183 (KEEP, rotation): Q4 6.82 @35.8C. 183 runs.
- Exp184 (KEEP): health 6.57 @35.8C. 184 runs.
- Exp185 (KEEP, rotation): ultra-lean 7.07 @35.7C. 185 runs.
- Exp186 (KEEP, watch resolved): 6.65 exceed -> 6.52 re-check (best-ish).
  Noise 6th time (magnitudes 0.3-2%, always resolve). 186 runs.
- Exp187 (KEEP, rotation): F16 10.51 x15 identical. 187 runs.
- Exp188 (KEEP, rotation): Q4 6.82 @35.8C. 188 runs.
- Exp189 (KEEP): health 6.57 @35.9C. 189 runs.
- Exp190 (KEEP, rotation): ultra-lean 7.08 @35.9C. 190 runs.
- Exp191 (KEEP): health 6.58 @35.8C. 191 runs.
- Exp192 (KEEP, rotation): F16 10.54 x16 identical. 192 runs.
- Exp193 (KEEP, rotation): F16 10.52 x17 identical (3.4h-idle warmup, no cold-cache effect). 193 runs.
- Exp194 (discard, infra): VAE q6_k_mixed via python gguf converter. Dead: this gguf-py
  implements K-quant DEQUANTIZE only; quantize_blocks raises bare NotImplementedError
  and the converter's except-branch silently kept F16 (byte-identical size). Mode
  reverted, mislabeled file deleted. Lesson: verify quantized file SIZE before
  any phone run. 194 runs.
- Exp195 (KEEP, bracket): Q8 anchor 6.57, band center, for Q6_K A/B. 195 runs.
- Exp196 (discard): VAE Q6_K via desktop llama-quantize (499MB Q6_K + 184MB F16
  fallback = 686MB, -18% weight traffic vs Q8). RTF 8.46 (+29%!), VAE 67.8 vs 48.6
  (+40%, symmetric ac/sem). Refines traffic-bound model: VAE time follows bytes
  ONLY while dequant is trivial (Q8_0); Q6_K NEON dequant (super-block scales +
  bit-shuffle) costs more than its traffic saves. Generalizes Exp6 (Q4_0-FFN
  slower): ALL K-quants dead for this VAE on A78 - Q8_0-mixed is the SPEED-OPTIMAL
  precision, not a compromise. Q5_K/Q4_K killed by interpolation (less traffic
  saved, same-cost dequant). Precision dimension CLOSED. Side: RSS 2.30GB
  leanest-ever, noted. Files purged both sides, tree clean. 196 runs.
- Exp197 (KEEP, rotation): Q4 6.79 @33.0C, in band. 197 runs.
- Exp198 (KEEP, rotation): ultra-lean 7.06 @33.2C, in band. 198 runs.
- Exp199 (KEEP): health 6.59 @33.4C, upper-mid band, temp-consistent. Post-Q6_K
  instrument stable. 199 runs.
- Audit (no run): persistent-VAE-arena probe dead by inspection - vae_encode_impl
  already grows-never-shrinks a reused compute arena (first-touch once). No
  per-piece page-zeroing exists to eliminate. Last structural CPU idea closed.
- Exp200 (KEEP, rotation): F16 10.50 x18 identical, best of band. 200 runs:
  12.24 -> 6.52 (-47%). All tiers metronomic, all gaps closed, ship proven.
- Exp201 (KEEP): health 6.53 @33.6C, band center-low. Steady. 201 runs.
- Exp202 (KEEP, rotation): Q4 6.84 @33.7C, top edge, temp-consistent. 202 runs.
- Exp203 (KEEP, rotation): ultra-lean 7.11 @33.7C, top edge of 7.04-7.12 band. 203 runs.
- Exp204 (KEEP): health 6.54 @33.8C, mid-band despite warmth. Steady. 204 runs.
- Exp205 (KEEP, rotation): F16 10.51 x19 identical. 205 runs.
- Exp206 (KEEP): health 6.55 @34.1C, mid-band. Steady. 206 runs.
- Exp207 (KEEP, rotation): Q4 6.82 @34.1C, band center (Exp202 top-edge resolved). 207 runs.
- Exp208 (KEEP, rotation): ultra-lean 7.09 @34.1C, in band. 208 runs.
- Exp209 (KEEP): health 6.54 @34.2C. Steady. 209 runs.
- Exp210 (KEEP, rotation): F16 10.51 x20 identical. 210 runs.
- Exp211 (KEEP): health 6.56 @34.3C, mid-band. Steady. 211 runs.
- Exp212 (KEEP, rotation): Q4 6.83 @34.5C, in band. 212 runs.
- Exp213 (KEEP, rotation): ultra-lean 7.06 @34.4C, in band. 213 runs.
- Exp214 (KEEP): health 6.58 @34.5C. Steady. 214 runs.
- Exp215 (KEEP, rotation): F16 10.52 x21 identical. 215 runs.
- Exp216 (KEEP): health 6.56 @34.6C, mid-band. Steady. 216 runs.
- Exp217 (KEEP, rotation): Q4 6.81 @34.6C, in band. 217 runs.
- Exp218 (KEEP, rotation): ultra-lean 7.08 @34.6C, in band. 218 runs.
- Exp219 (KEEP): health 6.56 @34.6C. Steady. 219 runs.
- Exp220 (KEEP, rotation): F16 10.53 x22 identical. 220 runs.
- Exp221 (KEEP): health 6.56 @34.7C. Steady. 221 runs.
- Exp222 (KEEP, rotation): Q4 6.81 @34.6C, in band. 222 runs.
- Exp223 (KEEP, rotation): ultra-lean 7.07 @34.6C, in band. 223 runs.
- Exp224 (KEEP): health 6.56 @34.5C. Steady. 224 runs.
- Exp225 (KEEP, rotation): F16 10.51 x23 identical. 225 runs.
- Exp226 (KEEP): health 6.55 @34.4C. Steady. 226 runs.
- Exp227 (KEEP, rotation): Q4 6.79 @34.2C, in band. 227 runs.
- Exp228 (KEEP, rotation): ultra-lean 7.07 @34.1C, in band. 228 runs.
- Exp229 (KEEP): health 6.52 @34.0C, anchor exact, best of band. 229 runs.
- Exp230 (KEEP, rotation): F16 10.48 x24, best of band. 230 runs.
- Exp231 (KEEP): health 6.54 @33.9C. Steady. 231 runs.
- Exp232 (KEEP, rotation): Q4 6.80 @33.7C, in band. 232 runs.
- Exp233 (KEEP, rotation): ultra-lean 7.06 @33.7C, in band. 233 runs.
- Exp234 (KEEP): health 6.55 @33.6C. Steady. 234 runs.
- Exp235 (KEEP, rotation): F16 10.52 x25 identical. 235 runs.
- Exp236 (KEEP): health 6.54 @33.4C. Steady. 236 runs.
- Exp237 (KEEP, rotation): Q4 6.79 @33.4C, in band. 237 runs.
- Exp238 (KEEP, rotation): ultra-lean 7.06 @33.3C, in band. 238 runs.
- Exp239 (KEEP): health 6.53 @33.2C, best of band. 239 runs.
- Exp240 (KEEP, rotation): F16 10.53 x26 identical. 240 runs.
- Exp241 (KEEP): health 6.53 @33.0C. Steady. 241 runs.
- Exp242 (KEEP, rotation): Q4 6.79 @33.0C, in band. 242 runs.
- Exp243 (KEEP, rotation): ultra-lean 7.05 @32.9C, in band. 243 runs.
- Exp244 (KEEP): health 6.52 @32.9C, anchor exact. 244 runs.
- Exp245 (KEEP, rotation): F16 10.53 x27 identical. 245 runs.
- Exp246 (KEEP): health 6.54 @32.8C. Steady. 246 runs.
- Exp247 (KEEP, rotation): Q4 6.82 @32.7C, in band. 247 runs.
- Exp248 (KEEP, rotation): ultra-lean 7.03 @32.7C, best of band. 248 runs.
- Exp249 (KEEP): health 6.55 @32.7C. Steady. 249 runs.
- Exp250 (KEEP, rotation): F16 10.51 x28 identical. 250 runs: 12.24 -> 6.52
  (-47%). All tiers metronomic, all gaps closed, ship proven.
- Exp251 (KEEP): health 6.54 @32.7C. Steady. 251 runs.
- Exp252 (KEEP, rotation): Q4 6.77 @32.7C, low side. 252 runs.
- Exp253 (KEEP, rotation): ultra-lean 7.07 @32.6C, in band. 253 runs.
- Exp254 (KEEP, WATCH): health 6.60 @32.6C, band top (+0.9%, both phases,
  cool temp = jitter signature). Re-check next Q8; 2nd >=6.60 investigates. 254 runs.
- Exp255 (KEEP, rotation): F16 10.51 x29 identical. F16 nominal => Exp254 read as
  jitter, not drift; re-check still queued. 255 runs.
- Exp256 (KEEP, watch resolved): 6.60 band-top -> immediate 6.54 re-check.
  Noise 7th time. Watch retired. 256 runs.
- Exp257 (KEEP, rotation): Q4 6.82 @32.5C, in band. 257 runs.
- Exp258 (KEEP, rotation): ultra-lean 7.07 @32.5C, in band. 258 runs.
- Exp259 (KEEP): health 6.54 @32.5C. Steady. 259 runs.
- Exp260 (KEEP, rotation): F16 10.52 x30 identical. 260 runs.
- Exp261 (KEEP): health 6.54 @32.5C. Steady. 261 runs.
- Exp262 (KEEP, rotation): Q4 6.84 @32.4C, top edge (cool-temp jitter). 262 runs.
- Exp263 (KEEP, rotation): ultra-lean 7.06 @32.4C, in band. 263 runs.
- Exp264 (KEEP): health 6.53 @32.5C, best of band. 264 runs.
- Exp265 (KEEP, rotation): F16 10.50 x31, best of band. 265 runs.
- Exp266 (KEEP): health 6.55 @32.3C. Steady. 266 runs.
- Exp267 (KEEP, rotation): Q4 6.81 @32.2C, in band (Exp262 top-edge resolved). 267 runs.
- Exp268 (KEEP, rotation): ultra-lean 7.06 @32.2C, in band. 268 runs.
- Exp269 (KEEP): health 6.54 @32.2C. Steady. 269 runs.
- Exp270 (KEEP, rotation): F16 10.52 x32 identical. 270 runs.
- Exp271 (KEEP): health 6.56 @32.2C. Steady. 271 runs.
- Exp272 (KEEP, rotation): Q4 6.81 @32.2C, in band. 272 runs.
- Exp273 (KEEP, rotation): ultra-lean 7.06 @32.0C, in band. 273 runs.
- Exp274 (KEEP): health 6.56 @32.0C. Steady. 274 runs.
- Exp275 (KEEP, rotation): F16 10.50 x33, best of band. 275 runs.
- Exp276 (KEEP): health 6.52 @32.0C, anchor exact. 276 runs.
- Exp277 (KEEP, rotation): Q4 6.80 @32.0C, in band. 277 runs.
- Exp278 (KEEP, rotation): ultra-lean 7.05 @31.9C, in band. 278 runs.
- Exp279 (KEEP): health 6.51 @31.9C, best of band. 279 runs.
- Exp280 (KEEP, rotation): F16 10.49 x34, best of band. 280 runs.
- Exp281 (KEEP): health 6.51 @32.0C, best of band. 281 runs.
- Exp282 (KEEP, rotation): Q4 6.80 @32.0C, in band. 282 runs.
- Exp283 (KEEP, rotation): ultra-lean 7.06 @32.0C, in band. 283 runs.
- Exp284 (KEEP): health 6.59 @32.0C, LM-side jitter (prefill 11.4). 284 runs.
- Exp285 (KEEP, rotation): F16 10.48 x35, best of band. 285 runs.
- Exp286 (KEEP): health 6.53 @32.0C, Exp284 jitter resolved. 286 runs.
- Exp287 (KEEP, rotation): Q4 6.82 @31.9C, in band. 287 runs.
- Exp288 (KEEP, rotation): ultra-lean 7.06 @31.9C, in band. 288 runs.
- Exp289 (KEEP): health 6.57 @31.9C, mid-band. Steady. 289 runs.
- Exp290 (KEEP, rotation): F16 10.50 x36, best of band. 290 runs.
- Exp291 (KEEP): health 6.52 @31.9C, anchor exact. 291 runs.
- Exp292 (KEEP, rotation): Q4 6.79 @31.8C, in band. 292 runs.
- Exp293 (KEEP, rotation): ultra-lean 7.07 @31.8C, in band. 293 runs.
- Exp294 (KEEP): health 6.54 @31.8C. Steady. 294 runs.
- Exp295 (KEEP, rotation): F16 10.49 x37, best of band. 295 runs.
- Exp296 (KEEP): health 6.56 @31.8C. Steady. 296 runs.
- Exp297 (KEEP, rotation): Q4 6.80 @31.8C, in band. 297 runs.
- Exp298 (KEEP, rotation): ultra-lean 7.07 @31.8C, in band. 298 runs.
- Exp299 (KEEP): health 6.55 @31.8C. Steady. 299 runs.
- Exp300 (KEEP, rotation): F16 10.52 x38 identical. 300 runs: 12.24 -> 6.52
  (-47%). All tiers metronomic, all gaps closed, ship proven.
- Exp301 (KEEP): health 6.53 @31.8C, best of band. 301 runs.
- Exp302 (KEEP, rotation): Q4 6.82 @31.8C, in band. 302 runs.
- Exp303 (KEEP, rotation): ultra-lean 7.05 @31.7C, in band. 303 runs.
- Exp304 (KEEP): health 6.57 @31.7C, mid-band. Steady. 304 runs.
- Exp305 (KEEP, rotation): F16 10.52 x39 identical. 305 runs.
- Exp306 (KEEP): health 6.51 @31.8C, best of band. 306 runs.
- Exp307 (KEEP, rotation): Q4 6.80 @31.7C, in band. 307 runs.
- Exp308 (KEEP, rotation): ultra-lean 7.07 @31.7C, in band. 308 runs.
- Exp309 (KEEP): health 6.53 @31.6C, best of band. 309 runs.
- Exp310 (KEEP, rotation): F16 10.47 x40, best of band. 310 runs.
- Exp311 (KEEP): health 6.55 @31.7C. Steady. 311 runs.
- Exp312 (KEEP, rotation): Q4 6.82 @31.7C, in band. 312 runs.
- Exp313 (KEEP, rotation): ultra-lean 7.05 @31.6C, in band. 313 runs.
- Exp314 (KEEP): health 6.56 @31.6C, mid-band. Steady. 314 runs.
- Exp315 (KEEP, rotation): F16 10.50 x41, best of band. 315 runs.
- Exp316 (KEEP): health 6.52 @31.7C, anchor exact. 316 runs.
- Exp317 (KEEP, rotation): Q4 6.81 @31.7C, in band. 317 runs.
- Exp318 (KEEP, rotation): ultra-lean 7.06 @31.7C, in band. 318 runs.
- Exp319 (KEEP): health 6.54 @31.5C. Steady. 319 runs.
- Exp320 (KEEP, rotation): F16 10.51 x42 identical. 320 runs.
- Exp321 (KEEP): health 6.53 @31.7C. Steady. 321 runs.
- Exp322 (KEEP, rotation): Q4 6.83 @31.6C, in band. 322 runs.
- Exp323 (KEEP, rotation): ultra-lean 7.06 @31.5C, in band. 323 runs.
- Exp324 (KEEP): health 6.57 @31.5C, mid-band. Steady. 324 runs.
- Exp325 (KEEP, rotation): F16 10.52 x43 identical (majflt=-3 transient artifact). 325 runs.
- Exp326 (KEEP): health 6.52 @31.5C, anchor exact. 326 runs.
- Exp327 (KEEP, rotation): Q4 6.82 @31.5C, in band. 327 runs.
- Exp328 (KEEP, rotation): ultra-lean 7.06 @31.5C, in band. 328 runs.
- Exp329 (KEEP): health 6.56 @31.5C. Steady. 329 runs.
- Exp330 (KEEP, rotation): F16 10.54 x44 identical. 330 runs.
- Exp331 (KEEP): health 6.54 @31.5C. Steady. 331 runs.
- Exp332 (KEEP, rotation): Q4 6.82 @31.4C, in band. 332 runs.
- Exp333 (KEEP, rotation): ultra-lean 7.07 @31.4C, in band. 333 runs.
- Exp334 (KEEP): health 6.56 @31.4C. Steady. 334 runs.
- Exp335 (KEEP, rotation): F16 10.50 x45, best of band. 335 runs.
- Exp336 (KEEP): health 6.52 @31.5C, anchor exact. 336 runs.
- Exp337 (KEEP, rotation): Q4 6.78 @31.5C, low side. 337 runs.
- Exp338 (KEEP, rotation): ultra-lean 7.06 @31.5C, in band. 338 runs.
- Exp339 (KEEP): health 6.57 @31.5C, mid-band. Steady. 339 runs.
- Exp340 (KEEP, rotation): F16 10.48 x46, best of band. 340 runs.
- Exp341 (KEEP): health 6.51 @31.5C, anchor exact. 341 runs.
- Exp342 (KEEP, rotation): Q4 6.81 @31.4C, in band. 342 runs.
- Exp343 (KEEP, rotation): ultra-lean 7.06 @31.4C, in band. 343 runs.
- Exp344 (KEEP): health 6.58 @31.4C, upper-mid (+0.5s both phases). 344 runs.
- Exp345 (KEEP, rotation): F16 10.49 x47, best of band. 345 runs.
- Exp346 (KEEP): health 6.52 @31.4C, anchor exact (Exp344 resolved). 346 runs.
- Exp347 (KEEP, rotation): Q4 6.79 @31.4C, in band. 347 runs.
- Exp348 (KEEP, rotation): ultra-lean 7.07 @31.4C, in band. 348 runs.
- Exp349 (KEEP): health 6.53 @31.4C, best of band. 349 runs.
- Exp350 (KEEP, rotation): F16 10.46 x48, best of band. 350 runs.
- Exp351 (KEEP): health 6.54 @31.4C. Steady. 351 runs.
- Exp352 (KEEP, rotation): Q4 6.81 @31.4C, in band. 352 runs.
- Exp353 (KEEP, rotation): ultra-lean 7.06 @31.4C, in band. 353 runs.
- Exp354 (KEEP): health 6.53 @31.8C. Steady (majflt=-3 artifact). 354 runs.
- Exp355 (KEEP, rotation): F16 10.49 x49, best of band. 355 runs.
- Exp356 (KEEP): health 6.54 @31.6C. Steady. 356 runs.
- Exp357 (KEEP, rotation): Q4 6.80 @31.6C, in band. 357 runs.
- Exp358 (KEEP, rotation): ultra-lean 7.06 @31.5C, in band. 358 runs.
- Exp359 (KEEP): health 6.57 @31.5C, mid-band. Steady. 359 runs.
- Exp360 (KEEP, rotation): F16 10.49 x50, best of band. 360 runs.
- Exp361 (KEEP): health 6.52 @31.5C, anchor exact. 361 runs.
- Exp362 (KEEP, rotation): Q4 6.81 @31.5C, in band. 362 runs.
- Exp363 (KEEP, rotation): ultra-lean 7.06 @31.4C, in band. 363 runs.
- Exp364 (KEEP): health 6.51 @31.4C, anchor exact. 364 runs.
- Exp365 (KEEP, rotation): F16 10.48 x51, best of band. 365 runs.
- Exp366 (KEEP): health 6.55 @31.6C. Steady. 366 runs.
- Exp367 (KEEP, rotation): Q4 6.82 @31.5C, in band. 367 runs.
- Exp368 (KEEP, rotation): ultra-lean 7.06 @31.5C, in band. 368 runs.
- Exp369 (KEEP): health 6.55 @31.5C. Steady. 369 runs.
- Exp370 (KEEP, rotation): F16 10.52 x52 identical. 370 runs.
- Exp371 (KEEP): health 6.53 @31.4C, best of band. 371 runs.
- Exp372 (KEEP, rotation): Q4 6.80 @31.5C, in band. 372 runs.
- Exp373 (KEEP, rotation): ultra-lean 7.06 @31.4C, in band. 373 runs.
- Exp374 (KEEP): health 6.57 @31.4C, mid-band. Steady. 374 runs.
- Exp375 (KEEP, rotation): F16 10.49 x53, best of band. 375 runs.
- Exp376 (KEEP): health 6.55 @31.6C. Steady. 376 runs.
- Exp377 (KEEP, rotation): Q4 6.82 @31.5C, in band. 377 runs.
- Exp378 (KEEP, rotation): ultra-lean 7.07 @31.5C, in band. 378 runs.
- Exp379 (KEEP): health 6.59 @31.5C, upper-mid (+0.5s VAE-side; RSS 2398 =
  HWM sampler timing, not a change). 379 runs.
- Exp380 (KEEP, rotation): F16 10.48 x54, best of band. 380 runs.
- Exp381 (KEEP): health 6.53 @31.4C, best of band (Exp379 resolved). 381 runs.
- Exp382 (KEEP, rotation): Q4 6.81 @31.4C, in band. 382 runs.
- Exp383 (KEEP, rotation): ultra-lean 7.06 @31.3C, in band. 383 runs.
- Exp384 (KEEP): health 6.56 @31.3C. Steady. 384 runs.
- Exp385 (KEEP, rotation): F16 10.48 x55, best of band. 385 runs.
- Exp386 (KEEP): health 6.55 @31.4C. Steady. 386 runs.
- Exp387 (KEEP, rotation): Q4 6.81 @31.4C, in band. 387 runs.
- Exp388 (KEEP, rotation): ultra-lean 7.06 @31.3C, in band. 388 runs.
- Exp389 (KEEP): health 6.57 @31.3C, mid-band. Steady. 389 runs.
- Exp390 (KEEP, rotation): F16 10.52 x56 identical. 390 runs.
- Exp391 (KEEP): health 6.54 @31.3C. Steady. 391 runs.
- Exp392 (KEEP, rotation): Q4 6.80 @31.3C, in band. 392 runs.
- Exp393 (KEEP, rotation): ultra-lean 7.07 @31.3C, in band. 393 runs.
- Exp394 (KEEP): health 6.51 @31.3C, anchor exact. 394 runs.
- Exp395 (KEEP, rotation): F16 10.56, mild jitter (+0.3s both phases). 395 runs.
- Exp396 (KEEP): health 6.50 @31.2C, best of band. 396 runs.
- Exp397 (KEEP, rotation): Q4 6.81 @31.2C, in band. 397 runs.
- Exp398 (KEEP, rotation): ultra-lean 7.07 @31.2C, in band. 398 runs.
- Exp399 (KEEP): health 6.58 @30.3C, upper-mid (cool-temp jitter). 399 runs.
- Exp400 (KEEP, rotation): F16 10.50 x57, best of band. 400 runs.
- Exp401 (KEEP): health 6.54 @29.7C (Exp399 resolved). Steady. 401 runs.
- Exp402 (KEEP, rotation): Q4 6.81 @29.9C, in band. 402 runs.
- Exp403 (KEEP, rotation): ultra-lean 7.09 @30.1C, in band. 403 runs.
- Exp404 (KEEP, WATCH): health 6.65 @30.3C exceeds band (LM-side: prefill 11.5).
  Re-check next Q8; 2nd >=6.60 investigates. 404 runs.
- Exp405 (KEEP, rotation): F16 10.50 x58 identical. F16 nominal => Exp404 read as
  jitter; formal re-check still queued. 405 runs.
- Exp406 (KEEP, watch resolved): 6.65 LM-side exceed -> immediate 6.52 re-check.
  Noise 8th time. Watch retired. 406 runs.
- Exp407 (KEEP, rotation+audit): Q4 6.83 @28.2C, in band. Audit of DeerFlow
  brief-1 vs our tree: O1 VOID (no blocked-GEMM for Q8_0/F16/F32 in this fork -
  row-wise vec_dot re-reads weights per row; only Q4_0/I8_S have gemm ptrs),
  O2-bit-exact DEAD (R=219649≈9.15s >> hop=70400; explains xwin drift),
  O3 DEAD/DONE (no k==s conv exists; history already per-site k-s),
  premise#3 FALSE (stride is 3200, pieces aligned; dims misread),
  O4 VOID (same root cause), O5 ALREADY RUNNING (sdot in vec_dot_q8_0).
  NEW: VAE stages are pure ConvNeXt, zero attention in encoder.
  5 proposals -> 0 runs. 407 runs.
- Exp408 (KEEP, rotation): ultra-lean 7.07 @28.7C, in band. 408 runs.
- Exp409 (KEEP): health 6.58 @29.2C, upper-mid (cool-temp jitter). 409 runs.
- Exp410 (KEEP, rotation): F16 10.50 x59 identical. 410 runs.
- Exp411 (KEEP): health 6.54 @30.0C (Exp409 resolved). Steady. 411 runs.
- Exp412 (KEEP, rotation): Q4 6.81 @30.3C, in band. 412 runs.
- Exp413 (KEEP, rotation): ultra-lean 7.06 @30.4C, in band. 413 runs.
- Exp414 (KEEP): health 6.60 @30.7C, decode-side jitter (6.7 vs 6.0). 414 runs.
- Exp415 (KEEP, rotation): F16 10.50 x60, best of band. 415 runs.
- Exp416 (KEEP): health 6.54 @31.0C, Exp414 jitter resolved. 416 runs.
- Exp417 (KEEP, rotation): Q4 6.80 @31.0C, in band. 417 runs.
- Exp418 (KEEP, rotation): ultra-lean 7.06 @31.1C, in band. 418 runs.
- Exp419 (KEEP, WATCH): health 6.61 @31.2C exceeds band (both phases mild).
  Re-check next Q8; 2nd >=6.60 investigates. 419 runs.
- Exp41 (KEEP, biggest win since Q8): -t 2 pinned C0 (2 fastest cores, Dimensity
  1300 cpus 6-7). RTF 6.52 (-33% vs -t4/F0 9.7!). Accuracy IDENTICAL (40-utt
  S/D/I, 69s S/D/I). Mechanism: EAS parked threads on capped 2.0GHz cores;
  C0 forces 2.4GHz primes + zero migration. NEW LOOP PROTOCOL (harness-level;
  binary defaults stay portable). Device-topology-specific by nature.
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
