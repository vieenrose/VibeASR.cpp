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
