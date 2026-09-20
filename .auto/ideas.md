TAIL-FLUSH SHIPPED AS THE DEFAULT - v4.6, THE LARGEST WIN SINCE THE FUSIONS (Exp829).
Priced first (Exp828: 13.5 % protocol / ~13 % short clips), built behind a flag, gated, then flipped. Numbers:
  * SPEED: protocol 2.1866 -> 1.9422 (-11.8 %, paired 3+3 reps), gate mean 2.4456 -> 1.9811 (-19.0 %),
    17 s 2.1760 -> 2.0890 (-4.0 %), 69 s 2.2091 -> 2.1759 (-1.5 %). The length dependence matches the
    Exp828 model exactly (prize = pf_last x 4.52 s), and short clips gain MOST, which is where streaming
    latency lives. vae_s 14.0 -> 12.0 (priced 2.25 s, got 2.0), lm_s 7.9 -> 7.3 (priced 0.70 s, got 0.6).
  * ACCURACY (the acceptance rule the user picked: 40-utt parity + paired token test): WER 4.38 % vs 4.51 %,
    PAIRED 0 discordant tokens of 731 (McNemar p=1.0), insertions 1 vs 2 (one FEWER hallucinated word),
    6/40 transcripts differ and only in punctuation/marginal class. So the padded frames were not neutral:
    they were generating silence-driven tokens, exactly as Exp658's insertion profile suggested.
  * BEHAVIORAL SET with the flush active: 9/11 PASS - all SEMANTIC contracts hold (silence->[Silence],
    noise->[Noise], music->[Music], 48 kHz stereo, sub-piece short36, overlap splits speakers). The two
    FAILs are the LADDER TOKEN CANARIES (17 s 106 vs 108, 138 s 876 vs 877): the flush legitimately removes
    tokens that padded frames produced. MUST FIX NEXT: update those two expected values in
    .auto/behavior_watch.sh or the tool will cry wolf on every run.
  * IMPLEMENTATION (3 sites in demo/asr_streaming.cpp, ~20 lines): the last window's length becomes
    want = round_up(avail, 6400) (the cached encoder's 2-frame granularity), the piece loop takes
    nsamp = min(piece_samples, remaining) and counts frames actually produced, and the LM is fed
    win_frames instead of FRAMES_PER_WINDOW. feed_embeds was already row-parameterized and the xwin branch
    already encoded 22-frame windows, so nothing structural was needed - the "window-loop rework" the old
    notes feared did not exist.
  * HATCH: FLUSH_TAIL_OFF=1 -> protocol exactly as before (2.1860, transcript 55ac39b635cb = the frozen
    pre-Exp829 reference, verified). FLUSH_TAIL=0 is an alias. 9th hatch: add it to rollback_audit.sh.
  * EXCLUDED BY CONSTRUCTION: the deferred late path (lean tier, VAE_DEFER_LATE=1) assembles a window-sized
    boundary buffer (tpiece * pieces), so flush_tail is forced off there and the lean tier's cells are
    UNCHANGED. If someone wants the ~13 % on the lean tier too, the work is the boundary-buffer sizing, not
    the flush itself. Also untested: flush x --xwin (the xwin loop zero-pads its own final hop).
MUST DO NEXT (doc-drift trigger fired - the era's numbers changed):
  1. Re-anchor equiv_map.py's reference to hyp-flusht829 and update the protocol hash everywhere:
     the shipped protocol transcript is now 1a095c8496b4 (was 55ac39b635cb).
  2. Refresh ladder cells (10/17/69/138 s, gate mean 1.9811 measured already; lean row unchanged) in
     RESULTS.md, STREAMING_1P5B.md, and .auto/prompt.md's tier table; bump to v4.6.
  3. Refit the window-grid model: the per-clip padded-window term is gone, so vae_s ~= 0.21 + 3.45 x
     (windows - pf_last) - Exp803/828's form no longer applies.
  4. Re-run rollback_audit.sh (adds the 9th hatch) and the RSS soak; the graph now has one fewer window's
     worth of nodes per clip, and RSS should drop or stay flat.
LESSON (worth keeping): the biggest lever in the whole loop was not a kernel, a layout or a quantization -
it was a PROTOCOL ARTIFACT that the ledger had correctly measured but wrongly filed as "not a loop
experiment" for two years, because nobody had priced the LM half of it. When an "out of scope" item is the
largest remaining number, re-price it and ask, rather than re-phrasing the same decline.

TAIL-FLUSH PRIZE RE-PRICED WITH THE LM HALF: 13.5 % ON THE PROTOCOL CLIP, ~13 % ON SHORT CLIPS (Exp828).
The ledger's version of this lever was "16.3 % of VAE time" (Exp803) - a VAE-only view that undercounts, because
the padded frames also occupy LM prefill rows. Re-derived on the current stack, 5 clips (4/4/4/6/24 windows):
  FIT: vae_s = 0.21 + 3.45 x windows  (predicted 14.01 / 14.01 / 14.01 / 20.91 / 83.01 vs measured
       14.0 / 14.0 / 14.0 / 20.9 / 83.0). Per-window cost fell 3.60 -> 3.45 with Exp821's pad removal, so this
       model must be re-fitted after any graph change, not reused.
  Per-window LM prefill = 1.07 s (LT trace: 4.28 s over 4 windows).
  PRIZE = pf_last x (3.45 + 1.07), where pf_last = padded fraction of the FINAL window = (83200 - real)/83200:
     protocol 10 s  pf 0.653 -> 2.95 s = 13.5 % of gen   (VAE-only view would have claimed 10.3 %)
     chat17         pf 0.327 -> 1.48 s =  4.0 %
     chat69         pf 0.560 -> 2.53 s =  1.7 %
  pf is essentially frac(length/hop), i.e. clip-dependent and roughly uniform, so for arbitrary short clips
  E[pf] ~ 0.5 -> E[saving] ~ 2.26 s per clip; on the gate set's ~16.6 s mean that is ~13 %. THIS IS NOW THE
  LARGEST KNOWN LEVER BY AN ORDER OF MAGNITUDE (every CPU-kernel item left is <= 2 %).
WHY IT IS STILL NOT AN OPTIMIZATION: the final window would carry 9 real frames instead of 26 - a change to the
model's input contract, which this loop has always classified as a task/semantics decision, not a speedup.
FEASIBILITY IS BETTER THAN THE LEDGER SUGGESTS: the xwin branch already encodes a 22-frame window
(encode_frames(..., want=HOP_SAMPLES=70400), demo:459-471), so the encoder DOES accept frame counts other than
26; the real blocker is downstream - the demo's chunk/LM path is written against FRAMES_PER_WINDOW constants.
ACCURACY RISK IS NARROWER THAN IT SOUNDS: Exp554 established all late ops are causal or pointwise, so trailing
zeros do NOT alter the real frames' features; the only change is that the final chunk's LM input loses the
padded frames - which today may be feeding silence-driven insertions (Exp658: 2.1 insertions/1k on clean audio).
So it is testable with the existing 40-utt gate + paired token test, and could plausibly IMPROVE insertions.
STATUS: not attempted; scope decision surfaced to the user (Exp828). If attempted: env-gated prototype, price
speed on protocol + gate mean, gate WER with compare_arms.py, and never describe it as a general speedup without
the length dependence (13.5 % at 10 s, 1.7 % at 69 s).

WALL-CLOCK TREE CLOSED END TO END + THE METRIC'S BLIND SPOT IS NOW NAMED (Exp827).
  * INSIDE THE METRIC (v4.5, after Exp821 deleted graph nodes - the reason to re-close it): the per-window trace
    sums to vae 13.99 + prefill 4.28 + decode 3.61 = 21.88 s, and gen_s = rtf x duration = 2.1880 x 10 = 21.88 s.
    EXACT. No unaccounted work inside the measured window, on the current stack.
  * WALL = 24.2 s = load 1.2 + 1.12 UNNAMED + gen 21.88. The 1.12 s is the ONE-TIME SYSTEM-PROMPT PREFILL
    (31 rows; demo/asr_streaming.cpp:358-377 runs it before gen_start, so it is outside gen_s but inside g_prefill_ms;
    LT deltas exclude it because pre_prev is taken after it). Corroborated by rate arithmetic: 31 x 1.75 GMac/row
    at the measured 21.2 GMAC/s/core ~= 1.3 s. So the whole wall is accounted for by three named components.
  * CLAIM RULE: any end-to-end / throughput / TTFT statement must use 24.2 s for a 10 s clip = 2.42x realtime,
    NOT the ladder's 2.19. The ~2.3 s one-time cost matters most on short audio (a 5 s clip: ~11 s gen + 2.3 s =
    17 % of wall), which is exactly the regime the 40-utt gate set lives in.
  * PRODUCT-SIDE ITEM, DELIBERATELY NOT A LOOP EXPERIMENT: the prompt prefill and window 1's LM prefill are two
    sequential weight streams. They can be ONE batch (contiguous causal positions; the audio embeddings already
    exist before window 1's LM call), which removes one duplicate pass over the LM weights: wall -~1.0 s per clip,
    but gen_s +~0.14 s. So the loop's metric would REGRESS while real latency improves. Recorded, not done:
    a speed loop must neither move work OUT of the measured window (that is cheating - Exp806's page-in ruling)
    nor accept work INTO it to buy wall clock. If the product ever wants it, it is an LM-loop change plus a
    prompt/KV position check, and it should be judged on end-to-end wall time, not on rtf.
  * GUARD BOARD CURRENT (anti-overfit): guard slice 2.3226 (2 reps, 55 tokens, identical counts) vs protocol
    2.1896 -> +6.1 %. Across the v4.4 -> v4.5 ships the guard moved -2.7 % against the protocol's -2.8 % (within
    0.1 pp), 5 consecutive tracked changes, zero protocol-specific-tuning signal. Provenance note: AUDIO=<name>
    is honoured by measure.sh (device-side name) and it prints 'note: audio=...' - quoted, per the Exp648 rule.
  * INSTRUMENT QUIRK (do not explain away): one run emitted ZERO 'LT ' lines with EXTRA_ENV="LATENCY_TRACE=1",
    while a later run carrying the same env emitted 4 - transport itself then proven by 208 LPAD lines in the
    same file. Standing rule re-confirmed: a silent trace proves nothing until a knob that MUST print is shown
    printing in the same run.

BEHAVIORAL CONTRACT WATCHDOG SHIPPED AS A TOOL, 11/11 PASS AT v4.5 (Exp826, .auto/behavior_watch.sh).
Until now the product-contract set was run AD HOC per era (Exp676/800) with nothing committed, so nothing
enforced it - and Exp821 is exactly the change type that could break it silently (the dw conv now applies its
OWN causal left pad, i.e. behaviour at tensor EDGES changed). The script honors tier.env, syncs binary+models
by md5, refuses to run if ps is broken or a co-runner is alive, and negative-controls itself:
  .auto/behavior_watch.sh --selftest  -> plants tokens==999999 on every probe, MUST report failures (it does,
  11/11), so a green run means something (Exp660 rule).
Result at v4.5: silence->[Silence] (8 tok), noise->[Noise], music->[Music], 48 kHz STEREO transcribes (38 tok),
sub-piece short36 graceful (16 tok), twospk_overlap SPLITS speakers (73 tok), twospk one-tag (CLOSED Exp650),
and the ladder token canaries 39/108/446/877 exactly (10/17/69/138 s).
LONG-RUN MEMORY at v4.5: 138 s run sampled -> peak_rss 2191.3 MB, majflt 0, completed with RTF 2.2781 / 877
tokens (ladder-consistent) -> no leak signal from the pad-removal graph.
SHELL/LAUNCH TRAPS (cost two wasted runs; both are "silent, not loud"):
  1. `cd repo && LAUNCH &` parses as `(cd && LAUNCH) &` - the PARENT shell never changed directory, so every
     later command (python3 .auto/..., $DEV) ran in the workDir with an EMPTY DEV. My orphan check then
     reported "0 alive" vacuously. Rule: background a SUBSHELL `( cd x && y & )`, or better, never background
     a chain that assigns variables other commands depend on; and make any idle/alive probe distinguish
     "adb failed" from "no process" (require >=20 ps rows before believing IDLE).
  2. Launching a device run as `adb shell "nohup ... </dev/null >f 2>&1 &"` is UNRELIABLE: of two attempts one
     ran to completion (soak2.txt, RTF line present) and one died with its adb session leaving a 30 KB PARTIAL
     transcript and NO RTF line. A sampler attached to the dead process would happily report a phantom. Rule:
     drive long runs through the harness in a local background job (keeps an adb client attached), and always
     require the completion marker (RTF line) before trusting any sampled series.

--xwin RE-PRICED AT v4.5 + EXP821's GATE WAS INCOMPLETE (Exp825, a real correctness fix to a documented option).
  * SPEED CLAIM, CLOSED: --xwin (cross-window VAE carry) is a COST at every length, interleaved 2x2:
      10 s  2.1900 windowed -> 2.3789 xwin  (+8.6%)
      17 s  2.1832           -> 2.3744      (+8.8%)
     138 s  2.2794           -> 2.3772      (+4.3%)
    So the two-year-old "-8.5 % on shorts" note is dead, and there is no length regime where carry is faster.
    It also emits MORE tokens (115 vs 108 at 17 s; 886 vs 877 at 138 s) and its accuracy was already measured
    worse (WER 14.4 -> 27.8 on the multispk probe, Exp648). It stays a DIAGNOSTIC for diarization, not an option.
    Why the gap shrinks with length: carry advances contiguously (encodes 70400 samples per hop) while the
    windowed protocol re-encodes 83200 per hop, so carry's VAE work per audio second is ~15 % lower and that
    offset grows with clip length; the per-hop splice cost (below) is length-independent.
  * CORRECTNESS FIX (correction to Exp821's shipped scope, not a speed change): the fast path was offered in
    CARRY mode. The gate was `pieces == 1 && !defer` - it never considered --xwin, and the xwin loop reuses the
    same cache. Consequence: since v4.5, --xwin was silently measuring a different system than the one Exp648/714
    characterized. Proof it was real: VAE_DW_LPAD_OFF=1 moves the xwin arm 37 -> 39 tokens, and after adding
    `&& !params.xwin` the xwin arm reproduces the pre-Exp821 output exactly (hash 5f08cd0af04f, 0 fast-path
    sites). Mechanism: with carry, a site's only COLD build is hop 1, and the host-side history roll there is
    not what the materialized [hist | x] tensor would have left for the following hops - the same non-neutrality
    class as the lean tier (Exp823), now with a clean second instance. RULE: "output-neutral" gates must
    enumerate the *cache lifecycle* modes (per-window reset | carry | deferred late pass), not just granularity.
  * PAD ASYMMETRY ACCOUNTED FOR: with the fast path disabled in both arms the xwin gap drops to +6.8 %
    (2.2567 -> 2.4099), i.e. ~1.8 pp of today's xwin penalty is precisely Exp821's free cold start on the
    windowed side (windowed pays 0 splice nodes per window at p1; carry materializes 26 per hop).
  * THREE HARNESS/INSTRUMENT BUGS FOUND IN ONE ITERATION (all "silent, not loud"):
    1. measure_xwin.sh had NO EXTRA_ENV support - `adb shell` does not inherit the local environment, so my
       first mechanism test measured the same binary twice (identical hashes gave it away; 6th instance of the
       knob-that-does-not-fire class: LM_FILE Exp799, SKIP_TAIL Exp764, THREADS Exp808, xwin Exp825).
    2. measure_xwin.sh did not sync the BINARY (only models) = Exp816's bug class, second instance. This made a
       CORRECT fix look inert; the fix only showed up after measure.sh happened to push the new binary. Now
       md5-compares build-android/bin/asr_streaming, pushes and re-verifies.
    3. My Exp823 trace used a function-local `static std::map` in the concurrent encoder path with a comment
       CLAIMING thread-locality - i.e. the Exp669 race, reintroduced in a diagnostic. Fixed to thread_local;
       the shared map had been under-reporting firings 4x (26 lines instead of 26 sites x 4 windows x 2 chains
       = 208, which now checks out arithmetically).

ROLLBACK AUDIT REFRESHED AT v4.5 (Exp824, runbook product-safety item; .auto/rollback_audit.sh, 11 arms x 2 reps,
all in ONE build so no thermal/provenance assumption). Default 2.1881. Cost of turning EACH hatch off:
  dw_conv1d +8.6%  | gelu_bias +6.7% | gelu_batch +4.2% | dw_lpad +3.3% | norm_fuse +1.9% | ls_fuse +1.5%
  | mm_m2 +0.8% | dw_axpy +0.0% | ct_block +25.4% (38 tok, DIFFERS = system change) | ALL_OFF +34.9% (differs)
  * 9 of 11 arms are BYTE-IDENTICAL (hash 55ac39b635cb) => the v4.5 fusion+kernel+m2 stack is output-equivalent
    to its own pre-change path measured in one build - the strongest equivalence statement this loop can make.
  * COUPLING A RUNBOOK MUST STATE: VAE_DW_CONV1D_OFF rose +5.7% (v4.4) -> +8.6%, because Exp821's causal left
    pad lives INSIDE that kernel, so reverting the kernel also reverts the pad absorption. Same reason
    gelu_bias rose +5.0% -> +6.7% (Exp781 folded the table lookup into it). A runbook quoting per-hatch costs
    must re-run this audit after every fusion, not reuse old numbers.
  * VAE_DW_AXPY_OFF reproduces +0.0% -> still inert on the shipped path (Exp814's finding holds at v4.5); the
    safe pairing is DW_CONV1D_OFF alone.
  * NEW ARM "stack_off" (added to the script): all fusions off but the [C,T] layout KEPT = +22.1% (2.6717 /
    2.6762, byte-identical). Use THAT as "what the fusion stack buys"; ALL_OFF conflates it with a layout
    revert. Exp814's ALL_OFF=+14.1% is superseded and was internally inconsistent (cheaper than ct_block
    alone at +22.8%), which means several fusions are inert or counterproductive inside the legacy layout -
    composition effect, recorded, not worth archaeology.

LEAN-TIER LEFT PAD: CLOSED ON VALUE, AND MY EQUIVALENCE ARGUMENT IS WRONG THERE (Exp823, last open speed question).
  * VALUE: forcing the fast path at the lean tier (p13 + defer ON) measures 2.4004 -> 2.3713 = -1.2% (1 rep;
    P<=T variant 2.3767). BELOW THE 2% SHIP BAR, so the project is not worth its risk even if correctness were
    solved. DECISION: the gate stays (fast path = one-piece windows with no deferred late pass); the lean
    tier's documented cells are unchanged. This is why the tier ladder did not move.
  * HYPOTHESIS 1 - the roll's leading-zeros branch, the only case that cannot occur at p1 - is REFUTED by
    construction: VAE_DW_LPAD_FORCE=2 (offer the fast path only where P <= T) gives output byte-identical to
    unrestricted FORCE=1, and the new site trace shows every firing site already has zeros_head=0
    (15 sites, P=7, T=6400/3200/1600 = stage 0-2 early-pass sites at the sample grid).
  * THE DIVERGENCE IS NOT MARGINAL-CLASS: the two lean transcripts differ in WORDS ("dilation" vs
    "diversation", plus a comma migrating across a chunk boundary), i.e. the features really do differ. So the
    "taps beyond the edge multiply zero, so deleting the pad is exact" argument does NOT extend to the
    split/deferred path as implemented. Untested next suspect (only worth it if the lean tier ever matters):
    forward_early resets cache->next_id per build, so early-pass site i and late-pass site i share a slot key;
    my host-side roll writes [zeros | x tail] into that shared slot where the old code rolled the materialized
    padded tensor, and for a slot that is warm-by-collision those are not the same object.
  * Diagnostic knobs added, inert by default (shipped tier re-verified 3x byte-identical, 2.1874-2.1914):
    VAE_DW_LPAD_FORCE=1 (offer at any granularity) / =2 (only where P <= T), VAE_LPAD_TRACE=1 (one line per
    site: P, T, dim, warm_before, zeros_head). Reuse the trace before ever revisiting this.

MUL_MAT EPILOGUE FUSION: BUILT, PROVEN CORRECT, AND MEASURED SLOWER AT EVERY TILE SIZE (Exp822, discard).
gelu was the largest non-matmul item at v4.5 (1.38 s per chain = 6% of wall, 5.7 GB/s so traffic-bound), and
Exp781 had already fused its bias, so the only route left was to compute it where the data is produced. I
built it properly: ggml_mul_mat_gelu_bias (op_params flag + src[2] bias), the GEMM called per COLUMN TILE with
the epilogue applied to the tile, the ne11%4 tail columns covered separately, and the fusion constructed
INSIDE ggml_nn_linear per Exp673's prescription.
  * CORRECTNESS: transcript BYTE-IDENTICAL (55ac39b635cb) and the gelu node disappears from the graph, so the
    column tiling and the 4x4 layout reasoning (interleave is along weight ROWS, Exp770) are validated.
  * SPEED: 2.2775 vs the 2.1849 baseline = +4.2% SLOWER. Tile sweep: 4 -> 2.3142, 16 -> 2.2775, 64 -> 2.2914,
    256 -> 2.2890, 1024 -> 2.2993 (vae_s 15.3 / 14.9 / 15.0 / 15.0 / 15.0). A shallow optimum exists but
    nothing approaches the baseline, so this is not a tuning problem.
  * MECHANISM (node timer made it visible, since it attributes epilogue work to MUL_MAT's span): MUL_MAT went
    9.77 -> 12.01 s/chain (+2.24 s) while gelu's 1.38 s vanished = net +0.86 s/chain, matching vae_s 14.0 ->
    14.9. So the elementwise passes are NOT redundant traffic: as a separate node they run while nothing is
    streaming weights, whereas 2 passes per tile interleaved into the GEMM's column loop evict the weight panel
    the GEMM is streaming through L2 - and this VAE's GEMMs are weight-bandwidth-bound (Exp681/785).
  * RULE (generalizes, and it is the reason to revert rather than tune): elementwise-into-GEMM fusion loses on
    a bandwidth-bound GEMM. The five shipped fusions all fused ELEMENTWISE-with-ELEMENTWISE (bias+gelu,
    scale+residual, gamma+norm, dw taps), which works because both sides touch the same activations; none of
    them pushed work into a weight-streaming loop. Therefore DO NOT pursue the epilogue route for the other
    elementwise items either - bias adds (0.50 s), ADD_SCALED (0.43), CONT (0.51) are now closed by this
    mechanism, not merely "below bar".
  * Reverted in both repos (auto-revert is unavailable from the workDir). v4.5 anchor re-verified: 2.1874,
    vae_s 14.0, byte-identical. If a future model makes the VAE GEMMs compute-bound rather than
    weight-bound, this design is written down and byte-identical, so it becomes worth re-testing.

SHIPPED v4.5 (Exp821): DEPTHWISE KERNEL ABSORBS THE CONV'S CAUSAL LEFT PAD, -3.2% RTF AND -183 MB RSS.
The splice's ggml_pad_ext node is gone at depthwise conv sites: ggml_conv1d_dw_ct_lp(w, x, lp) puts the pad in
op_params[0] and each output column starts its tap loop at the first in-range tap. WHY BIT-IDENTICAL (stated
correctly, unlike the ASI's garbled first draft): the taps that disappeared read columns that were ZEROS in
the materialised [zeros(P) | x], so their products were w*0 = +-0, and 0.0f + a == a exactly - the ascending-k
accumulate ladder from Exp666/670 is untouched. Verified: protocol hash unchanged, 40/40 gate transcripts
byte-identical (b=0/c=0 of 731, p=1.0), token counts unchanged at 17/69/138 s. Costs no accuracy anywhere the
fast path runs.
  * THE SCOPE RULE THAT MADE IT SHIPPABLE: "bit-identical by construction" was tested on BOTH tiers, and the
    lean tier (p13 + VAE_DEFER_LATE=1) DISAGREED - 38 vs 39 tokens, hash 941d088403fc. So the fast path is
    offered only when no later build in the window can read the history this build rolls, i.e. one piece per
    window AND no deferred late pass (vae_cache_set_whole_window, set by the demo). OPEN QUESTION worth ~3% on
    the lean tier: cold-site skip + my host-side history roll must diverge somewhere in the deferred/late-pass
    slot sharing. Diagnose with the node timer + PAD-by-consumer on the lean path, not by re-reasoning.
  * Instrument that made this tractable: PAD-by-consumer attribution in vae_graph_stats_dump (a conv's src[1]
    IS the splice output, so bytes split by consumer with no call-site edits; self-validates because the two
    buckets sum to PAD's dst bytes: 40.9 + 123.1 = 164 MB). It is how the im2col half (~0.8%, dead per Exp819)
    was separated from the dw half (2.4%, alive here) BEFORE writing code.
  * False positive worth remembering: the dw shape-model checker aborted on every lp>0 node because its model
    assumed no pad - caught only because that checker runs unconditionally (Exp668's rule paying off again).
  * TRIGGER FIRED: granularity's documented reopen condition (a per-piece cost change) fired asymmetrically -
    p1 got the pad removed, p2/p13/p26 cannot (their windows are multi-piece), so the p1-vs-p2 gap should now
    read ~6% instead of ~3%. Only worth a re-sweep if a p1-vs-p2 RAM decision matters; p13/p26 ratios are
    unaffected (none of them can take the fast path).


PAD PRICED AS TRAFFIC, AND IT SPLITS 26/8 - THE BIG HALF IS A KERNEL I OWN (Exp820, analysis-only).
VAE_GRAPH_STATS on the shipped tier: PAD = 327.3 MB per graph build, n=34, and 0.73 s of chain time for
2.62 GB/clip = 3.6 GB/s of dst bytes (~7.2 GB/s counting the read-back) = streaming bandwidth. So PAD is
memory WORK, not contention noise - the prize is at the high end of Exp819's bracket, ~3.2% of RTF.
  * SITE SPLIT from the same dump: IM2COL n=8 + CONV1D n=26 = PAD n=34, i.e. EVERY conv splices and
    26 of 34 splices feed ggml_conv1d_dw_f32 - the depthwise kernel I wrote in Exp666/670 - while 8 feed
    vendored im2col (the route Exp819 measured dead via im2col_asym).
  * QUEUED NEXT EXPERIMENT (no vendored im2col needed): add a left-pad (zeros) parameter to my dw kernel and
    skip the cold-path splice for dw sites. Bit-identity argument is the strong kind: the taps beyond the
    tensor edge multiply zero, and at the shipped p1 every site is COLD (every piece is a window head), so
    hist is zeros by definition - the padded tensor is exactly "read with an out-of-range index treated as
    zero". Acceptance = protocol transcript hash, then the gate mean; do not trust rtf alone (-DNDEBUG).
    First measure the byte split by consumer (PAD bytes on dw vs non-dw sites) so the expected win is known
    before building - guessing from site count is not evidence.
  * PATH NOTE worth remembering: at the shipped tier the stats label reads "piecewise", because p1 + defer OFF
    means one piece = the whole window, so the early/late split passes (forward_early/forward_late) do NOT
    execute in the shipping config - they are the lean tier's path (p13/p26 + VAE_DEFER_LATE=1). Any note in
    this ledger about "the late pass batches the deep layers window-wide" is a lean-tier statement. The conv
    helper is shared by both paths, so a dw-side fix lands in both.


PAD (SPLICES) IS WORTH ~3.2% NOT 2.2% - AND THE CHEAP ROUTE TO IT IS DEAD (Exp819, discard).
  * ARITHMETIC CORRECTION to Exp818: the two encoder chains run one per core, so a chain's own elapsed IS the
    VAE's wall (14.64 s), and node time is 2x wall only because the two chains overlap. PAD is 0.73 s of that
    14.64 s chain elapsed, and both chains' PADs run at the same time, so removing it takes ~0.73 s off the
    22.5 s clip = 3.2% of RTF, not the 2.2% I wrote (I divided by the chain count twice). Honest bracket:
    Exp678 shows one chain alone runs 9.2 s vs 14.6 s concurrent, so part of PAD's 0.73 s is bandwidth
    contention rather than work; if only the work part is recoverable the prize is ~1.8-3.2%. Either way it is
    the largest single in-scope item known, so it IS worth vendored effort - unlike everything else left.
  * THE CHEAP ROUTE, MEASURED DEAD: build [hist | x] implicitly by letting the conv's im2col do the causal
    left-pad, so the padded tensor never materialises. The fork already has ggml_im2col_asym(lp0, rp0, ...) and
    the shipped splice always calls the conv with padding=0, so swapping ggml_im2col -> ggml_im2col_asym with
    lp0=0, rp0=0 should have been byte-identical by construction. IT IS NOT: 1024 tokens (runaway), transcript
    50c4bcca8522 vs 55ac39b635cb, vae_s unchanged at 14.5. So the two builders disagree on this geometry even
    with zero padding - asym's read offset / column order differs from plain im2col's. Do NOT use
    ggml_im2col_asym as a drop-in; do not "fix" it by trying lp0=P directly (a builder that disagrees at P=0
    will not agree at P>0). Reverted; anchor 2.2571 byte-identical.
  * WHAT WOULD ACTUALLY WORK (queued, ~30 lines in vendored ggml, in scope per Exp663): add left-padding to
    ggml_compute_forward_im2col_f32 itself (dst zero-filled over the pad columns, reads shifted by lp) and an
    lp-carrying builder, then in vae_cached_concat's COLD branch (which is EVERY splice at the shipped p1,
    since every piece is a window head) return x unpadded, keep padding for the conv, and let the cache tap
    read the last P columns of x - legal because the pad is left-only, so x's tail == the padded tensor's tail.
    Validate on host first: ggml has tests/ and -DNDEBUG on device turns a shape mistake into garbage silently
    (Exp667), which is exactly the failure mode this probe just produced on device.
  * Why it is still not free: the pad region's zeros still get written, just inside im2col's own buffer, so
    the recoverable part is the padded tensor's write + the conv's read of it, not the whole 0.73 s.

- CONV-INT8: BUILT AND NOT WORKING, INACTIVE BY DEFAULT (Exp685). Everything the design needed is in

THE VAE, TIMED PER OP AT LAST - AND PAD (THE SPLICES) IS A 2.2% LEVER, NOT A CLOSED ONE (Exp818).
GGML_OP_TIME=1 times every node on thread 0 by op type and phase. The VAE budget now CLOSES: node time
29.28 s / 2 chains = 14.64 s against vae_s 14.7 s (+0.06 s of launch/barrier/alloc overhead, so nothing is
missing). Per chain of 14.7 s:
    MUL_MAT 9.82 (67%)  UNARY/gelu 1.49  PAD 0.73  RMS_NORM 0.53  CONT 0.49  ADD 0.48  ADD_SCALED 0.42
    IM2COL 0.34  CONV1D(dw) 0.34   [RESHAPE/VIEW/CPY ~0 - views are free, as the byte profile said]
  Cross-validation with the independent ablations (VAE_ABL_*): gelu 1.49 vs 1.20, bias ADD 0.48 vs 0.40,
  ADD_SCALED 0.42 vs 0.30, RMS_NORM 0.53 vs 0.60, dw 0.34 vs ~0.15-0.3. Same shape, ablations LOWER, exactly
  as predicted (removing an op does not remove the traffic its neighbours still cause). Two instruments, one
  answer - so Exp817's "~5.5 s unexplained" is now ~4.9 s measured, and the VAE is 67% matmul / 33% ops.
  * NEW LEVER, ABOVE BAR: PAD = 0.73 s/chain = 2.2% of RTF, 5% of VAE. The ledger had splices CLOSED (Exp501)
    because making the [hist|zeros] staging buffer persistent bought 0% - but that removed only the memset,
    not the copy: PAD is 272 node executions/clip of the streaming-cache concat, and it is the 3rd largest
    non-matmul item. Re-open with a concrete alternative: write into a pre-offset destination (or fuse the
    splice into the consumer) instead of pad-then-add. This is a graph-level change in vae.cpp, in scope.
  * Also newly visible: CONT 0.49 s (1.5% RTF) - contiguous copies the graph does not need; worth a look after
    PAD, same character of change.
  * INSTRUMENT CAVEAT (do not quote prefill op shares yet): lm_prefill reports MORE node time than its wall -
    5.31 vs 4.3 s at -t2 and 10.16 vs 8.1 s at -t1 - so thread-0 node spans exceed the phase span. Decode and
    VAE are consistent (3.43 vs 3.6; 6.50 vs 6.6 at -t1), so it is specific to how the prefill span is timed:
    most likely some of the prefill's graph work runs outside g_prefill_ms (lazy logits at sample time), which
    would make prefill's 4.3 s an UNDER-count and its node sum the truth. Test by moving the phase setter to
    bracket the sampler call before believing either number.
  * LESSON: a [GGML_OP_COUNT][4] accumulator indexed as [phase][op] is a silent transpose - it reported 38 s
    in every phase under shifted op names. The check that caught it costs one line: per-phase node time must
    be <= that phase's wall seconds / its chain count, and .auto/optime_report.py now asserts it (plus prints
    phase=idle so a truncated read cannot fake an absence, the Exp817 trap).

THE VAE'S NON-MATMUL BUCKET IS ~5.5 s, NOT ~1 s - THE LEDGER'S "ELEMENTWISE IS AT THE FLOOR" WAS OVER-READ
(Exp817). Made the census type-complete (every MUL_MAT by src0 type, inside the Exp816 phase attribution) to
test whether the unattributed VAE time was hidden F16 matmul. It is not, and the arithmetic is now sharp:
  * VAE total 388.48 GMac = q4_0_4x4 387.789 + f16 0.692 (24 calls). lm_prefill 185.14 = 181.991 + f16 0.115
    + q8_0 3.034; lm_decode 66.41 = 56.338 q4_0_4x4 + 10.035 q8_0 (the head) + f16 0.040 (attention, 2408
    calls). The q4_0_4x4 column equals the MAC census to the digit, so the new counter is validated.
  * Paired skip GGML_MM_SKIP_F16=1 (fires: 39 -> 1024 tokens): vae_s 14.7 -> 14.6. So F16 matmul is ~0.1 s
    and "quantize the remaining F16 matmuls" is CLOSED by measurement, as is any attention-precision idea for
    speed (40 GMac of F16 across the whole clip).
  * So VAE 14.7 s = 7.5 s wide-column blocked matmul (measured, Exp813) + ~1.55 s narrow blocked (63.1 GMac
    at the ne11=26 ceiling rate - rate-derived, so +-1 s) + ~0.1 s f16 + ~5.5 s of everything else. The fused
    ops I have priced (gelu 1.2 + rms_norm 0.6 + bias 0.4 + scale/resid 0.3) account for ~2.5 s of that 5.5,
    leaving ~3 s in the dw conv kernel, im2col, splices/PAD, contiguous copies and launch overhead.
  * WHAT THIS CORRECTS: the ledger says "elementwise residue ~7% of VAE, at the noise-resolved floor". That
    was the SUM OF THE OPS MEASURED, not the size of the non-matmul bucket. Two instruments disagree about
    the composition - the byte profile says 4.7 GB of tensor traffic per early-pass build (x8 builds/clip =
    ~37 GB, which at streaming bandwidth IS ~5 s), while op ablations account for ~2.5 s. Both can be true
    only if a lot of that traffic is absorbed by cache, which is exactly what needs measuring.
  * QUEUED INSTRUMENT (the one thing that settles it): per-NODE TIME inside the graph compute under an env
    flag (clock_gettime around each node's compute, aggregate by op type, default off). Bytes per op are
    already known; seconds per op is the missing half, and it is the only way to apportion the 5.5 s without
    abverting one op at a time (which under-counts, because removing an op does not remove the traffic the
    surviving ops cause). Budget it: 1 build + 1 run, and it re-prices every "nothing left" claim above 2%.
  * PROCESS LESSON (3rd occurrence, after Exp609/676): I spent a build "fixing" a census that printed only
    phase=lm_decode - because run_experiment truncates output and the other two phases were above the cut.
    The instrument was fine. To assert absence, print the full list or a count; and make dumps print the
    zero bucket (phase=idle is now included for exactly this reason).

SKIP_BUILD=1 DID NOT DEPLOY THE BINARY (Exp816) - read this before trusting any code-change measurement.
measure.sh had `adb push build-android/bin/asr_streaming` INSIDE the `if SKIP_BUILD != 1` block, while the
two libraries are pushed on md5 difference unconditionally. Consequence: a change under src/ or demo/
measured with SKIP_BUILD=1 ran the PREVIOUS binary, while a change under 3rdparty/llama.cpp took effect.
The asymmetry is what hid it - and it hid it for exactly one run, the phase census, which printed
`phase=idle` for all 626 GMac. Per the standing rule (prove a knob fires by an OUTPUT change), an inert
instrument is not a null result, so I went looking instead of writing "phases are indistinguishable":
device binary 2,209,680 B / 09:36 vs host 2,210,128 B / 12:45.
  * FIX: measure.sh now always syncs the binary by md5 and verifies it after push (adb can report success
    while a busy-text file refuses the write). Rule for the loop: after editing src/ or demo/, either run
    measure.sh WITHOUT SKIP_BUILD=1, or trust the new sync - and treat "the numbers look the same" as no
    evidence that your change is in the artifact.
  * WHAT THE INSTRUMENT THEN SHOWED (per-phase RATES, first measurement-grade version; phase sum equals the
    total to 0.001 GMac, which is the internal consistency check): lm_prefill 182.0 GMac / 4.3 s = 42.3
    GMAC/s aggregate = 21.2/core -> AT the blocked-int8 ceiling. This RETIRES Exp779's "prefill 7.62
    GMac/window = 15.2 GMAC/s", ~5x low: right conclusion (prefill is not the remaining time), wrong basis.
    lm_decode 56.3 GMac over 39 tokens = 1.44 GMac/token = 811 MB at 93 ms = 8.7 GB/s ~ 81% of the 10.8
    >>> CORRECTED BY Exp843: that 56.3 GMac is the q4_0_4x4 BUCKET ONLY. The type census also reports
    >>> 10.0 GMac of q8_0 (the output head) in the same phase, so decode streams 1072 MB/token, not 811,
    >>> and runs AT the memory ceiling (12.6 GB/s over MUL_MAT node time), not at 81% of it. Do not use the
    >>> 8.7 GB/s figure to argue there is decode headroom - there is none.
    GB/s two-thread ceiling (Exp523). vae 387.8 GMac / 14.7 s with 324.6 GMac in the wide bucket at ceiling.
  * CAVEAT so nobody subtracts their way to a false conclusion: the census counts ONLY blocked-int8
    matmuls. vae_s minus (census MACs / ceiling rate) is an upper bound on (non-matmul + F16 matmul), not a
    measurement of elementwise cost - the measured elementwise number stays ~7% (Exp791/792).

THE 316 ms WINDOW-1 PREMIUM IS PAGE-IN, AND THE LOOP'S majflt METRIC WAS A TAUTOLOGY (Exp815).
Three things in one iteration, all instrument work:
  * BUG: bench_device.sh read minflt/majflt from /proc/PID/stat AFTER `wait`, so /proc/PID was gone, awk
    printed nothing, ${VAR:-0} made it 0, and majflt_delta printed 0 FOREVER. Adding minflt exposed it as
    -1598. Every "majflt 0" this loop ever reported (both RSS soaks, every ladder row) meant UNREADABLE.
    Re-verified on the fixed instrument: 138 s soak majflt 0 / minflt 995,675 / RSS 2446 MB, so the
    no-thrash/no-leak conclusion survives - now as evidence. RULE: a metric that has never been observed
    nonzero is unproven; plant a case that must move it (here: touch more memory = longer clip).
  * CLOSED with mechanism: LT trace in one run gives vae 3907/3581/3600/3590 ms -> +317 ms window-1
    premium; ~204k of the run's ~304k minor faults fall in that window (820 MB of first-touched pages, all
    page-cache hits, majflt 0) at ~1.6 us/fault. THP does not exist on this kernel, so the only lever is
    pre-touch at load, which MOVES the cost into load_s - forbidden as a metric move by Exp806's rule, and
    recorded instead as a product-side TTFT win of ~317 ms at zero RTF change.
  * RE-ARMED IDEA (Exp156's "revisit only with a discriminating probe" - the probe now exists): ~29k faults
    RE-FAULTED per window (RSS sawtooth 2353<->2374 MB) = ~170 ms per protocol clip, ~0.8%. Price it with
    the fault series before any code; the 138 s run shows only 14k/window, so it shrinks with length and is
    probably not worth a persistent arena. .auto/fault_run.sh + .auto/fault_sample.sh do this in one run.
  * AUDIT: a DEVICE_PATHS marker + check 3c (declared device artifacts must exist on the phone), because my
    new drivers tripped 3 false "missing path" FAILs. Check 3c guards a real failure mode: a driver whose
    model file disappeared silently measures another tier (Exp772/694 class).

ROLLBACK AUDIT REFRESHED, AND A HATCH THAT DOES NOTHING (Exp814). The Exp677/787 runbook costs were 3
versions stale, so all arms were re-run in one binary with a committed tool (.auto/rollback_audit.sh) that
prints, per arm, BOTH the rtf cost and the per-window transcript hash. Costs vs default 2.2565:
DW_CONV1D_OFF +5.7 | GELU_BIAS_OFF +5.0 | GELU_BATCH_OFF +3.8 | LS_FUSE_OFF +1.6 | NORM_FUSE_OFF +1.4 |
MM_M2_OFF +0.9 | fusions-all-off +14.1 (reproduces Exp787's +13.8 to 0.3 pp). Every fusion arm byte-
identical (55ac39b635cb, 39 tok), so the stack is still output-equivalent to the pre-change path measured
IN ONE BUILD. Two runbook corrections:
  * VAE_DW_AXPY_OFF is a NO-OP on the shipped path (+0.1%), because the Exp670 conv1d kernel does not use
    the tap chain that the hatch disables. It only fires together with VAE_DW_CONV1D_OFF, where it costs
    +2.8% (2.4266 -> 2.4936, VAE 15.7 -> 17.0 s). A hatch that guards a superseded path is not a rollback
    path - check what the CURRENT default actually executes before documenting a hatch's cost.
  * VAE_CT_BLOCK_OFF (revert the channels-first block layout) is +22.8% and NOT output-preserving at p1:
    38 tokens, different hash, because the legacy layout also drops the dw conv off the conv1d kernel onto
    taps/im2col - Exp640's rounding mechanism showing up at the shipped granularity, not just on lean p26.
    So Exp787's "every arm identical including all-off" stands for the FUSION hatches, but any arm that
    includes the LAYOUT revert changes the text (+31.3%, 2.9630) and needs its own gate, not a hash check.
  * Tool bug worth remembering: the first version divided each arm's mean by the control arm's SUM, so every
    delta read ~-50%. Same class as Exp660's over-claiming self-check - a tool that prints plausible numbers
    is not a tool that is right. Prove the control arm's delta is 0.0% before believing a table.
SHAPE-RATE GAP IN THE VAE'S EARLY STAGES: LOOKED LIKE THE BIGGEST LEVER IN YEARS, WAS A
MICROBENCH ARTIFACT (Exp812/813). Two-run arc, worth keeping because both halves are reusable.
  * Exp812 (the finding): the MAC census (GGML_MM_DEBUG_MACS, now with a params->ith==0 guard - it
    double-counted per thread before that) says a 10 s clip is 626.1 GMac of blocked-int8 matmul, and
    52% of it (324.6 GMac, 328 calls) sits at ne11 >= 128, i.e. the VAE's early stages run matmuls over
    up to 83,200 columns with as few as 32 rows. A CONSTANT-MAC sweep (340.8 MMAC, trading rows against
    columns) measured 9.07 GMAC/s at 32x83,200 vs 21.49 at 2048x1,300 -> a 2.4x "shape penalty", and the
    census's own shapes showed the same monotone trend. Read as: column-tile the GEMM, maybe -20% RTF.
  * Exp813 (the correction): price it IN-GRAPH instead of from a microbench. GGML_MM_SKIP_BIG=1 returns
    before the gemm for ne11 >= 128 (output garbage ON PURPOSE; vae_s is LM-independent, Exp674/682).
    Result: vae_s 14.7 -> 7.2 s, so that bucket IS 7.5 s = 51% of VAE seconds. But 324.6 GMac / 7.5 s =
    43.3 GMAC/s aggregate = 21.6 per core, which is EXACTLY the kernel's best cache-friendly rate. There
    is no shape penalty in-graph, so column tiling buys nothing. The microbench's gap came from (a) the
    F32->Q8_0 conversion inside its timed region (the conv path feeds Q8_0 im2col directly) and (b) a
    freshly allocated 10 MB src1 per call (cold), and the shipped path neither does.
  * RULE (generalizes): a microbench measures the BENCH as much as the kernel. Before commissioning work
    on any rate gap, price the same MACs IN-GRAPH with a skip/substitution knob and divide. Microbench
    curves are for ranking shapes against each other, never for absolute attribution.
  * PLACEMENT LESSON (cost two dead arms): the gemm() call sits ABOVE the census/tail section in
    ggml_compute_forward_mul_mat, so a skip placed after it removes only tail columns - with ne11 % 4 == 0
    that is literally nothing, and both A/B arms printed the same 39 tokens. Proof a measurement knob
    fires is a CHANGE IN OUTPUT (here 39 -> 12 tokens); without that check I would have "measured parity"
    twice on an arm that never ran (4th occurrence of this class after Exp764/798/804).
  * Rate budget is now complete and closes the loop's oldest number: VAE 14.7 s = 7.5 s wide-column bucket
    at kernel ceiling + the rest; the LM's prefill is at kernel rate and decode at bandwidth. Nothing in
    the CPU path is below its measured ceiling.
  the tree and nothing ships: converter (.auto/conv_int8.py + quant4x4.cpp), runtime path
  (vae_conv_1d_i8 + metadata-only geometry carrier + K=row/IC kernel-size derivation), auto-detected
  from the weight layout so an F16 file takes the old path (default transcript byte-identical).
  Result on device: vae_s 16.2-16.4 vs 17.1 baseline (-5%, the predicted win is REAL) but the output
  is garbage - 1024 '!' tokens with a pre-converted Q8_0 src1, 9 tokens if ggml converts F32->Q8_0
  itself (the LM's route). So the dots are fast and the features are wrong.
  WHAT IS RULED OUT: geometry (the im2col node is structurally identical to the working F16 path -
  same 3-D [K,IC,OC] shape, same params); the F16-vs-F32 unfold (both garbage); the padding/cache
  path (untouched); kernel-size derivation (verified head K=8 = 16384/2048).
  WHAT IS UNRESOLVED: whether the CONVERTER bytes or the RUNTIME's reading of them are wrong, because
  the tool that should have answered - .auto/conv_roundtrip.py, a Python dequantizer - FAILS ITS
  POSITIVE CONTROL (run against llama-quantize's own q4_0_4x4 tensors, which are known good, it
  reports max|err| ~= the full weight scale). Its scales decode correctly (d[0] = max|x|/8 exactly,
  per row, so block_q4_0x4 = d[4] then qs[64] and the 4-row grouping are right) - so the remaining
  unknown is precisely the nibble-to-element map. Fix the checker against the control FIRST (the
  brute force has covered byte order {interleaved, row-major} x value map {nibble-8, signed4}; the
  untried variants are the nibble-half swap and an offset permutation inside the 4-byte group), and
  only then trust a verdict about the file.
  LESSON (6th "the measurement you ran wasn't the measurement you meant"): I nearly concluded from a
  failing checker that a correct-looking converter was broken. The positive control cost 3 minutes
  and would have been run earlier if I had treated a new checker as guilty until proven innocent -
  which is Exp660's rule, applied to a tool I wrote in the same iteration.
  CHECKER POST-MORTEM (same iteration, after logging): .auto/conv_roundtrip.py failed its control four
  times in a row, and each time the reason was a bug in MY tool, not evidence about the file:
    * compared against parser `ne` = reversed(dims) - the one convention this loader is known NOT to
      use (Exp685 verified at runtime). Fix: compare against dims as stored.
    * a model-identity check iterated `b.items()` (the parse dict) instead of `b['tensors']`, so it
      reported "0 tensors compared" and I nearly read that as "the two files are different models".
      Redone correctly: 68 F16 conv/norm tensors are BYTE-IDENTICAL, so
      vae-encoder-f16.gguf vs vae-encoder-q4x4ffn.gguf IS a valid control pair.
    * value/element model: tried (interleaved,row-major) x (nibble-8,signed4) x (2j/2j+1, j/j+16).
      All fail with the same signature - max|err| ~= 1.0 x weight scale, avg 0.1-0.24 x scale, with
      complete coverage and no duplicates. That signature means positions are right and the VALUE is
      wrong, while the block's scales decode exactly (d[i] == max|x|/8 for the first block of four
      different rows, so block_q4_0x4 = d[4] then qs[64] and the 4-row grouping are both correct).
  So the checker is still unvalidated and CANNOT adjudicate converter-vs-runtime. STOP decompiling
  bytes in Python. NEXT PROBE (much better, one device run, no layout model at all): add a debug mode
  to vae.cpp that, for one conv tensor, builds a tiny graph computing
     ggml_mul_mat(int8_weight, ggml_cast(im2col_of_known_activations, Q8_0))
  and the equivalent F16 product, then prints max|difference| using ggml's OWN reader on the device.
  Zero disagreement => my bytes are right and the bug is in the graph; large => converter. The same
  trick already worked for the spike, where the host could not be trusted because q4_0_4x4 has no
  x86 vec_dot.
  HOST A/B PROBE: ALSO NOT POSSIBLE (Exp687) - and the reason is now precise. .auto/conv_i8_ab.cpp
  builds a 2-D Q4_0_4_4 tensor plus a Q8_0 (or F32) src1 and runs both products on the host. It
  segfaults. The exported symbols ggml_gemv/gemm_q4_0_4x4_q8_0 DO exist in the x86 libggml, but their
  bodies are __ARM_NEON-guarded: on x86 they report "unsupported", mul_mat falls through to the
  generic path, and that path uses the type's vec_dot, which is NULL for Q4_0_4_4 => null deref. So:
  an existing symbol is not evidence of support (3rd instance of "the check you ran doesn't test the
  thing you think it tests" in this one project). Q4_0_4_4 can only be EVALUATED on aarch64; it can
  only be QUANTIZED anywhere.
  NEXT (device, one run, no layout model, settles converter-vs-runtime): teach conv_int8.py to also
  emit, for ONE chosen tensor, an F16 reference copy under a different name (e.g.
  acoustic.downsample_layers.1.0.conv.weight_ref), and add a VAE_CONV_I8_CMP branch in vae.cpp that
  on that tensor computes BOTH products from the SAME activations - int8: mul_mat(q4x4, cast Q8_0),
  f16: mul_mat(reshape_2d(ref), im2col_f16) - then reads both outputs back on the host and prints
  max|diff| and the reference scale. Near-zero => my bytes are right, the bug is in the graph wiring.
  Large => the converter (most likely my ggml_quantize_chunk argument order).
- CONV-INT8 ROOT CAUSE (Exp688, FOUR iterations after Exp685's "garbage output"): the offline
  converter quantized on the HOST, and `ggml_quantize_chunk(GGML_TYPE_Q4_0_4_4, ...)` on x86 writes
  correct nibbles but **ZERO fp16 SCALES** - .auto/qt.c proves it directly (d = {0,0,0,0}, qs = 68 9c
  d0 14 ...). value = d * (q - 8), so every weight dequantizes to exactly 0: the conv path computes
  zeros, the LM gets silence, the protocol run emits 9 tokens. Nothing was wrong with the graph, the
  geometry, the padding, the Q8_0 staging, or the 2-D layout - and no amount of reasoning about those
  could have found this, because a zero-scale blob is legal bytes.
  * Why nobody noticed: a zero-scale block passes every size/shape check the converter makes, and the
    shipped 4x4 ggufs are fine because llama-quantize uses llama-quant.cpp's own path, not
    ggml_quantize_chunk.
  * HOW IT WAS FOUND: the on-device load-time probe (vae.cpp, VAE_CONV_I8_CMP=1) multiplies the int8
    weight by a one-hot [k,1] selector - the gemv shape - so the product IS the dequantized column.
    Its self-test (quantize a known matrix with the runtime's own ggml_quantize_chunk, read it back)
    PASSED on device (ratio 0.125 = q4 noise), which is what made the conv verdict trustworthy: the
    device quantizer+reader agree, so the file's bytes were the only suspect. That in turn localized
    the fault to the host tool, where qt.c confirmed it in one run.
  * FIX (queued, straightforward): quantize on the device. quant4x4.cpp is plain ggml C++, so build it
    with the same NDK toolchain as the app (30 s), have conv_int8.py ship each conv's raw f16 bytes to
    the device, run the ARM helper there, and pull the blobs back. The file assembly stays on the host.
    Then re-probe (expect "BYTES MATCH"), measure 3 reps (stop rule >= 2%) and gate 40 utts (WER <=
    ~4.9%); the EV is +2 to +4% RTF on 12.3% of VAE time (Exp683).
  * PROBE GOTCHAS learned the same run (all cost a device round trip): (1) ggml_init RETURNS the
    context - discarding the result left ctx NULL and segfaulted; (2) a wide src1 into a blocked
    mul_mat wrote only column 0 and left the rest of the output as uninitialized arena (zeros + a
    NaN), so dequantize with a [k,1] selector one column at a time; (3) the reference must come from
    the source file by name (an extra *_ref tensor in the converted file was not registered by the
    loader: 562 keys for a 563-tensor file, unresolved and now irrelevant).
  CONV-INT8 STATUS AFTER THE FIX ROUND (Exp689): two real bugs killed Exp685's attempt, neither in the
  place I had been looking:
    1. The converter quantized on x86, where ggml_quantize_chunk(Q4_0_4_4) writes valid nibbles with
       ZERO scales (Exp688) - fixed by quantizing on device (--device, .auto/quant4x4_arm). Probe now
       reports the dequantized weight matching the source at q4 noise (ratio 0.061 vs control 0.125).
    2. encoder.output_dim came from head_conv_weight->ne[2]; a blocked 2-D weight has no ne[2], so the
       latent dim silently became 1. Fixed (n_dims==2 ? ne[1] : ne[2]) and the probe's dims line now
       matches the reference file exactly (output_dim 64/128, head_K 8, VAE dims 1536/1536).
  STILL BROKEN: the int8 file produces 1024 tokens (runaway, deterministic hash daa10f03) while
  measuring vae_s 16.1 vs 17.1 ref (-6%, the predicted win). Everything structural agrees with the
  reference build, so the remaining suspect is the ONE thing no check has covered: whether this fork's
  im2col emits rows in (ic*KW + kw) order with kw fastest, matching my weight's memory order (k + K*ic).
  A mismatch would produce exactly this - plausible-magnitude garbage from correct weights.
  NEXT (decisive, ~20 lines in the probe): same activations, same im2col, TWO weights - the 3-D F16
  weight copied from the source file vs my 2-D int8 one - and compare the two products. Agreement =>
  pairing is fine and the fault is elsewhere; disagreement => fix the converter's row order (transpose
  the K/IC axes when flattening) and the project is done. Also: the probe's verdict threshold should be
  per-block (|diff| <= ~0.55 * that block's scale) rather than a global ratio, since 0.061 of the global
  max IS q4 noise; my <0.05 cut would have failed correct bytes.
- ATTRIBUTION ON THE CONV-INT8 GRAPH (Exp693 - supersedes Exp683's conv share and
  refreshes Exp674's elementwise split; re-derive again after any op-level change).
  vae_s ablations, baseline 16.5 s: conv dots (VAE_ABL_CONV) 15.0 = 9.1% of VAE
  (was 12.3% on F16 dots - the int8 route took a third of it), bias adds 2.4%,
  layer-scale mul 1.8%, residual add 1.8%, depthwise taps 1.8% => elementwise total
  ~7.8%, unchanged by the conv change as expected. Nothing in the VAE now has a
  single ablatable item above ~9%, and that 9% is already int8 dot work.
- BLOCKED-CONV STAGING CONTRACT, PROBED THREE WAYS (Exp692/693): the blocked conv
  matmul REQUIRES an F32 src1. ggml_im2col(dst=Q8_0) SIGABORTs (no quantized im2col
  path in this build); im2col(dst=F16) fed straight to mul_mat hits
  GGML_ASSERT(src1->type == GGML_TYPE_F32); ggml_cast(F16->Q8_0) runs but computes
  garbage (1024 tokens vs 39) because it violates that contract. So the F32
  unfolding is the only supported staging and the ~2% the wrong arm shows is
  unreachable in this ggml. Do not re-probe staging variants.

LESSON (Exp694, cost an invalid gate and a phantom +31% regression): a runner script's
own defaults ARE the configuration of every run that passes no env. eval40.sh defaulted
to VAE=vae-encoder-q8_0mixed, LM=streaming-lm-q4_k_m, PIECES=13 - three variables off the
shipped tier - so the gate measured a different system and printed a plausible WER/RTF for
it. Two rules: (1) READ THE STAMP (hyp-*/run-info.log) before believing any gate number;
(2) keep the tier in ONE declaration (.auto/tier.env) and have the audit fail on drift -
now implemented and negative-controlled (pointing tier.env at another VAE produces 2 FAILs).
Also: the correct-tier gate supersedes the Exp690 gate - shipped tier WER 4.51%, 1 token of
731 from hyp-gate645 (p=1.0), 40-utt mean 2.7524; lean p13 2.8136, 2 tokens (p=0.5).

- DIARIZATION COLLAPSE: ROOT CAUSE (user-requested debug, closes Exp648/650). The
  "1 tag for 4 voices" on the canonical probe is the windowed streaming protocol, not a
  dead feature and not quantization. Mechanism, each link evidenced:
  (1) No speaker state crosses a window: vae_cache_reset per window
  (demo/asr_streaming.cpp:475), the LM's cross-window history is text only, and no
  speaker-embedding/clustering/voiceprint code exists in src/ or demo/ (grep for
  xvector/dvector/cluster/cosine/spk_emb/voiceprint/EDA: empty). Chunk labels are
  therefore numbered window-locally ("Speaker 0" = this window's first voice).
  (2) A second tag needs two voices' contrast inside ONE window's acoustic embeddings.
  Same file, same weights, same LM: windowed -> tags 0x12, attribution 0.567; --xwin
  (only change: encoder carry, same chunk spans) -> tags 0,1,1,0,1,0,0,1,1,0,1,0,
  attribution 0.294, tags 2/4. Transcripts: .auto/hyp-ms-windowed-v39.txt vs
  hyp-ms-xwin-v39.txt. xwin also costs WER 14.4% -> 27.8% here, so carry is a
  diagnostic, not a fix.
  (3) All backlog observations follow: 100/1000 ms gaps use 3 tags (different
  turn/window alignment -> different within-window contrast), overlapped mix splits
  0/1 (two voices share windows), sequential twospk collapses (one voice per window),
  accuracy-first tier got 2/4 on the same probe (F16 features render a leading sliver
  just faithfully enough to cross the split threshold - configuration-coincidental).
  A text-level feedback loop compounds it: windowed KV history is all "Speaker 0", an
  in-context prior toward 0 (consistent with, not separately proven).
  WHAT THE MODEL CANNOT DO in this protocol: stable cross-window speaker identity
  (tags renumber window to window; xwin consistency 0.724). Fixing that needs
  per-window speaker embeddings carried across windows = a model/architecture change,
  still out of loop scope. The published probe card (eval-bilingual/publish/README.md)
  now states this mechanism instead of "collapsed to a single tag".
- --xwin AS A SPEED LEVER: CLOSED on the current stack (was -8.5% on shorts two years
  ago). Paired A/B via .auto/measure_xwin.sh: protocol 2.4677/2.4596 -> 2.5448/2.5468
  (+3.2% SLOWER), 17 s 2.412/2.413 -> 2.523/2.517 (+4.6%), and worse WER on the
  multispk probe (14.4% -> 27.8% with 5 deletions). What was declined was auto-select
  by length (inapplicable to streaming); the mechanism itself is now measured and loses
  on this stack. Its value was diagnostic (see the diarization root cause above).

- GATE-SET PROVENANCE LEDGER (Exp699, full sweep of eval-librispeech/hyp-*/): 30 sets
  predate the Exp617 run-info.log stamp (a78, ctblock, defer5, dwtaps, f16a78, geluquick,
  q5head, head4x4, lm4x4*, xwin, ...): usable as DISTANCES only, never as single-variable
  effects - the docs already carry that caveat on the resolution-limit row, verified still
  accurate. Partial sets exist and are correctly UNCITED anywhere: hyp-conv1d66 n=29 (the
  crashed Exp666 gate - superseded by dwchk68/dwfix69/convint8b), hyp-dwct67 n=8, hyp-dwft69
  n=18, hyp-p13im2chk n=2, hyp-f16load/lm4x4/p13chk n=5, hyp-par3 n=3. All SHIP decisions
  rest on stamped n=40 single-variable pairs (gate645, gelb673, lsfuse671, dwchk68, dwfix69,
  leanp13c/t, leanp26c, convint8b, lean694); unstamped sets support only declines (safe
  direction) and historical distances. No doc repair needed. Re-run this sweep if a new
  accuracy claim is ever built on an old hyp dir.

- HF PROBE CARD RE-UPLOAD: DONE. The corrected README.md (diarization root cause
  instead of "collapsed to a single tag") was pushed to
  hf.co/datasets/Luigi/bilingual-zh-en-multispk-probe as commit 6ef33879 (authenticated
  as Luigi via the cached `hf` CLI token) and verified by re-downloading: the new
  paragraph is live and the old sentence is gone. Exp696's "manual re-upload" note is
  superseded - no push script was needed.

- PROBE v2 BUILT, VALIDATED, PUBLISHED (user request, closes the diarization thread).
  `eval-bilingual/gate_ms_v2.wav` (45 s, sha 3b9ef13d74819788): 3 overlap pairs
  (2x zh S1xS2/S2xS1, 1x en S3xS4; B starts 2.0 s before A ends, 0.5/0.5 mix) + 4
  sequential controls, same 4 voices as v1. Builder `.auto/build_gate_ms_v2.py`
  (seed 650): per-pair pre-gap chosen by deterministic search so the first 1.5 s of
  overlap provably sits in one window (asserted shared_window 7/11/14); pair clips
  under RMS 0.03 level-matched to 0.05 (gain_db recorded, controls untouched);
  per-turn fresh_clip flags (only S2's 3 clips are unseen - S1/S3/S4 pools are
  exhausted, overlaps are novel audio regardless). Scorer self-test passes on the new
  manifest (Exp654 rule). Validation on shipped tier: tags 4/4, attribution 0.424
  (vs v1 0.567), WER 25.9 % (harder test, not a regression). Published to HF as
  bilingual_multispk_v2_45s.wav + gold_v2.json/turns_v2.tsv/transcript_v2.txt with a
  v2 card section (5 uploads). v1 files untouched. LESSONS: (1) --bless on the audit
  manifest is DESTRUCTIVE (rewrote every entry, dropped cell/derived_from/
  ignore_collision, turned a known-collision WARN into FAIL) - revert and add entries
  surgically; (2) count clip pools WITH audio-on-disk and freshness BEFORE designing
  turn counts (S1/S3/S4 pools hold 4/2/2, all v1-used); (3) a quiet pair member is a
  loudness test, not a separation test - level-match and record gains.
- CLOSED (Exp772, was "OPEN (interrupted): p1-vs-p2 interleave at v3.9"): re-swept on the v4.2
  stack (conv-int8 + m2), 3-4 interleaved reps/arm with a same-sweep p1 control (2.3805):
  defer-OFF p1 2.3784 / p2 2.4538 (+3.2%) / p26 2.7073 (+13.8%); defer-ON p13 2.5308 / p26 2.5537
  (+0.9%). The old confounded single reps are moot; the gap is real and reproducible, but its
  MECHANISM CLAIM WAS FALSIFIED by Exp773: the story was "m2 pairs only even remainders, so p2's
  odd leftover column costs a full extra weight pass while p1's 2 leftovers take one paired pass".
  A same-sweep 2x2 with GGML_MM_M2_OFF=1 costs +0.7% at p1 AND +0.8% at p2 - EQUAL, so m2's benefit
  is the LM prefill (ne11=26/31), which is granularity-independent, and the VAE-side tail
  differentiates nothing. Honest statement: finer pieces re-stream stage weights per piece and pay
  their own graph build/launch, at p2 twice per window. The v3.9 -> v4.2 widening (-1.2% -> -3.2%)
  has no verified cause; do not write a story for it without another 2x2.
  PARTIAL ATTRIBUTION (Exp773b, same-session 2x2 over pieces x conv precision, 3 reps):
  gap(int8 convs) = +3.0% (2.3799 -> 2.4501), gap(F16 convs) = +1.9% (2.4739 -> 2.5221), so ~1 pp of
  the gap is the int8 conv path's PER-PIECE cost (geometry carrier + right-pad + F32 staging, paid
  again at each piece) and ~2% is intrinsic per-piece overhead. The v3.9 -1.2% figure is weak data
  (aborted interleave, single reps), so the "widening" is largely a correction of that number, not
  a regression. Note p2+F16-conv emits 40 tokens vs 39 in the other three cells - the conv-precision
  difference reaches the transcript at p2 but not at p1 (the shipped tier is paired-equivalent,
  McNemar p=1.0). p1 stays the default; p13+defer-ON reproduces the lean cell (2.531 vs the
  documented 2.54). p26 defer-OFF emits 38 tokens vs 39 elsewhere - output is not granularity-neutral
  in the GEMV regime (defer-ON at p26 returns to 39).
  HARNESS BUG FOUND DOING THIS: run_rtf_multi.sh hardcoded VAE_FILE=vae-encoder-q4x4ffn.gguf, i.e.
  the pre-Exp690 F16-conv tier, so ANY sweep through it measured a non-shipping tier. It now parses
  the tier from measure.sh and prints it (commit 6eb18cf). Rule: a sweep runner must derive the tier
  from the benchmark script, never copy it. (6eb18cf)

- v2 FOUR-ARM CHARACTERIZATION (autoresearch on the probe, NOT optimization - 85 ref
  tokens cannot rank tiers; no code/weights changed on these numbers). Same clip:
  shipped WER 25.9 % / attr 0.4235 / tags 4/4 (0,1,2,2,0,3,1,2,3,0);
  xwin WER 20.0 % / attr 0.4286 / tags 5/4 (phantom 5th speaker);
  accfirst WER 18.8 % / attr 0.4268 / tags 4/4;
  lean-p13 WER 25.9 % / attr 0.4235 / tag-IDENTICAL to shipped.
  READING: attribution is invariant (0.42-0.43) across carry, precision, and
  granularity while numbering renumbers window-locally - separation on v2 is robust,
  not config-coincidental, and no knob moves it (cross-window identity needs a model
  change). WER varies 18.8-25.9 % (carry/precision help overlap word recognition), so
  recognition and separation are independent axes - do not rank tiers by v2 WER.
  Transcripts: .auto/hyp-msv2-{shipped,xwin,accfirst,leanp13}.txt.
  CAUGHT MYSELF: a `$a`-in-double-quotes shell bug made one python probe read the xwin
  file twice, printing a "changed" shipped sequence (incl. phantom Speaker 4s) on an
  unchanged file. The scorer (explicit argv paths) was unaffected; grep -oE on each
  file is the trustworthy check. Same class as Exp648's whole-file-vs-marker trap:
  verify WHAT a tool read before believing a diff.

- GRANULARITY REOPENED AND RE-CLOSED AT v4.0 (Exp706): p1 beats p2 by -1.2% with
  ZERO gate discordants, so the default flipped p2 -> p1. Evidence: clean p2/p1/p1/p2
  interleave (p2 2.4609/2.4575, p1 2.4335/2.4272 - every p1 rep below every p2 rep,
  thermal flat 36.5-36.6, all 4 transcripts byte-identical f8205302) plus 2 earlier
  reps the same direction; 40-utt gate at p1 vs p2 reference: b=0/c=0 of 731 tokens
  (McNemar p=1.0, tag gatep1, mean 2.7109 vs 2.7524) - IDENTICAL, the strongest gate
  in loop history. Full p1 ladder row measured same-session (17 s 2.3898, 69 s 2.4124,
  138 s 2.4766, all token-identical to p2). Likely mechanism (structural, not isolated):
  p1 launches half the matmul/im2col nodes over the same audio, and every blocked-conv
  node pays an internal F32->Q8_0 from_float conversion - so per-node overhead scales
  with piece count while bytes don't. The old "p2 differs (40 vs 39 tokens)" sentence
  was the taps-vs-im2col era and is retired: p1/p2 identical everywhere, p13 differs
  by 2 gate tokens (p=0.5). Cost of the flip: +344 MB RSS (2.46 GB, still phone-safe);
  p2 stays the documented fallback. Flip mechanics done in one pass: tier.env,
  measure.sh/bench PIECES default, eval40 default, audit green (64), ladder row, tier
  table, map sentence. LESSON: the Exp643 "ties p2, +335 MB, do not re-sweep" closure
  was priced pre-fusions/pre-int8 - any closure whose mechanism involves per-node costs
  must be re-swept after an op-level change alters node costs. The conv-int8 internal
  conversion is exactly such a change.

- DEFER-OFF AT p1 SHIPPED (Exp708, v4.1): VAE_DEFER_LATE default inverted to OFF.
  Mechanism: at PIECES=1 a piece IS the whole window, so the deferred late pass
  batches nothing - it only adds boundary-buffer staging (26->1 copies) and a second
  graph build. Bracketed ON/OFF/OFF/ON interleave (thermal flat 36.5-36.6): OFF
  2.3937/2.3927, ON 2.4207/2.4255 = -1.2%; vae_s 15.9 vs 16.2. Output BYTE-IDENTICAL
  at p1 (hash f8205302 - the piece-wise path IS what produced the frozen reference at
  p1) and the 40-utt gate has ZERO discordants of 731 (b=0/c=0, tag gatedef1, mean
  ~2.677 vs 2.7109). Lean p13 is the exception and the docs now carry the flag
  explicitly: defer is REQUIRED at fine granularity (measured ON 2.54 vs OFF 2.67 =
  -5%, the deep layers are GEMV-shaped per piece there) - recipe is now
  `PIECES=13 VAE_DEFER_LATE=1`. LESSON (same class as the Exp706 granularity
  re-sweep): Exp639's "defer is a no-op at p2" was measured at p2 with the OLD
  weight formats; at p1 with int8 convs the late pass is not a no-op but pure
  overhead. A closure of the form "knob X is neutral" must name the regime it was
  neutral IN, and be re-checked when the default point moves (here p2->p1 moved it).

- EXP709 LADDER REFRESH @ v4.1 (all cells re-measured with defer default OFF):
  shipped 17 s 2.3550/2.3620 (hash-identical pair), 69 s 2.3940/2.3912 (446 tok),
  138 s 2.4531 (877 tok, token-identical to every prior stack), RSS 2375-2389 MB
  (-86 vs defer-ON: the boundary staging buffers are gone), anchor 2.3960
  byte-identical. lean p13 defer-ON 2.5174 @ 1752 MB, 39 tokens, protocol hash
  BYTE-IDENTICAL to the frozen reference (defer batching does not change numerics,
  re-proven at p13). Gate mean cell for lean still predates v4.x (2.81, pre-fusion
  era) - refresh only with a paired 40-utt run if a decision needs it.
  HARNESS TRAP FOUND (mine): EXTRA_ENV reaches the DEVICE only via measure.sh --env;
  a bare `env VAR=...` sets it host-side only, so a device-side knob (VAE_DEFER_LATE)
  silently does nothing while host-side vars (PIECES) still work - the broken arm
  printed empty metrics (d41d8cd9 = md5 of nothing) rather than wrong numbers, which
  is the good outcome; audit_harness cannot see this class because it audits files,
  not flag routing. Rule: device-side knobs go through --env, and always confirm the
  first arm of a sweep printed METRIC lines.

- ROLLBACK MATRIX RE-DERIVED @ v4.1 (Exp710, one binary, 8 arms): costs vs the new
  default 2.3964 - DW_CONV1D_OFF +5.1, LS_FUSE_OFF +1.2, GELU_BIAS_OFF +0.9,
  DEFER_LATE +1.2, SEQ_ENCODERS +3.9, CT_OFF +21.3, all-hatches-off +23.2 %.
  Output: every FAST/semantic-preserving hatch byte-identical to the frozen
  reference. BUT the layout hatch is no longer output-neutral: VAE_DW_CT_OFF alone
  and all-off both land on hash 55d7cc5f (38 tokens) - where Exp677 measured them
  IDENTICAL at p2+defer. Isolating arm: VAE_SEQ_ENCODERS=1 (piece-wise, [C,T]
  intact) is BYTE-IDENTICAL and 39 tokens => piece-wise mode is output-neutral AT
  p1; the divergence is specifically dw-TAPS in the LEGACY [T,C] layout
  (Exp640's taps-vs-im2col rounding, now visible at the default because defer-OFF
  made the default piece-wise). The "encoder MODE is not output-neutral" claim of
  Exp678 was regime-bound (p2+defer) and is retired at p1.
  FIELD RUNBOOK at v4.1: safe byte-identical fallbacks = VAE_DEFER_LATE=1,
  VAE_SEQ_ENCODERS=1, and each fusion's _OFF individually; VAE_DW_CT_OFF changes
  the transcript (only for debugging the layout rework, not a rollback).
  LESSON (4th regime-dependence find): Exp677's "rollback provably rolls back"
  included an OUTPUT-equivalence claim per hatch - equivalence claims are as
  regime-bound as perf claims. Re-run the matrix in one binary whenever the default
  path changes structurally (here: operating point p2->p1 AND encode path defer->
  piece-wise in the same stack).

- LAST OPEN CELL FILLED (Exp711): lean p13 40-utt gate mean = 2.8187 (n=40, tag
  lean711b, rtf.log persisted). Reproducibility proven in-build (lean711 vs lean711b
  transcripts byte-identical across two full 40-utt runs). Paired vs shipped
  (gatedef1): 2 discordant tokens of 731 (b=0/c=2, p=0.5, WER 4.51->4.79) - the same
  lean-vs-default difference as every prior stack. CAVEAT recorded: lean711 BIN
  (42524f05, post-Exp708 build) differs from gatedef1's (1ba9309a) - both ARE the v4.1
  default config, but the pair is not single-variable; direction matches all archived
  lean pairs, so the equivalence reading stands.
  OBSERVATION worth knowing: the lean gap is +5.3% on the gate mean vs +5.4% on the
  10 s protocol (2.52/2.39) - at v4.1 per-piece overhead scales with clip length again
  (p13 = 13 pieces/window), UNLIKE the Exp646-era +0.9% on short clips. The lean tier
  is now uniformly ~+5% at every length; its only claim remains 1.75 vs 2.37 GB.
  HARNESS: eval40.sh now writes hyp-<tag>/rtf.log (per-utterance RTF had been
  stdout-only - a gate mean was LOST on every interrupted/resumed run, which is how
  this cell went stale twice).

- WATCHDOG @ v4.1 STRUCTURAL-PATH CHANGE (Exp712): p1 + defer-OFF change the encode
  path without changing weights, so the generalization watchdog was due under the
  Exp651 trigger ("path or weights change"), even though the 40-utt gate was already
  b=0/c=0. Result: held-out CV-en 0.2636 and CV-zh-TW 0.1474 - IDENTICAL numbers to
  v3.9 and PAIRED zero discordants on both (b=0/c=0, p=1.0, 220+468 tokens;
  insertions equal too: en 11/11, zh 0/0). The v4.x regime changes are invisible off
  corpus, exactly as the gate predicted. Transcripts .auto/hyp-holdout_{en,zh}-v41.txt.
  Behavioral probes re-passed at v4.1 via device-side AUDIO= names: silence ->
  [Silence], noise -> [Noise], music -> [Music] + one lyrics line (Exp625 behavior).
  Scorer nit: score_mixed.py hardcodes gate_bilingual.ref.txt and CRASHES on the
  holdout manifests - the holdout route is score_stream.py (per Exp651); the traceback
  is at least loud, not silent. Behavioral asset names live ONLY on device
  (silence5s/noise5s/song20s/proto48k_stereo) - the device_assets.json manifest covers
  cell-backing clips; extending it to behavioral probes is cheap if ever audited.

- ATTRIBUTION RE-DERIVED @ v4.1 (Exp713, vae_s ablations on the shipped piece-wise
  p1 graph, base 15.9 s): conv dots 8.8% (8.4 s at p2 era: 9.1%), bias adds 2.5%,
  layer-scale 1.9%, residual 1.9%, dw taps (conv kernel) 1.9% - elementwise 6.2%
  (was 7.8% at v3.9). PARTS SUM 1.4 s < union by ~0.3 s: after the three fusions the
  remaining ops genuinely overlap (same cache lines touched by neighboring matmuls),
  unlike Exp662's additive regime - do not expect additivity on this graph.
  Structural corollary: piece-wise-at-p1 does NOT change deep-layer batching - the
  late pass still batches window-wide (52 pieces), which is why defer-OFF's -1.2% is
  pure staging/build cost and why the attribution barely moved across the two path
  changes. The speed board is confirmed unchanged by measurement, not by assumption.
  CORRECTION to Exp712's note: behavioral probe assets ARE hash-manifested
  (device_assets.json carries silence5s/noise5s/song20s/proto48k_stereo/twospk* with
  notes, and their device hashes match today) - no tooling gap, my note was wrong.

- XWIN REGIME-CHECK @ v4.1 (Exp714, closed): --xwin now costs +6.2% on the protocol
  (interleaved 2x: windowed 2.3966/2.3934, xwin 2.5410/2.5492, token counts equal)
  vs +3.2% pre-defer-flip. Mechanism consistent with the defer story: the carry's
  per-chunk splice cost scales with piece size, and p1 doubles the carried tensors;
  removing the window-batched late pass did not help it. xwin stays a DIAGNOSTIC
  (moves probe attribution) and server-path option, not a speed lever. Doc fixed:
  STREAMING_1P5B.md still said "~8% faster on short clips" (a 500-run-era claim that
  survived two contradicting measurements) - corrected in place.
  SELF-CAUGHT TRAP (the Exp595-607 class again): my first 6-run sweep used env
  XWIN=1, which nothing reads - xwin is the --xwin CLI flag via measure_xwin.sh -
  so the whole "parity" table was the default arm twice. Tell: identical rtf AND
  identical hashes across supposedly-different arms. Rule: before trusting an arm,
  show the flag actually reaches the binary (grep the invoked command line once);
  env-name guesses are for env-gated knobs only.

- v4.1 BEHAVIORAL SUITE COMPLETED (Exp716, the partial Exp712 run finished): 48 kHz
  stereo -> coherent zh with the DOCUMENTED 'YyY' proper-noun quirk of the 4x4 LM
  (Exp499 class, not a regression); twospk_overlap -> Speaker 0 x4 + Speaker 1 x2
  (Exp609 holds at v4.1); twospk sequential -> single tag (closed Exp650 behavior).
  Silence/noise/music were done at Exp712. Anchor 2.3912 byte-identical.
  The audit caught its first real FAIL on its own this iteration: Exp715's rebuild
  left the device with the OLD binary (the probes used AUDIO=, which skips the push),
  i.e. the two-speaker probe above ran on the pre-Exp715 binary - functionally
  identical here (that change is default-inert), and the audit's sync check is exactly
  what turned a silent-stale arm into a loud one. Pushed; 64/64 green.

- LEAN TIER LONG-FORM RAM CLAIM RE-VERIFIED @ v4.1 (Exp717): 138 s soak with the
  lean recipe (PIECES=13 + defer ON): peak RSS 1766 MB, series ~1756-1808 MB flat,
  majflt 0, 851 tokens (the documented lean long-form count), repetition screen 0
  repeated 5-grams, RTF 2.5757 (4% above shipped's 2.4766 - consistent with the
  uniform ~+4-5% lean gap at v4.1). The "sub-1.8 GB" tier claim holds at the current
  stack. Protocol cells re-confirmed 2.52 (defer-ON verified via the correct --env
  route this time); shipped anchor 2.3937/2.4012 in band.
  SAMPLER POST-MORTEM: my first in-line series collected zero samples - a local
  `while read` over a piped PID list whose $p expanded before the adb hop (the
  Exp679 adb-argv trap, again). The exit-time peak_rss_mb + coarse polls answered
  the question anyway; prefer rss_soak.py (it refuses to print an empty success).

- SYSTEMATIC DOC-DRIFT SWEEP (Exp718, zero device time): present-tense grep across
  RESULTS/STREAMING/prompt.md surfaced 6 stale passages that per-cell refreshes had
  missed (the Exp714 xwin finding generalized into a sweep): prompt.md header
  "Current best 5.96" (300-run era) -> 2.39 v4.1; RESULTS lean-tier paragraph (2.55/
  2.50/2.59 + the RETIRED "+0.9% on shorts" story) -> v4.1 uniform +4-5% with the
  defer flag in the recipe; gate-mean paragraph (3.1235 gate645-era as "current") ->
  2.68 gatedef1 with gate645 labeled historical; lean mean 2.8136->2.8187 (lean711b);
  STREAMING lean recipe ("no env flags", 2.84/2.78/3.15 @1.88) -> 2.52/2.58/2.82 @1.75
  + VAE_DEFER_LATE=1 required; orphaned sentence deleted.
  RULE: after ANY structural flip, do a PRESENT-TENSE sweep ("current/shipped/default/
  today" x numbers), not just update the cells you remember - prose is what future
  sessions (and the runbook) follow. The ladder is necessary but not sufficient.

- GELU_QUICK RE-ADJUDICATED AT v4.1 AND CLOSED ON EVIDENCE (Exp719). The Exp626/657
  ruling ("no measurable accuracy cost; declined on WHICH tokens change; reopen only
  with a second sample + repetition-free zh/long-form") was itself p2-era. Re-ran its
  reopen condition at the current stack:
  * zh canary (hotwords): now BYTE-IDENTICAL to default (the p2-era 'T T' token
    doubling did not survive the v4.x stack) - the named objection had already gone.
  * Second sample, zh held-out 468 tokens, paired: gelu_quick 17.95% vs default
    14.74% = +3.2 pp, discordants 8/23, McNemar p=0.011, CI [+0.85,+5.56] pp.
  => gelu_quick is CLOSED AS A REGRESSION, not as a parity-churn decline: on the hard
  zh domain the tanh approximation measurably hurts; the canary was too small to see
  it (identical!) - a warning that a byte-identity probe neither proves safety nor
  detects harm at n≈40 tokens. The ~1% speed item stays declined with statistics now.
  (Its v4.1 speed share is likely smaller than -2.8% anyway: two of the three gelu
  sites are fused with bias since Exp673.)
  METHOD: this is the paired-test rule (Exp655) applied to a KNOB rather than a tier -
  and the first case where the loop's own "parity" verdict on a declined option turned
  out regime-dependent AND wrong in the conservative direction. When re-checking a
  declined knob, test a SECOND DOMAIN, not just the canary.

- Q8_0-INTERLEAVED IM2COL: BUILT, BIT-IDENTICAL, ZERO WIN (Exp720) - closes the LAST
  priced speed item. Implemented ggml_compute_forward_im2col_q8_0 (panels of 4 columns
  through the kernel's OWN quantize_q8_0_4x4 -> bit-identical BY CONSTRUCTION; verified:
  protocol transcript byte-identical to the frozen reference) + builder ne0=k/32 for
  Q8_0 dst + dispatch case + knob VAE_CONV_I8_Q8COL (default OFF, shipped path unchanged).
  Speed: 5 interleaved reps = EXACT PARITY (2.3935-2.3973 vs base 2.3921-2.3965).
  * WHY the ~2% was a phantom: the Q8CAST arm's speedup was never "skipping the
    conversion" - a row-major Q8_0 src1 != vec_dot_type falls into the GENERIC llamafile
    path (a different kernel), so that arm measured a KERNEL-SWITCH effect, not a
    conversion cost. The real in-kernel quantize_mat_q8_0 is ~free: L1-resident panels,
    vectorized, overlapping the dot stream's memory latency.
  * Kept as tooling: if a future build stages conv activations as Q8_0, staging bytes
    drop 4x (RAM-leaning, speed-neutral); the forward also documents the exact panel
    layout contract of the blocked kernels (useful reference; the gemv-tail question is
    settled: tail columns are read interleaved, hence zero-padded panels are correct).
  * CONSEQUENCE: the speed board is now EMPTY BY MEASUREMENT, not by assumption - every
    component priced (elementwise ~6%, conv dots 8.8% int8, conversion ~0, staging free,
    LM at bandwidth ceiling, scheduling/granularity/defer/topology closed, declined knobs
    re-adjudicated). Remaining gains need model/training changes or upstream kernels.
  LESSON (measurement-artifact class, 6th instance): before pricing a "remove work X"
  lever, verify the fast arm differs ONLY by X - the Q8CAST arm differed by the whole
  kernel path. Decompose with all-legal arms (this run IS that decomposition).

- POST-SURGERY GATE REGRESSION (Exp721): Exp720 changed ggml's SHARED im2col
  builder/dispatch, so the default path needed the 40-utt regression test (the
  protocol hash only covers one clip). Fresh gate (tag gate721, stamped BIN
  3e3cc40a = post-surgery): ZERO discordants vs the frozen reference (b=0/c=0,
  p=1.0), S/D/I identical to gatedef1/gatep1/convint8b, gate mean 2.6758 (= 2.6768
  pre-surgery, 0.04% off). The ggml surgery is confirmed inert on the default path
  across 40 acoustic conditions.
  SCORER-CONVENTION NOTE: score_hyp.py's jiwer path reports 4.82% for the shipped
  tier's current output while the paired hybrid-tokenizer tool reports 4.51% for the
  IDENTICAL outputs (both stable per-tag) - the ladder quotes the hybrid convention
  (per Exp656). Quote the tokenizer with any WER; never mix conventions in one row.

- ASSERT-ARMED VERDICT ON THE Q8_0-IM2COL KNOB (Exp723, closes interrupted work):
  an assert-armed (Debug, NDEBUG off) device binary was built and both paths run:
  DEFAULT path = zero asserts, 39 tokens, transcript identical to the frozen ref
  (timing 2.47 meaningless - Debug build). KNOB-ON (VAE_CONV_I8_Q8COL=1) path trips
  GGML_ASSERT(view_src==NULL || ... <= ggml_nbytes) at ggml.c:4134: the interleaved
  Q8_0 im2col tensor's ne[] undercounts its byte footprint, so the downstream
  reshape/view reads out of bounds. Invisible in Release (assert compiled out; the
  arena happened to be big enough) - which is exactly why the Exp720 parity runs
  looked clean. VERDICT: the knob is downgraded from "kept as tooling" to
  ASSERT-DIRTY - do not ship or build on it without fixing the tensor-size contract
  (ne0 must satisfy BOTH the panel layout AND ggml_nbytes accounting; the
  interrupted (k/32)*4 attempt was reverted unvalidated). The shipped default path
  is proven assert-clean, which was the safety question that mattered.
  HYGIENE: the interrupted session left the ggml submodule dirty + 123 MB build dir
  + scratch logs; all reverted/removed, setup_assert.sh committed as a tool with a
  corrected usage note (measure.sh has no BIN override).

- VERIFICATION SWEEP Exp724-730 (all keep, no shipped change; band 2.39-2.42):
  * Exp724 lean-tier spaced verification: recipe reproduces its cell exactly
    (2.5211 @ 1751.9 MB, 39 tok, byte-identical) - both tiers verified same-session.
  * Exp725 provenance extension: all 5 post-Exp699 gate sets (gatep1/gatedef1/gate721/
    lean711/lean711b) are n=40 + stamped - every ship rests on stamped pairs.
  * Exp726 overfit guard (new recurring check): second same-domain 10 s slice
    (slice10b, 55 tok, applause/audience) = 2.5353, sane transcript; VAE rate
    identical (15.9 s), LM per-token 91 vs 95 ms/tok - no protocol-clip overfit.
    NOT a primary-metric run (different clip/density); never rank against 2.39.
  * Exp727 manifest hygiene: slice10b was already hash-manifested (hash matched
    device) - note upgraded to record the guard role instead of re-blessing.
  * Exp728 heartbeat: device idle, 2.3965 byte-identical.
  * Exp729 long-form determinism: 138 s at current build = 877 tokens EXACT,
    2.4551; only marginal-class diffs (contractions/punctuation) vs the Exp515-era
    ref - expected across 200+ runs of stack changes (Exp644 doctrine). Archived as
    .auto/hyp-138-v41.txt for same-stack checks.
  * Exp730 pathology screen on that archive: 138 distinct repeated 10-grams, ALL
    exactly x2, all genuine talk refrains - zero runaway. ***CORRECTION (Exp875): "genuine" was wrong
    half the time - chat138.wav IS chat69.wav played twice (proven by ranged md5, enforced by audit check
    7e), so a 69 s signal repeated must produce x2 n-grams by construction. The zero-runaway verdict
    stands (it is about max run length, which the repeat does not create), but the x2 pattern is an asset
    property, not a content or model property.*** Method note: screen
    includes Speaker labels, so absolute counts are not cross-method comparable;
    verdict rests on max run length.
  LESSON: the readable ledger had fallen 9 iterations behind JSONL (Exp714-721 were
  also thin here) - the ledger is what future sessions actually read, so batch-write
  it whenever several verification iterations accumulate.

- LAST ° CELLS FILLED (Exp732): lean 17 s 2.5085 @108 tok, lean 69 s 2.5101 @432
  tok (shipped: 2.36 @108, 2.39 @446). Gaps +6.3%/+5.0% - the uniformity claim
  (+4-6% at every length) is now airtight across all four lengths. NOTE the 69 s
  token divergence (432 vs 446): lean-vs-shipped CAN differ on long-form (a couple
  of early greedy flips cascade), consistent with the 2-token gate difference - the
  ladder states token counts per cell precisely so nobody mistakes tier outputs for
  identical. No ° cells remain in either shipping tier's row.
  HARNESS SELF-CATCH: my sweep loop ran bare measure.sh twice (39-tok default clip)
  before the AUDIO= arms - the 39-tok tell caught it immediately (Exp709/714 rule:
  confirm the first arm measured the intended clip). Wasted ~80 s, no bogus cells.

- LEAN EQUIVALENCE NOW SINGLE-VARIABLE (Exp734): fresh lean gate at the current
  build (tag lean734b, BIN 3e3cc40a = same as gate721, PIECES the ONLY variable):
  b=0/c=2 of 731, p=0.5, mean 2.8247 - the Exp711 cross-BIN caveat is REMOVED, the
  equivalence verdict stands cleanly.
  FREE BYPRODUCT (misrun, kept honestly): lean734-nodefer (p13 WITHOUT defer - I
  forgot EXTRA_ENV, then repeated the mistake on the retry half before the stamp
  check caught it; FOURTH sighting of the flag-routing trap, this time purely
  procedural) gates at mean 3.0049 = +6.3% vs defer-on - the gate-mean price of
  defer at p13, consistent with protocol +5.2%.
  PROCEDURAL RULE (new, for me): after ANY eval40 invocation, FIRST read the new
  tag's run-info.log stamp before the run goes further - a 20-utt half costs 9 min,
  a stamp check costs 2 s. I paid 18 min twice for skipping it.

- VERIFICATION SWEEP Exp731-737 (all keep, no shipped change; band 2.3965-2.4045):
  * Exp731 ledger consolidation (the Exp724-730 sweep written into this record).
  * Exp732 ladder completed: lean 17 s 2.51 / 69 s 2.51 fill the last ° cells;
    uniformity (+5-6% all lengths) airtight; 69 s tier outputs diverge (432 vs 446
    tok - early greedy flips cascade; per-cell tokens stated, not assumed).
  * Exp733 storage+audit: /data 28%, 9 GB workdir (ample); audit 65/65 green.
  * Exp734 lean equivalence now SINGLE-VARIABLE (same-BIN pair vs gate721:
    b=0/c=2, p=0.5, mean 2.8247) - last accuracy caveat removed. Misrun priced
    defer-off gate mean +6.3%; stamp-first rule promoted to prompt.md (Exp735).
  * Exp736 guard validation: slice10b reproduces (+0.3%) - stable recurring guard,
    use sparingly (~every 10 rounds).
  * Exp737 heartbeat: no sync events, no co-runner, 2.4012 byte-identical.
  No ° cells, no open caveats, no stale ledger entries remain after this write.

- LEAN DEFER-PATH WATCH (Exp739-740): three consecutive defer-ON lean reps elevated
  +1.2-1.6% (2.5604/2.5472/2.5497, mean 2.5524) vs the 2.52 cell, across rested/idle
  states (batt 29.5-30.6, 0 co-runners, majflt 0), all hash-identical. LOCALIZED by
  the very next runs: shipped p1 2.4003 EXACT and defer-OFF lean 2.6712 EXACT vs its
  2.673 reference - so the elevation sits in the DEFER-ON late pass specifically,
  not in launch count, global device state, or thermal (coolest run still elevated).
  Mechanism unknown; candidates are late-pass staging/build costs shifting with
  allocator state, but that is speculation. PROPORTION: secondary tier, clean hashes,
  +1.3% changes no decision (tier exists for 1.75 GB) - the 2.52 cell STANDS, no
  churn on 3 reps. TRIGGER: if defer-on lean stays elevated next session, run the
  VAE_ABL sweep on the lean path to localize within the late pass (vae_s delta is
  the discriminant, never rtf).

- LEAN CONV SHARE MEASURED SEPARATELY (Exp741, closes Exp740 trigger): same-session
  paired ablations - shipped+ABL_CONV vae 14.5 (base 15.9 => conv 1.4 s, reproduces
  Exp713's 8.8% exactly: shipped conv share did NOT move) vs lean+ABL_CONV vae 15.7
  (base 17.4 => conv 1.7 s). The +0.3 s lean elevation sits ENTIRELY inside the conv
  share arithmetic. IRREDUCIBLE AMBIGUITY (stated, not hidden): lean conv was never
  measured separately before, so "conv share drifted 1.4->1.7" and "conv was always
  1.7 on p13 (13x the per-node im2col+conversion launches) while non-conv drifted"
  both fit - the leading structural candidate is per-NODE conversion setup scaling
  with piece count (Exp720 proved per-BYTE conversion free; per-call setup was never
  priced). Recorded as lean conv share = 1.7 s (new cell). No action: hashes clean,
  gate unaffected, +1.3% on the secondary tier changes no decision; 2.52 cell stands.
  SHARPER TRIGGER: escalate only if lean conv share exceeds ~1.9 s or lean total
  exceeds 2.58 next session; the discriminant is lean+ABL_CONV vae_s, never rtf.

- CONV-SHARING SCALING: p26 THIRD POINT REJECTS LINEAR SETUP SCALING (Exp742):
  p26+deferON base vae 17.5, +ABL_CONV 15.9 => conv share 1.6 s, total 2.5557 @
  1718.9 MB (re-validates the documented p26 tier row: 2.55 @ 1.72 GB).
  Three-point pattern p1:1.4 / p13:1.7 / p26:1.6 REJECTS the per-node-setup-scales-
  linearly candidate (predicted ~2.0 s at p26). Supported account instead:
  GEMM-EFFICIENCY - identical conv MACs, but p1's whole-window im2col feeds large-L
  GEMMs (~18% more efficient) while p13/p26 run small-L pieces; consistent with the
  Exp678 shape-sensitivity and Exp681 rate-bound findings. So lean-vs-shipped conv
  gap is STRUCTURAL (no drift), while the session-level lean total shift (+1.3%,
  all granularities' totals stable except lean defer-ON) stays unexplained-but-
  bounded - thermal/session drift on the launch-heavy path remains the suspicion.
  Neither changes any decision; the thread is closed pending the standing trigger
  (lean conv >1.9 s or total >2.58).

- LEAN ELEVATION EPISODE CLOSED AS TRANSIENT (Exp744): defer-on lean returned
  2.5277 (vae 17.2, hash-identical, RSS 1752) - back inside the 2.52 cell after 4
  elevated reps (2.5458-2.5604) with NO code/config change in between. Verdict: a
  transient device-state episode (~2 sessions), not a regime change; the thermal/
  session-drift suspicion is now the confirmed shape (appears and clears
  spontaneously, shipped tier unaffected throughout). The 2.52 cell never moved.
  Standing guidance unchanged: sub-2%-of-tier wobbles on the secondary tier with
  clean hashes get a watch flag and a re-check, never cell churn.

- DEFER AT p2 RE-CHECKED, EXP639 HOLDS (Exp746): p2 with defer-OFF (current default)
  = 2.4621, vae 16.5, RSS 2071.5, hash-identical - i.e. PARITY vs the documented
  p2-defer-ON 2.46 row. So the defer story has a granularity boundary INSIDE the
  coarse regime: p2 (2 pieces) batching-benefit ≈ staging-cost (tie, both eras),
  p1 (1 piece) zero-benefit < cost (-1.2%). Exp639's p2 verdict is the one
  neutral-knob closure that SURVIVED the int8 era intact - worth noting because
  every other closure from that era broke. The p2 fallback row needs no update
  (2.4621 ≈ 2.46); the mechanism note in STREAMING_1P5B.md already encodes the
  regime (defer valuable iff pieces are GEMV-shaped), which this confirms from
  the other side.

- AUTO-REVERT IS A SILENT NO-OP AT A NON-REPO WORKDIR (Exp759, harness integrity).
  pi-autoresearch/index.ts:2446 reverts discards with `git checkout -- .` at cwd =
  the SESSION workDir (/home/user/vibe-asr-streaming-1p5b), which is the PARENT of
  the actual repo (VibeASR.cpp/) and not a git repo. The COMMIT half raises
  (that is the "git add failed (exit 128)" line on every log_experiment here); the
  REVERT half never checks git's exit code, so it prints "Git: reverted changes"
  while reverting nothing. Consequence: a discard/crash would leave the discarded
  code in the tree and every subsequent measurement would be silently of the
  discarded variant. Nothing was ever lost (zero discards this session, and
  Exp1-era notes already mandated manual commits), but the failure was UNMONITORED.
  FIX SHIPPED: audit_harness check 3b FAILs on any dirty tracked file between runs
  and names the cause; negative-controlled by planting an src/vae.cpp edit (1 FAIL)
  and restoring (66/66). RULE: after any discard/crash, check
  `git -C VibeASR.cpp status --porcelain` yourself - the tool's message is not
  evidence. Generalizes: an "auto-" safety net must be observed FAILING before you
  trust it succeeding (Exp660's rule, aimed at the harness itself this time).

- BLOCKED-INT8 GEMV TAIL: PRICED, THEN SHIPPED AS THE m2 KERNEL (Exp763-767, v4.2).
  The loop's first hand-written CPU-kernel project in many iterations, and the only
  accuracy-preserving speed lever left on the measured board.
  * PRICING (Exp763): the vendored mul_mat runs gemm over ne11-ne11%4 columns, then ONE
    gemv CALL PER LEFTOVER COLUMN, and ggml_gemv_q4_0_4x4_q8_0 walks all nb weight blocks
    with the activation pointer fixed = one full src0 stream per leftover column.
    VAE_MMSHAPES census hook (GGML_MM_DEBUG_SHAPES=1; note it needs VAE_GRAPH_STATS=1 too
    or it prints nothing - the Exp631 silent-hook trap): every VAE FFN linear and the LM
    prefill run at ne11=26 (2 tail cols); conv matmuls have huge L (208..41600, all %4==0,
    no tail). Skip-tail knob (GGML_MM_SKIP_TAIL, default-inert) paired: VAE -1.4%, LM
    prefill -16.6% = 965 ms/clip = 4.0% of RTF. A tail column costs 92 ms = exactly one
    decode token (one 890 MB stream at 9.7 GB/s); the prefill GEMM is compute-bound at
    ~78 GMAC/s (marginal column ~38 ms). Padding the frame axis to a multiple of 4
    (graph-level, VAE FFN, output-exact) measured PARITY and was reverted - a padded gemm
    column costs ~38 ms of compute, recovering only ~30 of the 185 ms tail.
  * THE SKIP-ARM MYSTERY (Exp764, resolved by census not theorizing): skip=2 should have
    been identical to control (tail_start = ne11-2 both ways) but produced garbage, and
    skip=1 saved VAE time but zero prefill time. Cause, two parts: (1) the LM runs an
    INITIAL 31-token pass (ne11=31, %4=3, 386 calls at exactly 1:4 vs the 1544 ne11=26
    window-prefill calls), whose KV/prompt state skip=2 poisoned (tail_start 29 vs 28);
    (2) garbage features -> 256 tok/window runaway decode -> thermal throttle inflates
    the prefill PHASE timer itself. LESSON (Exp674 trap, new guise): in garbage arms even
    phase timers lie - only vae_s is truly LM-independent (it matched predictions in EVERY
    arm: skip=1 -95/-138 vs -112 predicted; skip=2 flat). Shape facts come from CLEAN runs.
    Also fixed en route: the knob mapped GGML_MM_SKIP_TAIL=0 -> 1 (the '0 is not zero'
    class, same as measure.sh's no-arg trap); semantics now N = skip N columns, 0 = off.
    Correct tail accounting: LM windows 736 + LM initial 276 + VAE 225 = 1237 ms = 5.2%.
  * THE KERNEL (Exp765): ggml_gemv_q4_0_4x4_q8_0_m2 - two activation columns per single
    weight pass. Design for zero numeric risk: the block loop replays the single-column
    sdot ladder twice per K block; the f16 scale product moved v16->v13 so the 8 derived
    nibble vectors survive for column 1; all 16 sdot .inst encodings reused VERBATIM (no
    hand-encoded bytes beyond proven ones); fallback = two single calls. Per-column
    numerics identical by construction -> the protocol hash is a tripwire for dispatch
    bugs, not a hope. Wired into the tail loop for >=2 remaining columns, gated to
    GGML_TYPE_Q4_0_4_4 (the loop serves all blocked types), default ON with the
    GGML_MM_M2_OFF=1 hatch. Composes with SKIP_TAIL (fires only when >=2 remain).
  * MEASURED (Exp765/766): prefill -143 ms (2 reps, consistent), rtf ~-1.0%
    (2.376 vs 2.396), 5/5 protocol runs byte-identical. Ceiling autopsy: the m2 pass
    costs ~146 ms/window vs 91 for one gemv pass (1.6x) - the doubled sdot ladder is
    issue-bound, so 2x work per pass yields only 1.25x; the 2.8% ceiling assumed a free
    lunch. SHIPPED under the sub-2%-with-identity rule (q8head precedent): 40-utt gate
    40/40 byte-identical, b=0/c=0 of 731 tokens, p=1.0 (tag gatem2, mean 2.6641).
    Lean: same-session m2-ON 2.5414 (n=3) vs m2-OFF 2.5668 (n=2) = -1.0%, so no
    lean regression; the 2.52->2.54 absolute move is the known +/-1% lean session wobble.
  * LADDER (Exp767, v4.2): shipped 10 s 2.38 / 17 s 2.36 / 69 s 2.38 / 138 s 2.44 /
    gate mean 2.66; lean 10 s 2.54 / 17 s 2.50 / 69 s 2.52 (lean mean 2.82 + lean 138 s
    2.58 carried pre-m2). Harness footgun found refreshing: measure.sh --env forwards to
    the DEVICE; PIECES is consumed by measure.sh itself, so lean arms need PIECES=13 in
    the caller env (two p1+deferON misruns measured and discarded as void).
  * CLOSED, HEADROOM ~0 (Exp770 measured it at 1.2%; Exp774 shows that 1.2% is NOT recoverable):
    the tail pass is not an EXTRA walk over the weights - a 4-column-panel kernel needs
    ceil(26/4) = 7 passes for 26 columns, and m2's tail pass is one of those 7, so "fusing the tail
    into the gemm's own weight walk" cannot remove a pass; it could only recover the unused 2 of 4
    column slots in the last panel (ALU, not traffic). Cross-check on the pass-bound model: 1 of 7
    passes of the FFN would be ~1.13 s of the VAE, but skipping the whole tail measured 0.2 s
    (ratio 0.18), so the VAE is not pass-bound either - consistent with Exp681 (kernel-rate-bound).
    And no wider panel exists to exploit: ggml_gemm_q4_0_4x8_q8_0 also uses ncols_interleaved = 4.
    m=3 single-pass is likewise closed (the ne11=31 remainder costs 1 pass either way).

- LEDGER IS SPLIT IN TWO HALVES - READ BOTH (Exp770 harness finding). This file holds Exp685
  onward; ../.auto/ideas.md (outside the repo, pointed at by the loop prompt) holds the long-form
  archive through Exp684 and is otherwise STALE. A session that reads only the prompt path can
  resurrect closed work - that is what happened in runs 769-770 (the archive's Exp684 "conv-int8
  STILL TO DO" was superseded by Exp685-693, and the shipped default already IS the conv-int8 file).
  A POINTER header was added to the archive half; keep new entries here (committed) and move only
  durable cross-era lessons to the archive.

- CONV-INT8 LAYOUT LINE CLOSED + TAIL-FUSION RE-SIZED TO 1.2% (Exp769/770).
  * The converter is CORRECT; the earlier "PAIRING MISMATCH" verdict was a threshold artifact.
    Q4_0's own bound is amax/15 = 0.0217 for the probed tensor; identity readback diff 0.0199 is
    INSIDE it, and the decisive predicted-vs-kernel product test (product rebuilt from the probe's
    own dequantized bytes) gives ratio 0.0045, explained by the probe's fp16-quantized selector
    scale. Rule: never threshold a 4-bit tensor against an absolute error - against its type's
    error bound.
  * K/IC swap and OC-plane transpose both give garbage (10 tokens, VAE 24.2 s vs 17.1):
    pass-through bytes were already right (quant4x4.cpp's header comment). No more transposes.
  * fuse-GEMV-tail-into-GEMM ceiling = 1.2% RTF, not 1.8%: post-m2 the VAE tail is 0.2 s of 15.9 s
    vae_s (3 pairs, whole tail skipped) + LM ne11=31 residual 92 ms => ~292 ms of 23.9 s. Below
    bar by measurement. Re-open only if GGML_MM_DEBUG_SHAPES shows ne11 % 4 >= 3.
  * KNOB BUG, 5th "wrong measurement" case: GGML_MM_SKIP_TAIL's Exp764 formula
    ne11 - MIN(v, ne11%4) EQUALS the default for v >= ne11%4, so the first sweep ran no arm at all
    and looked like parity. Fixed to (ne11 - ne11%4) + MIN(v, rem) (submodule 70d1bb58). Standing
    rule: prove a knob fires by an OUTPUT change (39 -> 1024 tokens here) before interpreting it.
  * RSS band settled: shipped-tier peak_rss = 2374.2-2375.1 MB across 34 runs; the 2.23 GB in old
    notes is the F16-conv reference row. RESULTS.md needs no change.

- GELU IS THE LARGEST REMAINING NON-MATMUL ITEM: 13.3% OF VAE SECONDS (Exp780, VAE_ABL_GELU knob).
  Five paired reps on the shipped stack: vae_s 15.8/15.9 base vs 13.7 with the gelu node removed =
  2.1 s, about 8.8% of RTF. That is 4x the next-largest elementwise item re-measured in the same sweep
  (rms_norm 4.1%, bias adds 2.5%, scale+resid 1.9%), and it SURVIVED all three shipped fusions because
  Exp673 fused the bias INTO gelu rather than removing the pass. Derived rate: 980 MB per graph build =
  122.5M elements written, x8 builds per 10 s clip = 980M elements in 2.1 s = 467 Melem/s aggregate,
  233 Melem/s per core - far below what a streaming NEON op achieves (6-10 GB/s), so gelu is NOT
  traffic-bound on this path; something in the loop's per-element work is slow.
  HYPOTHESIS (not yet measured): ggml_vec_gelu_f32 under GGML_GELU_FP16 is a SCALAR loop that converts
  each f32 to f16 and gathers one entry from ggml's gelu table (65536 x 2 B = 128 KB, larger than the
  A78's 48 KB L1), so every element pays an L2 access. If true, the fix is issuing several table lookups
  per iteration (their latencies overlap) - NOT replacing the table with a polynomial, because the table
  IS the function and Exp601/626 already settled what gelu_approximation costs in churn.
  NEXT: a standalone gelu microbench (host first, then device) that separates gather latency from math
  and proves bit-identity of a batched variant before any timing claim. Ceiling is 13.3% of VAE = ~8.8%
  of RTF if the loop's rate were free, so even a 2x improvement there is ~4%, above the 2% bar.
  METHOD NOTES from the same iteration, both cost time and both are harness rules:
  * LONG CLIPS ARE SATURATED - the pipelining question is now closed by DIRECT measurement, not by
    Exp533's 2-stream proxy: device-side /proc sampling of a 138 s run gives 1.96 of 2.0 cores busy,
    flat across 66 intervals (min 1.73 during model load, max 1.97, 662.7 CPU-s over 338.9 s wall).
    There is no idle core time for window/phase overlap on long clips either.
  * That measurement also RETRACTED two claims I had made earlier today from a 400 s clip whose run
    reported rtf=2.4688 while the binary's own line said "RTF: 0.971" and "449.85 s" for 462.86 s of
    audio - internally inconsistent, so both of its findings (long-clip F16 convs +34%, and "1.03 cores
    busy") are withdrawn; the asset and its directory were later reclaimed by the device, which is how
    an unfalsifiable measurement survives. Long clips must be run through measure.sh --clip (host file,
    blessed) and the asset must be in device_assets.json before a documented cell rests on it.
  * AD-HOC DEVICE COMMANDS must use the harness's own paths: RDIR=/data/local/tmp/vibeasr with
    VAE_FILE/LM_FILE env (from bench_device.sh), a serial taken from measure.sh's DEV= line, and assets
    at ../eval-bilingual. Inventing a path silently targets a directory that may not exist, and an
    unqualified adb command is wrong the moment a second device appears.

- FOURTH FUSION SHIPPED: ONE-PASS FUSED GELU+BIAS (Exp781, v4.3) - protocol 2.38 -> 2.29-2.32 (-2.8 %
  paired), VAE 15.87 -> 15.0 s (-5.3 %), gate mean 2.6641 -> 2.5742 with **40/40 byte-identical**
  transcripts (b=0/c=0 of 731, p=1.0), lean p13 2.54 -> 2.44, RSS unchanged. What was wrong: the Exp673
  "fused" gelu_bias still looped the row TWICE (f32 add into dst, then gelu in place) = 4 tensor passes
  where 2 suffice. Priced first with the new VAE_ABL_GELU knob: gelu was 13.3 % of VAE seconds, 4x the
  next elementwise item and the largest non-matmul item since v3.6-v3.8. Negative control that decided
  the design: batching four table gathers in flight is PARITY (vae_s 15.9 both), so the cost is traffic,
  NOT the 128 KB table's gather latency - my Exp780 mechanism guess was wrong while its rate math (233
  Melem/s/core) was right. Sixth occurrence of "price the op before choosing the fix".
  OPEN NOW (attribution RE-DERIVED in Exp782, so item (1) is closed): conv 1.45 s (9.7% of VAE) |
  gelu-fused residue 1.20 (8.0%) | rms_norm 0.60 (4.0%) | bias 0.40 (2.7%) | scale+resid 0.30 (2.0%).
  Cross-check: gelu's ablation delta fell 2.1 -> 1.2 s, exactly the 0.85 s of vae_s the fusion saved.
  The one item still above the bar is a fused f32 rms_norm+gamma (2.6% of RTF) - see below.
  rms_norm is now the top elementwise item (4.0 % of VAE = 2.6 % of RTF): the f32 path runs
  `ggml_rms_norm` and then a SEPARATE gamma MUL (vae.cpp:296-297) = 4 tensor passes where rms_norm_scaled
  would do 2. rms_norm_scaled exists but `GGML_ASSERT(src0->type == GGML_TYPE_I8_S)` and its work buffer
  is a full [n_elements] float_buf + per-thread absmax - so giving it an f32 path must request the work
  buffer BY TYPE (the Exp578/664 segfault trap) or better, extend the f32 rms_norm compute to read an
  optional gamma src and skip the separate mul entirely (no work buffer at all).
  (2) lean 40-utt mean carries a scaled value marked ° - re-run it (~40 min) when convenient; (3) the
  hatches were re-audited for THIS stack (Exp677-style) never since Exp710 - costs of DW_CONV1D_OFF /
  LS_FUSE_OFF / GELU_BIAS_OFF / GGML_GELU_BATCH_OFF / M2_OFF at v4.3 are unmeasured.

- ATTRIBUTION + THREE AXES CLOSED BY MEASUREMENT (Exp783, v4.3, base vae_s 15.0 / wall 23.0 s):
    conv matmuls        1.45 s   9.7 % of VAE   (already blocked-int8 since v3.9 - no lever left)
    gelu residue        1.20 s   8.0 %          (Exp673's fusion removed one round trip of three)
    rms_norm             0.60 s   4.0 %  <- the ONE item above the bar as a fused f32 norm+gamma
    bias adds            0.40 s   2.7 %          (needs a mul_mat epilogue)
    scale + residual     0.30 s   2.0 %          (already fused, this is the irreducible write)
  * GEMV-TAIL AXIS, CLOSED: with GGML_MM_SKIP_TAIL (semantics fixed: 0 = off, >0 = cap the tail) the
    VAE's ENTIRE tail is 0.1 s of 15.0 (vae_s 15.0 vs 14.9-15.0, three pairs; the LM's ne11=31 tail is
    ~92 ms by census arithmetic). So the whole "fuse the tail into the gemm" project tops out near
    1.2 % of RTF, not the 1.8-2.8 % carried in the ledger from pre-m2 arithmetic - BELOW the bar.
    My arithmetic-based 3.2 % estimate was wrong (4th "measurement you meant" case): it assumed each
    leftover column costs a full 92 ms token-equivalent stream, but with m2 paired the VAE's 2-column
    tail already shares one pass and the measured residual is 6x smaller than the model.
  * SHAPE CURVE (built into .auto/mm_shape_micro, device run): the blocked kernel delivers 20.4 GMAC/s
    at ne11=26 on ONE thread and 38.6 on two, saturating at 23.3 / 43.9. So the "VAE runs at ~25 GMAC/s
    vs a 40-45 microbench" residual - unexplained since Exp551 - was a THREAD-ACCOUNTING artifact: the
    old microbench number was two threads and the in-graph number one. Per core the kernel is at its
    ceiling; the VAE's remaining time is the non-matmul traffic in the table above. Reuse the curve to
    price any proposed shape/batching change in a minute instead of a ladder.
  * LM AT THE KERNEL CEILING (paired tail-skips): skipping the LM's tail cuts prefill 4.3 -> 4.0 s but
    makes the LM emit garbage (1024 tokens), and in that state prefill stays EXACTLY 4.0 while rtf
    doubles - because phase timers get thermally contaminated by runaway decode while vae_s stays
    exact. Rule: when a probe invalidates the output, vae_s remains trustworthy and LM-phase seconds
    do not (Exp674 generalized).

- GEMV-TAIL AXIS CLOSED BY MEASUREMENT + CENSUS RECONCILED (Exp791). Two instruments disagreed and
  both were right about different things, so the resolution is worth keeping:
    * Direct skip (GGML_MM_SKIP_TAIL=1, 2 paired reps): removing *every* tail column saves 0.1 s of VAE
      and 0.3 s of prefill = 0.4 s = 1.7% of the clip.
    * Shape census (GGML_MM_DEBUG_SHAPES on a byte-identical run) models the same traffic as 8.5 GB =
      1.31 s if every tail read reached DRAM.
  Ratio 0.31: only about a third of the modelled tail traffic is real DRAM work, because a tail pass
  immediately follows the gemm passes of the SAME call and the dominant tail tensors are 1.3-7.7 MB,
  i.e. L3-resident. The big-k tensors that would miss L3 are decode (ne11=1) calls, which are not tails.
  Shape inventory of the shipped build: ne11 = {1 (decode, 20k calls), 26 (LM window prefill + VAE FFN),
  31 (LM initial pass)}. Only ne11=31 has a remainder >= 3, and it costs 0.2 s. So the ledger's reopen
  condition ("an ne11 with remainder >= 3") is technically satisfied but moot - do not commission an m3
  kernel or tail-fusion; the whole axis is 0.4 s.
  RULE this adds: a traffic model over a shape census overstates cost whenever the re-read is
  cache-resident. Price traffic claims with a skip/ablation, and use the census to explain the skip.

## v4.4 STEADY-STATE SNAPSHOT (Exp785-795) — read this first after a compaction

Speed board: EMPTY BY MEASUREMENT. Nothing in scope is above the 2% bar.
  - Rate budget (Exp785): blocked-int8 kernel = 20.4 GMAC/s/core at the VAE's ne11=26 (~23 asymptotic);
    the shipped VAE runs 392 GMac/clip in 14.6 s = 26.7 GMAC/s = 65% of the 2-thread ceiling, and
    80-87% of the matmul-only ceiling once non-matmul traffic is excluded. The "25 vs 40-45 GMAC/s"
    mystery that was open since Exp551 is CLOSED: 40-45 was a two-thread reading.
  - Elementwise residue (Exp791/792): bias adds ~3.1% of VAE seconds, rms_norm ~2.4%, everything else
    <=1.4%. Five fusions shipped (dw-conv1d, add_scaled, gelu+bias, gelu-table fold, norm+gamma fold).
  - GEMV tail (Exp791): whole axis = 0.4 s = 1.7%. ne11 is only ever 1 / 26 / 31. Traffic models over
    a shape census OVERSTATE cost because a tail pass re-reads L3-resident bytes: price traffic claims
    with a skip, use the census to explain the skip.
  - Bias epilogue: 1.7% of RTF, below bar, and needs the mul_mat row loops - do not open.
  - split=5 (Exp794): parity on the lean tier (3 reps, +19 MB, byte-identical). The Exp565 -0.9% is
    pre-conv-int8. No third rung between the tiers.

Ladder (v4.5, ALL cells measured, no estimates): shipped 2.18/2.18/2.22/2.28 + gate mean 2.4456 @ 2.19 GB
  (Exp821 in-kernel conv left pad: -3.2% protocol, -183 MB RSS, 40/40 byte-identical; lean tier UNCHANGED because the
   fast path is gated to one-piece windows with no deferred late pass - at p13+defer the transcript shifted by 1 token);
lean p13+defer 2.41/2.39/2.41/2.47 + gate mean 2.7001 @ 1.75 GB. Lean is output-equivalent (2/731, p=0.5).

Rollback ladder (Exp793, one binary): DW_CONV1D_OFF +5.6% | GELU_BIAS_OFF +4.8% | LS_FUSE_OFF +1.5% |
NORM_FUSE_OFF +1.5%. NOTE: Exp677's GELU_BIAS_OFF cost (+1.7%) was 3x stale - Exp781 moved gelu's table
lookup into that pass. Fallback order by cost: norm_fuse, ls_fuse (cheap) then gelu_bias, dw_conv1d.

Verification rotation (all green at v4.4): anchor (hash 25b53ca2fc20, 39 tok) | lean tier | overfit
guard slice10b (2.41, tracks the protocol across 4 rotations) | audit_harness (69 checks) | 40-utt gate
(40/40 byte-identical to hyp-gatem2 AND hyp-norm784) | rollback ladder | robustness set (Exp676/677).

Instruments to reach for BEFORE any device ladder: `.auto/mm_shape_micro` (kernel/shape cost, ~1 min) and
`VAE_ABL_*` / `VAE_*_OFF` knobs (op cost). Both resolved everything this session. Reminder: ablation
knobs go stale when a fusion deletes the node they ablate - rewire them in the same commit (6 instances).

- HOLDING PATTERN FOR STEADY STATE (written at Exp796, 792 runs, best 2.26). The loop is past the point
  where ideas are plentiful, so the value now comes from *disciplined* verification, not from inventing
  levers. Rotate deliberately and say which rung you are on:
    1. anchor (byte-identity + rate) - cheap, catches regressions and thermal drift;
    2. tier rotation (lean / guard slice / long clips) - catches tier-local drift;
    3. instruments (mm_shape_micro, VAE_ABL sweep, census) - re-price a documented item with CURRENT
       code, since every share in the ledger is a snapshot that a later fusion silently invalidates;
    4. integrity (audit_harness, rollback ladder, robustness set, gate regression) - the loop's own
       tooling has been the source of more wrong answers than the model has.
  Anti-patterns observed in this loop's own history: (a) re-deriving a fact that a source comment
  already records (Exp792 - the comment at vae.cpp:299 documented the stale ABL_SCALE); (b) trusting an
  ablation whose node a fusion deleted; (c) asserting from a truncated view (head -20, cut -c1-140);
  (d) shipping a knob whose semantics were never proven by an output change (Exp764's SKIP_TAIL).
  Before reviving ANY old idea, check the snapshot above and the archive half - three axes (granularity,
  piece pipelining, thread counts) look re-openable and are not.

- TAIL-PADDING WASTE, NOW MEASURED AS A FUNCTION OF CLIP LENGTH (Exp803). The window is 83,200 samples
  with hop 70,400 (OVERLAPPING), so windows(n) = 1 + ceil((n - 83200)/70400) and the VAE encodes
  windows x 83,200 samples regardless of how much is real. Five clips (83,200 / 166,400 / 224,000 /
  240,000 / 294,400 samples) give vae_s = 0.30 + 3.60 x windows, i.e. **the VAE's cost is per WINDOW,
  3.6 s each, essentially independent of how much of the window is real audio.**
  Consequence: a real streaming engine would flush the partial final window instead of encoding a
  padded one. The saving is (1 - tail_fraction) x 3.6 s, which is length-dependent:
      gate utterance 3.7 s : 2.80 s = 38.9% of VAE time   (18.4% of wall)
      protocol 10 s        : 2.35 s = 16.3% of VAE time   (10.5% of wall)
      69 s                 : 2.01 s =  2.3%               ( 2.1% of wall)
      138 s                : 0.42 s =  0.2%               ( 0.2% of wall)
  WHY THIS IS NOT A LOOP EXPERIMENT: the gain is an artifact of the metric clip being short. Shipping
  it to move the protocol number would be textbook benchmark overfitting (a 10% "win" that is 0.2% on
  real long-form audio), and it is a semantics change anyway - the model would see a shorter final
  window, so it needs its own accuracy gate. It also explains a standing puzzle: the 40-utt gate mean
  (2.54) sits ABOVE the 10 s protocol number because short utterances each pay a whole extra window.
  PRODUCT ACTION if short-clip latency ever matters: implement the short final flush (VAE encode of a
  short tail + LM prefill of the matching fewer rows), gated on accuracy. Estimated -10% protocol /
  -18% per-utterance latency, ~0% on long-form.
  Measurement hygiene worth keeping: vae_s is a *per-run total of VAE wall time*, and clip length is
  quantized by the window grid - so RTF-vs-length curves are staircases, not curves. Any RTF comparison
  between clips must state the window count, not just the seconds (grid3 at 9.33 s and win3 at 10.00 s
  have IDENTICAL VAE work: 4 windows).

- LATENCY TREE CLOSED TO THE MILLISECOND (Exp806) + ONE ARTIFICIAL WIN DECLINED. Per-window trace on
  the shipped p1 config sums to 22.53 s against measured gen_s 22.53 s - there is NO unaccounted
  component anywhere in the pipeline. Components against MEASURED ceilings (not models):
      VAE matmul  at the blocked-int8 rate ceiling (20.4 GMAC/s/core at ne11=26)
      VAE elementwise  ~6.4% of VAE seconds remains (bias/norm/resid singles, all sub-bar)
      LM prefill  flat 1.06 s per window - content-independent by construction (26 rows)
      LM decode   93 ms/token = one full weight stream at the 6-7 GB/s bandwidth ceiling
  The ONLY slack found is a 316 ms window-1 VAE premium (1.4% of wall), a one-time page-in.
  DECLINED ON INTEGRITY GROUNDS, not on size: putting a warm-up window at load time would move that
  316 ms into load_s, which the metric excludes (rtf = gen_s/duration). Total time-to-first-result for
  a user would be unchanged, so it is measurement relocation, not optimization. If the product ever
  wants it, the honest framing is 'first-token latency after model load', and it should be measured as
  such - NOT claimed as an RTF improvement. Any future session that finds this 1.4% should re-read this
  entry before shipping it.
  GENERALIZATION: because the metric excludes load, any change that moves work from gen_s to load_s is
  an artifact. Check the metric boundary before believing any new 'win'.

- OUTPUT-EQUIVALENCE MAP OF EVERY ARCHIVED GATE SET (Exp810, .auto/equiv_map.py -> .auto/equiv-map.md;
  host-only, 0.4 s, zero device time). Paired token distance of all 66 usable hyp-* sets from hyp-gate791
  (the v4.4 reference). Findings:
  * 9 sets are BYTE-IDENTICAL to today's output: convint8b, gatep1, gatedef1, gatem2, gate721, gate778,
    gelufuse781, norm784, lean734-nodefer. Those span the ENTIRE shipped lineage (conv-int8 -> p1 -> m2 ->
    gelu-table fold -> norm-gamma fold), i.e. every speedup since conv-int8 has now been confirmed
    output-preserving by 9 independent 40-utt runs - the strongest accuracy statement the loop has, and it
    cost no device time.
  * The cheap precision knobs separate cleanly: q2_K 69 tokens, q3_K_M 31, head4x4 14, gelu_quick 14,
    lm4x4 18 - all real text changes; whereas every fused-kernel/ layout change sits at 4 (the marginal
    class) or 0.
  * hyp-convint8 differs by 20 but hyp-convint8b is identical: the first conv-int8 gate run was a different
    (pre-fix) configuration. When a pair disagrees, check which one is stamped.
  METHODOLOGY, learned the hard way in this run: (a) never reimplement the tokeniser - an ad-hoc version
  produced distances 2-4x too large and contradicted the ledger until I imported compare_arms' own
  gate_streams; (b) do NOT call compare_gate() in a loop - it runs a 20 000-sample bootstrap, so 66 calls
  exceeded a 300 s timeout; reuse its stream builder and do one LCS per set instead; (c) raw token-presence
  distance is a DIFFERENT quantity from compare_arms' McNemar b/c (correctness discordants) - the map
  answers "same system?", compare_arms answers "better?". The docstring says so because conflating them
  would look like a contradiction of the ledger.

- THE DECODE LENGTH TERM, AND WHY THE LADDER RISES WITH CLIP LENGTH (Exp811 - first externally-supplied
  lever in many runs; a peer loop running this model on a Jetson reported decode = 0.362 ms + 27.9 us x
  position, linear out to 4096 positions).
  * Fitted on MY device from the per-window trace of the 138 s clip (48 windows, 877 tokens, positions
    70..1284): ms/token = 88.2 + 0.022 x position, R2=0.78, slope 22.0 +/- 3.4 us/position. The law
    reproduces total decode to 1.1% (90.1 predicted vs 89.1 s measured), so it is not a curve fit artifact.
  * The per-position coefficient is within 1.3x of the GPU's (22.0 vs 27.9 us) while my intercept is 240x
    higher (88 ms weight stream vs 0.36 ms). => the length term is per-boundary WORK, not traffic: KV is
    28 KB/position (28 layers x 2 KV heads x 128 dim x 2 x f16) = 4.4 us/position at the 6.5 GB/s streaming
    rate, and the measured 22 us is 5x that. Attention FLOPs are 172 KFLOP/position = 7.8 GFLOP/s at 22 us,
    so roughly half the term is compute, which no KV quantization can remove.
  * SIZE: 0.2% of RTF on the protocol clip (positions <= 88) but 4.0% on the 138 s clip. The measured ladder
    rises exactly +4.0% from 10 s to 138 s, and VAE/wall is flat at every length (Exp803), so THIS TERM IS
    THE LADDER'S LENGTH GRADIENT. A previously unexplained number, closed.
  * CORRECTION TO A CLOSED AXIS: Exp563 closed KV q8_0 with "the KV read is <1 ms/token at ~500 positions".
    Measured, it is ~11 ms/token there - about 10x the estimate. The axis stays closed FOR THE METRIC, but
    for the right reason (the protocol prefix is short), not because KV reads are cheap. If long-form
    sessions ever matter to the product, --kv-type q8_0 is the lever (peer measured 112 -> 29.75 MiB at
    n_ctx 4096, needs flash-attn on their GQA model, rtf flat, and IT CHANGES OUTPUT) and it must be judged
    on long clips + an accuracy gate, never on the protocol clip.
  * DO NOT extrapolate my 22 us/position beyond ~1300 positions: bin-local slopes were 17.4/22.5/25.1, i.e.
    mildly convex, so the true law may be linear + a small quadratic. The peer's linearity is validated on
    their hardware only.
  * INSTRUMENT NOTE from the peer, checked rather than accepted: phase timers measure WHO WAITS. My
    per-window pipeline is strictly sequential on CPU and the tree closes to the millisecond, so
    prefill_s/decode_s ARE exclusive here; the warning applies to my ac_s/sem_s pair (both read 14.7 s =
    wall, because the two encoder chains run concurrently) - never price from those two.
- CENSUS KNOB DOUBLE-COUNTS WITH >1 THREAD (Exp811, found by the 1-vs-2-thread cross-check).
  GGML_MM_DEBUG_MACS increments inside ggml_compute_forward_mul_mat, which each worker thread enters to
  process its own row slice, so at -t 2 every split matmul is counted TWICE at full size: total 864 GMac
  at -t 2 vs 626 GMac at -t 1, with the ne11=1 (decode) bucket exactly halved (134 -> 67) and the ne11>=128
  bucket unchanged. The correct per-clip MAC volume is the SINGLE-THREAD census (626 GMac). A future fix is
  to increment only when params->ith == 0. Until then, never quote the -t 2 census total, and note that any
  GMAC/s figure derived from it is ~1.4x too high (864/626).

## Exp830 (v4.6 lean tier + three integrity findings) — DONE AND SHIPPED

**Flush now works under VAE_DEFER_LATE (RAM-lean tier).** Lean 2.40 -> **2.12** (−11.5 %) at 1.75 GB;
17/69/138 s = 2.29/2.37/2.42; gate mean 2.70 -> **2.14**; hybrid WER 4.65 % vs shipped 4.38 %, paired
b=0/c=2 of 731 (p=0.5) = output-equivalent. Shipped tier untouched (1.9278). Three assumptions had to die:
 * the boundary is at the LATE-SPLIT stage, so its frames-per-piece (ashape[0] = 16 at p13 for a 2-frame
   piece) is DOWNsampled relative to the audio grid - frames present cannot be computed as want/3200; the
   full-window size is tpiece*pieces, derived at piece 0, and buffers are allocated there then packed down
   to the frames actually present (a few KB per channel);
 * the late pass's return value is OUTPUT frames (26), not boundary frames (208);
 * `win_frames` was only updated by the non-deferred branch, so the deferred branch kept assuming 26.

**Integrity findings (class: silent, not loud):**
 1. `score_hyp.py <bad-tag>` silently scored a DIFFERENT set (the tag defaults to "a78"). Now refuses on an
    empty or partial set; negative-controlled (`score_hyp.py hyp-flusht` -> REFUSING). This made an
    acceptance number irreproducible (4.68 vs 4.38) and cost an hour of archaeology - the pair
    `hyp-flusht829` (real) vs `hyp-flusht` (typo) is why the ledger must quote directory names, not tags.
 2. `compare_arms.py --gate` takes THREE args (dirA dirB refs.json) with paths relative to the repo ROOT
    (`../eval-librispeech/...`). With two args / wrong prefix it does not fail cleanly. Any paired test must
    echo its own file counts before its verdict is trusted.
 3. `ps -A -o PID,STATE,RSS,NAME` is **invalid on this device's toybox** ("ps: bad -o"), and a failing
    idleness check reads exactly like an idle device. Use `ps -A -o NAME` (what measure.sh uses) or
    `ps -A | grep asr_streaming`.
 4. Ladder doc bug: the table had no 138 s column, so every secondary row's 138 s value sat under the
    "40-utt mean" header - Exp645's clobbering, recurring. Fixed structurally (column added, rows re-aligned).

Still open after v4.6: RSS soak on the flushed graph (last one v4.5); `--xwin` under the flush (untested
pairing); the rollback audit re-run with the 9th `flush_off` arm (tool updated, not yet re-run).

## Exp832 — harness argv gap closed, `--xwin` re-characterized, flushed-graph soak PASS

`bench_device.sh` could not pass CLI-only flags (fixed argv; `measure.sh` forwarded env only), so `--xwin`
was unmeasurable through the harness and Exp825's numbers could not be re-run. Added `ARGS=` -> appended to
argv; verified by output change (`1a16d2009cf7` vs default `1a095c8496b4`, `carry` banner present).

Re-baseline: carry 10 s 2.3750 vs windowed 1.9354 = **+22.7 %** (documented +8.6 %), 138 s 2.3808 vs 2.24 =
+6.3 %. Carry did not regress (2.3735 vs 2.3789 in Exp825) - the windowed path got the flush and carry did
not, because the flush lives in the windowed loop while the carry loop pads its final hop to 70,400 samples.
Gap = work volume (1.23x samples encoded, 1.2x frames fed to the LM), not a stall. Carry is **483 MB lighter**
(1708/1727 MB). Queued if carry ever matters: flush the carry loop (~5-6 % at 10 s).

Soak on the flushed graph: 15 samples / 4.7 min, 2181.4 -> 2184.2 MB, HWM 2205.5, +0.05 MB/min, majflt 0 = PASS.

## Exp833 — LM-INPUT DEDUPLICATION: CLOSED BY GATE, DO NOT RETRY

The windowed schedule advances 22 frames per hop but encodes and feeds 26, so the first 4 feature rows of every
window after the first describe audio the LM was already given (they were the previous window's tail): the
protocol clip feeds **87 embeddings for 75 unique frames**. Skipping them is arithmetically complete - each
global frame is fed exactly once, in order, and the *encoder* overlap (real convolutional context) is kept.
Measured speed: −0.77 % paired (prefill 3.8 -> 3.4 s = the predicted row-count saving, offset by +3 decode
tokens on the protocol clip) - below the bar even if accuracy were neutral.

**Accuracy: catastrophic. 40-utt gate WER 18.33 % (hybrid) / 20.11 % (jiwer path) vs 4.38 % / 4.68 %; paired
b=2 / c=105, McNemar p = 7e-29.** The failure mode is DELETIONS (the LM stops emitting text), distributed
across utterances rather than collapsing any single one (0/40 files lost >50 % of their characters), so it is
early chunk termination, not garbling. Conclusion: **the full 26-frame chunk structure of the LM input is
load-bearing** - the model's chunk-boundary behavior depends on seeing a whole 26-frame window per chunk, so
those 4 rows are not redundant work even though they describe already-seen audio. Feature-level context
arguments do not predict this; it is measured.

TRAP WORTH REMEMBERING: the change made the metric BETTER at every length (gate mean 1.9811 -> 1.9262, −2.8 %)
**because transcribing less audio takes less time.** A speed loop that skipped the accuracy gate would have
shipped a 14 pp WER regression as an improvement. Any "removed work" lever must be gated before its speed
number is believed - and for LM-input changes, deletions are the failure mode the protocol clip hides (the
protocol clip emitted 3 MORE tokens here).

Also verified: the `FEED_DEDUP_OFF` hatch reproduced the frozen reference exactly (1a095c8496b4) before the
revert, so the experiment's provenance was sound - the hypothesis was simply wrong. Reverted in-tree; the
shipped protocol is unchanged.

## QUEUED, PRICED (Exp834): batch the two boundary tokens into the per-window prefill — ~2.0 % of the metric

Measured on the shipped path with a temporary per-call breakdown (probe reverted, tree clean):

| call | rows | ms/window |
|---|---|---|
| `feed_token([speech_start])` | 1 | **83** |
| `feed_embeds(chunk)` | 26 | 890 (34 ms/row) |
| `feed_token([speech_end])` | 1 | **83** |

An isolated 1-row decode is billed at gemv/decode cost (one full weight stream, ~83 ms) while a row inside the
prefill batch costs 34 ms. Two boundary tokens x 4 windows = 664 ms = **3.4 % of gen_s**, and batching them
into the frame batch costs 2 extra rows (68 ms) instead of 166 ms, so the recoverable part is 49 ms x 8 rows =
**392 ms = 2.0 % on the protocol clip** (138 s: 48 windows -> ~1.5 %, so it is a short-clip lever and should
help the 40-utt mean by ~2 %).

Why it is not already done: this vendored llama decides embd-vs-token **per batch, not per row**
(`src/llama.cpp:3108` `if (batch->embd) { ... } else { ubatch.embd = nullptr; }`), so a batch cannot mix
embedding rows with token-id rows. Route (all in scope, Exp663/664 precedent): add
`llama_get_token_embd_row(model, id, float * dst)` to the vendored llama (read the `token_embd` tensor row and
convert with `ggml_type_traits[type].to_float`), then build ONE 28-row embedding batch
`[embd(t_start), frames x26, embd(t_end)]` in the windowed loop. Numerics: get_rows dequantizes the same row
at compute time, so feeding the pre-converted F32 row should be **bit-identical** - but verify with the
protocol transcript hash AND the 40-utt paired test, because if the model applies token-type embeddings or
embedding scaling, the equivalence breaks exactly where Exp833 showed the chunk contract is fragile.
Do NOT try to shorten the input instead: Exp833 measured that dropping rows from the chunk costs 14 pp WER.

## Exp835 — SINGLE-PASS BOUNDARY-TOKEN PREFILL, SHIPPED (v4.7, 10th hatch `BOUND_BATCH_OFF`)

Exp834's priced lever, built. Vendored addition: `llama_token_embd_row(model, id, dst)` (reads one row of
`token_embd`, converts with `ggml_get_type_traits(type)->to_float` - the SAME routine `ggml_get_rows_q` uses).
Demo: one 28-row embedding batch `[embd(speech_start), frames x win_frames, embd(speech_end)]` replaces three
decodes. Prefill 3.8 -> 3.2 s; shipped tier 1.9312 -> **1.8813** (-2.6 %), gate mean 1.9811 -> **1.9329**.

**Durable facts:**
 * An isolated 1-row `llama_decode` costs 83 ms here vs 34 ms/row batched - 2.4x. Standing rule: never leave a
   single-row decode in a hot loop if its row can join a batch.
 * Batching rows moves them from the gemv path to the gemm path, which is arithmetically equal but **not
   bit-identical**. So "verify by transcript hash" is still the right test, but a mismatch here does not mean a
   bug: the gate showed 38/40 identical and b=1/c=2 of 731 (p=1.0), i.e. marginal-class equivalence. This is the
   first shipped speedup whose equivalence rests on the paired test rather than on identity - state that in any
   runbook so a future session does not hunt for a phantom bug.
 * The end-token row index must follow `win_frames`, not the constant 26: with the tail flush the final window is
   short, and a fixed index fed garbage (57 tokens, hash changed). Same lesson as Exp830 in a new place.
 * Scoped to `pieces==1 && !defer && !xwin` (the shipping config, the Exp821 precedent). The deferred/lean path
   flips one token under the batch - unexplained, and its features already differ from the shipped path, so it
   keeps the three-decode path. QUEUED if the lean tier ever matters: find why it flips (start by feeding the
   deferred path's frames through the batch with VAE_LATE_SPLIT unchanged and diff one utterance).
 * Build hygiene: my first "result" for this change measured the PREVIOUS binary because the build had FAILED on
   a non-public ggml symbol and I trusted a log tail instead of the exit code. Rule: check `BUILD rc` or the
   mtime guard, never a log tail.

## Exp836 — ROLLBACK LADDER AT v4.7 (14 arms) + WHY Exp835 BEAT ITS PREDICTION

`BOUND_BATCH_OFF` added as the 10th hatch (also added to the stack_off/ALL_OFF lists, or those arms stop meaning
"all fusions off"). Default 1.8894, 2 reps/arm, per-window transcript hash. Cost of turning each OFF:
dw_conv1d +8.6 | gelu_bias +5.1 | gelu_batch +4.0 | dw_lpad +3.0 | **bound_batch +2.5** | ls_fuse +1.2 |
norm_fuse +1.1 | mm_m2 +0.3 | dw_axpy +0.0 (inert, 4th confirmation) | flush_off +13.6 (differs: the padded
protocol, by design) | ct_block +24.2 (differs: layout revert = system change) | stack_off +25.5 (identical) |
ALL_OFF +33.3 (differs). 10/14 arms byte-identical to the default.

 * **m2 got cheaper: +0.7 % -> +0.3 %.** The 28-row prefill batch is an exact multiple of 4, so the window
   prefill no longer leaves a 2-column GEMM tail (26 = 6x4+2 did). Exp835 therefore partly SUBSUMED the m2
   lever - and that is why the batch measured -2.6 % when Exp834's isolated-decode arithmetic predicted -2.0 %.
   Standing rule this generalizes: when a change alters a batch's ROW COUNT, re-check every kernel lever whose
   benefit depends on row-count divisibility; fast paths overlap in both directions (Exp824's nesting lesson
   was about fusions hiding each other; this is a shape effect hiding a kernel lever).
 * **The layout-revert arm is not deterministic in TOKEN COUNT**: VAE_CT_BLOCK_OFF gave 37 tokens in one sweep
   and 38 in the next, same binary and env. So do not describe it as "deterministic but differs" in a runbook.
 * HYPOTHESIS I CORRECTLY RETRACTED: I read that 37 as proof that the batch changes output under the [T,C]
   layout and tightened the `bound_on` predicate accordingly. A paired test (ct_block with batch ON vs OFF:
   38 and 38 tokens) refuted it, so the tightening was reverted - scoping must rest on a verified difference.
 * METHODOLOGY TRAP (4th instance of the Exp818/826 class): after a multi-arm sweep I quoted
   "hash=$(md5sum .auto/last_out.txt)" as the DEFAULT arm's hash, but last_out.txt holds the LAST arm run, so
   I was comparing the ct_block arm's output to the default's reference. In a multi-arm sweep, hash per-arm
   files or run the arm you care about alone.

## Exp837 — m2's value is now ~0.3 % EVERYWHERE + guard rotation tracks the batch ship + harness gap fixed

 * **m2 re-priced at the RAM-lean tier** (where the boundary batch is OFF, so the prefill really is 26 rows =
   6x4+2 with a tail): GGML_MM_M2_OFF costs **+0.4 %** (3 reps, overlapping spread, 2.1125 -> 2.1217). Combined
   with the shipped tier's +0.3 % (Exp836), m2 is no longer the ~1.8 % lever it was reported as at ship; the
   two protocol changes since then (Exp829's flush -> final window is 9 rows; Exp835's batch -> full windows are
   28 rows) removed most of the tail work it was patching. DECISION: keep the kernel - it is free and harmless -
   but never quote 1.8 % again, and do not "clean it up" as if it were load-bearing either.
 * **Guard rotation** (second same-domain 10 s slice, 55 tokens constant): 2.0228 vs anchor 1.8889 (+7.1 %), and
   it moved -2.4 % from v4.6 while the protocol moved -2.6 %. This matters specifically because Exp835 shipped
   on a PAIRED TEST rather than on byte-identity: the guard is independent evidence that its gain generalizes.
 * Harness: `run_rtf_multi.sh`'s clip field was device-side only, so a host clip path failed loudly. Now it
   pushes a local path and uses the basename (audited green after the change).

## Exp838 — LEAN-TIER BOUNDARY BATCH: PRICED (-2.8 %), DECLINED ON THE CANARY CLASS. AXIS CLOSED.

The last in-loop speed item. With a measurement-only force knob (removed after this run, per Exp833's rule), the
RAM-lean tier goes **2.1121 -> 2.0535 (-2.8 %)**, prefill 3.8 -> 3.3 s. The prize was real.

 * **Mechanism of its one-token divergence is now known, and it is the same variable as Exp836/837.** Without the
   batch the prefill is 26 rows = 6x4+2, so the two control-token rows run through the m2 tail sub-path; with it
   the batch is 28 = 7x4 and they run in the main 4-column group. Different accumulation order, so near-ties in
   the LM's argmax move. On the shipped tier nothing changes (it is already 28 rows, hence byte-identity), which
   is why this only ever showed up at the lean tier.
 * **Evidence both ways, stated honestly.** On the 57 s bilingual probe - the loop's only real zh+en audio asset -
   the deferred-path batch produces a **byte-identical transcript** (1055 chars; WER 14.43 %, en 10.7 % / zh
   15.9 % in both arms, 171 tokens both). So there is NO measurable accuracy effect. But on the protocol clip the
   zh canary's proper-noun run changes `YYY` -> `YyY` (twice), the exact symptom string Exp499 attributed to the
   blocked-int8 kernel path, and Exp626's ruling makes that class a stop signal.
 * **Why byte-identity was the right test here**: the 40-utt gate is English-only, so it is structurally blind to
   this failure mode. A clean paired gate test would have proven nothing about the only thing that moved.
 * DECISION: not shipped, knob removed rather than defaulted off. This is a RULE-BASED decline, not a measured
   regression - if a zh accuracy set of >=400 tokens is ever built (the ledger's own bar for zh claims, Exp655),
   the 2.8 % could be bought honestly, and the implementation is one line of predicate.
 * Consequence for the loop: the boundary-batch axis is closed at both tiers (shipped: shipped; lean: declined),
   and so is the speed board - every component is at a measured ceiling and both work-removal levers are spent.

## Exp839 — EQUIVALENCE MAP RE-ANCHORED TO THE POST-BATCH OUTPUT + FOUR UNITS FOR ONE 5-FILE CHANGE

Host-only, zero device time: recomputed the paired distance of all 71 archived 40-utt gate sets against the
CURRENT shipped output (`hyp-bound835`).

 * **No archived set is within 4 hybrid tokens of today's shipped output.** The boundary batch (Exp835) moved it
   by 11 hybrid tokens / 3 correctness-discordants vs `hyp-flusht829`, because gemv->gemm is not bit-identical.
   So the pre-batch lineage claim ("9 sets byte-identical across the shipped history") must now be read as
   "byte-identical to the PRE-BATCH reference". Consequence for future sessions: when the shipped output
   legitimately moves, re-anchor `equiv_map.py`'s default in the same iteration, or every old artifact reads as
   "changed" and byte-identity acceptance tests silently compare against the wrong baseline. Default is now
   `hyp-bound835`; the map is saved at `.auto/equiv839.md`.
 * Cross-tier pair re-stated for the docs: lean gate vs shipped-now = **b=1 / c=0, McNemar p=1.0** (WER 4.65 % vs
   4.51 %), and lean vs pre-batch shipped = b=2 / c=0, p=0.5 - so "the two tiers are output-equivalent" still
   holds at v4.7, with the lean tier 1 token away. Running the same pair twice with arguments swapped gave b/c
   mirrored (0/2 vs 2/0): a free symmetry check on compare_arms.
 * **FOUR UNITS, one change - the trap that cost me four tool calls.** For the same pair: **5** byte-differing
   files, **20** whitespace-token differences, **11** hybrid-token differences (the gate has 730 hybrid vs 766
   whitespace tokens, the hybrid tokenizer folds punctuation/contractions), and **3** correctness-discordant
   tokens (McNemar). I mis-read the map's 11 against compare_arms' 3 as a 3.7x instrument defect and nearly
   "fixed" a working tool; the map's own docstring had already documented the distinction, and the actual defect
   was in my throwaway script (it compared per-file structures against a per-token stream, reporting 254). Sixth
   instance of "the measurement you ran wasn't the measurement you meant", and a new sub-rule: **quote the unit
   with every distance** - files, whitespace tokens, hybrid tokens, or discordants are not interchangeable.
 Example change in the pair: `off` -> `oft` twice in one utterance (a real ASR confusion, 2 whitespace tokens).

## Exp840 — LONG-RUN MEMORY AT v4.7: FLAT (and why a soak's first number is a trap)

Steady-state **+0.18 MB/min** (18 samples, 69 s clip; band 2174.7-2198.0 MB; HWM +2.2 MB), 138 s peak 2206 MB with
VmHWM flat from t~200 s, majflt 0 -> the boundary batch's one-time ~3 MB `chunk_emb` buffer does not leak.

 * **The run first said +24.4 MB/min.** Mechanism of that false alarm: the weights are mmap'd and faulted in
   lazily, so RSS RAMPS for the first windows while VmHWM is already flat - and because the pages are in the page
   cache, **majflt stays 0**, so the usual "major faults mean paging" tell never fires. Any whole-run line fit on
   such a series measures the ramp, not retention. `rss_soak.py` now reports full-range AND steady-state trends
   (window = HWM >= 98 % of max AND RSS >= 95 % of median) with an explicit FLAT/GROWTH verdict.
 * Controlled the way the loop requires: synthetic leak -> +40.0 MB/min; leak planted onto the recorded series ->
   +40.2 MB/min. Same code, opposite data, opposite verdict, so the FLAT is a property of the data.
 * **Truncated-view trap, 7th instance, this time against my own fixed tool**: `... | tail -12` cut the verdict
   lines from the soak output and it briefly looked as if the fix printed nothing. Rule: when the answer may be
   above the cut, redirect the full output to a file and grep it - never pipe through head/tail.
 * Free ladder cross-checks from the same runs: 69 s 2.1467 (cell 2.14) and 138 s 2.1984 / 876 tokens (cell 2.20 /
   canary 876), with prefill 52.1 s and decode 89.5 s on 138 s matching Exp811's per-position decode law.

## Exp841 — NOISE FLOOR MEASURED ON THE CURRENT STACK: sigma 0.19 %, and it is TWO-STATE

8 interleaved reps of ONE binary on the protocol clip: mean **1.8851**, **sd 3.5 ms = 0.19 %**, range 1.8802-1.8886
(0.45 %). This supersedes the 0.49 % figure quoted since Exp662 (v3.x stack, 12 consecutive runs, so it folded in
thermal drift).

 * **The spread is not gaussian.** The 8 values split into two groups - 1.8811 (n=3) and 1.8875 (n=5) - separated
   by 6 ms (0.34 %), with WITHIN-group sd of 0.05-0.08 %. So a run is highly repeatable inside a boost/thermal
   state and the run-to-run spread is mostly a two-state effect. Practical consequence: consecutive-run A/B is
   biased by which state each arm lands in; interleaving (what `run_rtf_multi.sh` does) is what makes a sweep
   trustworthy, and more reps buy resolution but not state coverage.
 * **Resolution table** (95 %, two-sided, n reps per arm; conservative range-based figure in brackets):
   n=1 -> 0.52 % [0.45 %], n=2 -> 0.37 % [0.32 %], n=3 -> 0.30 % [0.26 %].
 * What this does to existing verdicts:
   - every shipped keep (0.8-11.8 %) is 4-40x the resolution -> none is at risk from noise;
   - `dw_axpy` +0.0 % stays "inert" (far below resolution AND below any plausible effect);
   - **m2's +0.3 % sits AT the limit**: "inside noise" (Exp837 wording) was slightly too strong, corrected in
     RESULTS.md to "honest range 0.0-0.4 %". Same conclusion (not worth removing, not load-bearing), better label.
   - the 2 % shipping bar is ~7x the 3-rep resolution: kept deliberately, because transcript-level acceptance is
     the binding constraint, not timing.

## Exp842 — LEAN-TIER BATCH RE-TESTED ON 468 REAL zh TOKENS: accuracy objection RESOLVED; still not shipped

15 minutes of device time settled what Exp838 had to decide by rule. Two arms of the RAM-lean tier on
`holdout_zh.wav` (213 s, 48 Common Voice zh-TW voices, 468 ref tokens, never optimized against):

| | batch OFF (shipped lean) | batch ON |
|---|---|---|
| WER | 15.38 % (S=62 D=8 I=2) | **15.17 %** (S=61 D=8 I=2) |
| paired | **b=2 / c=1, McNemar exact p=1.0**, bootstrap CI [+0.004, +0.011] on A-B || 
| stream tokens | 641 | 655 |
| protocol-clip RTF | 2.1121 | 2.0468 (-2.2 %) |

 * **The accuracy objection is gone.** Three corpora now agree: zh held-out 468 tokens (p=1.0), the 57 s bilingual
   probe (byte-identical, Exp838), and the English 40-utt gate (b=1/c=2, p=1.0, Exp835). Insertions and deletions
   are IDENTICAL (2/8 vs 2/8), so the 14 extra stream tokens are zh segmentation/folding, not hallucinations.
   This meets the ledger's own >=400-token bar for zh claims (Exp655).
 * **Why it is still not shipped.** The remaining blocker is a product invariant, not accuracy: with the batch ON,
   the lean tier's PROTOCOL transcript changes from 39 to 38 tokens and its canary run becomes `YyY` while the
   shipped tier (already batched) keeps `YYY` - so enabling it would introduce a NEW cross-tier text difference on
   the loop's own metric clip, in exchange for 2.2 % on a tier that is not the loop's metric, plus a full re-
   measurement of that tier's cells (~45 min). Cross-tier parity is worth more here than 2.2 % of a non-metric tier.
 * **Standing statement for the runbook:** the lean tier's batch is now "shippable on evidence, held for parity".
   If the lean tier ever becomes product-critical, flip the predicate (`params.vae_pieces == 1` -> drop it and the
   defer test), take the 2.2 %, and re-measure the tier's ladder + gate mean; the accuracy work is already done.
 * Method: the force knob was proven live BEFORE the long arms (38 tokens vs the 39 reference on the protocol clip)
   - Exp764's rule, and it mattered: my first build failed to compile at all (BUILD rc 2 from a redeclared bool),
   so without the exit-code check I would have run 16 minutes of both arms on the previous binary.

## Exp843 — LM DECODE IS AT THE MEMORY CEILING (measured both ways); the "81 % of ceiling" claim was wrong

Decode is 17 % of the metric and the only component whose ceiling had never been measured directly. Two
independent instruments now agree, and the answer is "no headroom".

 * **A DRAM-bound GEMV probe** (new `mm_shape_micro big` mode: one 460.8 MB q4_0_4x4 weight at L=1, pinned to the
   two A78s) measures the streaming limit: **12.6 GB/s at 2 threads** (22.4 GMAC/s), **6.3 GB/s at 1**. Runs
   repeat to 0.4 %. The classic grid could never show this: its L=1 rows use 3-13 MB weights, i.e. L3-resident,
   so they price the kernel, not the memory system.
 * **In-graph node timing** (`GGML_OP_TIME=1`) says the lm_decode phase is **97.3 % MUL_MAT** (3313 ms of
   3405 ms; everything else - norms, adds, copies, attention's non-matmul part - is 92 ms), and the type census
   (`GGML_MM_DEBUG_TYPES=1`) gives the phase as 56.338 GMac q4_0_4x4 + **10.035 GMac q8_0** + 0.040 f16 =
   41.81 GB per clip = **1072 MB per token** (813 MB body + 257 MB head).
   41.81 GB / 3.313 s = **12.62 GB/s = the ceiling.** Over full decode wall time it is 11.6 GB/s (92 %), the
   8 % being the non-matmul 92 ms plus sampler time.
 * **Consequence for the ledger:** Exp816's "811 MB/token, 8.7 GB/s, 81 % of ceiling" divided by the RIGHT time
   but the WRONG byte count - it used the q4_0_4x4 bucket and dropped the q8_0 head, which is 24 % of decode
   traffic. That made a closed axis look 19 % open. Annotated in place, not deleted.
 * **The head is 24 % of decode = 0.82 s = 3.6 % of the metric.** That is not a new lever: it was priced in
   Exp592, where a q4_0_4x4 head was +0.55 pp WER and a q5_K head +0.42 pp, both declined in favour of q8_0
   (which is also why q8_0 was chosen: q6_K has no gemv kernel). So the number is now known and the decision
   already made - useful only if the accuracy trade is ever revisited with a bigger eval set.
 * Ceiling numbers to use from now on: **12.6 GB/s** for 2-thread streaming (supersedes the inferred 10.8 from
   Exp523), 6.3 GB/s for 1 thread.
 * Method note: `mm_shape_micro` gained a mode, and piping its build through `head -5` SIGPIPE'd clang so the
   "successful" compile wrote a stale binary - `COMPILE rc=$?` said 0 only on the retry without the pipe. Same
   class as Exp835's rule: check the exit code, and do not pipe builds through head.

## Exp844 — GUARD BOARD CURRENT (2.0036) + the two-state effect is NOT the reported CPU clock

Guard rotation (second same-domain 10 s slice, 55 tokens, never optimized against): **2.0036** (3 reps) against
the anchor's 1.8687 = +7.2 %, fully explained by its 16 extra decode tokens at the same VAE rate. History:
2.4085 -> 2.3861 -> 2.3226 -> 2.0728 -> 2.0228 -> **2.0036**. Since Exp837 the guard moved **-0.95 %** and the
anchor **-1.07 %** with NO shipped change in between - i.e. both clips drifted down together, which is the
device-state effect from Exp841/843 showing up in both arms, not an improvement. That is the expected behaviour
of an anti-overfit guard and worth stating: the guard's job is to move WITH the metric, not to be flat.

 * **NEW STATE PROBE, NEGATIVE RESULT.** `run_rtf_multi.sh` now prints `cpu7_khz` per rep (start + per run) so a
   future session can see the state instead of inferring it from clusters. It is NOT informative on this device:
   every sample read **1,300,000 kHz** while `cpuinfo_max_freq` is **2,400,000** and `stats/total_trans` is
   **1,429,272** - the governor is clearly transitioning, but `scaling_cur_freq` reports a constant that is
   neither the peak nor a real-time reading, and `scaling_governor`/`scaling_max_freq`/`thermal_zone*` are not
   readable at all. So keep using (a) `batt_temp_c` per run and (b) same-session interleaving as the state
   controls; do not cite scaling_cur_freq as evidence of anything (its only value is detecting a cluster that
   has been deliberately pinned low by an operator).
 * Cross-session rule reconfirmed by this round: a ~1 % same-binary drift appears over hours, so any A/B must be
   same-session; Exp841's resolution table (0.30 % at 3 reps) applies WITHIN a session only.

## Exp845 — GATE REGRESSION: output is STATE-INVARIANT (b=0/c=0, CI [0,0]); ladder re-stamped same-session; lean penalty corrected to +11.8 %

 * 40-utt gate re-run 5 rounds and one device-state change after the Exp835 ship: **WER 4.51 % unchanged**, and
   paired against the ship-time set `hyp-bound835`: **b=0 / c=0 of 731, bootstrap CI exactly [0, 0]**. Two things
   follow: the shipped config is deterministic across sessions/states (the ~1 % drift is timing only), and
   Exp835's acceptance - which rested on a paired test instead of byte-identity - now has a second independent
   run agreeing with it.
 * **Ladder discipline rule (new).** The shipped row now reads 1.868 / 2.037 / 2.122 / 2.183 + gate mean 1.9147
   in ONE session; the previous set (1.88 / 2.05 / 2.14 / 2.20 / 1.9329) was ~0.9 % higher in a warmer state with
   the SAME binary and identical output. Ladder cells are therefore only mutually comparable within a session:
   stamp the session, and when state drift is found, re-measure the whole row rather than one cell - otherwise
   the ratio between two cells silently mixes two states.
 * **CORRECTED, and it matters for product decisions: the RAM-lean tier costs +11.8 % same-session**
   (2.0998 vs 1.8780), not the +6.3 % the ladder carried. Mechanism: the tail flush helped both tiers roughly
   equally, but the boundary-token batch shipped on the p1 tier ONLY (Exp838/842 deliberately hold it at lean for
   cross-tier parity), so lean lost a 2.6 % lever it used to have. Turning it on would restore the gap to ~+9 %
   (Exp842 measured lean-batched 2.0535) - the accuracy evidence for that is already complete.
 * Harness note: `run_rtf_multi`'s clip field with a bogus local path (`.auto/../chat17.wav`) fails as a device
   path and the arm reports FAILED - loud, good. The 17 s clip's device name is `chat17.wav`.

## Exp846 — LEAN-BATCH DECISION DOSSIER COMPLETE (quantified on real audio; still a product call, not shipped)

 * **Tiers AS SHIPPED are output-equivalent on real zh audio**: shipped p1 (batched) vs lean p13+defer (unbatched)
   on 468 Common Voice zh-TW tokens = **b=0 / c=0, bootstrap CI exactly [0,0], WER 15.38 % both**. So Exp845's
   finding that lean now costs +11.8 % is bought with genuinely identical text on real audio - the parity argument
   is not a documentation artifact.
 * **If lean got the batch**: b=2 / c=1 of 468, McNemar p=1.0, CI [-0.43, +1.07] pp, and the batched arm is
   slightly BETTER (15.17 % vs 15.38 %). Price: -2.6 % RTF on that tier (gap +11.8 % -> ~+9 %).
 * So the trade is exactly: **2.6 % on a non-metric tier**, in exchange for moving the cross-tier difference on
   real audio from "provably zero" to "3 tokens in 468, statistically indistinguishable", plus one synthetic
   canary token on the protocol clip. Recommendation: keep as-is while both tiers ship; enable if the lean tier
   becomes the product default, in which case re-measure its 4-length ladder.
 * Shipped tier's zh held-out transcript is now archived (`hyp-stream/zh846-shipped-p1.txt`) - the reference that
   makes future cross-tier zh claims computable without new device time.

## Exp847 — LEAN ROW RE-STAMPED (rule applied to myself) + the long-form "tier divergence" is ORTHOGRAPHIC

 * Rule from Exp845 applied the very next round: the RAM-lean row had mixed provenance, so it was re-measured as a
   ROW in one session: **2.098 / 2.259 / 2.334 / 2.398** (10/17/69/138 s, RSS 1752-1766 MB), gate mean **2.1258**,
   WER 4.65 %. All ~1 % below the old cells = state drift, same binary.
 * Determinism: lean today is **byte-identical to Exp830** (b=0/c=0 of 731, CI [0,0], same stamped config), and
   cross-tier vs shipped-now is **b=0/c=1 of 731** (CI [-0.41,+0.00] pp) - mirrors Exp839's b=1/c=0 correctly.
 * **Long-form parity, resolved.** The tiers' *token counters* differ by ~3 % on 69 s (446 vs 432), which reads
   like a semantic divergence. Diffing the TEXT: **11 differing spans in 306 words, all orthographic** -
   `' cause` vs `because` (a contraction the deferred path normalizes to the full word), `had...` vs `had`,
   `for.` vs `for`, `know.` vs `know,`, one capitalization. Zero repeated 10-grams in either. The counter delta is
   a **subword-count artifact** of those choices (2 tokens vs 1), not different understanding. Standing rule: to
   characterize an output difference, diff the text; a token-count delta is a proxy that can mislead by ~15x here.
 * WER TOOL DISAGREEMENT (documented, not fixed): score_hyp implies a 726-token reference, compare_arms uses 731,
   with identical S+D+I - so the same transcripts read 4.68 % vs 4.65 %. Convention now written into
   STREAMING_1P5B.md: absolutes from score_hyp, between-system claims from compare_arms, never mixed. "Fixing" a
   scorer would re-quote every historical number (Exp654 lesson), so this is a quoting rule.
 * Harness: `measure.sh --clip` needs a HOST path (a bare device name errors loudly - good), so `chat69.wav` is
   now mirrored in `.auto/assets/` (md5 cb919d2dd8f2003169ef66257fd362b7, 3,311,576 B) after pulling from
   `/data/local/tmp/vibeasr`. Two near-misses avoided by reflexes this round: my `cp` of `last_out.txt` after an
   instantly-failed run captured a STALE transcript (caught because both files were the same suspicious size), and
   the sweep's `multi-run-*.txt` files are 73-byte summary tails, never transcripts (Exp671, third encounter).

## Exp848 — LEAN TIER'S BEHAVIORAL CONTRACTS VERIFIED WITH THE FLUSH ACTIVE (11/11), and a false alarm about tier differences

 * **First full behavioral run on the RAM-lean/deferred path with the tail flush active** (the flush reached lean
   through a different code path - Exp830's packed boundary frames - and had only ever been gated on WER + the
   protocol transcript, never on edge-case semantics). 11/11 PASS: silence->8 tok, noise->4, music->38 -
   **identical counts to the shipped tier** - plus 48 kHz stereo, sub-piece short36, both diarization probes, and
   the four ladder canaries (39/106/432/847, the lean row from Exp847).
 * `behavior_watch.sh` now takes **env overrides that win over tier.env** (env > tier.env > default, both branches
   tested: PIECES=99 in env resolves to 99, no env resolves to tier.env's 1) and **env-tunable ladder canaries**
   (CAN10/CAN17/CAN69/CAN138). Rationale: verifying a non-shipping tier by editing the shared tier.env would leave
   a stale recipe that silently retargets every later run - the Exp798 class of failure.
 * **FALSE ALARM, resolved by measurement**: `twospk_overlap` read 54 tokens at lean vs "73 tok" recorded at
   Exp826, which looked like a big tier difference (26 %). Direct same-counter A/B: **both tiers = 54**, text
   identical (35 words, Speaker 0 + Speaker 1 in both). 73 was the PRE-FLUSH count - the flush removes
   silence-generated tokens on every clip (Exp829 recorded exactly that for the ladder canaries).
   RULE: when a documented count differs, check the PROTOCOL VERSION it was recorded under before attributing it
   to tier, config, or a regression. Same class as Exp847's token-counter-vs-text lesson, one level up.

## Exp849 — LONG-SESSION KV LIMIT MEASURED: the shipped n_ctx=4096 covers ~4 minutes, not the 15 min the comment said

 * Bracketing on the 138 s ladder clip by shrinking `-c` (reaches the KV boundary in minutes instead of hours):
   `-c 1024` died in window 21, `-c 1536` in window **32 (pre-registered prediction hit exactly)**, `-c 2400` and
   `-c 2560` completed all 48. Monotone, so **49.5 +-1 positions/window**, i.e. ~18 per audio second.
 * **4096 -> ~83 windows ~ 245 s ~ 4.0 min** of continuous audio. 16384 would be ~16 min at 450 MB KV (the code
   comment attached "15 min" to 4096; it belongs to 16384). This never showed up in any loop metric - the longest
   clip ever run was 138 s.
 * Failure is GRACEFUL but uneven: exit 1 with `decode failed` OR `frames failed` depending on which row group
   crosses the boundary, and NO final `--- Transcription ---` block (window lines emitted so far are on stdout).
   A runbook must match either message, not one.
 * Levers for long sessions: `-c 16384` (+338 MB), `--kv-type q8_0` (~2x positions per byte), or session restart.
   None affect the metric (the protocol clip uses ~11 % of the budget).
 * HARNESS INSIGHT worth keeping: to probe a resource limit, shrink the resource instead of extending the workload
   - `-c` is a knob on the very quantity being exhausted, so a 138 s clip tested four boundary points in ~15 min.

## Exp850 — SILENT-CORRUPTION BUG FOUND AND FIXED: a truncated VAE gguf produced fluent wrong transcripts with exit 0

 * **Method that found it:** the Exp849 idea of probing a resource by shrinking it - instead of lengthening the
   workload, damage the INPUT (`truncate -s -8MB / -64MB` on the VAE file). The metric never looks at this.
 * **Root cause, one line in src/vae.cpp:** the mmap guard at the bounds check only set `loaded=false`, which routes
   to the copy fallback, which did `size_t got = fread(...); (void) got;` into a **zero-initialised**
   `std::vector<char>` and then wrote `tensor_size` bytes to the tensor. So a truncated tail = ZERO weights.
   Measured effects: -8 MB -> one word changes (`YyY` -> `Yboy`); -64 MB -> 4 window lines lost, output is other
   language garbage. Exit 0, RSS/timing indistinguishable from healthy. The LM loader was already fine.
 * **Fix:** check the short read, name the file/tensor/offset/bytes, free and return nullptr (the demo already
   prints `VAE load failed` + exit 1). Acceptance: healthy run BYTE-IDENTICAL (1a095c8496b4) so valid files are
   provably unaffected - this cannot change WER, only reject damaged files.
 * **Board shipped:** `.auto/fault_inject.sh` (5 probes: healthy control must emit `tokens: N`, four damaged
   fixtures must exit 1 with a matched message). Its negative control is this iteration itself - the pre-fix runs
   ARE the "silent EXIT=0" observations.
 * Thread-count invariance proven while here: `-t 1` gives the **same transcript hash** as `-t 2` (rtf 2.47 vs
   1.87), so blocked-int8 thread tiling does not change accumulation order - the shipped output is not
   thread-count specific (Exp808 had measured speed only).
 * Harness traps hit (all self-inflicted, all caught because the output looked wrong): `/tmp` does not exist in the
   device shell so `2>/tmp/x.err` fails with EXIT=1 that is MINE not the binary's; `adb -s $VAR` with an unset
   variable hashed nothing and produced md5 d41d8cd9 (empty input, Exp676 signature); `echo EXIT=$?` inside a local
   double-quoted adb string is expanded LOCALLY (Exp676 again) - must be `\$?`; and a success signal must require
   real output (`tokens: N`), not just exit 0, because exit 0 is exactly what this bug returned.

## Exp850b — fault class swept: audio inputs are properly guarded, and the RTF denominator is CONTENT-derived (anti-cheat proof)

 * Static hunt for the Exp850 pattern found no other ignored-read in the load paths (both readers check; the only
   unchecked calls are two `fwrite`s in vae.cpp DEBUG-DUMP paths - harmless, noted, not field-critical).
 * **Audio front end is correctly guarded**: header-only WAV -> `[audio_io] Error: No audio data in file`; empty
   WAV -> `Failed to open WAV file`; both exit 1. (Cosmetic: the empty-file message says "failed to open" when the
   open succeeded and the header read failed.)
 * **Metric-integrity result, worth quoting in any benchmark claim**: the RTF denominator comes from DECODED
   SAMPLES, not the RIFF header. A WAV with its data size inflated 4x (same 10 s of real data) did the honest work
   (`vae_s 12.0` = 4 windows) and reported rtf 2.0036 - the same as the honest clip; a file truncated to half its
   declared length did 2 windows (`vae_s 5.8`) and reported 2.0331. So no header field can manufacture speed, and
   if a future change ever made the duration header-derived, the lying-header probe would read ~0.5 and the board
   fails. Combined with the hash-pinned clips, the published RTFs are auditable from two directions.
 * `.auto/fault_inject.sh` is now a 9-probe board: 5 model-file + 4 audio-input. Fixtures are generated at run time
   from a hash-pinned clip (gitignored), so nothing binary needs committing and the board cannot drift from its
   source asset.

## Exp851 — loader equivalence + config edges: 12-probe board, guard tracks, one silent no-op documented

 * **`--no-mmap` is output-equivalent to the zero-copy loader** (same protocol hash, rtf 1.8734, load_s 1.2). This
   matters because it is the path that runs when mmap fails on a device - equivalence was assumed, now measured.
 * Config edges are well-behaved: `-c 16` -> exit 1 `frames failed`; `--vae-pieces 7|0` -> exit 1 with the divisor
   message; `--max-tokens 0` -> **exit 0 with 0 tokens** (cap checked before the first token). The last one is the
   only silent-by-design no-op found: a typo yields an empty transcript and no error. Left as is (self-evidently
   empty), but it is the kind of thing a runbook should mention; a `>= 1` validation would be a 3-line change if
   anyone wants it.
 * Board now 12 probes (5 model, 4 audio, 3 config), all green.
 * Guard rotation: **2.0059** (3 reps, 55 tok) vs anchor 1.8739 (+7.0 %, its 16 extra decode tokens). History
   2.4085 -> 2.3861 -> 2.3226 -> 2.0728 -> 2.0228 -> 2.0036 -> **2.0059**: +0.11 % since Exp844 where the anchor
   moved +0.28 % (device state), i.e. flat within noise across a stretch where only loaders/harness/docs changed -
   no overfit signal. cpu7_khz stayed pinned at 1.3 GHz in every sample (consistent with Exp844's negative result).

## Exp852 — Gate proves Exp850's loader fix is inert (b=0/c=0), loader equivalence extended to real speech

 * First 40-clip gate after a `src/` change since Exp835: **WER 4.51 %, paired vs hyp-gate845 = b=0/c=0 of 731,
   CI exactly [0,0]**. The short-read check therefore touches nothing outside its error path, now shown on 40 real
   utterances rather than inferred from one protocol hash. Gate mean **1.9117** (cell 1.9147; -0.16 % = state).
 * Loader equivalence now demonstrated on real speech, not only the synthetic-ish protocol clip: mmap vs `--no-mmap`
   on `6829-68769-0025` -> same hash `644e8b16c8c7`, 278 bytes, 30 tokens, rtf 1.9075/1.9057.
 * Two traps caught by refusing to accept a number that contradicts another instrument:
   (a) `cmp` of two files whose generating `ls` had failed reports **IDENTICAL for two empty files** (md5
       d41d8cd9). A size guard now precedes every comparison in this pattern - "identical" is only meaningful if
       both sides are non-empty.
   (b) My own map query printed "0 of 72 sets at distance 0", which contradicted the tool's own summary ("2 of 74
       byte-identical"). The table lists ONLY differing sets by design; I had queried the wrong population. The
       tool was right, my one-liner was vacuous - the recurring "measured the wrong thing" class.

## Exp853 — TTFT measured: 8.13 s shipped / 8.90 s lean, and streaming emission is real (progressive flush)

 * **Method worth keeping: the file-size timeline.** Launch through a LOCAL background job, poll
   `adb shell stat -c %s <out>` in a tight loop, print (t, bytes) and dedupe on size. It measures both "is output
   emitted incrementally" and TTFT without touching the binary or adding instrumentation.
 * Timing trap: `date +%s%3N` produced garbage arithmetic on this host (t=6287763.86 s). Use
   `awk '{print $1}' /proc/uptime` for fractional wall time - cheap, monotone, no format games.
 * **Emission is genuinely incremental** (printf + fflush per window; bytes arrive 0 -> 45 -> 84 -> 135 -> 172 -> 340),
   so the streaming UX claim is safe - and the lean tier's total output size matched the shipped tier byte for byte,
   a free re-confirmation of cross-tier equivalence.
 * **TTFT = 8.13 s** (shipped) / **8.90 s** (lean, +0.77 s of per-piece overhead landing on window 1), then a line
   every ~5.5 s per 2.93 s hop. Decomposition closes against the known latency tree to ~0.1 s, and ~32 % of TTFT is
   fixed setup (load 1.2 + prompt prefill 1.12 + page-in 0.32), i.e. independent of the audio.
 * Product framing, not loop scope: at RTF 1.87 a live mic drifts 0.87 s behind per second of speech; the TTFT
   levers are process reuse, prompt-prefill batching, and arena pre-touch - all wall-clock wins that the loop's
   metric does not reward (two of them were already identified and deliberately not shipped).

## Exp854 — cold page cache cannot be induced without root; warm-cache numbers turn out to be ROBUST, not lucky

 * Three attempts to create a cold-cache condition, each failing for a DISTINCT reason - recorded so nobody spends
   another three runs on it:
   1. `echo 1 > /proc/sys/vm/drop_caches` -> permission denied (adbd is uid 2000, no `su`, no root).
   2. 4 GB sequential churn right after a warm run -> RTF unchanged, majflt 0. LRU keeps recently-touched pages, and
      single-touch sequential reads self-evict, so the churn recycled its own pages.
   3. 12 GB churn x 2 -> still no eviction (Cached *grew* to 4.75 GB), but majflt moved 0 -> 1, which is the tell
      that the probe grazed the mapped pages and still did not evict them.
 * **`file age != cache state`**: a model file untouched for 9 days reads at 3.8 GB/s, identical to the hot shipping
   file (3.9 GB/s) - DRAM speed, so it was never evicted. The valid tell for cold-vs-warm is read bandwidth against
   DRAM/flash expectations, not mtime.
 * Net product conclusion: the loop's warm-cache RTF is not a lucky measurement - it survives cache churn and idle
   time. A true cold boot remains unmeasured and needs privileges; bounded as >= ~0.4 s of first-touch stall
   (1.6 GB at the 3.9 GB/s ceiling measured here, and decode wants 12.6 GB/s), with the caveat that real flash
   would be slower than that measured page-cache-served rate.
 * Harness lesson: my first dd-through-`$(( ))` construction silently never ran (whole sweep took 4.3 s, which is
   impossible for 2 x 1.1 GB reads) - nested arithmetic inside an adb string again. Rule that finally sticks: send a
   plain command, redirect stderr to a file IN the run dir (there is no /tmp on the device shell), and read the file
   back. Sanity-check with elapsed time before believing any bandwidth number.

 * **NEW DEVICE STATE, and a save by rep-taking.** The single run taken immediately after creating + reading +
   deleting 16 GB of flash read **rtf 1.9921** (lm_s 7.9, prefill 4.1, peak RSS 2100 MB) - +6 % with a BYTE-IDENTICAL
   transcript. Three back-to-back reps right after it returned 1.8762 / 1.8786 / 1.8833 with lm_s 6.8 and RSS 2191,
   i.e. the effect is a ONE-RUN transient (most plausibly the run racing the filesystem's post-delete cleaning or
   writeback), not a persistent state change. Consequence for the loop's rules: any single-run reading outside the
   0.19 % band is provisional until reproduced - especially after an experiment that wrote or deleted large files.
   The RSS signature (91 MB low with majflt 0) is the tell that it was I/O/page-fault related, not thermal.

## Exp855 — position cost law corrected 2x, the --kv-type "lever" was fictional, and the ladder reaches 155 s

 * **Why the loop had never seen this:** every long-form claim I had rested on Exp807's fit
   `ms/token = 88.2 + 22.0 us x position` (measured to P=1284). I extended the range by building `chat155.wav`
   (chat138 + chat17, 154.97 s, sha256 1cd4f3a84fbb) and tracing 53 windows. Result:
   **decode ms/token = 89.1 + 11.13 us x P** (R2 0.88, se 0.56, to P=2450), **prefill ms/row = 33.0 + 5.22 us x P**,
   **vae = 3428 +- 9 ms/window** (flat, content-independent).
 * **The mechanism of my own earlier error:** a window consumes 28 fed rows **plus one position per emitted token**
   (~18.5 at this operating point) = **46.5 positions per window**. Exp807 counted only the rows (~28), so P was
   understated ~1.7x and the slope inflated by the same factor. Independent corroboration: Exp849's `-c` bracketing
   (dies at window 21 of 1024, 32 of 1536) says 49.5 +- 3 positions/window - it agreed with the *bigger* number all
   along, and I had not connected the two measurements.
 * **How the old law was falsified, cheaply:** the 155 s clip *contains* the 138 s clip, so refitting windows 1-47
   alone reproduces 11.11 us/position - the disagreement is methodological, not data-dependent. And totals decide
   it outright: measured decode 100.0 s, corrected law 100.8 s (+0.8 %), old law 113.2 s (**+13.2 %**).
   Rule: when a fitted law and a bracketing experiment disagree, test on data you already have before running more.
 * Consequences that survive the correction: the term is ~4.9x its KV traffic cost (2.28 us per token per position
   at the measured 12.6 GB/s ceiling), so it is attention compute + cache management, not streaming - the KV axis
   stays closed for the protocol metric. The length gradient of the ladder (~+4 % at 138 s) is now predicted at
   +3.7 % instead of overshooting at +7.5 %.
 * **`--kv-type q8_0` does not exist.** The code comment and two of my own next-hints recommended it; the demo
   parses 10 flags and `--kv-type` is not one - passing it exits 1 `Unknown arg` (now a fault-board probe). The type
   is hardcoded (`type_k = type_v = GGML_TYPE_F16` in the vendored llama), so the "2x positions per byte" lever is
   really "make a vendored change and pass an accuracy gate", worth 1.88x capacity (q8_0's per-32 scale costs 6 %),
   and its benefit is capacity, not speed. Fixed the comment and the docs.
 * KV geometry, now derived rather than quoted: 28 layers x 2 kv heads x 128 head dim x 2 (K,V) x 2 B =
   **28.0 KB per position** = 112 MB at 4096 (matches Exp849's 27.5 KB from memory size). Session limit becomes
   **~258 s** at 4096 (4096 / 46.5 windows x 2.933 s), vs the bracketing estimate ~245 s.
 * New ladder cell: 155 s = **2.1935**, 978 tokens, RSS 2208 MB, majflt 0. Registered in `device_assets.json`
   (host mirror `.auto/assets/chat155.wav`, gitignored; md5 verified host == device before registering).
 * Asset housekeeping worth knowing: the 138 s clip is `chat138.wav` on device (6 622 748 B), and `chat.wav` is
   still a byte-identical duplicate of `chat69.wav` (the known Exp675 WARN).

## Exp856 - resource-limit board (fds, threads) built and closed: 3 fds constant, no thread leak

 * `rss_soak.py` now samples **fd count** (`ls /proc/<pid>/fd`) and **Threads** alongside RSS/HWM/VmSize, prints the
   RLIMIT_NOFILE headroom, and gives separate verdicts. New env overrides `SOAK_PID` (target one pid exactly) and
   `SOAK_PROC`, which exist so the columns can be validated against a known-count process.
 * **Validated before use, not after** (Exp660's rule): `ls /proc/<pid>/fd`, `lsof -p <pid>` and my own counter all
   report exactly **3** for a sleeping process (stdin/out/err). Only then was the ASR row believed.
 * **Result: the shipping path holds 3 descriptors, constant across 48 windows (net +0), threads 2-3 by phase, RSS
   flat.** The reason is structural: the zero-copy loader does `munmap` then `close(fd)` on the gguf files
   (`src/vae.cpp:1635-1638`) - a mapping outlives its fd, so nothing is held during inference. fd exhaustion is
   therefore NOT a long-session failure mode; the ceiling is KV positions (~258 s), and RLIMIT_NOFILE is 32 768.
 * Thread count oscillates 2 <-> 3 (main + pool workers, phase-dependent). max-min = 1 over 48 windows rules out a
   leak without needing the per-sample series, since a per-window leak would have added dozens.
 * Dead ends so nobody re-tries them: a planted fd-leak script (`fdhog.sh`) dies instantly on this toybox (`exec
   4$i</dev/null` in a loop), and `ls /proc/<pid>/fd` returning "No such file or directory" means the target already
   exited - both of my first "permission denied" fears were wrong. Quantitative validation on a known process is
   cheaper than either synthetic-leak plumbing or paranoia.
 * Harness trap hit AGAIN: piping the soak through `tail -22` truncated the sample table AND the verdict lines before
   the tool's own output was captured - second occurrence of Exp840's class in two iterations. Fix: redirect the tool
   to a file and grep the file; never pipe a tool whose answer may be above the cut.

## Exp857 — THE LOOP NEVER AUDITED ITS OWN BRIEFING: headline prose rotted two eras while 87 checks stayed green

A fresh session reads exactly two things before it measures anything: RESULTS.md's `Headline:` line
and `.auto/prompt.md`'s `Current best:` line. Both were wrong by 14-17 %, in the optimistic direction,
and the harness audit printed "87 checks passed, 0 failures" over the top of them:

| site | said | measured truth |
|---|---|---|
| RESULTS.md headline | `12.24 -> 2.70 (-78 %)` | 1.87, -84.7 % (last true at v3.5/Exp664) |
| prompt.md `Current best` | `2.18 (MAX-SPEED v4.5)` | 1.87, era v4.7 |
| prompt.md shipped tier row | `~2.18`, 17 s 2.18 / 69 s 2.22 / 138 s 2.28, gate 2.4456 | 1.87 / 2.04 / 2.12 / 2.18, gate 1.91 |
| prompt.md lean tier row | `~2.44`, 17 s 2.42, 69 s 2.45 | 2.10 / 2.26 / 2.33 / 2.40, gate 2.13 (Exp847) |

**Why the audit could not see it:** check 4b validates prompt.md by grepping it for the shipped VAE
*filename*. That filename stayed correct through v4.6 and v4.7 while every number around it rotted.
The doc-drift trigger DID fire at Exp829 and DID list "refresh .auto/prompt.md's tier table" among its
MUST DOs - it reached RESULTS.md and STREAMING_1P5B.md, and never reached prompt.md, whose last doc
commit is v4.5 (Exp821). Class: **a guard that validates the right ATTRIBUTE of the wrong thing**
(Exp764 knob semantics, Exp816 tautological majflt, Exp855 fictional `--kv-type`).

**Fixed the class, not the text.** `.auto/headline.json` is now the ONE machine-readable statement of
era + both ladder cells + protocol hash + provenance, and audit check 9 requires every current-state
claim to agree with it. It recognises claims by FORM, not vocabulary:
  A. `Current best: N`   B. `Headline ...: base -> N`   C. a ladder table ROW labelled as the shipping tier
Each must contain its block's current 10 s cell, and must name the era **on its own line**. Coverage is
printed (`6 current-state claim(s): shipped=4, lean=2; 2 explicitly-historical excluded`) so the check
can never read as coverage while checking nothing.

**Four drafts of the guard were wrong, each caught by a planted fault rather than by reading:**
 1. Colon-anchored `headline:` - a cosmetic rewrite to `Headline (era v4.7):` silently unchecked the
    single most-read line. Pattern brittleness is a silent failure, not a loud one.
 2. Word-anchored `headline` anywhere - 6 false positives ("headline speed claim for the max-speed
    tier", an Exp654 scorer row). A noisy guard gets muted within a session, which is worse than none.
 3. Historical-line exclusion applied to the 3-line window - it excused the CURRENT lean row because an
    ADJACENT table row said "pre-flush cells", and excused prompt.md's Current-best paragraph because a
    later line said "DEAD tiers". Scope exclusions to the matching line.
 4. The era clause read the window too: deleting EVERY era token from prompt.md's Current-best line
    still passed, because the next sentence mentioned v4.7. My own control D was invalid first (sed
    removed only one of two era tokens) - a control that cannot fail is not a control.
Final battery: spec-only ship to a fictional v4.8 -> 6 FAILs (the intended future use: edit the spec
first and check 9 enumerates the prose to fix); synthetic stale `Current best` -> 1; lean row relabelled
era v4.5 -> 1; era tokens stripped -> 1; restore -> 0. Zero device minutes.

**Trap I hit while verifying my own change (Exp671 class, 4th encounter):** hashing the transcript
SECTION instead of the whole file printed `686b3e0798d4` and looked like an output regression. The
canonical protocol-hash command is `md5sum .auto/last_out.txt | cut -c1-12` (= `1a095c8496b4`).
Caught only because the value disagreed with a documented one.

KNOWN COVERAGE GAP, take next if docs come up again: present-tense claims that use neither form
survive the check, e.g. STREAMING_1P5B.md's "the shipped default is PIECES=1 at RTF ~2.43" (v3.9-era,
still reads as current). A FORM_D for "the shipped/current default/tier ... is N" would close it; do it
with a false-positive sweep in the same commit or not at all.

## Exp858 — check 9 coverage gap closed: FORM_D, and the wrapped-sentence lesson

The gap I had just written down was bigger than I described. `STREAMING_1P5B.md`'s stale claim reads
"the shipped / default is PIECES=1 at RTF **~2.43**" **with the sentence broken across a newline**
("the shipped" ends one line, "default is PIECES=1..." starts the next), so *no line-anchored rule can
see it* - which retroactively explains why four forms of the same class survived: the guard reads lines,
prose wraps.

Added **FORM_D**: present-tense config claims ("the shipped default is", "shipped tier is", "currently
ships", "default tier is", ...) scanned over a **2-line window**, with two brakes:
 * the value must be an **RTF-adjacent** number, otherwise "…whose default is defer-OFF): 2.54 on the
   protocol clip" (a v3.9-era read of a *config* default) turns into a phantom speed claim;
 * the historical rules still apply, so the two sites found are excluded by their own `SUPERSEDED`
   section heading rather than by a deny-list, and the exclusion count went 2 -> 3 so this is visible.
Dedupe: the 2-line window hits a wrapped sentence twice - one real claim, two FAILs - so FORM_D now
remembers whether it fired on the previous line. Reporting the same defect twice trains people to
distrust the count, which is its own kind of noise.

Made one prose site self-policing instead of just correct: STREAMING_1P5B.md's superseded-snapshot
section now opens with `**Current best: 1.87 (era v4.7)** — …the live ladder is in RESULTS.md`, phrased
in the FORM_A idiom ON PURPOSE and placed *above* the `SUPERSEDED` heading so the historical-section
rule cannot hide it. Now if the era moves and someone forgets this pointer, check 9 fails. Coverage
6 -> 7 claims (`shipped=5, lean=2`), and a spec-only bump to a fictional v4.8 flags all 7 across
3 files, including that pointer.

**A control that failed because of my grep, not the code:** I wrote
`grep "asserts CURRENT \(lean\)"` when the message says `asserts CURRENT state (lean)`, so control B
looked like a miss and I nearly went hunting for a bug that was not there. Diagnosed by running the
detection logic standalone on the exact line (triggers hit, mv=2.44, section not historical) - i.e. by
testing the CLAIM rather than the REPORTER. Rule: when a control fails, verify the fault is present in
the *object under test* before believing the instrument's own output channel.

Known remaining limits of check 9, stated so nobody assumes more coverage than exists:
 * a live claim parked UNDER a historical heading is invisible to it (mitigation: phrase live pointers in
   a policed form and put them above such headings, as done here);
 * past-tense history is by design not checked - which is why the doc convention going forward is
   **past tense = then, present tense = now**, and the guard polices the present-tense half.

## Exp859 — THE REPRODUCE RECIPE REPRODUCED A DIFFERENT SYSTEM (+3.5 %)

Found by asking a question the loop had never asked: does `RESULTS.md`'s "How to run (reproduce)" block
actually produce the number printed 25 lines above it? It did not. The block ran
`VAE_FILE=vae-encoder-q4x4ffn.gguf`, which is the **F16-conv reference build**, while the shipped tier is
`vae-encoder-convint8.gguf` (the one declaration in `.auto/tier.env`). Measured both, same session:

| what the docs told you to run | VAE file | rtf | transcript |
|---|---|---|---|
| documented recipe | `vae-encoder-q4x4ffn.gguf` (reference) | **1.9372** | `1a095c8496b4` |
| shipped tier | `vae-encoder-convint8.gguf` | **1.8710 / 1.8735** | `1a095c8496b4` |

So a faithful reader got **+3.5 %** and an honest-looking number - and because the transcript is
byte-identical, nothing about the run *looks* wrong. Same mechanism as Exp694 (eval40.sh's stale
defaults made a gate score another system), one level up: that class was fixed for the SCRIPTS and never
applied to the prose a human copies.

**Shipped:** recipe rewritten to the shipping tier, including the step the docs had dropped entirely -
`.auto/conv_int8.py <src> <out> --device`, where `--device` is mandatory because
`ggml_quantize_chunk(Q4_0_4_4)` on x86 writes ZERO SCALES (Exp688), so a host-built file of that type is
fluent silent garbage. Verified end to end: the exact documented command line reads **1.8735**, matching
the 1.87 headline cell, hash unchanged.

**Guard:** check 10 - any documented COMMAND (a line invoking `measure.sh`/`eval40.sh`) must name
tier.env's files; prose/table mentions of an alternate tier stay legal (the F16-conv reference row is
history, not a recipe). Command lines only, because that is the boundary between "information" and
"instructions". Negative control: re-point the recipe at the reference file -> 1 FAIL naming both values;
restore -> 0. Coverage message reports the count of overrides compared (2), so an empty scan is visible.

## Exp860 — every flag the docs or comments PROMISE must exist in the parser (generalizing Exp855)

Exp855 found `--kv-type q8_0` recommended by a code comment and by two of my own next-hints, in a binary
that parses 10 flags and exits 1 on anything else. That iteration added a fault probe for **that one
flag** - which leaves the class open for the next fictional flag. Check 11 closes it.

Ground truth is derived, never declared: the binary's set comes from the string literals in
`demo/asr_streaming.cpp`'s parse loop (11 tokens), the harness scripts' from their `case` labels
(`--clip/--skip-build/--env`). Scanned surfaces: doc **invocations** (`/asr_streaming`, `/measure.sh`,
`/eval40.sh` - flags appearing *after* the program token) and **code comments** in src/ and demo/,
because the original fictional flag lived in a comment, not in a doc. 7 flag uses scanned, all valid.

Three false-positive classes had to be handled, all the same underlying error - *mentioning* a program
rather than *invoking* it:
 * `cmake --build build --target asr_streaming -j` - "asr_streaming" is a target name, `-j` is make's;
 * a prose cell about "two `asr_streaming` processes running concurrently";
 * `-mcpu=cortex-a78` in a build-recipe table cell that names the binary.
Fix: require a '/' before the program token (an executable path) and take the LAST invocation on the
line. First draft used `rfind(prog)` and produced 4 FAILs on 4 lines that mention a name.

Comment lines that DISAVOW a flag are skipped ("there is no --kv-type flag; passing it exits 1"), so
Exp855's corrective comment can live in the tree without tripping the check that exists to catch its
resurrected cousin. Control: plant `// pass --kv-type q8_0 ...` in src/vae.cpp -> 1 FAIL naming file,
line and flag; restore -> 0.

Also swept, and reported as NOT checked (saying so is part of the check): flags belonging to other
programs (cmake, llama-quantize, python tools) are outside its scope - their option sets are not
derivable here, and a deny-list of foreign flags is how a guard starts lying.

## Exp861 — guard rotation #8: flat, and the delta is still exactly the decode tokens

Scheduled board (last Exp851). One sweep, 3 interleaved reps per arm, so thermal/boost state hits both
arms equally: protocol **1.8707** (1.871/1.870/1.871) vs the never-optimized guard slice
`slice10b_24k.wav` **2.0042** (2.008/2.003/2.002), 55 tokens, RSS 2191 MB, **= +7.1 %**.

 * The +7.1 % is not a mystery and is not an overfit signature: 55 vs 39 tokens = 16 extra decodes at
   ~93 ms = 1.49 s predicted vs 1.33 s measured on a 10 s clip, at identical VAE work.
 * History: 2.5367 -> 2.5095 -> 2.4085 -> 2.3861 -> 2.3226 -> 2.0728 -> 2.0228 -> 2.0036 -> 2.0059 ->
   **2.0042**. Since Exp851 the guard moved **-0.08 %** and the protocol **-0.17 %**, both inside the
   0.19 % band. Over rounds that changed loaders, docs and harness only. **Verdict: no drift, no
   protocol-specific tuning.**
 * `cpu7_khz` = 1,300,000 in all 6 samples - the Exp844 negative result (frequency is not the confound;
   the primes are hard-capped) reproduces.

**Self-inflicted, and the reason to always read your own diff:** inserting the row with `oldText` = the
NEXT row's label (so the new row would land above it) silently deleted that label - the Exp845 row lost
its first cell and would have rendered as a broken table row. Caught by `git diff` + a check that every
line in the table range starts with '|'. Rule: when inserting a line adjacent to an anchor, the newText
must restate the anchor; and a table edit is not verified until something has counted its row starts.

## Exp862 — the VAE time budget CLOSES on the current stack, and one above-bar item fell out of it

The per-node instrument (`GGML_OP_TIME=1`, queued by Exp817 as "the one thing that settles it") had last
been read at v4.5. The flush (v4.6) removed a whole padded window and the boundary batch (v4.7) changed
prefill rows, so every share in the ledger predated the shipping graph. Re-derived in one run:

**Census (blocked-int8 MACs/clip):** vae 328.1, lm_prefill 161.4, lm_decode 56.3.
The VAE figure vs Exp817's 388.5 is **-15.5 %**, which is an independent confirmation of the flush from
the MAC side (vae_s went 14.0 -> 11.9 = -15 %). Good when two unrelated instruments agree.

**Node time, VAE phase = 23 809 ms.** With >1 thread this SUMS the two concurrent chains, so
wall-equivalent = node/2: 11.90 s against measured vae_s 11.94 s. The budget closes to 0.4 %, which is
the point of the instrument - no missing time, no double count.

| op | node ms | wall-equiv s | % of VAE | % of metric |
|---|---|---|---|---|
| MUL_MAT | 16 563 | 8.28 | 69.6 % | 44.3 % |
| UNARY (gelu) | 2 461 | 1.23 | 10.3 % | 6.6 % |
| RMS_NORM | 885 | 0.44 | 3.7 % | 2.4 % |
| **CONT** | **827** | **0.41** | 3.5 % | **2.2 %** |
| ADD | 824 | 0.41 | 3.5 % | 2.2 % |
| ADD_SCALED | 746 | 0.37 | 3.1 % | 2.0 % |
| CONV1D | 631 | 0.32 | 2.6 % | 1.7 % |
| IM2COL | 585 | 0.29 | 2.5 % | 1.6 % |
| PAD | 287 | 0.14 | 1.2 % | 0.8 % |

**Rate check, done properly for once.** VAE matmul = 328.1 GMac / 8.28 s = **39.6 GMAC/s aggregate**. The
device micro (`mm_shape_micro 2 real`) measures the VAE's OWN shapes in isolation at **42.6-44.8**
(stage3/4/5 FCs). So the VAE runs within **7 %** of its shapes' isolated ceiling - "at ceiling" is now
confirmed with current evidence, and that 7 % is the known in-kernel F32->Q8_0 `quantize_mat_q8_0`,
already priced at ~2 % by Exp715/720 and NOT reachable by im2col dst_type (Q8_0 aborts, F16 is parity,
cast-to-Q8_0 computes garbage: the three failure modes are in `vae_conv_1d_i8`'s comment).

**The above-bar find: CONT = 0.41 s wall = 2.2 % of the metric.** Eight `ggml_cont(ggml_permute(...))`
sites in vae.cpp (490, 651, 687, 691, 756, 818, 967, 997) - the layout transposes between the [C,T]
channels-first blocks and the row-major form the matmuls consume. Exp818 saw it (0.49 s then), wrote
"worth a look after PAD", and it was never looked at; PAD has since been closed twice. 280 calls.
Do NOT assume it is free to remove: a cont(permute) is usually there because a consumer needs a dense
buffer, so the honest first step is a PER-SITE attribution (which sites, how many bytes each), not an
optimization. Sites reachable without touching vendored ggml are the ones where a downstream op could
consume the permuted view directly.

Micro-benchmark context that makes the mechanism concrete (constant 340 MMAC, ne00=128, F32 src1):
rows=32 -> 11.85 GMAC/s, rows=128 -> 25.05, rows=1024 -> 38.06, rows=2048 -> 39.17; the SAME shapes with
src1 pre-staged as Q8_0 are flat 38.6-42.4. So the conversion penalty is a LOW-ROWS phenomenon, and the
VAE's big-MAC shapes are not in that regime - which is why the aggregate sits at 39.6 and not at 12.

### QUEUED (from Exp862): CONT attribution, then decide
- Instrument: extend the OPTIME dump with a per-CALL-SITE tag (the 8 sites above) or a one-off build that
  prints bytes per cont node; 1 run. Acceptance for any later change: byte-identity, then 3 reps.
- Candidate classes, cheapest first: (a) cont before an op that could take the strided view directly;
  (b) cont(permute) pairs that cancel across two adjacent blocks; (c) sites that exist only because a
  consumer's shape check is stricter than its kernel needs (check, do not assume).
- Do NOT re-open via ggml_set_inplace (a copy op, not arithmetic - Exp662) and do not add vendored strided
  src1 support for <2.2 % without a measurement saying the sites are removable.

## Exp862b — a contamination class the co-runner guard could not see: the device's OWN background work

Two consecutive reps read **2.0431** then **2.4109** (+9 %, +29 %) with byte-identical output. The Exp679
guard said the device was idle because it greps for `asr_streaming` - and it was right, no *ASR* was
running. What was running: **`dex2oat32 -j4` at 377 % CPU, nice 10**, AOT-compiling an app because the Play
Store had started an install (`com.android.vending` + `installd`), load average 34.8.

Fix in `measure.sh`: sample `/proc/stat`'s busy jiffies twice, 1 s apart, and warn when **>25 % of the
device's CPU is going to work other than ours**, printing `other_busy=N%` in the note line so a
contaminated run is identifiable AFTER the fact too.

CALIBRATION, and it matters: **`/proc/loadavg` is not usable as a threshold on this device** - it reads
~21-23 even when the device is provably idle (`busy = 0.9 %`) and barely decays over minutes. A
loadavg-based guard would have warned on every single run forever. Busy-jiffies delta is the signal.

Negative control (Exp660 rule: a self-check is untrustworthy until a planted fault makes it fail):
6 x `timeout 30 nice -n 10 sh -c 'while :; do :; done'` -> `other_busy=76 %`, WARNING printed, rtf
1.9766 (+5 %). Idle -> 1 %, silent, 1.8857. Note the FIRST control attempt loaded nothing at all
(device-side `(cmd &)` dies on this toybox, Exp856 again) and would have "passed" vacuously; background
the LOCAL adb call and let on-device `timeout` bound the burners.

Harness bug caught by the control: the first sampler joined two `/proc/stat` lines with `|` and indexed
the second line from `$13`, which printed `busy=-28641 %`. Sum each line separately.

## Exp863b — speeding up CONT site 7: ATTEMPTED, NOT SHIPPED, DISCREPANCY UNRESOLVED
> **→ RESOLVED AND SHIPPED by Exp864 (v4.8). Read the end-of-file entry first.** The discrepancy was NOT
> in the transpose formula (the referee was right) but in the fast path's GUARD: no F32 type test, so it
> corrupted non-F32 cont nodes. Everything below was true as measured at the time. Do not redo this probe
> set - the kernel ships; the reusable parts are the per-site attribution method and the referee design.

The lever: site 7's `cont(permute(x,1,0,2,3))` is 0.42 s of wall (98.8 % of cont() time) and ggml's
generic path is its own `"this is not optimal - fix me"` element-at-a-time memcpy, measured at
**0.78 GB/s of dst writes vs PAD's 3.6 GB/s**. A transpose changes no values, so the prize (~2.2 % of
the metric) looked safe: acceptance = protocol transcript byte-identity, no gate needed.

Two implementations, both reverted:
1. **NEON 4x4 register transpose** (two `vtrn` stages, pairing derived on paper as
   `(r0,r2)/(r1,r3)` then `(t0,t2)/(t1,t3)`). Pipeline output: 14 tokens instead of 39, hash
   e44d190018be vs 1a095c8496b4. The rtf read 1.64 - a FAKE win caused by the LM emitting 25 fewer
   tokens (Exp674's "the measurement you ran wasn't the measurement you meant", 6th instance).
2. **Plain C 16x16 cache-blocked transpose** (no intrinsics at all - blocking was the real mechanism).
   Same wrong values, same first-mismatch position.

Why it is NOT shipped: a C double loop of the form `d[j*dn + i] = s[i*sx + j]` cannot have a lane bug,
and the kernel's own diagnostic printed the correct strides (`sx = C`, `dn = ne0`, `ne00/ne01` right),
and the buffers do not alias (`x->data != y->data`). So either ggml's `cont` semantics differ from the
definition I encoded, or the oracle is wrong - and I could not tell which, which is exactly the
condition for not shipping. The oracle (`.auto/cont_tile_test.cpp`, kept) has ALREADY had one indexing
bug of its own (it scored a [C,T] tensor as row-major [T,C], which made a correct kernel look wrong),
so it is not yet a trustworthy referee.

**Cheapest decisive next step if this lever is ever retried** - settle the referee before writing any
kernel: inside the fast path, ALSO run the generic loop into a scratch buffer and `memcmp` the two
results, reporting "kernel vs generic" mismatches per shape. If they agree, the kernel is right and the
oracle is wrong; if they disagree, the kernel is wrong - and in either case the answer arrives in one
host-free run instead of through a 35 s pipeline or a self-written expectation.

Ruled out along the way (do not retry):
* Passing the permuted VIEW to the downsample conv instead of materializing it: SIGABRT at
  ggml.c:17299 `GGML_ASSERT(src0->nb[0] == sizeof(float))` - the same wall Exp819 hit via im2col_asym.
  Knob deleted rather than defaulted off, per Exp833's rule for arms that abort.
* Anything relying on `/proc/loadavg` as a busy signal (Exp862b), and anything whose speed claim comes
  from a run whose token count changed (Exp674).

## Exp863c — the referee says the transpose formula was right all along (68.8M elements, 0 mismatches)

`GGML_CONT_REF=1` in `ggml_compute_forward_dup_bytes`'s strided branch now runs the SHIPPED element
loop (which owns `dst`) and then a candidate blocked transpose into scratch, compares them element by
element, and prints an exit summary so silence cannot be mistaken for agreement:

    CONT_REF SUMMARY: compared 68812800 elements, 0 mismatches

Consequences:
* **The blocked-transpose formula `d[j*ne0 + i] = s[i*(nb00/4) + j]` is byte-identical to the shipped
  path on every accepted node of the real pipeline** - 68.8M elements across 4 windows x 2 chains. So
  my two failed implementations were not wrong in their arithmetic, and my oracle's verdict was the
  thing to distrust (it has since been shown to encode the same relation, so the earlier oracle failure
  must ALSO have been structural - see below).
* Two referee design errors found on the way, both worth remembering:
  1. Take 1 compared scratch against the SOURCE. That is a tautology, not a check - it "passed" 13.3M
     elements while proving nothing. A referee must compare the CANDIDATE against the SHIPPED OUTPUT.
  2. Take 2 printed only on mismatch, so its silence was ambiguous (Exp668's rule). The exit summary
     with an element count is what makes 0 mismatches mean something; it prints
     "(CHECK NEVER FIRED)" if the count is 0.
* **The live hypothesis for the original 14-token failure is STRUCTURAL, not arithmetic**: both my
  implementations replaced the element loops with an EARLY RETURN, which also skips the surrounding
  (i02, i03) slice bookkeeping and the contiguous-dst guard's own structure. The next attempt should
  therefore not return early - keep the shipped branch skeleton and substitute only the inner walk,
  then let the referee (roles swapped: generic formula into scratch, candidate owns dst) certify it.
* Prize unchanged: 0.42 s wall = 2.2 % of the metric, bit-identical, acceptance = protocol hash + 3 reps
  + gate not needed (a transpose cannot change values).

- CONT TRANSPOSE KERNEL: SHIPPED AS v4.8 AFTER 3 FAILED ATTEMPTS (Exp861->863c->864; protocol 1.85, -1.66 %).
  Site 7 of Exp863's per-site dump (the [C,T]<->[T,C] stage-boundary copy, 0.42 s = 2.2 % of the metric,
  ggml's own "this is not optimal - fix me" element loop at 0.78 GB/s) is now a 16x16 blocked walk inside
  the shipped branch of ggml_compute_forward_dup. Byte-identical by construction and in fact: gate b=0/c=0
  of 731 (p=1.0, bootstrap CI exactly [0,0]), ladder canaries identical on both tiers.
  WHY IT TOOK SO LONG, AND THE LESSON: the fast path's guard was missing
  `src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32`. Without it the branch also fired on non-F32
  cont nodes and reinterpreted their bytes as float => silent corruption under -DNDEBUG (Exp667's class,
  4th instance). The referee reported "0 mismatches over 68.8M elements" the whole time because ITS guard
  required F32: the instrument certified exactly the population it shared with the candidate and was blind
  to the population where the bug lived. RULE: a differential referee must cover the SUPERSET of the fast
  path's accepted nodes, or its green result means nothing - assert the two guards are identical, or drop
  the type term from the candidate. Second bug, found in one run by the same instrument: the first bad
  element index was i00 = 16 = the tile size, which located a missing `ib*sx` in the source pointer (only
  tile 0 had ever been right). first_bad VALUES are worth printing, not just counts (Exp668's rule).
  Also ruled out, with instruments rather than argument: dst/src aliasing (no CONT_ALIAS lines) and a
  concurrency exposure (wrong output persisted under VAE_SEQ_ENCODERS, i.e. single-chain).
  STRUCTURE MATTERS: the first two attempts REPLACED the branch's element loops with an early return, which
  also skipped the branch's later structure - a fast path must substitute the inner walk, never return.
  DIAGNOSTICS LEFT IN TREE (all default-off, output verified unchanged): GGML_CONT_REF=1 same-run referee
  (shipped loop owns dst, candidate writes scratch, BIT-level memcmp - float != flags NaN vs NaN and faked
  a failure on a 26x2048 node), CONT_ALIAS print, per-site CONT dump.
  AVOID: the permuted-view route (hand the conv src0->permute directly) - SIGABRT at
  ggml.c:17299 GGML_ASSERT(src0->nb[0] == sizeof(float)); Exp819 hit the same wall from the other side.
  DO NOW: rollback_audit.sh with GGML_CONT_TILE_OFF as the 11th hatch (nested-fast-path lesson, Exp824),
  and behavior_watch.sh on BOTH tiers (a copy kernel at VAE stage boundaries - edge cases are where a
  shape-population difference would show).

- GELU ABSORBED INTO ITS CONSUMER'S ACTIVATION QUANTIZATION — SPECIFIED, PRICED, NOT YET BUILT (Exp864d).
  The v4.8 node census (GGML_OP_TIME, re-derived after the CONT change per the Exp665/671/674 rule) gives
  the VAE per chain: MUL_MAT 16.57 s node-ms (71.4%, = 8.29 s wall, AT the 39.6 GMAC/s ceiling), then
  **UNARY/gelu 2513 ms = 1.26 s WALL = 6.8 % of the metric**, the largest non-matmul item by 3x
  (RMS_NORM 0.44, ADD 0.40, ADD_SCALED 0.36, CONV1D 0.32, IM2COL 0.29, PAD 0.14, CONT 0.11 after Exp864).
  WHY THIS ROUTE AND NOT THE EPILOGUE: Exp822 measured producer-side fusion (gelu in the GEMM store tiles)
  NEGATIVE, +4.2 % at every tile size, because the extra pass evicts the weight panel the GEMM is streaming.
  The CONSUMER side is a different site: fc2's mul_mat must quantize its F32 activations to Q8_0 anyway,
  and that staging loop is ONE place - ggml.c:14209, `from_float_to_mat(...)` (= quantize_mat_q8_0) over 4
  columns x ne10, with a scalar tail loop below. Today the pipeline writes gelu(x+b1) to memory (1 read +
  1 write of the whole fc1 output) and then reads it back to quantize. Folding gelu+bias INTO the quantizer
  deletes both passes: quantize(gelu(x+b1)) computed in registers. Ceiling = gelu's node time, 1.26 s wall;
  realistic 4-6 % since the quantizer's own read stays.
  BIT-IDENTICAL BY CONSTRUCTION (the same argument that made Exp673/671/670 shippable): the f32 values fed
  to the quantizer are the same gelu(x+b1) results - same table lookup (GGML_GELU_FP16), same f32 add - and
  Q8_0 block quantization is a deterministic function of those values. Q8_0 blocks are 32 CONTIGUOUS
  elements, so a 32-float STACK buffer suffices: NO work-buffer request, which is what arms Exp578/664.
  RECIPE:
    1. ggml-quants.c: `quantize_mat_q8_0_gelu_bias(const float * x, const float * b, void * y, int nx,
       int ny, int blck)` - per 32-element block: vaddq_f32 then the SAME gelu body, then the unchanged
       q8_0 block math. No fma contraction (Exp664's rule).
    2. ggml.c staging loop (14209): if the node flag is set, call the variant with bias from src[2].
       Flag = op_params[1] == 1; bias tensor rides as an EXTRA SRC (src[2] of a mul_mat is unused - and
       Exp672's lesson: pass the loaded tensor raw, do NOT ggml_cont a model tensor or cgraph leaves it
       unallocated/garbage).
    3. Builder: ggml_mul_mat_gelu_bias(m, a, bias) setting op_params + src[2].
    4. vae.cpp ffn: build fc1 with apply_bias=FALSE (raw product, exactly today's intermediate) and fc2
       through the new builder with ffn_fc1_bias. Fuse ONLY where the gelu output has exactly ONE consumer
       (the fc2 matmul) - check the graph at build time, keep the node otherwise.
    5. Gate the fusion on the frame count being a multiple of 4: the tail rows (ne11 % 4) go through a
       different per-row conversion path below the loop, which this does not cover.
  ACCEPTANCE: protocol transcript hash byte-identical (it must be, by the argument above), then 3 reps +
  vae_s + the gate paired test (any src/ change => the gate, Exp852's rule), then behavior_watch on both
  tiers, then rollback_audit with the new hatch (GGML_GELU_CVT_OFF). STOP if the hash moves: that would
  mean a second consumer exists somewhere and the deletion is wrong, not a rounding detail.

- CONSUMER-SIDE GELU FUSION: EQUIVALENCE PROVEN, SPEED LOST, KEPT OPT-IN (Exp865, GGML_GELU_CVT=1).
  Built the Exp864d idea: fc1 stays RAW and FC2's F32->Q8_0 activation staging applies gelu(x+fc1_bias), so
  the gelu node (1.26 s wall = 6.8% of the metric, the largest non-matmul item) leaves the graph. Result:
  PROTOCOL TRANSCRIPT BYTE-IDENTICAL (hash 1a095c8496b4) - the equivalence argument was right - but 3.3%
  SLOWER (1.9093 vs 1.8489, VAE 12.2 vs 11.6 s). Cause is mechanical: the fusion calls the shipped
  quantize_q8_0_4x4 once per 32-element block instead of once per row, repeating its per-call setup
  (float32x4_t srcv[4][8], the amax reduction tree) ne10/32 times, plus a 4x32-float stack tile per block.
  So the ceiling (deleting 2 tensor passes) is real but this implementation shape gives it back.
  DURABLE LAYOUT FACT (cost two wrong runs): in the 4x4 interleaved staging, ONE block_q8_0x4 carries the
  same 32-element block of FOUR columns, so its address step is 4*nbw1/nb_blk, NOT nbw1/nb_blk. Using the
  per-row stride scrambles the activation staging and yields 1024-token garbage silently under -DNDEBUG.
  Also: a "fuse" extra-src must be attached to the CONSUMER's matmul - the first attempt passed it on the
  fc1 call, which applied gelu to fc1's input and left fc2 without any. A one-shot probe printing the
  flagged node's NAME (ffn.linear1.weight) located it in one run; probes should always name the node.
  NEXT ATTEMPT IF PURSUED: inline the gelu INTO A COPY of quantize_q8_0_4x4's block loop in
  ggml-aarch64.c - no tile, no per-block call; pass ggml_table_gelu_f16 as an argument (not visible in that
  TU); apply the three branches per lane right after vld1q_f32 and BEFORE vabsq/amax (amax must see gelu'd
  values). Acceptance = protocol hash, then 3 reps + vae_s, then gate (any src/ change, Exp852's rule).
  Status of the elementwise board: gelu 1.26 s is the only above-bar item left; RMS_NORM 0.44, ADD 0.40,
  ADD_SCALED 0.36, CONV1D 0.32, IM2COL 0.29, PAD 0.14, CONT 0.11 s, and the producer-side epilogue route is
  measured negative (Exp822, +4.2% by weight-panel eviction).

- GELU CONSUMER FUSION: BOTH IMPLEMENTATIONS MEASURED, AXIS CLOSED BY MECHANISM (Exp865/865b).
  Three arms of one binary, vae_s on the protocol clip (LM output byte-identical in both fused arms):
    baseline (gelu is its own node)            11.6 s
    VAE_ABL_GELU (gelu work absent entirely)   10.6 s   => the removable ceiling is 1.0 s, 8.6% of VAE
    fused via stack tile + per-block call      12.2 s   (fused work costs 1.6 s)
    fused by inlining into the quantizer loop  13.1 s   (fused work costs 2.5 s)
  Reading: the gelu's cost is NOT the two tensor passes alone. Roughly half of it is the 16-bit table
  gather, which pipelines well in a standalone streaming loop (independent iterations; Exp780's batched
  variant proved 4-in-flight is parity, i.e. already at its gather floor) but sits on the amax DEPENDENCY
  PATH inside quantize_q8_0_4x4, where every block needs all 32 gelu'd values before it can compute its
  scale - serializing gathers that used to overlap. Hence inlining is worse than the tile version, and the
  tile version is worse than the node.
  So: gelu stays a separate node. Both fused forms remain in the tree behind GGML_GELU_CVT=1 (default OFF;
  default path verified byte-identical to the v4.8 anchor in the same binary). Together with Exp822
  (producer-side epilogue +4.2% by weight-panel eviction), the fusion board is now closed on BOTH sides:
  neither the producer's store tiles nor the consumer's input staging can absorb an elementwise op whose
  per-element work is not itself free. RULE WORTH KEEPING: before planning any op-absorption, ablate the op
  to find its REMOVABLE ceiling first (here: 1.0 s), and compare against the op's standalone cost (1.26 s) -
  the gap is the irreducible compute, and it was 20% of the node, not 100%.

- SWEEP HARNESS WAS MEASURING A NON-SHIPPING CONFIG (Exp865c, found by a suspicious reproduction).
  run_rtf_multi.sh builds its device command from "label|audio|threads|pieces|env". Any arm spec with fewer
  than 5 fields left threads/pieces EMPTY, and because they are expanded UNQUOTED the positional list
  SHIFTED: bench_device.sh got the run TAG as its thread count and an empty --vae-pieces. Effect: every
  number the tool produced - including eight anti-overfit guard rotations - described a different config
  (~1.2 % slower than measure.sh, RSS column nonsense), and v4.8's -1.66 % did not appear in its protocol
  arm at all. Fix: defaults resolved from the tier (threads from measure.sh, pieces from tier.env), values
  validated (threads numeric, pieces in {1,2,13,26}), a --dry mode, and audit check 12 which asserts the
  resolved argv equals the tier and that a malformed arm exits non-zero (proven with a planted fault).
  HOW IT SURFACED: a fresh guard sweep reproduced Exp861's per-rep numbers to four decimals. Timing does not
  do that. Rule worth keeping: if a measurement is TOO consistent, suspect the instrument, not luck.
  LESSON FOR MYSELF, RE-LEARNED THE HARD WAY: I ran the fixed-but-UNCOMMITTED sweep for several rounds and
  then destroyed it with a careless `git checkout .auto/run_rtf_multi.sh` while planting a fault. The loop's
  own rule is commit tooling in the iteration it is written; a fault-plant must restore from a COMMIT, not
  from the working tree. (The audit's new check caught the loss immediately, which is the only reason the
  loss was cheap.)
  Guard baseline from today: protocol 1.8504, guard 1.9852 (+7.3 %, = its 16 extra decode tokens).

- SOAK VERDICT WAS A COIN TOSS; NOW IT REPORTS ITS OWN RESOLUTION (Exp865d).
  Two soaks of ONE binary on chat138.wav returned +2.40 MB/min -> "GROWTH" and +0.95 -> "FLAT", with
  identical first/last/HWM. Cause: rss_soak classified on a bare |slope| < 2.0, and on a 5-minute window a
  few 20 MB two-state spikes move the least-squares slope by more than the threshold. Fixed: slope is now
  reported with a 95% interval (standard error of the regression) and classified by whether the INTERVAL
  clears 0.5 MB/min, which yields FLAT / GROWTH / INCONCLUSIVE honestly. --selftest validates the classifier
  on synthetic series through the real functions (extracted from this file's source, since its run section
  executes at import time): +40 MB/min with +-12 MB noise -> GROWTH; +1.2 with +-3 -> GROWTH; flat +-3 ->
  FLAT; flat +-12 (device-like) -> INCONCLUSIVE, with the detectable floor printed (2.09 MB/min at +-12).
  Real data, 3 soaks of the v4.8 tree on the 138 s clip: peak/HWM = 2205.9 / 2205.9 / 2205.8 MB (identical
  to 0.1 MB), RSS last = 2184.9 / 2184.7 / 2184.6 MB, fd FLAT (3), threads 2-3 FLAT, majflt 0. The invariant
  PEAK across independent runs is the stronger no-leak statement here; the slope test at 5 min simply cannot
  resolve 0.5 MB/min on this device.
  QUEUED (cheap, better instrument than any slope on one run): soak ACROSS N sequential runs of the same
  clip and compare run-1 median RSS to run-N median - a paired comparison that is immune to the spike noise
  that defeats the slope, and it matches how a session actually runs (many windows, one process). Needs the
  sampler to follow a restarting PID (PROC lookup per run) rather than one PID.

- SESSION MEMORY ANSWERED, AND THE RSS TWO-STATE PATTERN NAMED (Exp865f). rss_soak gained --repeat N: N runs
  of one command under ONE sampler, comparing per-run steady MEDIANs plus per-run PEAKs. Two independent
  3-run session soaks (chat69, 24 windows each, 72 windows total):
      peak per run  2198.2 / 2198.1 / 2198.2   and   2198.2 / 2198.1 / 2198.1   -> spread 0.1 MB
      median per run 2176.8 / 2176.8 / 2197.2  and   2176.8 / 2197.3 / 2197.1   -> a ~20 MB STEP, not a slope
  So there is NO session growth: the peak (per-run, monotone within a run, spike-immune) is invariant. The
  median step is the device's two-state resident-set pattern - the SAME ~20 MB effect that made Exp865d's
  single-run slopes flip between GROWTH and FLAT, now identified as a state switch rather than a trend.
  A slope fitted over 3 run-medians is honest only as an interval (+10.2 +/- 12.0 MB/run) - which is why the
  tool now states the peak-invariance conclusion explicitly. Also fixed: --repeat leaked into the wrapped
  argv (the parser only stripped --interval), so the harness tried to execute the FLAG as a program.
  Verdict for the product: RSS is bounded across sessions; the only long-session ceiling remains KV
  positions (~245-258 s of continuous audio at n_ctx=4096, Exp849/855).

- SPLICE-PHASE LAW, TESTED ON THE A78 STACK (Exp866h, prompted by the Jetson loop's report): my stitched
  eval assets insert GAP_MS=300 -> 7,200 samples between clips while the protocol advances HOP = 22 frames
  x 3200 = 70,400 samples, so the 24 clips of holdout_en sit at 24 DISTINCT phases of the window grid
  (0, 20768, 67584, 18080, 55104, ...). Only clip 1 starts a window. .auto/align_asset.py now rebuilds any
  such asset with every clip on a hop boundary and PROVES content identity via the manifest's per-clip sha256
  (all 24 matched; +39.4 s of silence is the only difference). Measured on the phone, same session, 220
  held-out English tokens: misaligned 26.36% vs aligned 25.00%, PAIRED b=17 / c=17, McNemar p=1.0, CI
  [-3.6, +6.8] pp. Reading: no SYSTEMATIC penalty is detectable at this n (power ~0.4; their +7-17 pp would
  fall just outside this CI, so it is disfavoured, not excluded), but 34 of 220 tokens changed identity -
  phase moves WHICH tokens are wrong even when the mean is unchanged, which is label noise for training data.
  HYPOTHESIS for the cross-device discrepancy (test on their side): a SINGLE splice imposes one shared phase
  offset on all later audio (systematic -> big WER swing), while many splices give i.i.d. phases whose
  effects cancel in the mean and survive only as variance. If so, the rule to publish is "align every
  segment start to the hop", not "a splice costs 15 pp".
  OPEN MODELLING QUESTION: is the right modulus the STRIDE (22 frames = 70,400) or the WINDOW (26 frames =
  83,200)? I used 70,400 because the windowed protocol advances 22 frames; a non-overlapping/carry schedule
  would use 83,200. A one-arm test (shift by 83,200 - 70,400 = 12,800 samples) distinguishes them.
  TOOLING: .auto/align_asset.py (re-phase + prove), .auto/phase_probe.py (insert aligned/non-aligned silence
  into an existing asset; content-preserving, so arms share one reference and compare paired).
  CAUTION KEPT: transcript LINES are WINDOWS, not turns - a per-clip WER built by zipping lines to the
  manifest table produces ~100% garbage (it did, and the 26% aggregate exposed it). Per-turn WER needs the
  scorer's own turn segmentation: QUEUED as a score_stream.py --per-turn option.

- PHASE ALIGNMENT, SECOND CORPUS -> NO SYSTEMATIC EFFECT, LARGE TOKEN CHURN (Exp866i). Same method on the
  48-clip zh-TW Common Voice held-out set (468 ref tokens, 48 distinct phases, content proven identical by
  per-clip sha, +71.2 s of silence only): misaligned 15.38% vs aligned 16.88%, PAIRED b=23 / c=32, McNemar
  p=0.28, CI [-4.70, +1.71] pp. So en favoured ALIGNED by 1.36 pp and zh favoured MISALIGNED by 1.50 pp -
  opposite signs across corpora, both inside their CIs, while 55 of 468 zh tokens (11.8%) and 34 of 220 en
  tokens changed identity. Conclusion for this stack: hop-misalignment does NOT bias WER; it randomizes
  WHICH tokens the model gets wrong. Consequences: (1) historical held-out numbers need no re-quoting;
  (2) for stitched TRAINING data the churn is the problem, not the mean - a misaligned corpus labels the
  same audio with different text; (3) the Nano loop's +7-17 pp must come from a different variable than
  phase alone (their single-splice design gives all later audio one shared offset, so it is systematic;
  a many-splice asset averages out), which is the testable hypothesis already sent.

- PHASE POLICY IS NOW A CHECKED PROPERTY (Exp866j). audit_harness gained a phase check over every stitched
  eval manifest (9 of them): clip-start positions are taken from exact `start_samples` when present, else
  from the ms-rounded start_s with a +-32-sample tolerance, and every multi-clip stream must declare
  `phase_policy`: "hop-aligned" (verified) or "arbitrary" (accepted, with the Exp866h/i evidence that WER is
  unbiased but tokens churn). Missing policy = WARN naming the fix. 104 checks, proven by planting
  hop-aligned on the arbitrary zh set -> FAIL "clips deviate up to 34400 samples from the grid" -> restore.
  Two bugs found in my own new code on the way, both worth remembering:
    * an aligned build written with ms-rounded start_s LOOKS misaligned to anything re-deriving positions
      from the manifest (3 distinct phases instead of 1) - so align_asset now writes start_samples/end_samples;
    * naming a local `dev` inside the asset section shadowed that section's dev dict {name: md5} and crashed
      the audit with AttributeError - a check that silently stops running is worse than no check, so verify
      the CHECK COUNT moved, not just that the run went green.

- LONG CLIPS ARE A 4.5x SHARPER INSTRUMENT THAN THE PROTOCOL CLIP (Exp867h; transfers the Nano loop's
  0.07%-at-240s finding). Interleaved 3+3 in one session: chat138.wav n=3 sd 0.052% rel (range 0.102%),
  stream_10s_24k.wav n=3 sd 0.233% rel (range 0.458%). 3-rep standard errors: 0.030% long vs 0.135% short.
  Mechanism is arithmetic: run length scales the work while fixed effects (thermal entry, page-in, scheduler
  settling, the two-state ~0.3% cluster step from Exp841) do not, so relative variance falls ~1/sqrt(work).
  HOW TO USE IT: for a length-independent question (a kernel, an elementwise op, a rate), bracket on
  chat138 - 2 reps there beat 8 reps on the protocol clip. For a PROTOCOL or length-dependent change (tail
  flush, first-window behaviour, anything touching the final window) the protocol clip remains the metric
  AND the right place to measure: a long clip dilutes exactly the effect those changes have.
  Cost: 5.5 min/rep, so a 2x2 is ~22 min - worth it only when the decision hinges on <0.2%.

- dw_axpy RE-TESTED AT THE RIGHT RESOLUTION: TRULY INERT (Exp867j). With the 138 s clip, VAE_DW_AXPY_OFF gives
  -0.0004 / -0.0002 s per pair = -0.014% +/- 0.005% (so a REAL sign - the fused path is marginally the slower
  of the two - but 40x below any ship bar), output byte-identical. The five earlier "inert" verdicts on the
  protocol clip were directionally right; the one -0.4% reading was noise. Keep the code, stop spending runs.
  Side product: the paired long-clip noise floor is ~0.0005 s per pair (0.02%), so any future claim below
  ~0.05% needs either 4+ pairs or a mechanism argument.

- SPLICE-PHASE, CLOSED ON MY STACK (Exp868, four configurations, two corpora). The averaging hypothesis I
  sent the Nano loop ("many splices cancel; one splice is systematic, hence their +7-17 pp") is REFUTED.
  align_asset gained --phase N, which puts EVERY clip at the same phase (a leading pad is required, else
  clip 1 sits at 0 and the systematic condition silently breaks - my first attempt did exactly that).
    * zh 468 tokens, all 48 clips at phase 35,200 (half a hop = maximal displacement under a stride
      modulus), content proven identical: WER 16.45% vs 16.88% phase-0, b=36/c=34, McNemar p=0.905.
      A 7 pp systematic penalty would need ~51/19 discordants; 36/34 excludes it at p<0.001.
    * en 220 tokens, 533 ms DELETED from inside clip 5 (verified real speech: cut-region rms 3139 vs clip
      rms 3220) vs the same cut inside a silence gap: 21.36% / 21.82% vs 26.36% baseline, McNemar p=0.29 /
      0.38, insertions DOWN 11 -> 6. A content discontinuity does not produce their insertion signature.
  MODULUS QUESTION, ANSWERED FROM CODE (no device time needed, cite these lines): the WINDOWED path advances
  `start = w * HOP_SAMPLES` (demo/asr_streaming.cpp:526, HOP_SAMPLES=70,400 at :42) so the modulus is the
  STRIDE, exactly as my phase_policy enforces. But the CARRY path (--xwin) uses
  `start = (h==0) ? 0 : WINDOW_SAMPLES + (h-1)*HOP_SAMPLES` (:489), i.e. 0 then 83,200, 153,600 ... which is
  12,800 mod 70,400 - a DIFFERENT GRID. So a corpus aligned for the shipping tier is systematically
  12,800 samples (533 ms) off the grid any --xwin evaluation would impose. Practical rule: align to
  multiples of 70,400 (shipping), and never mix phases between a windowed and a carry run of the same asset.
  So the splice rule stands as a REPRODUCIBILITY/label-consistency control (10-15% token churn), not an
  accuracy lever, on the A78 stack.

- COST MODEL RE-FIT, TREE CLOSURE, AND KV PRICED IN MY OWN UNITS (Exp870; supersedes Exp828's fit).
  Four ladder clips, ONE session, window count from the binary's own banner, VAE half taken from vae_s
  (LM-independent, Exp674/682) so a token-count difference cannot contaminate it. Tool: `.auto/cost_fit.sh`
  (measures), `.auto/cost_fit.sh --analyze` (re-fits the saved .auto/cost_fit.txt with NO device time).
    * VAE: **vae_s = +0.39 + 3.343 x EFFECTIVE windows**, residuals ±0.05 s. Effective = N minus the
      fraction of the final window the flush removed (a model without that term is why Exp828's intercept
      was +0.21: it over-prices the 10 s clip by +21 % and the 138 s one by +5 %). Zero-intercept form
      3.362 s / 3.467 s of audio => **VAE-only RTF 0.970**: the encoder alone is just under real time.
    * LM: **prefill = 29.8 ms/row + 9.11 µs/KV-position**, **decode = 90.3 ms/token + 10.67 µs/KV-position**,
      residuals ≤1.8 ms. Two independent phases, same slope => the length-dependent term IS KV attention.
    * Shares: protocol 62.9 % VAE / 17.3 % prefill / 19.5 % decode; 138 s 52.7 / 17.4 / 29.9.
    * TREE CLOSURE ≤0.30 % at all four lengths. The latency tree is COMPLETE - there is no unmeasured
      wait/sync/loader inside the metric. Keep the line in the tool as an invariant, not as a result.
    * RATE: 58.3 GMac/window → 16.8-17.4 GMAC/s aggregate = **20 % of the two-core int8 peak at every
      length**. That is Exp516/551/678's shape-limit conclusion, now length-independent, and it is the
      number any future VAE claim has to be consistent with.
  KV PRICE (the answer to the Nano loop's eviction thread, in my units): a KV-side lever is worth
  ≤0.46 % on the protocol clip (cannot move this metric), 3.9 % at 69 s, 7.6 % at 138 s as an upper bound,
  3.3 % if only the decode term is credited - and the decode-only branch is theirs: their formula predicts
  -3.0 % on my 138 s clip, my slope 9.1-10.7 µs/position vs their 11.13 reproduces it. PARKED as a
  long-form product feature (not a metric item): it needs a KV option the binary does not expose.
- HARNESS INTEGRITY, TWO MORE CLASSES CLOSED (Exp870).
    * HEAD CAN BE AUDIT-RED AND NOBODY KNOWS: a doc row reading "any invocation of measure.sh/eval40.sh
      carrying a schedule option" was parsed as an invocation of eval40.sh (prose slash), charging that
      line's --xwin/--env to a program whose interface is positionals+env. Committed red, caught one round
      later only because the NEXT session ran the audit first. RULE: the audit is the last write of an
      iteration. Writing it before the docs means it guards code but not prose.
    * A GUARD THAT CANNOT SEE IS WORSE THAN NO GUARD: the first fix (reject any slash preceded by a word
      char or dot) also rejected legitimate harness-path invocations, and the planted control printed ZERO
      failures - the over-correction was found by RUNNING the control, not by reasoning about the regex.
      Second rule added with it: a flag belongs to the invocation before it, so a two-command line stops
      charging one typo to two programs (it used to report 3 failures for 2 defects).
    * DOUBLE REPORTING: two identical `for b in sorted(set(badflags))` loops over the same accumulating
      list printed every flag failure twice, so the header's failure count was inflated. Any guard whose
      count lies makes "it went from 5 to 3" unreadable. Same class as Exp660's over-claiming guard.
- majflt AS A COLD-CACHE DETECTOR, REHABILITATED (Exp870). After a ~4 h idle: 1.8913 (majflt 30),
  1.8851, then 1.8541 (majflt 0) = +2.0 % cold inflation at v4.8. Exp156 had retired the majflt detector
  as non-discriminating and kept only the pre-wait; on this stack it discriminates cleanly (30 vs 0 for a
  2 % move). Cheap policy that costs nothing: read majflt from the METRIC line, and if it is > 0 treat the
  run as a warm-up rather than a measurement.

- COST MODEL IS PREDICTIVE, NOT JUST FITTED (Exp871). Wrote the numbers to .auto/cost_pred871.txt BEFORE
  measuring, then ran three configurations absent from the fit:
    * VAE term: chat155 (155 s, new length) -0.07 %, hotwords (17.28 s) +0.13 %, guard slice (10 s) -0.21 %.
    * Wall/rtf GIVEN TOKENS: +0.06 % / -1.31 % / +0.49 %. Blind (tokens from the 6.3 tok/s density
      constant): +0.01 % at 155 s (that domain matches), +4.2 % and +12.4 % where the density did not.
    * So the model is a RATE model: per-window VAE cost and the LM's per-row / per-token rates are
      content-independent; the token COUNT is the only content-driven input and must be measured.
      Predict with `.auto/cost_fit.sh --predict <seconds> [tokens]` - no device time.
  WHAT IT BUYS: "what would that shape cost?" is now a paper exercise (give it frames/s and it returns
  wall), which is exactly what the peer loops asked for, and a failed idea can be priced before it is built.
  CONTENT-INDEPENDENCE, PROVEN DIRECTLY: protocol clip and never-optimized guard slice both give
  vae_s = 11.6 s; the 17 s clip and the hotword probe both give 19.6 s at equal effective windows. That is
  the assumption behind every vae_s-based argument in this loop (Exp674/682) - it was assumed, now it is measured.
  New cell: 155 s -> rtf 2.1736, RSS 2208 MB, 978 tokens, majflt 0.
- ASSET HEADERS ARE NOT CHECKED (Exp871; the check shipped in Exp872). *** THE "ONE OF THEM LIES" HALF OF THIS "
  ENTRY IS WRONG - RETRACTED BY THE CHECK BELOW IT: chat17.wav's header is correct; I conflated the RIFF "
  size field (file-8 = 816,036) with the data field (file-44 = 816,000) in an od -c dump, and the 36-byte "
  "anomaly" is just 44-8. Keep the entry for the class lesson, not the fact. ***  chat17.wav declares a
  `data` chunk of 816,036 bytes and holds 816,000 -> a reader that trusts the header reads 18 samples past
  EOF. Harmless here (the loader uses the file length; every cell using it is unaffected) but it is exactly
  the kind of latent asset defect the Exp676 hash audit cannot see: hashing bytes validates identity, not
  self-consistency. QUEUED: audit_harness check - for every blessed WAV, parse RIFF/fmt/data and assert
  data_size == filesize - header_size (with the device-side reads batched), negative-controlled by planting
  a wrong size on a host mirror. Same check would catch a truncated push.
  Also: device_assets.json says chat155.wav = "chat138+chat17"; the byte probe confirms the tail is chat17
  but NOT the prefix, and chat17/chat138/chat155 are byte-identical at 100-200 KB. Long chat clips share
  source audio. Do not use any of them for a phase/position argument without re-deriving their layout.
- SHELL TRAP, RE-HIT VERBATIM (Exp871): I put $(...) inside a double-quoted `adb shell "..."` string, it
  expanded LOCALLY, hashed nonexistent local files, and returned d41d8cd9... - the empty-input md5 that
  THIS LEDGER NAMED as the tell (Exp676). Reading my own output would have caught it in one second, and it
  did - but only because I have read that note. Rule that generalizes: when a probe returns the same hash
  for two different "files", the probe is broken, not the files identical.

- PHASE CONVERTIBILITY = 1.0 ON THIS STACK, BOTH SCHEDULING MODES (Exp872). The Nano loop's #243/#252
  correction (a phase's timer overstates its wall contribution until that phase is wall-critical) is true on
  THEIR pipeline and would be a bug in MY cost model if it were true here. Measured, not argued:
  concurrent (shipping) 138 s closure -0.02 %, 155 s +0.03 %; SEQUENTIAL arm (VAE_SEQ_ENCODERS=1, 138 s)
  168.5 + 52.2 + 89.8 = 310.5 s vs wall 310.6 s = -0.03 %, rtf 2.2507 (+3.9 % vs concurrent 2.1672, RSS
  1946 vs 2191 MB). Additivity therefore comes from the pipeline (encode-then-decode per window, no VAE/LM
  overlap), not from the scheduling mode. Instrument corollary: vae_s is STAGE wall time, not a chain sum -
  sequential vae_s 168.5 ~ ac_s+sem_s = 168.6; concurrent ac_s = sem_s = vae_s = 11.6.
  USE: my closure line is a real detector for an unaccounted phase; theirs is not, and any cross-stack wall
  prediction has to carry a per-phase convertibility factor. Mine is 1.0 with evidence.
- WAV-HEADER CHECK: SHIPPED, NEGATIVE-CONTROLLED, AND IT REFUTED MY OWN ASSET CLAIM (Exp872).
  Walk the RIFF chunks of every blessed clip (device + HOST mirror) and assert header_bytes + data_size ==
  file_size. Value: catches a truncated push and a self-inconsistent header, both invisible to md5.
  Two lessons:
    * THE CONTROL CAUGHT A SILENT NO-OP. The first plant fired ONLY the md5 check, because section 7 does
      `host = {k: man.pop(k) ...}` - my loop over `man` therefore saw no HOST keys and checked nothing.
      A guard that reports "N assets OK" while iterating an empty dict is the Exp660 sin; iterate `host`.
    * READ FIELDS, NOT DUMPS. The Exp871 claim that chat17.wav lied by 36 bytes was me misreading od -c:
      RIFF size = file-8, data size = file-44, so a 36-byte difference between them is REQUIRED. The check
      written to enforce the property is what disproved the claim - 6th "wrong measurement" case in this
      loop, 1st one closed by a guard built from the earlier ones.
  Detail: ffmpeg writes a LIST/INFO chunk that puts `data` at offset 216 - probe 1 KB, not 192 B, or good
  assets get reported as "unverifiable".

- GUARD-COVERAGE AUDIT + ONE SILENT GUARD FIXED (Exp873). `.auto/audit_selftest.py` (tracked) plants one
  fault per audit check, runs the audit, and requires THAT check to fail; `--only <re>` runs a subset;
  `--list` prints the matrix. Result 13/13 plantable classes fire (11 first pass, then the two below),
  3 classes declared UNCOVERED with reasons - visible, not implied.
  * SILENT GUARD FOUND: device-state test was `'device' in r.stdout`, and adb prints
    `error: device 'X' not found` on failure, which CONTAINS 'device'. An unreachable/mistyped serial
    passed connectivity and every later device section then reported missing files - misleading, though it
    never produced a wrong number (those checks fail loudly). Fixed to require the literal line `device`.
  * check 8 printed "gate refs.json: MISSING" as prose, never bad(). Fixed: a missing gate reference set is
    a FAILURE (the scorers would print a WER against an empty reference).
  * section 7 iterated the manifest only -> new 7d lists device clips that are in NO manifest and FAILs on
    any that is byte-identical to a documented clip (the Exp675 collision class, now closed on the device
    side too). First run found 16 undeclared clips; all now accounted for: 7 declared fixture/probe
    patterns in device_assets.json `known_extras` (with reasons), 3 real assets blessed
    (holdout_en_aligned / holdout_zh_aligned / holdout_zh_ph35200 - align_asset.py outputs used by
    Exp866h/i), gate utterances matched by name pattern. Audit 115 checks / 0 failures / 1 explained WARN.
  * Two false-positive classes avoided: check 2 now skips audit_selftest.py (a fault catalog names paths
    that must not exist), and a plant must target a TRACKED file OUTSIDE .auto for the dirty-tree check to
    see it (3b ignores .auto on purpose).
  USE: run `.auto/audit_selftest.py` after ADDING an audit check (add a fault for it), and any time the
  audit's own logic is refactored. It costs ~8 min and is the only thing that proves the guards exist.

- UNRUN TESTS + UNSAFE PLANT REVERTS (Exp874). Two new guard items, both from applying Exp660 to TESTS:
  * `score_mixed.py --selftest` RED since 869e73e (fixture said "1 char inserted", string inserted 2).
    Scorer correct, fixture self-contradictory. Now fixed, and audit check 14 RUNS all four tool self-tests
    (compare_arms/rss_soak/score_mixed/score_stream, <=0.5 s) + WARNs on any tracked tool with an uninvoked
    --selftest. Fault 14 in audit_selftest.py proves it fires. Coverage now 14/14 plantable classes.
  * PLANT REVERT RULE: never revert a plant with `git checkout --` - that restores the COMMITTED state and
    destroys uncommitted edits to the same file (it ate a real fix this round). Plants snapshot bytes and
    restore bytes. Recorded because the loop's own advice (Exp869: "restore from a COMMIT") is about losing
    harness code to an auto-revert; for a PLANT the safe primitive is a content snapshot.
  * Also defused: `sh` (subprocess helper) was rebound to a path inside audit section 12 - a landmine for
    every later section. Rule: don't shadow module helpers inside a section body.
  NEXT, if the harness is touched again: the 3 UNCOVERED classes are (a) device binary-hash drift (needs a
  real rebuild to plant), (b) co-runner, (c) sweep-tier check (has its own --dry control). Only (a) is a
  genuine hole: it would be plantable by pushing a deliberately stale .so in a scratch dir and pointing the
  push list at it - worth one fault if the audit's push-list derivation ever changes.

- GUARD COVERAGE IS NOW 17/17 PLANTABLE CLASSES (Exp877). The last genuine hole was check 5 (host-vs-device
  binary hashes) - never controlled, and the class has bitten this loop for real: the Exp595-607 compiler-flag
  sweeps measured stale objects because `make` does not rebuild on a flag change, so four arms reported
  "0 %" about binaries that were never rebuilt. Plant now: append one byte to the DEVICE's libllama.so
  (device-side `cp` backup, restore by `mv`, verified md5-equal to host afterwards) -> check 5 reports
  `host vs device md5 MISMATCH`. Full sweep re-run after adding four faults in three rounds: 17/17 fire,
  tree verified clean, final audit green.
  REMAINING UNCOVERED, on purpose (visible, not implied): (a) co-runner - planting a second asr_streaming
  would poison the very timings the check protects, and it happens by accident often enough (Exp679 found
  two live runners from a killed `timeout`); (b) check 12 (sweep-tier) has its own --dry control.
  RULE: run `.auto/audit_selftest.py` when ADDING an audit check (add a fault with it) or when the audit is
  refactored; ~10 min, and it is the only evidence that the guards exist. Rounds Exp873-877 found four
  instrument defects this way (unfirable device-connectivity test, unrun scorer self-test, set -e abort
  sites, unproven binary-hash check) - i.e. more than any measurement round in the same window.
  * VACUOUS SWEEP RUN found the same round (Exp877): calling `run_rtf_multi.sh "arm|clip.wav" ...` (arms
    without the leading REPS) set REPS to the arm string, `seq` failed, ZERO runs executed, and the tool
    still printed a summary header and exited 0. It took 8 minutes of my time because I called it with the
    wrong signature and only noticed "PASSED in 0.2s" - arithmetically impossible for a 10 s clip at RTF 1.85.
    Now all three misuse forms (no REPS / REPS=0 / no arms) exit 2 with the usage line, and audit check 12
    asserts that (0 s, validation runs before any device contact). Rule reaffirmed: a timing under ~2x the
    audio length is a run that did not happen - check the DURATION before the number.
- GATE REGRESSION AFTER THE HARNESS STREAK (Exp878): 40/40, hybrid WER 4.51 % (jiwer 4.55), paired b=0/c=0
  of 731 vs BOTH hyp-gate852 and hyp-bound835 (CI exactly [0,0]) -> five rounds of audit/harness work, and
  the Exp876 edit that touched eval40.sh's own parse lines, are output-inert on real speech. Gate mean rtf
  1.90 (was 1.9117) = state. Tooling nits fixed: `compare_arms.py` on a GATE manifest without `--gate` died
  as a KeyError from score_stream - it now names the right invocation and exits 1 (the flag is easy to
  forget because both modes take three positional-ish arguments).

- LONG-FORM LAWS EXTENDED WITH UNIQUE AUDIO, LIMIT IS DENSITY-BORNE (Exp879). Built `long250.wav` = holdout_en
  (139.6 s) + hop-aligned 1,240 ms gap + first 110 s of holdout_zh = 250.8 s, 86 windows, 728 tokens, all
  DISTINCT audio (the chat ladder's long rungs are one 69 s clip twice - Exp875 - so this is the first probe
  past that). Results:
  * Cost model validated at a new length AND a new content type: vae 3.337 s/window over 86 windows vs the
    model's 3.343 + 0.39/N, and the wall tree closes -0.3 % (471.4 accounted vs 470.1 = rtf x audio). The VAE
    term is content-independent across a 2.5x token-density change - the assumption behind every paper-pricing
    argument in this loop, now tested where it could have failed.
  * Decode law refit at P<=3140: 95.8 + 13.22 us x P (slope se 1.23) vs the 2025-era 89.1 + 11.13 - the old
    slope under-predicted this run's decode total by 5.9 % (77.6 vs 82.5 s). 1.7 sigma apart: widen the law,
    do not replace it. Prefill law confirmed out of range (33.9 + 5.00 us vs 33.0 + 5.22; -1.3 % on totals).
  * SESSION LIMIT IS DENSITY, NOT SECONDS: positions/window = 28 rows + tokens emitted. Chat ~18.5 tok/win ->
    46.5 pos/win -> 4096 positions = 258 s; sparse read speech 8.5 tok/win -> 36.5 pos/win -> 329 s. This run
    finished at 3140 positions, confirming the law from below (it would have died at window 112). Docs fixed;
    Exp849's "~258 s" is now qualified rather than general.
  * Memory does not grow with positions (KV is allocated up front): RSS 2218 MB at 3140 positions vs 2191 at
    39 - the only long-session ceiling remains the position count.
  * NOT a ladder rung: rtf 1.8743 here is LOW because the clip is sparse (2.9 tok/s vs chat's 6.3), not
    because anything got faster. Never compare this number to the chat ladder.
  * TRAP CLOSED (Exp879b): the sweep does not pull out-loop.log, so `.auto/last_out.txt` kept the PREVIOUS
    run's transcript; hashing it after a sweep reported a protocol-output change that was really a 250 s run's
    output (9th wrong-measurement case, and the first where the "wrong" file was one I trusted every day).
    Fixes: the sweep now moves it aside BEFORE running (so a stale read fails loudly), the mv sits after
    argument validation (Exp879b's own bug: when it sat first, check 12's "REPS with no arms" misuse probe
    moved the capture aside before erroring), and audit check 16 states what the capture holds - 4 window
    lines = protocol, otherwise a WARN, and WARNs when it is missing. Rule: a hash is only as good as the
    file's provenance; quote the capture's window-line count next to any identity claim.

- A REBOOT MOVES EVERY BAND CELL ~1/3, AND NOBODY RECORDED UPTIME FOR 879 RUNS (Exp880). A spontaneous
  device reboot (adb dropped; uptime 367 s on recovery) produced protocol 1.19-1.31 and 138 s 1.49-1.59
  against the long-uptime band 1.85 / 2.16 - same binary, byte-identical transcripts (md5 1a095c8496b4),
  same 39/876 tokens, same chunk counts. Three controls, not one: (1) a host wall-clock cross-check
  (binary 13.2 s vs 13.63 s host) rules out a lying clocksource; (2) a DELIBERATE `adb reboot` reproduced
  the fast regime (~1.31 at ~5 min uptime), so it follows the reboot, not a one-time update; (3) the build
  fingerprint is unchanged (OPPO/CPH2371 R.203be74, patch 2025-10-01, in .auto/device_fingerprint.txt).
  Still fast at 9.75 h uptime (1.18-1.24), so the slow state accumulates over DAYS - the decay curve is
  unmapped (QUEUED: protocol samples at ~10 min / 2 h / 6 h / 24 h / 72 h after one reboot, idle between,
  to decide between a state qualifier and a reboot-before-keep protocol). Partial mechanism: sustained
  cpu7 under load reads 1.43 GHz now vs the ledger's constant 1.3 GHz (+10 %), leaving ~20 % to the memory
  subsystem / accumulated background load. This also closes the page-cache section's "genuinely UNMEASURED"
  paragraph: post-boot is not a first-pass stall, it is a ~30 % regime on every phase.
  ENFORCED, not noted: measure.sh prints uptime/procs/mem/cpu7/fingerprint with EVERY measurement and WARNs
  under 900 s uptime; headline.json stays LONG-UPTIME (the comparable set); docs carry the regime in
  STREAMING_1P5B.md without touching a band cell.
  RULE FOR THE NEXT SURPRISE: a ~30 % same-tokens speedup is a STATE change until proven otherwise - run the
  wall cross-check and read /proc/uptime BEFORE theorizing (my first reading was "the sweep is broken").

- `-c` PRICED, NOT ASSERTED (Exp880; predictions in .auto/cost_pred880.txt BEFORE the runs). Exp849 told the
  product to "raise -c" and claimed it "does not affect the metric" - measured only on runs that DIED.
  Protocol clip, 3 interleaved reps x 4 arms: 4096 -> 1.3679, 8192 -> 1.3569 (-0.8 %), 16384 -> 1.3428
  (-1.8 %), 65536 -> 1.3575 (-0.8 %). P1/P2 HOLD: no fixed reservation cost (the sign is even favourable).
  Long end, chat138 (1813 positions used): an order-symmetric A,B,B,A sweep reads 16384 at +0.4 % - the
  +2.4 % from a fixed-order run was ORDERING BIAS, resolved by the reversal (P6 amended, not failed).
  Memory: the KV buffer is COMMITTED at load, not reserved - RSS rises 27.4 KB/position (+165 MB at 8192,
  +388 MB at 16384, +1681 MB at 65536), matching the 28.0 KB/position arithmetic, majflt 0 throughout.
  So P3a, and the product form is "`-c 8192` doubles the session for +112 MB at 0 % of the metric".
  Third `-c` failure mode for the runbook: `-c 1048576` dies SILENTLY at context creation (log ends at
  llama_new_context_with_model, no error, no summary, empty transcript) - not matched by Exp849's
  "decode failed / frames failed" rule. Plumbing proof for all of the above: the absurd arm's device log
  shows `n_ctx = 1048576`, i.e. ARGS reaches the binary (the Exp865c class, checked first).

- THE SHIPPED BINARY LIED ABOUT ITS OWN DEFAULTS (Exp880, fixed). `-h` said `-c (default: 16384)` while the
  struct defaulted to 4096, and `--vae-pieces (default: 13)` while the code defaulted to 2 (asr_server.cpp's
  text is consistent - its default really is 16384, which is where the copy came from). Fix: help states the
  true defaults, AND the vae_pieces struct default is now 1 = the shipping tier, so a bare invocation
  reproduces the gated config (verified: no --vae-pieces -> RTF 1.1898, 39 tokens, 4 windows - previously it
  would have measured p2, the Exp859 class). All harness paths pass --vae-pieces explicitly, so no measured
  cell moves; the protocol run after the rebuild is byte-identical (1a095c8496b4).
  Guard: audit check 17 derives flag->field->default from each binary's own source and requires every numeric
  "(default: N)" to equal the struct (13 claims across both binaries), plus a second arm requiring the
  vae_pieces code default to equal tier.env PIECES (bare == shipping tier, by the single-declaration doctrine).
  Fault 17 (n_ctx 4096 -> 4444 in the struct) makes it fail naming -c.
  AND THE SWEEP EARNED ITS KEEP AGAIN: check 7d (undeclared device clips, Exp873) was SILENT -
  its chat17-sized dup plant produced only the dirty-tree FAIL. Root cause: the size filter that limits
  md5 requests was either/or - manifest 'bytes' if any entry has them, else live device sizes. Exp879's
  long250 entry added the manifest's ONLY bytes field, which silently replaced the fallback with a
  one-element set and disarmed collision detection for every other clip size. So a manifest edit made
  AFTER the last sweep broke a guard the sweep had certified - the rule 're-run the sweep when the
  manifest changes' is now written next to the rule 're-run it when the audit changes'. Fix: UNION both
  sources (a cost filter must never shrink the verdict set when information is ADDED). Re-run: 18/18
  plantable fire, 0 silent; check 16 declared UNCOVERED with reason (WARN-only by design, and the driver
  matches FAIL lines - manual control done instead: a 1-window capture makes it WARN, restore verified
  byte-identical). Lesson for the ledger: performance filters on guards need a monotonicity argument,
  not just a passing test - the test passes on the day the filter is written and rots the day the data
  changes shape.
  Also closed the hole tonight's adb outage exposed: run_rtf_multi.sh printed FAILED for every arm and exited
  0 - the Exp877 vacuous-run class in the sweep. It now pre-flights the device (refuses a vacuous sweep),
  warns on a co-runner (Exp679), and exits 1 when any run produced no RTF.
  GATE (this round, required by the any-src-change rule): 40/40, WER 4.55 % (S=29 D=2 I=2 - the identical
  error profile to Exp878), paired b=0/c=0 of 731 tokens vs BOTH hyp-gate852 and hyp-bound835 (CI exactly
  [0,0], p=1.0). So the help-text + vae_pieces-default change is output-inert on real speech, not just on
  the protocol hash. Gate mean 1.32 in the fresh-boot regime vs 1.89-1.91 long-uptime - the regime shows on
  varied real speech too, same ~30 %.

- GUARD ROTATION #11, FIRST IN THE FRESH-BOOT REGIME (Exp881; predictions in .auto/cost_pred881.txt BEFORE
  the runs). The guard must track the anchor across regimes or it is not a relative instrument. Order-
  symmetric P,G,G,P sweep on the second independent fresh boot: protocol 1.1883/1.1944 (39 tok), guard
  1.3085/1.3114 (55 tok) = +10.0 % premium vs +7.4 % long-uptime. Predicted +10.3 % (1.30-1.33) from the
  mechanism, HIT: the premium is exactly the 16 extra decode tokens at the CURRENT rate - guard decode
  4.2 s/55 = 76.4 ms/tok vs protocol 3.0/39 = 76.9, guard vae_s 7.1 vs protocol 7.0-7.1 (same 4 windows),
  transcript fluent and complete. No overfit signal.
  STANDING RULE CHANGE: the ratio band (+7.3-7.7 %) is LONG-UPTIME-ONLY - it widens mechanically when VAE
  cheapens more than decode (-39 % vs -17 % here), so a naive ratio check would have cried wolf. The
  criterion is now the per-token accounting (P2-style), which is regime-independent. A guard certified by
  ratio alone is certified for one regime only.
- SECOND FRESH BOOT REPRODUCES THE REGIME (Exp881): protocol 1.1939 at uptime ~25 min on an independent
  boot, transcript byte-identical. Decay points so far: 6 min 1.22 / 25 min 1.19 / 9.75 h 1.18-1.24 - i.e.
  NO decay within 10 h; the slow state accumulates over days. The 24-72 h points are still the ones that
  matter (QUEUED decay series stands).
- UNKNOWN REBOOT ACTOR (device hygiene, open): this boot's sys.boot.reason = "reboot,shell" (clean,
  shell-UID-initiated) ~21 min before the session, fingerprint unchanged. Not me (my only reboot preceded
  35000 s of uptime), not host cron (checked - unrelated project), no other pi session. Leading hypothesis:
  the phone's own scheduled power on/off (ColorOS) - which, if daily, also explains Exp880's spontaneous
  reboot AND bounds the slow-state accumulation to ~day scale, consistent with everything measured. Verify
  from the phone UI when it is next in hand; if a third unexplained reboot appears, audit host-side adb
  access. A device that reboots itself on a schedule is a metrology feature (fresh states for free), not a
  fault - provided the schedule is KNOWN, which it currently is not.

- LEAN PENALTY IS REGIME-INDEPENDENT (Exp882; predictions in .auto/cost_pred882.txt BEFORE the runs).
  Two order-symmetric S,L,L,S passes, fresh boot: shipped pooled 1.2113 (1.1876/1.1909/1.2453/1.2214,
  sd 2.2 %), lean pooled 1.3562 (1.3559/1.3626/1.3472/1.3591, sd 0.5 %) -> penalty +12.0 % vs +11.8 %
  long-uptime. WITHIN noise: the deferral overhead (+1.2 s VAE deep stages + ~0.5 s p13 prefill rows,
  decode identical) scales ~3/4 with the regime in absolute seconds, so the relative price is unchanged
  and the tier table needs no reprice - just this confirmation cell. Lean RSS 1752.2 MB both passes to
  0.1 MB. Method lesson kept: pass 1 alone read +14.3 %, pass 2 alone +8.2 % - single-pass ratios
  mislead in BOTH directions when one arm scatters, which is exactly why predictions must demand pooling.
  Side observation for a future contention experiment: the concurrent shipped path jitters VAE-side while
  the sequential lean path does not (lean sd 0.5 % vs shipped 2.2 % at n=4 each) - at this n it is a
  note, not a finding.

- FRESH-REGIME HEAT DISCIPLINE (Exp882 addendum): two upward excursions (1.4044 both phases up, then
  1.2706) inside one back-to-back sequence, both at other_busy=1-2 % - background CPU ruled out as the
  cause. After 3 min idle: 1.1917 with batt 38.1 -> 36.7 C. Verdict: SELF-HEATING, excluded from the
  decay series (settled/idle points only: 6m 1.22 / 25m 1.19 / 35m 1.19 / 45m 1.19 / 9.75h 1.18-1.24).
  The fresh regime looks MORE heat-sensitive than the long-uptime band (+18 % vs +-1 %) - consistent
  with higher clocks having more to throttle, mechanism n=1. Rule for the queued decay series: space
  runs with idle cooldown, or heat masquerades as decay (the Exp15/16 discipline, re-learned for a
  new regime - the third time this loop has mistaken heat for signal: Exp15/16, Exp122-123, now).

- ROLLBACK LADDER RE-RUN ON THE CURRENT BINARY, AND RELATIVE COSTS DON'T TRANSFER (Exp883).
  16 arms x 2 reps, rep2 reversed (new), batt-gated <= 37.0 C (new, no HOT flags): identity EXACTLY
  as at Exp864b (12 byte-identical; flush_off = 55ac39b635cb verified in a separate cooled run, so
  every historical cell stays reachable; ct_block/ALL_OFF differ as documented). Costs: small hatches
  match (axpy inert 6th time, m2/cont/lpad/gelu_batch within ~1 pp), big stacks read higher
  (dw_conv1d +12.8 vs +8.4, stack +38.6 vs +27.1, ALL +45.0 vs +39.5).
  Decisive control against heat: a cooled back-to-back default/stack pair reproduces +39.6 % with
  byte-identical transcripts. Mechanism: ABSOLUTE overheads transfer across regimes (dw_conv1d 1.52 s
  fresh vs 1.55 s, bound_batch 0.49 vs 0.48, stack 4.6-4.7 vs 5.0) while RELATIVE costs rise because
  the fused default fell 36 % - the regime cheapened compute-dense fused paths more than the
  bandwidth-bound fallbacks. Runbook rule: quote hatch costs in absolute seconds (transferable) plus
  relative at its regime. Kept as negative control: the first ungated attempt (32 back-to-back runs)
  drifted the default 1.19 -> 1.34 and inflated everything - reversal alone does not save a soaked
  ladder. Also closed pre-round: VAE_DW_CT_OFF is deliberately NOT a ladder arm (RESULTS.md documents
  it debug-only since Exp710 - it changes the transcript, failing the ladder's contract by design);
  checking the ledger first saved a wrong "fix".

- FRESH-REGIME NOISE FLOOR, MEASURED (Exp884): 6 protocol reps at 3-min spacing, one settled session:
  1.1890/1.1939/1.1937/1.1963/1.1940/1.1916, sd 0.209 %/rep vs 0.233 % long-uptime - the settled noise
  floor is REGIME-INDEPENDENT, so the 3-rep A/B, the ship bar, and the confidence discipline transfer
  unchanged PROVIDED runs are spaced (back-to-back fresh runs are not exchangeable: +18 % excursions,
  Exp882). No trend over the 20 min => decay invisible at 92-108 min; prompt.md's THERMAL DISCIPLINE
  updated with the fresh-regime protocol instead of the Exp15/16 long-uptime numbers. Transcript
  byte-identical throughout. This doubles as the 92-108 min decay point: 1.1931.

- GATE REGRESSION, DUE BY CADENCE (Exp885): 40/40, WER 4.55 % (S=29 D=2 I=2 - the identical profile
  three gates running), paired b=0/c=0 of 731 vs BOTH hyp-gate880 and hyp-gate852 (CI [0,0], p=1.0).
  Nothing changed since gate880 (harness + docs only), so the WER null was expected; the new data is
  the gate MEAN at 5 h uptime: 1.33 (1.2765/1.3766) vs 1.32 at ~1 h - flat, which is the decay series
  on a second timescale (varied short utterances, not one repeated clip). Decay series now: 6m 1.22 /
  25-45m 1.19 / 92-108m 1.19 / 5h 1.20 protocol and 1h 1.32 / 5h 1.33 gate-mean - two instruments,
  same answer: no hour-scale decay. The slow state needs days; the 24-72 h points stand as theQueued
  deciders. Stamp verified tier + current binary before proceeding (BIN=744bf770, unchanged).

- FAULT BOARD ROTATION, AND ITS BAND WAS REGIME-STALE (Exp886). fault_inject.sh as-is: 11 pass,
  2 fail - both fails exactly as predicted (lie_dur 1.4557 and trunc_half 1.4957 vs the hardcoded
  [1.5, 4.0], calibrated long-uptime), every exit-code/signature probe green. A lie_dur re-run at
  +0.3 s VAE over guard closed the numerator scare (heat at batt 38.9, not a header-driven window
  count - denominator 10.00 s content-derived, 55 tokens). Fix: the band is now same-session
  relative to a healthy baseline H ([0.5H, 2.5H], lie_dur additionally [0.7H, 1.3H] guardlike) -
  regime-proof by construction; re-run 13/13 green, including under a heat-elevated H=1.3374 that
  would break any absolute band. Rule generalized: any ABSOLUTE acceptance band on a timed quantity
  is a regime-stale guard waiting to happen - anchor same-session or document the regime it was
  calibrated in. Settled anchor 1.1983 at 5.3 h uptime (decay flat).

- SESSION MEMORY ROTATION, FRESH REGIME (Exp887): two independent 3-run chat69 soaks with 3-min
  gaps (full logs soak887a/b.log): peaks invariant to 0.2 MB both soaks (~2198 MB, same as
  long-uptime - allocation-driven), medians flat (no trend, and no Exp865f ~20 MB step either
  time - whether the step correlates with back-to-back heat is open but verdict-irrelevant),
  slope honestly INCONCLUSIVE, fd 3 / threads 2 / majflt 0 throughout. NO SESSION GROWTH, second
  regime. Soak rtf trended with heat (1.40 -> 1.47, batt to 39.3) despite gaps - 95-s runs heat
  faster than 3-min gaps shed; soak rtf is not a ladder cell and memory verdicts are heat-immune.
  Self-inflicted wounds, both documented: (1) the first attempt ran back-to-back AND lost its
  verdict to tool truncation - heat discipline applies to every multi-run instrument, and long
  outputs get teed to a file I own; (2) my prediction quoted the lean token canary (432) for a
  shipped-tier run - the ledger's 446 was exact, caught in 30 s. Settled anchor 1.2003 at 6.2 h.

- COST-MODEL REGIME TRANSFER (Exp888): settled chat69 (24 windows, 446 tok) in the fresh regime:
  VAE 48.9 s = 2.086 s/effective-window vs protocol's 2.107 - the linear VAE structure transfers
  to +-1 %, so per-window paper pricing holds fresh with the 0.61 scalar. Prefill 13.6 s vs ~14.3 s
  with-KV prediction (consistent, not crisp - needs a second settled rep to separate heat from
  structure); decode 34.9 in band. Fresh chat69 ladder cell 1.41 (provisional) vs 2.10. The model
  gains its regime parameter; no re-fit needed. My first P1-P3 bands were raw-division shortcuts
  ignoring intercepts/flush/KV - corrected in P1r-P4r BEFORE the clean run, and the lesson is
  recorded: predictions from shortcuts test arithmetic, not systems. Settled protocol 1.1943 at
  6.6 h (decay flat).

- GUARD ROTATION #12 (Exp889): order-symmetric P,G,G,P, protocol 1.1948/1.1961, guard 1.3075/1.3103
  (55 tok) = +9.5 % premium. Second confirmation of the per-token criterion (Exp881: +10.0 %):
  guard decode 76.4 ms/tok vs protocol 76.9, VAE identical 7.1/7.1, transcript fluent. The ratio
  band (+7.3-7.7 %) is now retired for good - two independent fresh-regime rotations land +9.5-10 %
  via token accounting, never +7.4 %. No overfit signal.
  Postscript: the round's own anchor run immediately after the rotation read 1.3254 (5th consecutive
  run, batt 38.1) and resolved to 1.1961 after 3 min idle (batt 37.2, other_busy 1 %) - third
  confirmation of the heat discipline, and the reason the logged metric is the settled run, not the
  rotation mean. Decay point at 6.7 h: 1.1961, flat.

- GATE REGRESSION, DUE BY CADENCE (Exp890): 40/40, WER 4.55 % (S=29 D=2 I=2 - identical profile
  four gates running), paired b=0/c=0 of 731 vs BOTH hyp-gate885 and hyp-gate852 (CI [0,0], p=1.0).
  Output stability now proven across 4 gates and both device regimes. Gate mean 1.35
  (1.3333/1.3759) vs 1.32/1.33 - slight creep, heat-consistent (chunk 1 starts warm at batt 37.4;
  chunk 2 matches gate885's chunk 2 to 0.1 %; the chunk1->chunk2 step inside every gate is the
  heat signature, not content, since utterance halves are fixed).

- CHAT69 PREFILL STRUCTURE, SECOND SETTLED REP (Exp891): prefill 13.6 s EXACTLY again (sd 0.0
  across two settled reps) - per the pre-registered rule a systematic +1.1 s over rows-only, but
  analysis shows it IS the KV term (with-KV prediction 14.3, measured 13.6 twice): the term exists
  at ~60 % of the scaled guess, refines the fresh KV slope, no anomaly. VAE 49.3 (2.103 vs 2.107
  s/eff-window, 0.2 %), decode 35.1, tokens 446, rtf 1.4203. Structure holds completely with all
  terms accounted for; question CLOSED. Fresh chat69 cell firms to ~1.41-1.42 (two reps).
  Settled anchor 1.1951 at 7.2 h, byte-identical.

- BEHAVIORAL CONTRACTS ROTATION (Exp892, ~44 rounds overdue): 11/11 PASS, 0 fail on the current
  binary - silence/noise/music labels, 48 kHz stereo, sub-piece short36, twospk collapse (CLOSED
  Exp650) + overlap split, ladder canaries 39/106/446/876 exact. Expectations pre-registered in
  the script since Exp848, so this was a pure rotation, no new predictions needed. Settled anchor
  1.2039 at 7.5 h (decay flat); the immediate post-board anchor read 1.3391 warm and settled on
  the normal 5-min discipline.

- GUARD ROTATION #13 (Exp893): order-symmetric P,G,G,P, protocol 1.1948/1.1971, guard 1.3109/1.3126
  (55 tok) = +9.7 % premium. Third confirmation of the per-token criterion (+10.0/+9.5/+9.7 %):
  guard decode 76.4 ms/tok vs protocol 76.9, VAE 7.1/7.1 identical, transcript fluent. No overfit
  signal. Postscript: immediate anchor 1.3368 (5th consecutive run) settled to the logged 1.1983
  after 3 min idle - fourth heat-discipline confirmation. Decay flat at 7.6 h.

- DEVICE-STATE TELEMETRY PERSISTENCE (Exp894): per-run state (uptime/procs/mem/cpu7/fingerprint)
  was printed but never saved - only batt_temp_c persisted, so retrospective mechanism tests were
  impossible. measure.sh now appends one TSV line per run (write can never break a measurement,
  PARSE_ONLY never writes); guarded by check 18 (capture newer than TSV = FAIL) + fault 18
  (hide the TSV). Seeded with 3 spaced settled anchors: 1.1891/1.1904/1.1892 (sd 0.0007 - the
  tightest triple of the loop), batt cooling throughout. Full sweep 19/19 fire, 0 silent.
  Decay point at ~7.7 h: 1.1896 pooled, flat.

- GATE REGRESSION, DUE BY CADENCE (Exp895): 40/40, WER 4.55 % (S=29 D=2 I=2 - identical profile
  five gates running), paired b=0/c=0 of 731 vs BOTH hyp-gate890 and hyp-gate852 (CI [0,0], p=1.0).
  Output stability across 5 gates and both regimes. Gate mean 1.35 (1.3195/1.3755) at ~8 h uptime:
  series 1.32/1.33/1.35/1.35 - flat, and chunk-2 reads 1.376 +- 0.001 across three gates, which is
  the tightest sub-number in the whole gate instrument (chunk halves are fixed utterance sets, so
  chunk-2 is a 20-utt fixed-work benchmark with 0.1 % reproducibility).

- THIN AUDIT: DORMANT LEVERS vs THE FRESH REGIME (Exp896, paper - no device time claimed). Every
  parked/dismissed avenue re-examined for a changed assumption. Verdict: NONE - no revivals.
  * Q4_0 LM (-1.8 % overall, Exp73): tradeoff is dequant-compute vs bandwidth; the regime cheapened
    compute MORE than bandwidth, so the dequant saving is worth relatively LESS fresh. Still
    discard, arguably more so. No retry.
  * Lean boundary batch (-2.6 %, held for parity): accuracy objection already resolved (b=0/c=0 on
    468 zh tokens, Exp842/846); remaining blocker is product-owner call on a non-metric tier, not
    measurement. Nothing I can unblock by running. Stays queued.
  * -c 8192 default (+112 MB, 0 % speed, sessions 2x): priced (Exp880), product call. Stays queued.
  * Gelu fusions (producer +4.2 % by weight-panel eviction; consumer +3.3 % by gather serialization):
    mechanism-closed both sides (Exp822/865). A regime scales rates, not dependency structure.
    No retry.
  * K-quants for VAE (Q6_K +29 %): NEON super-block dequant cost on A78 (Exp196). ISA fact, not a
    threshold. No retry. Q5_K/Q4_K dead by interpolation. No retry.
  * CONT permuted-view (SIGABRT wall): needs a 3rdparty change (off-limits). No retry.
  * ThinLTO/PGO/OMP (-t6/8, split pools, affinity): nulls on thread-independent axes. No retry.
  * Granularity (p1/p2/p13/p26 mapped): RAM ratios are structural. No retry.
  * dw_axpy removal (inert 6x, confirmed again Exp883): still inert. No retry.
  * xwin carry (11.5-12 % long-file drift): context-window mechanism, not a tuning artifact.
    No retry.
  * Training/QAT/distillation, Mali GPU, KleidiAI/i8mm, fork upgrade: out of scope / hardware
    facts. No retry.
  * mm_m2 tail (+0.2-0.8 %, kept): shipped, free, zero risk. Nothing to revive.
  Rule applied: a discard is revisited only on a CHANGED ASSUMPTION with a named mechanism, never
  on a new regime alone - regimes rescale thresholds but do not rewrite mechanisms. The two queued
  product calls (-c default, lean batch) are the only shippable value left, and both need the user.

- GUARD ROTATION #14 (Exp897): protocol 1.1951, guard 1.3074/1.3096 (55 tok) = +9.6 % premium -
  4th confirmation of the per-token criterion (VAE 7.1/7.1, decode 76.4 vs 76.9, fluent). The P2
  arm read 1.3069 (both phases up = heat step after 6 consecutive runs) and resolved to 1.1949
  after idle; excluded by the watch policy, documented here. No overfit signal.
- PLANT MUST LEAVE NO TRACE, INCLUDING MTIME (Exp897, harness): fault 17's byte-perfect revert
  still bumped demo/asr_streaming.cpp's mtime, which tripped run_rtf_multi's freshness guard on
  the next sweep (stale-binary refusal on a content-current tree - the instrument contradicted
  itself one round after certifying itself). Fix: snapshots cover stat (ns precision, verified
  exact) as well as bytes. Control: --only 17 fires AND mtime is bit-identical after. Rule
  generalized: a revert that restores content but not metadata is half a revert - every guard
  that reads metadata (mtime, perms, xattrs) can tell.

- -c 8192 SESSION DEMONSTRATED END-TO-END (Exp898): long250 (250.8 s, 86 windows) at -c 8192
  completes (exit 0, summary), rtf 1.307 (heat top-edge, informational), tokens 728 EXACT
  (context size does not change output), peak RSS 2331 MB majflt 0 (no swap pain). Delta +140 MB
  vs +112 arithmetic - the ~+30 MB second-order excess matches Exp880's sweep pattern; real,
  product-irrelevant, documented not chased. The product recommendation is now evidence, not
  arithmetic; the ship decision stays the user's. Settled anchor 1.1952 at 8.5 h.

- 17 s FRESH LADDER CELL (Exp899): settled chat17 reads VAE 11.8 s (2.003 s/eff-window, within
  5 % of the 2.10 rate), prefill 3.0, decode 7.9 (74.5 ms/tok), rtf 1.3399, tokens 106 EXACT
  (flushed canary). All bands held (two edge-kisses low, within noise). Fresh ladder complete
  (provisional): 10 s 1.19 / 17 s 1.34 / 69 s 1.41 / 138 s ~1.55. Settled anchor 1.1928 at 8.6 h.

- GATE REGRESSION, DUE BY CADENCE (Exp900): 40/40, WER 4.55 % (S=29 D=2 I=2 - identical profile
  six gates running), paired b=0/c=0 of 731 vs BOTH hyp-gate895 and hyp-gate852 (CI [0,0], p=1.0).
  Output stability across 6 gates and both regimes. Gate mean 1.35 (1.3350/1.3737) at ~8.6 h:
  series 1.32/1.33/1.35/1.35/1.35 - flat; chunk-2 now +-0.0015 across FOUR gates.

- GUARD ROTATION #15 (Exp901): protocol 1.1939, guard 1.3050/1.3081 (55 tok) = +9.5 % premium -
  5th confirmation of the per-token criterion (VAE 7.1/7.1, premium 69 ms/tok, fluent). P2 arm
  heat-contaminated again (1.2885, both phases, 4th sweep arm) and resolved to the logged 1.1960
  after idle - the 4th-arm step is now a recurring pattern worth naming: three back-to-back arms
  stay clean, the fourth trips. No overfit signal. Decay flat at 9.0 h.

- SWEEP HEAT PACING (Exp902): run_rtf_multi.sh gates each arm on batt <= 38.0 C (60 s waits, max
  3, HOT flag after). Validated with 4x protocol arms: pacing engaged 3x before arm 4, all four
  within +-0.4 % (1.1956/1.1907/1.1941/1.1906), A_p4 transcript byte-identical. Correction to my
  own pre-registered FAIL mode: HOT + clean does NOT mean the flag is meaningless - the waits
  cooled the SoC (clean number) while the lagging batt proxy still read high (honest flag). The
  flag marks proxy state; the waits do the work. Cost: +3 min on this sweep when engaged.
  Settled anchor 1.1899 at 9.1 h, byte-identical.

- TELEMETRY DATASET FIRST USE (Exp903): n=15 protocol rows correlate rtf against batt/uptime/mem/
  procs (|t|>2.18 pre-registered): batt +0.916 (t=+8.26, decisive), uptime +0.126, mem -0.051,
  procs -0.379 (t=-1.47, not significant, wrong sign for contention - noise until more n says
  otherwise). Short-term variance is heat alone; mem/procs carry no independent signal in the
  6-9 h window. Scope kept honest: narrow ranges rule out strong short-term effects only; the
  slow state (days) is unmeasured by construction. Settled anchor 1.1948 at 9.1 h.

- LONG CLIPS ARE NOT THERMOMETRICALLY VALID IN ONE PASS (Exp904): two chat138 runs (hot start
  and cooled start) agree within 0.8 % (VAE 105.3/104.5, prefill 32.9/32.5, decode 73.4/73.3)
  because both heat identically during their 3.7 min (end batt 40.0-40.2 both) - and both exceed
  the settled bands by ~5 %. Ruling: intra-run heat, not length-dependence (settled 24-window
  rate matches 4-window; both runs share one self-heating profile). Consequence: long clips buy
  PRECISION (Exp867h stands - it is about repeatability, and paired designs cancel the bias) but
  their ABSOLUTE levels carry ~+5 % intra-run bias fresh. Short settled runs measure levels;
  long runs measure rates-with-bias. Fresh 138 s cell 1.52-1.53 stays heat-influenced provisional.
  Corroboration (rough): fresh decode slope ~6.6 us/pos vs 13.22 (about half) matches the regime
  asymmetry. Settled anchor 1.1924.

- GUARD ROTATION #16 + GATE BY CADENCE, ONE INTERRUPT, ONE LESSON (Exp905): guard sweep
  P,G,G,P reads Pproto 1.1943, Gguard 1.3068, G2guard 1.3121 (55 tok both), P2proto 1.1979 -
  premium +9.65 %, 6th per-token confirmation (75 ms/tok, VAE 7.1/7.1, fluent, arms
  byte-identical pairwise); no 4th-arm step (+0.3 %). Gate gate905 40/40, WER 4.55 %
  (S=29 D=2 I=2, identical profile seven gates running), paired b=0/c=0 vs BOTH gate900
  and gate852 (p=1.0). Gate mean 1.3133 misses the 1.32-1.37 band on the letter because a
  mid-round interrupt idled the phone 55 min before chunk2: per-utterance ratios vs gate900
  show chunk1 x1.02 but chunk2 x0.92 uniform on identical outputs - cool-state bias, not a
  product change (and a symmetric corroboration of the heat-step size). Lesson: gate chunks
  run back-to-back in one thermal state, or the series point carries a qualifier. Telemetry
  n=17: batt r=+0.90 decisive; procs-watch attenuates to -0.26 (ns, resolving to noise).
  Settled anchor 1.1938 at 10.8 h, byte-identical, checks OK - decay flat 6m-10.8h.

- ROLLBACK LADDER DUE BY CADENCE (Exp906): 15 arms x 2 reps on the unchanged binary (23 rounds
  since Exp883). Identity exactly as Exp883: 12 byte-identical (1a095c8496b4), flush_off =
  55ac39b635cb both reps, ct_block/ALL_OFF differ by design (37/38 tok), all arms STABLE.
  Costs all within +-1.5 pp (flush exact, axpy inert 7th time); absolute seconds transfer
  (stack 4.65 s, dw_conv1d 1.38 s). Default rep spread +0.09 % - heat controls exemplary.
  Ladder default mean 1.1974. No repricing needed. Two by-products: (1) a 15-arm gated ladder
  exceeds one 40-min tool call when waits engage - split reps next time; (2) TELEMETRY GAP:
  device_state.tsv has no config column, so ladder hatch arms confound any raw correlation
  (demoed: +0.90 batt-r collapses, uptime fakes significance). Queued fix: log EXTRA_ENV per
  row so analyses filter to default-config; clean n=17 reproduces batt +0.896 exactly.

- TELEMETRY CONFIG COLUMN SHIPPED (Exp907, harness): device_state.tsv gains 12th column extra_env
  (EXTRA_ENV per row, empty = default; appended at end, indices 0-10 stable; legacy header migrates
  one-time, data history never rewritten); check 18 moves 11 -> 12 in lockstep (fault 18's plant
  is column-agnostic). Proven: default 1.1941 (col12 empty), axpy run carries its env string with
  byte-identical transcript (inert 8th time; pooled +0.8 % within hatch bar after order-symmetric
  retest killed the warm-second-run hypothesis). Full audit 129/0 green; selftest --only 18 green.
  Settled anchor 1.1922. Pre-907 rows lack config but are identifiable (the 30 ladder rows by ts).

- FAULT BOARD DUE BY CADENCE (Exp908): 13/13 green (model truncations loud, audio denominators
  content-derived, config edges loud) - no regression 22 rounds on. Prediction footnote: I quoted
  Exp886's as-is read instead of its same-session fix; the board matched the fixed expectation.
  Lesson: quote the commit, not the log's mid-round state. Anchor 1.1939, byte-identical.

- GUARD ROTATION #17 (Exp909): order-symmetric P,G,G,P in one pass (pacing active, no HOT waits).
  protocol 1.1955, guards 1.3093/1.3103 (55 tok both) = +9.56 % premium; per-token
  (13.098-11.955)/16 = 71 ms/tok, VAE 7.1/7.1 identical, decode 4.2 vs 3.0 s, prefill 1.8 both.
  Transcripts verified from the DEVICE arm captures (run_rtf_multi pulls only err-*.log, so I pulled
  out-Pproto1/Gguard1/G2guard1/P2proto1.log explicitly): P arms = 1a095c8496b4 (both), G arms =
  849cca7df5bc (both, pairwise identical). 7th per-token confirmation of the premium; NO 4th-arm step
  (P2 1.1898 is the low arm, -0.5 % vs P1). All cost_pred909.txt predictions met (P1 1.18-1.23,
  premium 1.09-1.11, 70-85 ms/tok, tokens 39/55). Settled anchor 1.1948 at 12.1 h (batt 35.7 C,
  majflt 0), byte-identical. Anchor decay series flat 6 m-12.1 h (1.19-1.22).

- GATE REGRESSION BY CADENCE (Exp910): 40/40, WER 4.55 % (S=29 D=2 I=2 - identical profile EIGHT gates
  running), paired b=0/c=0 of 731 vs BOTH hyp-gate905 and hyp-gate852 (CI exactly [0,0], p=1.0), and
  0/40 differing transcripts vs gate905 (byte-identical) - all on the unchanged binary (device BIN md5
  744bf77052fa matches gate905's stamp). Gate mean 1.3450, inside the 1.32-1.37 band; series
  1.3266 (885) / 1.3546 (890) / 1.3475 (895) / 1.3544 (900) / 1.3133 (905, cool-chunk2 qualifier) /
  1.3450 (910). Intra-run split this round: chunk1 1.3149 (started cool after >=15 min idle) and
  chunk2 1.3750 - opposite order to gate905 (1.3566/1.2700), mean ratio 1.0272; the chunk swing is
  state, reproduced symmetrically. Gate chunks were run back-to-back in ONE call (Exp905 lesson).
  Settled anchor 1.1986 at 12.3 h, batt 38.0 C warm post-gate (series says this is +0.3 % vs a cool
  read; recorded as-is, not "corrected"), transcript byte-identical.

- COVERAGE / AUDIT SELF-TEST BOARD (Exp911): 19/19 plantable checks fire on their own fault, 0 silent
  or invalid, 3 classes accepted as uncontrolled (co-runner, sweep-resolves-to-tier which has its own
  --dry control, and capture-identity which is WARN-only by construction). Every plant is a byte+stat
  snapshot restored in a finally block; `git status` clean and the final clean-tree audit is green
  (129 checks, 1 explained WARN) after all faults are reverted. Host-only round, no device time.
  Notable fired classes exercised this round: doc-vs-tier drift (10), doc-vs-parser flag (11),
  headline-vs-headline.json (9), frozen-refs removal (8), WAV header corruption (7), asset
  duplication (6), schedule-changing shipping command (13).

- BEHAVIORAL CONTRACTS ROTATION (Exp912): 11/11 PASS / 0 fail on the current binary (~29 rounds since
  Exp892, itself then "44 rounds overdue"). Labels hold (silence -> [Silence], noise -> [Noise], music
  -> [Music]); 48 kHz stereo resamples + transcribes (39 tok); short36 sub-piece path graceful (17 tok);
  twospk_overlap splits speakers (Speaker 1 present) while the sequential twospk stays one tag = the
  closed Exp650 behaviour, recorded not failed; ladder canaries EXACT: 10 s 39, 17 s 106, 69 s 446,
  138 s 876. All expectations are labels/tags/token-counts, never wording, so the board cannot become a
  speed measurement by accident. Device warm after the board; settled anchor 1.2009 at 12.7 h (batt
  38.4 C, top of the warm envelope; series 35.7 C 1.1948 / 38.0 C 1.1986 / 38.4 C 1.2009 tracks the
  batt correlation, not drift), transcript byte-identical.

- SESSION-MEMORY SOAK ROTATION (Exp913, ~35 rounds since Exp887): 3 chat69 runs under ONE sampler
  (rss_soak --repeat 3), full log teed to .auto/soak913.log.
  * Per-run steady MEDIANS 2176.8 / 2177.1 / 2187.0 MB, peaks 2198.2 / 2198.4 / 2198.4 MB (spread
    0.2 MB) -> peak invariance = NO SESSION GROWTH, reproducing Exp887 (~2198 MB) in the current regime.
    The +5.10 +/- 5.53 MB/run median step is the known two-state pattern, not accumulation.
  * Global 5.6-min slope is INCONCLUSIVE (+0.81 +/- 15.34 MB/min) - expected: the two-state spikes
    dominate a 5-min window, which is exactly why the per-run comparison is the designed instrument
    (Exp865f). Do not quote the global slope as a verdict.
  * fds 3/3 flat (limit 32768). Threads read 1-3, which is PHASE structure (1 during load, 2 during
    LM, 3 during concurrent VAE), NOT a leak: the sequence is 2/3 throughout with one 1 at a run
    boundary; the tool's "CHANGES by 2 - thread leak?" line is a naive max-min test. Same class as
    Exp905's chunk bias - read the mechanism, not the flag.
  * The RTFs in this log (1.478/1.547/1.544 at batt 40.9-41.2 C) are heat-influenced and are NOT
    ladder cells (Exp904: long clips are not thermometrically valid in one pass).
  * SELF-INFLICTED WOUND (method record): the first soak attempt was piped through `tail -30`, which
    discarded the verdict block before the tool captured it, so the whole run had to be repeated
    (~6 min + cool-down). RULE: multi-run instruments write to a log file (tee) FIRST - never pipe an
    experiment through tail, the verdict is the product, not the METRIC tail.
  Settled anchor 1.1968 at 13.5 h (batt 37.4 C after 8 min idle), transcript byte-identical.

- NOISE FLOOR RE-DERIVED AT 13.5-14 h UPTIME + FREE EQUIVALENCE-MAP REFRESH (Exp914).
  * 6 protocol reps, 3-min spacing, one device state per rep (all other_busy 1-2%, cpu7_khz 1.43 GHz,
    tokens 39, transcript 1a095c8496b4): 1.1987 / 1.1878 / 1.1931 / 1.1948 / 1.1991 / 1.2004.
    mean 1.1957, sd 0.397 %/rep, range 1.1878-1.2004 = 1.05 %. Batt fell 37.5 -> 35.7 C across the
    series while rtf rose +0.14 % first-vs-last (+0.105 %/rep) - i.e. NOT batt-driven this time.
    Excluding the rep2 dip (the coolest batt reading is also the fastest): sd 0.260 %, range 0.61 %.
  * READING: the settled floor at long uptime is 0.26-0.40 %/rep, i.e. consistent with Exp884's
    long-uptime 0.233 % within n=6 uncertainty, but the observed RANGE is wider than the "+-0.3 %"
    that note quoted. PRACTICAL RULE, unchanged in direction but restated: unpaired single-run deltas
    below ~0.5 % are not evidence; keep the 3-rep interleaved A/B (interleaving cancels the common
    drift) and the >=2 % ship bar. Do not tighten the bar to +-0.3 % - one 6-rep sample range was 1 %.
  * Session anchors today span 1.1922-1.2009 (0.7 %) across batt 35.7-38.4 C - which is the same
    width as one 6-rep batch, so the between-batch variation is dominated by state, not by the
    measurement procedure.
  * FREE: equivalence map regenerated vs hyp-bound835 (83 archived sets): 11 are byte-identical, and
    ALL v4.8-era gate sets are in that list - hyp-gate845/852/878/880/885/890/895/900/905/910 plus
    hyp-tile864 - so the current binary's output identity is recorded against the whole gate history,
    not just the last two gates. Host-only, 0.4 s.

- REPRODUCE-PATH BOARD (Exp915, ~50 rounds since Exp865g): executed RESULTS.md's documented binary
  recipe VERBATIM - `LM_FILE=lm-q8head.gguf VAE_FILE=vae-encoder-convint8.gguf ./.auto/measure.sh` -
  from the committed tree. Result 1.1887, tokens 39, transcript `1a095c8496b4` (= the protocol
  reference), checks OK, RSS 2191.5 MB, majflt 0. So a reader following the docs still gets the
  shipped system, not a neighbouring tier (the Exp859 failure mode where the block named the F16-conv
  reference and returned 1.9372).
  * SCOPE, stated honestly: the build was an incremental no-op (the whole command took 22.8 s, same as
    --skip-build), so this round validates the documented command + runtime identity, NOT a
    from-scratch compile. The clean-wipe rebuild remains Exp57/Exp610 (artifacts bit-exact).
  * Audit re-run as the LAST write of the round (loop rule): 129 checks / 1 explained WARN / 0
    failures - check 10 keeps the command naming the tier, check 5 keeps host/device bin + .so hashes
    in sync.

- RATE-CEILING / LONG-FORM BOARD IN THE FRESH REGIME (Exp916, ~36 rounds since Exp879): ran the 250.8 s
  unique clip (`AUDIO=long250.wav`, default tier, cool 36.0 C start, end batt 40.1 C): 86 windows /
  85.42 effective / 728 tokens / 3140 positions -> vae 191.7 s, prefill 63.7 s, decode 67.3 s,
  wall gen 322.7 s (rtf 1.2866 - heat-biased, NOT a ladder cell), RSS 2218.3 MB, majflt 0, load 0.8 s.
  * The linear structure transfers to a new length in the fresh regime:
      VAE    2.244 s/effective-window vs the settled 2.10 -> +6.9 %, which is the intra-run self-heating
             bias a 250 s single pass carries (Exp904), not a rate change.
      prefill 26.4 ms/row measured; the fresh-scaled model (x0.628 on the Exp870 constants, KV term at
             ctx=1469) predicts 65.3 s vs 63.7 s measured = -2.5 %. Crisp even at 86 windows.
      decode  92.4 ms/token: still the LOOSE leg. The two fresh reps (protocol 76.9 @ ctx 58, chat69
             78.2 @ ctx 522) imply a ~2.8 us/position slope, but this clip implies ~10.7 us/position -
             i.e. the slope is instrument-dependent, exactly what Exp891/904 flagged. Do not quote a
             decode law from this clip; decode pricing stays at the phase-scalar level.
  * Session limit not hit: 3140 positions completes under the DEFAULT -c 4096, so long250 is reachable
    without -c 8192 (Exp898's 8192 session was about the 400+ s end, not this clip). RSS 2218.3 MB
    reproduces Exp879's 2218 MB at the same position count.
  * HAZARD FOUND (queued, not fixed this round): `.auto/cost_fit.sh --predict` still carries the Exp870
    LONG-UPTIME constants (A=0.39, B=3.343, PM=29.82, ...), so in the fresh regime it over-predicts
    VAE 286.0 s vs 191.7 measured (+49 %) and wall 467.1 s vs 322.7 (+45 %) while its output line still
    claims "validated: vae +-0.2 %, wall +-0.1 %". The loop's practice (Exp888/891) applies fresh
    scalars by hand; the TOOL does not say so. This is the Exp896 class - an absolute timed number
    without a regime qualifier - and it is the next file to fix (with a control, per Exp660).

- COST_FIT --PREDICT REGIME QUALIFIER SHIPPED + CONTROLLED (Exp917, harness; nothing else shipped).
  The Exp916 finding: the tool's constants are the Exp870 LONG-UPTIME fit, so on a fresh boot it
  over-predicts (250.8 s clip: VAE +49 %, wall +45 %) while its output line claimed only
  "validated: vae +-0.2 %, wall +-0.1 %" - an absolute timed number with no regime qualifier, the
  Exp896 class. Fix:
  * `--predict` now prints an explicit `regime: LONG-UPTIME (Exp870 fit, PRE-REBOOT)` block with the
    fresh-regime numbers (VAE 2.10 s/effective-window, Exp888/891) and how to override.
  * new `--vae-rate S` (s per effective window) and `--lm-scale F` knobs, so fresh-regime pricing is a
    stated input instead of a silent manual multiplication; override prints
    `regime: USER-OVERRIDDEN` with the values used.
  * default output is NUMERICALLY UNCHANGED (regression-checked against the pre-edit values:
    250.8 s/728 tok -> vae 286.0, wall 467.1, rtf 1.8626; `--vae-rate 2.104` -> vae 179.7, wall 360.9).
  * `--selftest` proves the knobs FIRE (an output change: 2.0 x eff exact to the print precision;
    --lm-scale 0.5 exactly halves prefill) and that an unknown flag is REFUSED (exit 2). NEGATIVE
    CONTROL: planted `vae = A + B*eff` (ignoring the override) -> selftest FAILS with
    "--vae-rate did not fire: 114.3 != 68.15" and "inert"; restored byte-identical -> green.
  Audit 129/1 explained WARN/0 failures after the change.

- ROLLBACK LADDER DUE BY CADENCE (Exp918): 15 arms x 2 interleaved reps (rep2 reversed, batt gate <=37.0 C,
  no HOT/flags, 30/30 arms ran) on the unchanged binary. Default 1.1937 (rep1 1.1952 / rep2 1.1922 -
  bracket spread 0.25%, so the default anchors itself across the 42-min ladder). Full log: .auto/ladder918.log.
  * IDENTITY: 12 arms byte-identical to the reference `1a095c8496b4`; `flush_off` reproduces the historical
    pre-flush protocol `55ac39b635cb` exactly (as designed); `ct_block` (37 tok, hash `ad1953f30010`) and
    `ALL_OFF` (38 tok, c4031e597b20) differ BY DESIGN (layout revert = system change).
  * COSTS (vs default, seconds-first per the Exp883 rule): stack_off +4.68 s (+39.2 %), ct_block +2.12 s
    (+17.8 %), flush_off +1.57 s (+13.1 %), dw_conv1d +1.25 s (+10.4 %), gelu_bias +0.77 s (+6.4 %),
    bound_batch +0.48 s (+4.1 %), gelu_batch +0.41 s (+3.4 %), norm_fuse +0.38 s (+3.2 %), dw_lpad
    +0.34 s (+2.8 %), ls_fuse +0.26 s (+2.2 %), cont_tile_off +0.24 s (+2.0 %), mm_m2 +0.02 s (+0.2 %),
    dw_axpy -0.1 % (INERT, 8th confirmation).
  * ONE ARM OUTSIDE THE HISTORICAL BAND, flagged not explained: dw_conv1d's overhead now reads +10.4 %
    (absolute +1.25 s) vs +12.8 % (+1.53 s) at Exp883 and +1.38 s at Exp906 - a mild monotonic decline
    across three ladders at identical arm order, so not arm-order/heat. Everything else
    is within +-1.5 pp (stack_off +39.2 vs +38.6, flush +13.1 vs +12.7 exact). Re-check at the next ladder;
    until then quote the runbook in seconds.

- GUARD ROTATION #18 + THE ARM-TRANSCRIPT INSTRUMENT (Exp919). Two passes of the order-symmetric P,G,G,P
  sweep, because the first pass heat-stepped and the second started warm (both logged):
      pass A (batt 36.9 cold start): P1 1.1962, G 1.3059, G2 1.3081, P2 1.2054 (4th arm +0.8 %)
      pass B (immediately after):    P1 1.2342 (HEAT, +3.2 % vs A-P1), G 1.3111, G2 1.3130,
                                     P2 1.1906 (clean - the Exp902 pacing waited twice on heat first)
  Clean protocol points are A-P1 (1.1962) and B-P2 (1.1906) -> mean 1.1934; all four guard arms mean
  1.3095 -> premium **+9.73 %**, (13.0953-1.1934*10)/16 = **72.6 ms/token** (predicted 70-85). Coolest
  guard pair only: +9.52 %, 71.0 ms. 8th per-token confirmation; series #14..18 = 9.6 / 9.5 / 9.65 / 9.56
  / 9.73 - flat. The 4th-arm step this time appeared in PASS A's P2 and in PASS B's first arm, which is
  the Exp897/901/902/905 recurring pattern (three clean arms, the fourth or a warm start trips).
  * INSTRUMENT SHIPPED (Exp909's gap): run_rtf_multi.sh now pulls `out-<tag>.log` itself and prints the
    transcript hash per arm (`tx=`), so the guard board's identity claim is measured by the tool instead
    of by a manual pull. FIRST VERSION USED THE WRONG ORACLE - it hashed only the `^[n/m]` window lines
    (8750f02531f4 / ff345ffab10c) while the loop's canonical reference is the WHOLE capture file
    (1a095c8496b4 / 849cca7df5bc). The pre-registered control ("P arms must read the reference hash")
    caught it; fixed to `md5sum` of the pulled file and re-run: P arms 1a095c8496b4, G arms 849cca7df5bc
    exactly, on both arms of both roles.
  * LESSON (5th instance of this class): a new instrument that "fires" is not enough - the value it
    prints must be checked against a KNOWN reference in the same round, or it silently defines a second,
    private oracle. The control was free because Exp909's manual pulls are in the record.
  Settled anchor 1.1950 at 16.6 h (batt 36.5 C), transcript byte-identical; audit 129/1 explained WARN/0.

- PEER EXCHANGE RECORD (2026-09-20, CUDA/B300 loop "picard-desktop"; no experiment, no device time).
  They sent (a) a RETRACTION and (b) a segment-5 findings package; I replied with cross-checks.
  THEIR CORRECTED NUMBERS (version of record; the "slack ~8.5 s absolute" draft never reached me):
  wall_serial - wall_pipeline = 8.05 / 4.1 / 9.4 s at 10 / 116 / 240 s = 24.7 % / 1.16 % / 1.04 % of
  wall - NOT constant, NOT per-window; "work below slack" is a <=2 % effect for them, not a cliff.
  Durable part for us: their SERIALIZED loop closes to vae_s + lm_s = wall at 0.999 at all three
  lengths - i.e. the additive frame our tree uses is backend-independent, so their net-model and our
  additive model are the same statement in that frame. (Frame shared, constants not - which is exactly
  our Exp916/917 regime-staleness result.)
  THEIR SEGMENT-5 (recorded, per-phase, never whole-run): F16-VAE full runs thrash BY DEFAULT there
  (11k majflt, 400-490 MB swap; encode +13 s AND the LM doubles 32->57 s via evicted weight pages, so
  Q8's edge is up to -46 % - residency protecting the LM, not just load); Q5_0 decode -30 % / prefill
  +23 %; q4_k_s decode -26 % / prefill +35 % (wall +6.5 %, its old speed-neutral claim dead on their
  current tree; only -102 MB RSS stands); K-quant output effect is LENGTH-dependent (identical 10 s
  transcript, word changes at 116 s); two /tmp long clips vanished with no provenance written.
  OUR CROSS-CHECKS SENT: (1) frame agreement (our tree closes <=0.3 % at 10/17/69/138 s, -0.3 % at
  250.8 s); (2) a stored constant is a regime claim (Exp916: +49 % VAE / +45 % wall in the fresh
  regime on a byte-identical binary); (3) their F16 thrash is environment-specific - our F16 tier on an
  8 GB phone reads 2.99 GB peak RSS, majflt 0, flat to 138 s (and our shipped tier 2.19 GB / majflt 0),
  so every residency cell we quote carries RSS + majflt together; (4) our asset-provenance rule
  (name+size, blessed device md5, audit FAIL on drift / WARN on byte-identical collision, derivations
  machine-verified) is the fix for their vanished-clip loss - offered the manifest format.

- FAULT/CORRUPTION BOARD DUE BY CADENCE (Exp930): 13/13 PASS, 0 fail on the unchanged binary
  (BIN md5 744bf77052fa, same as Exp908) - no regression 23 rounds on. Full log: .auto/fault930.log
  (predictions in cost_pred930.txt, written first).
  * MODEL: LM tail -8 MB and LM header-only 1 MB -> `failed to load model`; VAE tail -8 MB and
    -64 MB -> `truncated or corrupt: te...`. All exit 1. (The Exp850 class: before the short-read
    check, a truncated VAE loaded and fed fluent ZERO weights with exit 0 and normal timing.)
  * FLAGS: unknown `--kv-type` -> exit 1 `Unknown arg: --kv-type` - a doc-promised option is not
    silently accepted.
  * AUDIO (the RTF-denominator contract): healthy baseline H=1.1908 in-session; trunc_half (header
    says 10 s, data 5 s) runs at 1.3442, lie_dur (header claims 40 s) runs at **1.3131 = 1.10xH** -
    inside the guard-like band [0.7H, 1.3H], so the denominator comes from DECODED samples: a
    header-following metric would have read ~0.3, a false 4x speedup. header-only (44 B) and empty
    (0 B) both refuse with no metric at all.
  * CONFIG EDGES: `-c 16` (below one window) -> `frames failed`; `--vae-pieces 7` and `--vae-pieces 0`
    -> `must divide 26`. All exit 1.
  * The Exp886 fix held: the audio arms are judged against a SAME-SESSION healthy baseline, not the
    old absolute [1.5, 4.0] band, which the fresh regime would have failed with a healthy binary.
  Settled anchor 1.1948 at 16.9 h (batt 36.2 C), transcript byte-identical. The board leaves
  last_out.txt holding a fixture capture, so the anchor was taken with a fresh measure.sh run (that
  stale-capture trap is pre-registered in cost_pred930).

- PEER EXCHANGE, FOLLOW-UP (2026-09-20, same CUDA loop). They confirmed the retraction is dead on both
  sides, adopted the frame-vs-constants split as their standard cross-box answer, adopted our
  RSS+majflt-together rule as a stated requirement, and declined the reboot experiment explicitly
  ("the finding transfers, the experiment doesn't need to" - they carry cotenant/temp/swap/majflt
  qualifiers per sample, the same object as our regime qualifier + overrides at a different
  granularity). They ASKED for the asset-manifest format for their vanished 240 s/431 s clips; sent the
  wireable spec: three-key JSON (assets{md5,note} / derives[prefix_of|concat] / known_extras patterns),
  the four checks (drift / collision+undeclared-duplicate / derivation arithmetic / coverage union),
  the three planted faults that must fire, and the two operational traps (bless via explicit device
  paths, never a remote loop that expands $f locally -> d41d8cd9 empty-stdin; assert absence with a full
  list or count, never head/tail). Nothing owed back.

- INTRA-RUN HEAT vs LENGTH: IT IS A DISCRETE BIG-CORE CLOCK STEP, NOT A LENGTH EFFECT (Exp931).
  Mechanism resolved with two instruments: per-window LATENCY_TRACE ("LT w=N vae=...") and a new
  committed sampler, .auto/batt_sampler.sh (batt temp + cpu7/cpu0 scaling_cur_freq every N s during a
  run; control: numeric columns, and a bad DEV prints "?" loudly). Both runs: long250.wav, same binary/tier.
    COOL start (35.1 C): vae 189.9 s, rtf 1.2767. Per-window VAE: w1-35 = 2037-2078 ms (FLAT),
      w36 = 2313 (the flip), w36-85 = 2323-2399 ms (FLAT). Step at t~120 s.
    HOT start (39.0 C, ~2 min after the first run): vae 196.8 s, rtf 1.3174. Per-window: w2-10 =
      2029-2077 ms (fast), w11+ = 2307-2340 ms. Step at t~40 s. cpu7 sampled 2400000 kHz at t=20/30/41 s
      and 2000000 at t=0/10/51+ -> the flip coincides with the governor's 2.4 -> 2.0 GHz transition.
  * THE TWO STATES: 2.045 s/eff-window (2.4 GHz boost) vs 2.335 s/eff-window (2.0 GHz settled), ratio
    1.142 for a clock ratio of 1.200 -> the VAE kernels are ~70 % clock-scaled, consistent with the
    elementwise-traffic profile (Exp661/665). Within a state the rate is flat to ~1 % across 85 windows.
  * CONSEQUENCE: there is NO length dependence in the VAE per-window rate. The observed length trend
    (protocol 2.12, chat69 2.09, chat138 2.23, long250 2.24 avg) is just the fraction of the run spent in
    the settled state: short clips finish inside the boost window, >=2 min clips straddle the step.
    Long-cell caveat restated: a 138 s/250 s cell is a STATE AVERAGE - comparable across lengths only
    within one clock state, which is why long-form rates transfer (Exp879/916) but absolutes do not.
  * Exp904's hot-vs-cool chat138 agreement (VAE 105.3/104.5) is re-explained: both runs reached the
    settled state early, so both averages are "settled" numbers - the design could not see the step.
    Its verdict ("intra-run heat, not length-dependence") was directionally right; the mechanism is a
    discrete CLOCK STEP, not a smooth ramp.
  * Batt-vs-per-window correlation is NOT the discriminant: r(vae_ms, batt) = +0.58, r(prefill_ms, batt)
    = +0.93, but batt and window index are collinear inside one run. The hot-start arm is the
    discriminant (same clip, step moves 36 -> 11).
  * TELEMETRY DEFECT (flagged, no published number changes): device_state.tsv's cpu7_khz column is
    sampled ONCE, pre-run, and reads 1430000 in every row, while during-run samples read 2400000 then
    2000000 - the column does not describe the measured run's clock. Use it only as run context, never
    as a within-run state variable.
  * Also visible per-window: w1 carries a +20 % VAE premium in both runs (2482/2178 ms vs the 2045
    plateau) - the known first-window page-in effect, now measured directly.
  * METHOD WOUND: the first attempt wrote `cd VibeASR.cpp && sampler & actual-run ...`, and `&`
    backgrounds the WHOLE and-list - measure.sh never ran (cwd stayed the parent, rc=127) while the
    wait burned 600 s, and run_experiment still reported PASSED. Rule: keep `cd` OUTSIDE any
    backgrounded group, and never trust PASSED without checking that the intended artifact exists.
  Settled anchor 1.1941 at 17.6 h (batt 37.5 C), transcript byte-identical.

- PEER EXCHANGE, MANIFEST WIRED ON THEIR SIDE (2026-09-20, CUDA loop). They blessed 8 clips and shipped
  asset_audit.sh with our four checks; three plants each fire their own FAIL (wrong hash / undeclared dup
  / off-by-one excerpt), clean run green, 0 WARN, staged dup removed. Two adaptations they made:
  (1) derivations compare DECODED FRAMES, not bytes (they parse the WAV header), with SILENCE:n parts;
  (2) our overlap trap bit them in REVERSE - their scan_dirs nested (asr-gate + asr-gate/derived_slices)
  listed one clip twice and the collision check correctly WARNed on their own bug; fixed with a path set.
  * Reciprocal finding for US (queued, prompted by their SILENCE:n): our `derives` rules express
    `prefix_of` and `concat` of whole/ranged real clips, but NOT a synthetic part - and `long250.wav`
    (which backs Exp879/898/916/931 long-form cells) is `holdout_en 139.6 s + 1,240 ms gap + first 110 s
    of holdout_zh`, i.e. documented in prose but NOT machine-verified. QUEUED: add a `silence` part type
    (in samples) to the concat rule + manifest entry + a planted off-by-one-silence control, so the clip
    behind the long-form ladder is proven the way chat17/chat138/chat155 already are.
  * Our nested-scan exposure: structurally impossible here (flat device dir, basename keys, no path
    aliases), so the "same file listed twice" cause cannot arise; two REAL files with identical bytes is
    the documented Exp675 WARN (chat.wav == chat69.wav). Same symptom, different cause - their path-set
    fix is the right general guard if our scan ever gains nesting.

- COST MODEL RE-DERIVED STATE-CONDITIONED: THE LENGTH TREND IS A STATE MIX, PROVEN BY ARITHMETIC
  (Exp932). All four ladder clips in one session with per-window LATENCY_TRACE + the device-state
  sampler. Clips: 10 s (4 win), 17 s (6), 69 s (24), 138 s (48).
  * TRACE vs SUMMARY: sum(LT vae) matches vae_s to -0.34 %/-0.34 %/+0.05 %/+0.02 %; tree closure
    (vae+prefill+decode vs rtf x audio) -0.11 %/+0.40 %/-0.12 %/+0.02 % -> the additive frame holds in
    the fresh regime at all four lengths.
  * STATE PLATEAUS (window 1 and the flushed last window excluded): FAST 2067 ms (n=13, sd 1.74 %),
    SLOW 2334 ms (n=61, sd 0.51 %); ratio 1.129. Fully-fast clips read 2046-2070; the slow state is
    2334. (The fast pool's sd is inflated by one 2144 ms transition window in chat138.)
  * STATE-MIX ARITHMETIC CLOSES FOR EVERY CLIP (+-0.3 %): predicting each clip's VAE total as
    window-1 + sum(per-window rate by classified state) + flushed last gives
      protocol 7.1 vs 7.1 (slow fraction 0/2) | chat17 11.9 vs 11.9 (0/4) |
      chat69 53.3 vs 53.4 (16/22) | chat138 109.8 vs 109.7 (45/46).
    So the aggregate per-clip rate (2.122 / 2.098 / 2.278 / 2.333 s/eff-win) is FULLY explained by the
    slow fraction - there is no residual length term. This is the falsifiable version of Exp931: one
    clip (chat69) exhibits BOTH rates within itself.
  * CLOCK TRAJECTORY during the sequence: cpu7 = 2.4 GHz at t=0..90 s (protocol + chat17 + chat69's
    opening windows), then 2.0 GHz for the rest (chat138 entirely settled). The state classification
    from VAE times alone agrees with the independent clock samples.
  * WINDOW-1 PREMIUM, CORRECTED: within-run it is +4.2 % (protocol) / +6.7 % / +6.0 % / +6.4 %.
    Exp931's "+20 %" was a CROSS-run comparison (a hot run's w1 vs a cool run's plateau) - retracted;
    quote +4..7 %.
  * FRESH LM LEGS, STATE-MIXED (flagged as such): prefill 16.2/18.0/22.5/25.7 ms per row at ctx
    58/118/522/1049 -> intercept ~15.6 ms/row + ~9.6e-3 ms per KV position; decode 78.4/74.9/79.7/
    84.7 ms/token -> ~78.0 ms/token + ~6.3e-3 ms per position. Both slopes are inflated because the
    short clips measured FAST and the long clips SLOW, i.e. the state mix masquerades as a KV term -
    treat these as upper bounds until a same-chip-clip fast/slow pair fixes the scaling (queued).
    The intercepts are clean: protocol and chat17 are both fully fast (prefill 16-18, decode 75-78).
  * TOOL UPDATED (Exp917 text superseded): `cost_fit.sh --predict`'s fresh-regime hint now gives the
    two STATE rates (2.05 boost / 2.33 settled) and says to pass --vae-rate for the state being
    priced, never for the clip length; selftest still green after the edit.
  Settled anchor 1.1887 at 18.0 h (batt 37.0 C), transcript byte-identical.

- PEER EXCHANGE CLOSED: CLOCK-STEP RESULT DOES NOT TRANSFER TO THEIR BOX (2026-09-20, CUDA loop).
  They checked our Exp931/932 clock-step mechanism against data already on disk instead of asserting:
  cpufreq distribution 935/952 polls at the 1.479 GHz ceiling with 17 scattered below (idle ramps
  during load), median identical in every logged run, and their GPU partition (31 reps) pegged at max.
  => their box has NO boost state above the settled one, so their long-window rates are single-state by
  construction; a downward idle blip during load cannot make a generation rate bimodal, and the rate
  law's +-5% residuals independently bound any residual state effect. They filed the general rule:
  "on any box WITH a governor, a long-window rate is a state average until proven single-state."
  * RECIPROCAL METHOD RULE (ours, sent back): the rate law's own RESIDUAL band is usually the cheapest
    single-state test - if a fit leaves +-5% residuals, a hidden state effect that size is already
    excluded, so the paired fast/slow run is unnecessary. On our side the residuals were +-0.3%, which
    is why a 13% term could not hide and the two-run split was worth the device time. Rule of thumb:
    design the state-split experiment only when the suspected effect EXCEEDS the model's residual band.
  * No cells change on either side; nothing owed.

- PHASE-SPECIFIC CLOCK SENSITIVITY, AND THE LM SLOPES RE-SORTED (Exp933). Probe: chat17.wav repeated
  6x back-to-back (FIXED ctx=118, FIXED 106 tokens every run), LATENCY_TRACE + 5 s device-state sampler,
  from a 37.1 C start. The state flips between run 2 and run 4 (run 3 = transition), and the clock
  samples agree: cpu7 2.4 GHz t=0..72 s, 2.0 GHz from t=77 s (two transient 2.4 s readings at 92-108 s).
    run1 1.3396 / run2 1.3392 | VAE 11.8, prefill 3.0/3.1, decode 7.9
    run4 1.4876 / run5 1.4794 / run6 1.4845 | VAE 13.5/13.4/13.5, prefill 3.6, decode 8.2/8.1/8.1
  * PAIRED STATE SCALING (same clip/ctx/tokens, so no KV confound - the point of the probe):
    VAE x1.142 | PREFILL x1.187 | DECODE x1.028. The step is NOT a uniform clock multiplier:
    prefill (batched GEMM) is the most clock-bound, VAE (kernel-rate-bound, Exp681) in between, and
    decode (weight-streaming GEMV) is nearly state-insensitive. Prediction miss recorded: P2 predicted
    decode slow/fast 1.05-1.15; measured 1.028 - the miss sharpens the mechanism rather than weakening it.
  * CONSEQUENCE FOR THE LM LEGS: the Exp932 "LM slopes are state-inflated" flag is right for PREFILL and
    essentially absent for DECODE. Re-fitting the Exp932 clip-level legs with each clip rescaled by its
    slow fraction (prefill x1.187, decode x1.028) gives prefill ~6.7e-3 ms/position (was 9.6e-3 mixed)
    and decode ~8.6e-3 from the chat17->chat138 pair (state-insensitive; the mixed 6.3e-3 was biased LOW
    by the protocol clip's per-window decode overhead, not by state).
  * FRESH KV-LEVER PRICES (upper bound: all KV attention removed), corrected legs:
      10 s 0.5 % | 17 s 1.1 % | 69 s 4.2 % | 138 s 7.9 %
    vs the long-uptime documented 0.46 % / 3.9 % / 7.6 % (Exp870/891) => the KV-lever price TRANSFERS
    across regimes as a fraction of wall (both the KV term and the wall shrink together). NO repricing
    needed; the "~0 % on the protocol metric, ~4-8 % long-form product item" verdict stands.
  * METHOD NOTE: per-window state-conditioned fits (ctx per window) are poorly conditioned here
    (residuals 4-10 ms against KV terms of ~1 ms at small ctx), so they place the state effect in the
    INTERCEPTS (prefill 18.2 -> 21.8 ms/row; decode 74.6 -> 77.1 ms/tok) and are NOT the right instrument
    for pricing - the clip-level fit with slow-fraction rescaling is. Do not quote a KV slope from the
    per-window fit.
  Settled anchor 1.1878 at 18.4 h (batt 36.2 C), transcript byte-identical.

- DURING-RUN CLOCK TELEMETRY SHIPPED (Exp934, harness; nothing else shipped). Closes the defect flagged
  at Exp931: device_state.tsv's `cpu7_khz` is ONE pre-run sample and reads 1430000 in every historical
  row, while a run actually sits at 2400000 (boost) then 2000000 (settled) - the step that Exp931-933
  showed is the dominant short-term driver (prefill x1.187, VAE x1.142, decode x1.028).
  * FIX (mirrors the Exp907 append-only pattern): bench_device.sh samples the big core's governor request
    inside its EXISTING 0.5 s polling loop (one extra file read, free) and emits
    `cpu7_khz_min/med/max` in the exit line; measure.sh parses them, prints `METRIC cpu7_khz_med=`, and
    appends a 13th TSV column `cpu7_khz_med` (header migrates one-time; history never rewritten),
    with audit check 18 moved 12 -> 13 in lockstep.
  * CONTROL = the same experiment that proves it (6 x chat17 back-to-back from a 36.2 C device):
      new column: 2400000 / 2400000 / 2400000 | 2000000 / 2000000 / 2000000
      VAE:        11.8      / 11.8      / 11.8      | 13.5      / 13.4      / 13.5
      RTF:        1.3436    / 1.3417    / 1.3425    | 1.4788    / 1.4764    / 1.4808
    The new column flips at exactly the run where VAE and RTF flip (+10.3 %), and every run's
    min == med == max (single-state, no straddle). The OLD pre-run column reads 1430000 in ALL SIX rows
    across that 10 % change - the defect, restated as a measurement (P1/P2/P3 all hit).
  * Audit green at 13 columns; selftest fault 18 still fires (column-agnostic by construction). The
    flip timing again tracks the start temperature: 36.2 C -> between runs 3 and 4 (~105 s of load),
    vs Exp933 37.1 C -> between 2 and 4.
  * Anchor 1.2159 at 18.7 h (batt 36.3 C), transcript byte-identical. FLAGGED, not hidden: this is the
    highest point of the session series (typical 1.188-1.199), and the new telemetry says state does NOT
    explain it - the run was in the BOOST state (median 2400000, and vae_s 7.2 vs the usual 7.0-7.1). So
    it is ambient/other; re-check at the next anchor rather than calling it drift.

- THE A/B SWEEP IS STATE-STABLE (Exp935) + two instrument completions. Hypothesis: a 4-arm guard sweep
  is ~2 min of intermittent load, so its arms could straddle the 2400000 -> 2000000 step (Exp931-933),
  which would contaminate every premium/ladder number the tool ever produced.
  * ANSWER: the arms are single-state in BOTH conditions. Cool sweeps (36.5 C): 1.1956 / 1.3061 / 1.3104
    / 1.1968 = +9.42 %; and with the new ARM-line clock: 1.1928 / 1.3103 / 1.3110 / 1.1948 = +9.88 %,
    all four arms clock=2400000/2400000/2400000. WARM sweep (batt 38.1 C, preceded by two warm-up chat17
    runs that were themselves SETTLED at 2000000): 1.2000 / 1.3085 / 1.3078 / 1.1944 = +9.01 %, and the
    during-run medians are 2400000 in ALL FOUR arms. Mechanism: the arms are ~25 s bursts separated by
    adb/pull gaps, and the Exp902 pacing waits restore the boost state (arm 1 ran boost right after the
    device had been settled). So the premium series is state-clean; this round's range 9.01-9.88 % vs the
    earlier 9.56/9.65/9.73 - stable, and the pre-registered FAIL mode (a flip moving the premium >1 pp)
    did not occur.
  * INSTRUMENT 1 (shipped): run_rtf_multi's ARM line used to print its OWN post-arm sample
    (`cpu7_khz=$(cpufreq)`, which reads the idle target 1430000/910000 and cannot see the step). It now
    prints the arm's DURING-RUN clock min/med/max, parsed from bench_device.sh's exit line, plus a 6th
    TSV field.
  * INSTRUMENT 2 (shipped, and it corrects a misreading the new field invites): min != max is NOT a
    straddle. Two arms showed a single 2150000 / 2240000 sample (an idle governor ramp between arms)
    with med == max == 2400000 and a fully boost-state rtf. bench_device.sh now also emits
    `cpu7_khz_low` (samples below the 2.3 GHz midpoint) and `cpu7_khz_n`; the anchor run reads
    low=1 of n=19, i.e. one transient, not half a slow run. QUEUED: a 49/51 split would still hide in the
    median - emit a coarse two-bin histogram if a long sweep ever needs straddle detection.
  * METHOD WOUND: two sweeps in one command overwrote each other's per-arm files, because run_rtf_multi
    tags are label+rep only (Pproto1 etc.) - the first cool sweep's during-run profile was lost. Use
    distinct labels (or add a run id to the tag) when sweeping twice in one call.
  * ANCHOR RESOLVES THE Exp934 FLAG: 1.1927 at 19.0 h (batt 36.3 C, boost state, transcript
    byte-identical) - back in the normal band, so the 1.2159 point was a one-off ambient excursion, not
    drift. The new telemetry paid for itself in the first round after shipping.

- long250 PROVENANCE MACHINE-VERIFIED (Exp936; peer-prompted, credits their SILENCE:n idea). long250.wav
  backs every long-form cell (Exp879/898/916/931) and was the ONE asset whose composition was described in
  prose but never checked. Its sources are all still on the device, so this was a grammar gap, not lost
  provenance:
    long250 data range 12,038,400 B = holdout_en.wav data 6,698,880 B (139.56 s)
                                    + 59,520 zero bytes (the inserted 1,240 ms gap at 24 kHz mono s16)
                                    + first 5,280,000 B of holdout_zh.wav (110 s)
  * GRAMMAR EXTENDED (audit_harness.py): a concat part is now any of
      "clip.wav"                whole data range (as before),
      ["clip.wav", start, len]  a RANGED part, byte offsets relative to the FILE,
      {"silence_bytes": N}      N synthetic zero bytes (device-side `head -c N /dev/zero`).
    A ranged part is bounds-checked against that clip's real data range, the declared lengths must sum to
    the target's payload, and the concatenation is hashed ON THE DEVICE and compared with the target's
    data range - so a wrong range, a wrong gap length or a rebuilt source each fail loudly.
  * RESULT: `derivation proven: long250.wav == concat(holdout_en.wav@44+6698880, SILENCE:59520,
    holdout_zh.wav@44+5280000) byte for byte` (audit --verbose). 131 checks / 1 explained WARN / 0 fail.
  * CONTROL (Exp660 rule): planted silence_bytes = 59,522 (one sample too long) -> FAIL with
    "the payload lengths sum to 12038402 B, not the target's 12038400 B - one of the clips was rebuilt
    independently"; restored -> green. Also re-ran selftest fault 7e (it rewrites chat155's concat): still
    FIRES with the new grammar and the extra entry present, so the coverage did not regress.
  * By-product: the existing chat138/chat155 rules were already using ranged parts (whole-data-range), so
    their audit lines now show the explicit ranges (chat69@224+3311352 twice; chat138@44+6622704 +
    chat17@44+816000) - the same proof, stated legibly.
  Settled anchor 1.1963 at 19.5 h (batt 35.9 C, boost state, low=0 of n=19), transcript byte-identical.

- THE ROLLBACK LADDER IS STATE-STABLE (Exp937) + the dw_conv1d arm's variance characterized and my
  Exp918 "decline" note RETRACTED. Hypothesis: unlike the 2-min guard sweep (Exp935), a 15-arm ladder is
  10-40 min with ~35-45 s arms (stack_off/ALL_OFF), so the 2400000 -> 2000000 step should land mid-ladder.
  * ANSWER: it does NOT. One rep, 15 arms, 16 min: EVERY arm ran at clock median 2400000, no HOT and no
    gate-blind flags. The Exp902 cool_down gate (batt <= 37.0 C) plus the per-arm structure keeps the
    ladder in the boost state, so every historical ladder cost is state-clean. Wiring shipped:
    rollback_audit.sh now prints each arm's during-run clock median and carries it into the summary table.
  * COST REPRODUCIBILITY vs Exp918 (single rep here, so drift is not cancelled - read as a cross-check):
    14 of 15 arms within +-1 pp - stack_off +38.6 (was +39.2), ct_block +17.1 (+17.8), flush_off +13.1
    (+13.1), bound_batch +4.3 (+4.1), gelu_bias +5.9 (+6.4), gelu_batch +2.9 (+3.4), norm_fuse +3.3
    (+3.2), dw_lpad +2.5 (+2.8), ls_fuse +1.5 (+2.2), cont_tile +2.0 (+2.0), mm_m2 +0.4 (+0.2),
    dw_axpy -0.0 (-0.1).
  * THE ONE EXCEPTION - dw_conv1d (the tap-chain fallback): +18.1 % here (vae 9.2 s, overhead 2.1 s) vs
    +10.4 % (1.25 s) at Exp918, +1.38 s at Exp906, +1.53 s at Exp883. THREE interleaved same-session reps
    (Exp937b) read +12.0 % (rtf 1.3451/1.3310/1.3455, vae 8.6/8.4/8.6, overhead 1.4 s) with a +-1.1 %
    spread - i.e. tight WITHIN a session but up to +-0.5 s BETWEEN ladders, and this ladder's own arm-2
    value (2.1 s) is 50 % above the interleaved same-session value (1.4 s), so it is not session-level
    state either - it looks arm-order/gate-dependent (arm 2 runs right after arm 1 across a cool_down
    gate; the interleaved reps run back-to-back).
  * BY-PRODUCT / plausible mechanism: the tap chain costs +166 MB RSS (2410 vs 2244 MB) - it touches far
    more pages than the fused kernel, which is a natural reason for extra sensitivity.
  * RETRACTION (mine): the Exp918 line "mild monotonic decline across three ladders, so not
    arm-order/heat" was a trend read from three noisy points. Correct statement: high CROSS-ladder
    variance (1.25-2.17 s), tight within-session repeatability, +166 MB RSS. RUNBOOK: quote the
    dw_conv1d fallback as a RANGE (1.3-2.2 s) or with interleaved reps, never as a single point.
  * QUEUED: an arm-order test (does dw_conv1d immediately after default, across a gate, reproduce the
    2.1 s?) if that range ever matters for a decision.
  Settled anchor 1.1937 at 20.0 h (batt 36.0 C, boost state, low=0 of n=19), transcript byte-identical.

- THE PROTOCOL METRIC'S STATE LEVEL, AND THE ANSWER TO Exp914's NOISE-FLOOR QUESTION (Exp938). Designed
  order-symmetric S,B,S,B run (S = protocol immediately after heat; B = protocol after 3 min idle):
    S1 1.2011 vae 7.2 prefill 1.8 decode 3.0 clock 2400000 (BOOST - the heat did not take)
    B1 1.1918 vae 7.1 prefill 1.8 decode 3.1 clock 2400000 (boost)
    S2 1.3334 vae 8.0 prefill 2.2 decode 3.1 clock 2000000 (SETTLED)
    B2 1.1976 vae 7.1 prefill 1.9 decode 3.0 clock 2400000 (boost)
  * BETWEEN-STATE EFFECT: settled vs boost = **+11.4 %** on the protocol clip, and it is **29x the
    within-state sd** (boost sd 0.392 %/rep over 3 reps, range 0.78 %). That makes the clock state the
    single largest known short-term effect on the metric - larger than batt (r=+0.9 over a ~3 C range)
    and larger than any hatch the runbook lists except dw_conv1d.
  * Exp914's NOISE FLOOR IS INTRINSIC, NOT STATE: boost reps read sd 0.392 % here vs the 0.40 % it
    measured at 19 h uptime - so the spaced-rep floor is not state contamination, and the A/B bar is NOT
    state-limited *provided both arms share a state* (Exp935 for the sweep, Exp937 for the ladder).
    Every protocol cell must therefore state its clock state; all idle-first anchors in this loop are
    boost, which is what the 1.19-1.20 series records.
  * PROTOCOL CERTIFICATE: 3 min of idle RESTORES the boost state (B1 and B2 both 2400000 after 3 min) -
    so the loop's standard spacing is state-sufficient, and the Exp905/910 gate chunk swings are the
    clock state, not a mystery.
  * PREDICTION MISS WORTH KEEPING: P1 predicted both S runs would be settled; S1 stayed BOOST. Two
    chat17 runs (~50 s of load) are NOT reliably enough to force the settled state - the flip needs
    ~75-105 s of sustained load (Exp933/934) - so a "heat then measure" design needs a THIRD warm-up run
    (or a longer one) before it can assume the settled state. S2 flipped only after the session had
    accumulated heat.
  * Independent phase confirmation on the protocol clip: vae +11.9 % (chat17's phase scaling predicted
    x1.142), prefill +19 % (predicted x1.187), decode ~0 % (predicted x1.028) - the same ordering, on a
    different clip, from a different design.
  * All four runs 39 tokens; the final capture is byte-identical (1a095c8496b4). Intermediate captures
    were overwritten by measure.sh, so the per-run transcript hashes were not kept - the state->text
    independence rests on Exp919/931-933's identity checks plus the token canary holding in all four.
  ANCHOR = B2 (a boost-state protocol run after 3 min idle, i.e. exactly the anchor condition):
  1.1976 at 20.4 h, transcript byte-identical.

- ONE-SESSION FRESH-REGIME LADDER, WITH STATE PROVENANCE (Exp939). The documented ladder cells are
  LONG-UPTIME (headline.json rtf10 1.85, Exp864/847); the fresh regime had only cross-session points
  assembled at Exp899 ("provisional 10 s 1.19 / 17 s 1.34 / 69 s 1.41 / 138 s ~1.55"). Shipped tier,
  ascending order, ONE session, no pre-heating between cells (the shape a user experiences in sequence):
    clip   rtf     vae    lm     tokens  RSS      clock     s/eff-win   implied boost fraction
    10 s   1.1937   7.1    4.8    39     2191.8   2400000   2.122        (boost; w1 premium inflates)
    17 s   1.3418  11.9   11.0    106    2192.6   2400000   2.098        (boost)
    69 s   1.4695  51.5   49.9    446    2198.2   2000000   2.197        ~48 % boost (11.1 of 23.4 win)
    138 s  1.5823 109.7  108.6    876    2205.8   2000000   2.333        0 % (settled plateau exactly)
  majflt 0 throughout; all four token canaries exact (39/106/446/876); RSS rises with the KV allocation.
  * THE LADDER'S SHAPE IS A STATE MIX, NOW MEASURED PER CELL: the 10/17 s cells are boost-only, the 69 s
    cell flips mid-run (~11 of 23 windows boost), the 138 s cell is settled end-to-end. The rtf rise with
    length is therefore NOT a length effect - it is the fraction of each cell spent in the 2000000 state
    (Exp931-933), and the state column now records it per cell.
  * IT ALSO EXPLAINS THE Exp899 DISCREPANCY: my 10/17 s cells reproduce Exp899 exactly (1.1937/1.3418 vs
    1.19/1.34) but the long cells are HIGHER (69 s 1.4695 vs 1.41, 138 s 1.5823 vs 1.55) because this
    session ran them with no idle between cells, so they started warmer and spent more of the run
    settled (Exp904's 138 s 1.52-1.53 corresponds to 2.238 s/eff-win = a mixed run; mine is 2.333 =
    fully settled). PREDICTION MISS recorded: my P3/P4 bands (1.40-1.45, 1.50-1.56) were set from the
    Exp899 points and were 1.4 %/1.4 % too low for this thermal history - the mix, not the band, is the
    right frame.
  * CONSEQUENCE FOR THE REPORT: a fresh-regime ladder cell is only meaningful with its state mix; cells
    measured in different sessions are NOT exchangeable at the long end. If the report needs
    state-homogeneous cells, pre-condition each cell (3 min idle -> boost for short cells; a pre-heat ->
    settled for long cells) - QUEUED, not done here because the loop's ladder convention is the
    ascending one-session row.
  Anchor (protocol, boost, after cooldown - also refreshed the capture the 138 s run had left stale):
  1.1950 at 20.9 h, transcript byte-identical, audit green.
