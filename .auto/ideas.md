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
    exactly x2, all genuine talk refrains - zero runaway. Method note: screen
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
