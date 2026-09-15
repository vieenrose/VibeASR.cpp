- CONV-INT8: BUILT AND NOT WORKING, INACTIVE BY DEFAULT (Exp685). Everything the design needed is in
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
  OPEN NOW: (1) elementwise attribution is stale again (5th time) - re-derive VAE_ABL_*; gelu should now
  read ~6-8 % and rms_norm (4.1 %) is the top item, and rms_norm_scaled already exists at block level;
  (2) lean 40-utt mean carries a scaled value marked ° - re-run it (~40 min) when convenient; (3) the
  hatches were re-audited for THIS stack (Exp677-style) never since Exp710 - costs of DW_CONV1D_OFF /
  LS_FUSE_OFF / GELU_BIAS_OFF / GGML_GELU_BATCH_OFF / M2_OFF at v4.3 are unmeasured.
