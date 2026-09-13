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
- OPEN (interrupted): p1-vs-p2 interleave at v3.9. Single reps suggested p1 2.4267
  beats p2 2.4644 (-1.5%, 3x sigma) with byte-identical transcripts, but order was
  fixed (p1 ran coolest) so thermal confound is open; the 4-arm interleave aborted
  after 2 arms (p2a 2.4616, p1a 2.4260 - consistent direction, still confounded).
  Needs a clean p2/p1/p1/p2 run before touching the granularity closure (p2 default)
  or the map sentence in STREAMING_1P5B.md.

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
