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
