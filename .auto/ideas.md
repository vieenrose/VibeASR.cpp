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
