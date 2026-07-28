# Full-LiteRT decoder for VibeVoice-ASR — measured status

Goal: run the whole model on LiteRT so Android needs no ggml. Everything below is
measured on a **Boox Tab Mini C** (Snapdragon 662, Cortex-A73, ARMv8.0 — no
dotprod), batch 1, real weights from the shipped `vibeasr-lm-i2_s-embed-q6_k.gguf`.

## The comparison baseline

ggml (`asr_infer`) decodes at **141 ms/token** measured end to end. Per token it
moves 328 MB of I2_S layer weights plus 467 MB of F16 LM head = **795 MB**, so this
device sustains about **5.6 GB/s**. That figure matters: it is the real ceiling, and
for a long time I mistook my own kernel's 2.64 GB/s for the hardware limit.

## Where this port is

| stage | bytes/token | measured | GB/s |
|---|---|---|---|
| 28 decoder layers | 328 MB | ~125-140 ms | 2.3-2.6 |
| LM head, **int8 per-channel** (from a 1/8 slice) | 233 MB | **~28 ms** | 8.3 |
| **total** | **561 MB** | **~153-168 ms** | |
| ggml, same work | 795 MB | **141 ms** | 5.6 |

**~1.9x slower than ggml.** Earlier notes in this repo claimed 1.12x; that compared
the layer graph alone against ggml's total, because the LM head is a separate
467 MB F16 tensor (`output.weight` is NOT tied to `token_embd.weight` here) and the
graph stopped before it.

## Efficiency, which is the actionable part

| kernel | GB/s | share of the 5.6 GB/s this device sustains |
|---|---|---|
| `ternary_gemm` (ours), small shapes | 1.9 | 34% |
| `ternary_gemm` (ours), MLP shapes, 4 threads | 4.6-5.8 | 82-100% |
| XNNPACK f32 matmul | 4.29 | 77% |
| ggml I2_S | ~5.6 | ~100% |

So the ternary kernel has roughly **2x of headroom**. "The MLP is already at
bandwidth" was wrong — it is at *our kernel's* bandwidth.

## Route to parity

1. ~~Kernel efficiency~~ **DONE for the large shapes.** Four-row blocking took the
   MLP projections from 1.43 to 5.84 GB/s (bit-exact), and the 28 layers from ~158
   to ~125-140 ms. What remains slow is the SMALL shapes: the fused qkv (786 KB) and
   o (590 KB) sit at ~1.9 GB/s and do not benefit from threads at all — verified by
   lowering the parallelism gate, which made the whole decoder *slower*
   (124.8 -> 141.5 -> 160.6 ms at gates of 1024 / 256 / 64 KB).
2. ~~Quantize the LM head to int8~~ **DONE, and it beat the projection.** A 1/8
   slice measures 3.53 ms, so the full head is ~28 ms — not the ~54 ms the byte
   count predicted. XNNPACK's int8 matmul runs it at ~8.3 GB/s, well above the
   ~5.6 GB/s this device sustains on f16/ternary traffic, because int8 halves the
   bytes AND uses a faster kernel. Accuracy: cosine 0.999703 vs the f32 head.
   Weights bake in as constants (29.4 MB for the slice, ~233 MB full) rather than
   needing runtime binding, since no custom op is involved.

Total now ~153-168 ms against ggml's 141: about **1.1x**. The remaining gap is the
small projections (fused qkv 786 KB, o 590 KB) which sit at ~1.9 GB/s and gain
nothing from threading.

## End-to-end generation works — and the whole is 2.3x the sum of its parts

`tests/generate.cc` runs the full pipeline: host Q6_K embedding lookup -> 28-layer
graph -> int8 head -> argmax -> next token, with the KV cache aliased so state
carries across steps. It generates real tokens: different prompts give different,
evolving outputs (785 -> 198 -> 151643 -> 198 alternates; 9707 -> 151645 settles on
EOS, which is expected for a model whose prompt format wants audio embeddings).

**2.7 tok/s, ~366 ms/token.** But benchmarked separately the same graphs cost
~125-140 ms (layers) + ~28 ms (head) = ~155-170 ms. Inside the loop they cost
312 ms and 53 ms.

Six hypotheses tested and rejected:

| hypothesis | test | result |
|---|---|---|
| staging copies double the weights | free blobs after upload | no change (410 ms) |
| denormals from the softmax mask | set FPCR FZ | no change (378 ms) |
| advancing `pos` costs more than fixed | VIBEASR_FIXED_POS=3 | no change (322 vs 335) |
| thermal throttling | 2 min cooldowns | no change |
| page-cache eviction | (implied by the first two) | no change |
| memory pressure from the embedding table | table is only 191 MB | no change |

What remains, untested: the two graphs ALTERNATE, so each token streams 344 MB of
layer weights and then 233 MB of head weights. 577 MB cycling per token gives an
effective ~1.5 GB/s against the ~2.5 GB/s the layer graph achieves alone. That
smells like TLB/page-walk pressure at this working-set size, which would also
explain why it is insensitive to everything above. Not yet demonstrated.

## What is already settled

* Ternary custom op runs on the **stock** LiteRT runtime — `LiteRtAddCustomOpKernelOption`
  is exported by both prebuilts, no fork needed.
* I2_S weights lift straight out of the GGUF (layout verified against ggml,
  cosine 1.000000).
* 28-layer graph with KV cache: **cosine 0.994938** against a dense reference.
* XNNPACK still delegates 1798 of 1972 nodes around 112 custom ops.

## Lessons that cost time

* **Synthetic probes flattered this port at every scale.** A one-op graph hid that
  per-call thread spawning made small GEMMs 1.8x slower; an 8-token context hid that
  the one-hot KV scatter is O(ctx); `torch.randn` input manufactured an accuracy
  collapse (cosine 0.816) that vanished with a real embedding (0.994938).
* **Three optimizations helped for reasons other than the ones I predicted.** Fusing
  projections won on kernel-invocation count, not partition count (which did not
  change). Hoisting RoPE tables and aliasing KV buffers bought no time at all.
* **Test defects twice masqueraded as system failures** — a runner comparing one
  file against two layouts, and a pass check latched after the timing loop.
