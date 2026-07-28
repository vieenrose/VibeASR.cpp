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
| LM head (F16, projected from a 1/8 f32 slice) | 467 MB | ~109 ms | 4.3 |
| **total** | **795 MB** | **~235-250 ms** | |
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
2. **Quantize the LM head to int8** (467 -> 233 MB), worth ~55 ms. Changes numerics
   where ggml keeps F16, so it needs an accuracy gate.

Both together give ~144 ms, i.e. parity. Either alone does not.

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
