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

## End-to-end generation, measured back to back against ggml

Steady state (run 2 onward), both on the same device minutes apart:

| | ms/token | tok/s | peak RSS | peak RssAnon |
|---|---|---|---|---|
| **LiteRT (this port)** | **123.5-124.2** | **8.1** | 786 MB | **241 MB** |
| ggml (`asr_infer`) | 123.1-128.3 | 7.9 | 1297 MB | 507 MB |

Phase split: embed 0.0, layers ~84, head ~29, argmax 0.5 ms/token.
Accuracy: cosine 0.994938 against a dense reference; the ternary kernel is
bit-exact against its scalar reference.

So the port now matches ggml on speed and uses **half the unevictable memory**.

FIRST RUN IS SLOWER — 190-207 ms/token against 124 steady. `madvise(WILLNEED)`
recovered part of it (207 -> 190), but the weight files were already in page cache,
so most of the remainder is the `schedutil` governor ramping the cores rather than
faulting pages. It matters little for real use: transcribing a minute of audio is
thousands of tokens, all of them steady-state.

### The measurement trap that cost hours

Absolute timings on this device are **not comparable across time**. The governor is
`schedutil`, and a memory-bound workload stalls often enough to read as low
utilization, so the big cores sit at ~1050 MHz of 2016 (52%). No root, so the
governor cannot be pinned.

Chasing a 2.5x "regression" between two binaries, I tested and rejected eight
hypotheses — duplicate staging buffers, denormals, advancing `pos`, thermal, page
cache, embedding-table pressure, input-write invalidation, two-graph alternation.
All wrong. The control I should have run first was the OLD binary again: it had
also gone from 124.8 to 326.8 ms on the identical graph. There was no regression
between binaries at all.

**Rule for this device: only trust A/B measurements taken back to back.**

### What actually fixed the speed

The thread pool. `TernaryPool` handed work off through a condition variable, and a
decoder dispatches it ~84 times per token (3 MLP projections x 28 layers) for a few
milliseconds of work each. That is far too fine-grained for a futex round trip:

| threads | condvar | spin-then-yield |
|---|---|---|
| 1 | 188 ms | 187 ms |
| 2 | 468 ms | 115 ms |
| 4 | 341 ms | **79.5 ms** |

With the condvar, "parallelism" made the decoder 1.8x SLOWER than single-threaded.
Spinning (8192 `yield` instructions before falling back to `sched_yield`) makes 4
threads 2.4x faster than 1. ggml's threadpool spins for exactly this reason.

## Prefill: batching gains only 1.2x, and the reason is instructive

A 16-token prefill graph exports and runs correctly (cosine 0.986 against a dense
reference over the same 16 steps), and the kernel gained a batched path where the
weight row is outermost so each row is read once for all m activation vectors.
Bit-exact at m=1 and m=16.

  decode  (1 token)   82 ms        -> 82 ms/token
  prefill (16 tokens) 1090 ms      -> 68 ms/token

Only 1.2x. The batched kernel measured across m shows why:

  m=1    0.801 ms/call
  m=4    1.751 ms
  m=16   6.431 ms

Time scales with m at about 8x for 16x the work. If weight reads were the
bottleneck, batching would make it nearly flat. They are not: **on ARMv8.0 without
dotprod this kernel is COMPUTE-bound**, needing ~16 SIMD ops (8 unpack + 8
`vmlal_s8`) per 16 bytes of weights. Batching amortizes the reads; it cannot
amortize the arithmetic.

So prefill batching is worth having but is not the order-of-magnitude win it is on
bandwidth-bound stacks. On an ARMv8.2 device with `vdotq_s32` the unpack-multiply
halves and the balance shifts toward bandwidth, where batching should pay much
better — untested, no such device here.

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
