# Ideas backlog (phone RTF, CPU-only)

- Persistent VAE graph: pieces are constant-size (6400 samples) — rebuilds the
  ~1700-node graph 26x/window. Hoist to build-once + input swap. Unknown gain
  (build is likely ms vs seconds of compute); profile first.
- `n_batch` sweep for LM prefill/decode (currently 512) + VAE-side batching.
- Arena base 512 MB -> sized per piece class (RAM win ~0.4 GB; maybe locality win).
- `-t 6` pinned (F0 mask is cpus 4-7; try mask covering 2 little + 4 big variants).
- Compiler flags: `-mcpu=cortex-a78` (vs baseline), LTO on/off, `-ffast-math`
  (accuracy-gated! changes FP associativity — WER-gate mandatory).
- OpenMP env: `OMP_PROC_BIND`, `OMP_NUM_THREADS`, `KMP_AFFINITY` — runtime already
  linked; thread placement may beat taskset granularity.
- Fused per-block kernels (norm+conv+scale+res+GELU+FFN): the big kernel project.
  Only after exhausting the above; needs WER gating (fusion changes FP order).
- FP16 VAE compute: needs accuracy validation; moderate refactor.
- imatrix-guided Q3/Q2 LM (needs importance matrix collection first).
- Eliminate 1.18x window overlap via lookahead-delayed emission (protocol change;
  affects streaming latency semantics — needs care, small gain).
- Mali GPU offload: measured ceiling ~RTF 2 best case + weeks + thermal risk.
  Parked unless CPU track stalls above RTF ~6.
