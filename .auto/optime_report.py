#!/usr/bin/env python3
"""Per-node op-time report (Exp818).

Parses the GGML_OP_TIME=1 dump out of a run's stderr file and prints, per phase, the ops sorted by seconds,
the sum of node time, and the RESIDUE against that phase's wall seconds. The residue is the number that
matters: op ablations structurally under-count (deleting an op does not delete the traffic its surviving
neighbours cause), so "wall minus measured ops" is the honest size of launch/barrier/allocation overhead.

Two self-checks it prints because both have bitten this loop:
  * PHASE COUNT - an absent phase here means absent from the FILE, not from the program. Never conclude from
    a tail-truncated view (Exp817: a working census looked like it counted only lm_decode).
  * node_time vs wall - if a phase reports MORE node time than its wall seconds, the instrument is broken
    (that is how a transposed [op][phase] accumulator was caught, showing 38 s in every phase).

Usage: .auto/optime_report.py [stderr-file] [vae_s lm_s_prefill lm_s_decode]
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '.auto/last_err.txt'
# CONCURRENCY: the VAE computes TWO encoder chains as two separate graphs, each with its own thread 0, so
# both threads satisfy `ith == 0` and the VAE's node time is 2x its wall seconds. The LM runs one graph with
# 2 threads and only its thread 0 counts. Getting this wrong reads as "instrument broken" - the first report
# flagged vae at 29.3 s vs 14.7 s wall, which is exactly 2 x vae_s and correct.
NCHAINS = {'vae': 2, 'lm_prefill': 1, 'lm_decode': 1}
wall = {'vae': 14.7, 'lm_prefill': 4.3, 'lm_decode': 3.6}
if len(sys.argv) > 5:
    wall = {'vae': float(sys.argv[3]), 'lm_prefill': float(sys.argv[4]), 'lm_decode': float(sys.argv[5])}

ops, tot, cur = {}, {}, None
n_lines = 0
for line in open(path, errors='ignore'):
    if 'OPTIME' not in line:
        continue
    n_lines += 1
    m = re.search(r'OPTIME phase=(\S+)\s+node_time=\s*([\d.]+) ms', line)
    if m:
        cur = m.group(1)
        tot[cur] = float(m.group(2)) / 1e3
        ops.setdefault(cur, {})
        continue
    m = re.search(r'\s+OPTIME\s+(\S+)\s+([\d.]+) ms\s+n=(\d+)', line)
    if m and cur:
        ops[cur].setdefault(m.group(1), [0.0, 0])
        ops[cur][m.group(1)][0] += float(m.group(2)) / 1e3
        ops[cur][m.group(1)][1] += int(m.group(3))

print(f"parsed {n_lines} OPTIME lines from {path}")
print(f"phases present: {sorted(tot)}   (an absent phase means absent from the FILE - check for truncation)")
for p in sorted(tot, key=lambda k: -tot[k]):
    w = wall.get(p)
    s = sum(v[0] for v in ops[p].values())
    print(f"\n== {p}: node time {tot[p]:.2f} s x{NCHAINS.get(p,1)} chain(s)   wall {w if w else '?'} s", end='')
    par = tot[p] / NCHAINS.get(p, 1)
    if w:
        flag = '  *** EXCEEDS WALL EVEN / CHAINS - INSTRUMENT BROKEN ***' if par > w * 1.02 else \
               f"   per-chain {par:.2f} s vs wall {w} s -> residue {w - s / NCHAINS.get(p, 1):+.2f} s"
        print(flag)
        print(flag)
    else:
        print('   (idle = work outside a timed phase span)')
    for op, (sec, n) in sorted(ops[p].items(), key=lambda kv: -kv[1][0])[:12]:
        per = sec / NCHAINS.get(p, 1)
        print(f"     {op:16s} {sec:7.2f} s  = {per:6.2f} s/chain  n={n:<8d} {100 * sec / tot[p]:5.1f}%")
