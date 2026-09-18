#!/usr/bin/env bash
# Exp870: re-fit the cost model on the CURRENT stack and re-close the latency tree.
# Why: every "what would that shape buy?" estimate in this loop (and the one I sent the Jetson loop for
# training-side pricing) rests on Exp828's fit, vae_s = 0.21 + 3.45 x windows, which was measured BEFORE the
# final-window flush (v4.6), the boundary-token batch (v4.7) and the blocked transpose (v4.8). A cost model
# does not stay valid across graph changes - the ledger says so three times about profiles.
# Method: measure the four ladder clips in ONE session, take the window count from the binary's own banner
# (stderr "Windows: N (window=83200 hop=70400)"), and fit components against it. Uses vae_s, not rtf, for
# the VAE half: vae_s is LM-independent (Exp674/682), so a length change cannot contaminate it via tokens.
set -u
cd "$(dirname "$0")/.."
LOG=.auto/cost_fit.txt
: > "$LOG"
for clip in stream_10s_24k.wav chat17.wav chat69.wav chat138.wav; do
    out=$(AUDIO=$clip ./.auto/measure.sh --skip-build 2>&1)
    n=$(grep -aoE "Windows: [0-9]+" .auto/last_err.txt | tail -1 | awk '{print $2}')
    v=$(echo "$out" | grep -aoE "METRIC vae_s=[0-9.]+" | cut -d= -f2)
    pf=$(echo "$out" | grep -aoE "METRIC prefill_s=[0-9.]+" | cut -d= -f2)
    dc=$(echo "$out" | grep -aoE "METRIC decode_s=[0-9.]+" | cut -d= -f2)
    tk=$(echo "$out" | grep -aoE "METRIC tokens=[0-9]+" | cut -d= -f2)
    rt=$(echo "$out" | grep -aoE "METRIC rtf=[0-9.]+" | cut -d= -f2)
    echo "clip=$clip windows=${n:-?} vae=$v prefill=$pf decode=$dc tokens=$tk rtf=$rt" | tee -a "$LOG"
done
python3 - <<'PY'
rows = [dict(kv.split('=') for kv in l.split()) for l in open('.auto/cost_fit.txt')]
print('\n--- fit: VAE seconds vs windows (the flush makes the LAST window partial, so expect a small bias) ---')
n = [int(r['windows']) for r in rows]
v = [float(r['vae']) for r in rows]
mx, my = sum(n)/len(n), sum(v)/len(v)
b = sum((a-mx)*(c-my) for a, c in zip(n, v))/sum((a-mx)**2 for a in n)
a0 = my - b*mx
resid = [c - (a0 + b*t) for t, c in zip(n, v)]
print(f'  vae_s = {a0:+.2f} + {b:.3f} x windows   (residuals {" ".join(f"{r:+.2f}" for r in resid)})')
print('  per-window cost by clip: ' + '  '.join(
    f"{int(r['windows'])}w:{float(r['vae'])/int(r['windows']):.3f}s" for r in rows))
print('\n--- tree closure: do the components sum to the measured generation time? ---')
for r in rows:
    s = float(r['vae']) + float(r['prefill']) + float(r['decode'])
    wall = float(r['rtf']) * {'stream_10s_24k.wav': 10, 'chat17.wav': 17, 'chat69.wav': 69, 'chat138.wav': 138}[r['clip']]
    print(f"  {r['clip']:20s} vae+pre+dec = {s:7.2f} vs rtf x audio = {wall:7.2f}  ({(s-wall)/wall*100:+.1f}%)")
print('\n--- rows/sec converter (what a training-side shape change is worth HERE) ---')
sec = {'stream_10s_24k.wav': 10, 'chat17.wav': 17, 'chat69.wav': 69, 'chat138.wav': 138}
per = [float(r['vae'])/sec[r['clip']] for r in rows]
pre = [float(r['prefill'])/sec[r['clip']] for r in rows]
print(f'  VAE seconds per audio second: {" ".join(f"{x:.3f}" for x in per)}  (mean {sum(per)/4:.3f})')
print(f'  prefill seconds per audio second: {" ".join(f"{x:.3f}" for x in pre)}')
print(f'  rows fed per audio second today = 26/2.933 = {26/2.9333:.2f}; VAE cost per row = '
      f'{(sum(per)/4)/(26/2.9333)*1000:.1f} ms, prefill cost per row = {(sum(pre)/4)/(26/2.9333)*1000:.1f} ms')
print('  => halving encoder rows/sec would remove roughly half of (VAE + prefill) per audio second,')
print('     but the LM decode term is per TOKEN, so it does not shrink with rows - that is the bound.')
PY
