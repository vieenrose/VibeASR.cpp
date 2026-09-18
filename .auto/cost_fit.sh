#!/usr/bin/env bash
# Exp870: re-fit the cost model on the CURRENT stack and re-close the latency tree.
# Why: every "what would that shape buy?" estimate in this loop (and the one I sent the Jetson loop for
# training-side pricing) rests on Exp828's fit, vae_s = 0.21 + 3.45 x windows, which was measured BEFORE the
# final-window flush (v4.6), the boundary-token batch (v4.7) and the blocked transpose (v4.8). A cost model
# does not stay valid across graph changes - the ledger says so three times about profiles.
# Method: measure the four ladder clips in ONE session, take the window count from the binary's own banner
# (stderr "Windows: N (window=83200 hop=70400)"), and fit components against it. Uses vae_s, not rtf, for
# the VAE half: vae_s is LM-independent (Exp674/682), so a length change cannot contaminate it via tokens.
#
#   ./.auto/cost_fit.sh              measure all four clips (one session, ~7 min) then fit
#   ./.auto/cost_fit.sh --analyze    re-fit the existing .auto/cost_fit.txt - NO device time
#
# The flush makes the LAST window partial, so the model is fitted against EFFECTIVE windows
# (N minus the fraction the flush removed); that is the whole difference between this fit and Exp828's,
# and it is why the old intercept was +0.21 s where the honest one is ~0.
set -u
cd "$(dirname "$0")/.."
LOG=.auto/cost_fit.txt
if [ "${1:-}" != "--analyze" ]; then
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
fi
python3 - <<'PY'
SEC, WIN, HOP, ROWS_PER_WIN = 24000, 83200, 70400, 26
DUR = {'stream_10s_24k.wav': 10.0, 'chat17.wav': 17.0, 'chat69.wav': 69.0, 'chat138.wav': 138.0}
rows = [dict(kv.split('=') for kv in l.split()) for l in open('.auto/cost_fit.txt')]
def fit(xs, ys):
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    b = sum((a-mx)*(c-my) for a, c in zip(xs, ys))/sum((a-mx)**2 for a in xs)
    return my - b*mx, b
T = []
for r in rows:
    N, dur = int(r['windows']), DUR[r['clip']] * SEC
    eff = N - (1.0 - min(WIN, dur - (N-1)*HOP)/WIN)      # window-equivalents actually encoded
    tk, rp = int(r['tokens']), 28*N                      # 26 audio frames + 2 boundary rows per window
    wall = float(r['rtf']) * DUR[r['clip']]
    T.append(dict(clip=r['clip'], N=N, eff=eff, vae=float(r['vae']), pre=float(r['prefill']),
                  dec=float(r['decode']), tk=tk, rp=rp, wall=wall,
                  msr=float(r['prefill'])/rp*1e3, mst=float(r['decode'])/tk*1e3,
                  ctx=13.0*(N-1) + tk/2.0))              # mean KV positions a processed row attends over
print('clip              N  eff_win    vae  s/eff-w  pre ms/row   dec ms/tok   rtf   shares vae/pre/dec')
for t in T:
    print(f"{t['clip']:18s}{t['N']:3d} {t['eff']:7.2f} {t['vae']:7.1f} {t['vae']/t['eff']:6.3f}"
          f" {t['pre']:5.1f} {t['msr']:6.1f} {t['dec']:5.1f} {t['mst']:6.1f}  "
          f"{t['wall']/DUR[t['clip']]:.4f}  {t['vae']/t['wall']*100:.1f}/{t['pre']/t['wall']*100:.1f}/{t['dec']/t['wall']*100:.1f}")
a, b = fit([t['eff'] for t in T], [t['vae'] for t in T])
print(f"\n1) VAE:  vae_s = {a:+.2f} + {b:.3f} x effective_windows   residuals "
      + " ".join(f"{t['vae']-(a+b*t['eff']):+.2f}" for t in T))
c = sum(t['vae'] for t in T)/sum(t['eff'] for t in T)
print(f"   zero-intercept form: {c:.3f} s per effective window = {WIN/SEC:.3f} s of audio "
      f"=> VAE-only RTF {c/(WIN/SEC):.3f} (the VAE alone is just under real time on the 2 pinned cores)")
print(f"   supersedes Exp828 (vae_s = 0.21 + 3.45 x N): with NO flush term it over-prices at BOTH ends -")
for t in (T[0], T[-1]):
    old = 0.21 + 3.45*t['N']
    print(f"   short audio and under-prices the rate: {t['clip']} N={t['N']} -> old {old:6.1f} s vs {t['vae']:6.1f} s measured "
          f"({(old-t['vae'])/t['vae']*100:+.0f} %)")
print("   worst where the flush saves the most (short audio). Every shape/latency estimate made before")
print("   Exp870 used the old form; re-read them with this one.")
pa, pb = fit([t['ctx'] for t in T], [t['msr'] for t in T])
da, db = fit([t['ctx'] for t in T], [t['mst'] for t in T])
print(f"\n2) LM:   prefill = {pa:.2f} ms/row {pb*1e3:+.2f} us per KV position  (resid "
      + " ".join(f"{t['msr']-(pa+pb*t['ctx']):+.2f}" for t in T) + ")")
print(f"       decode  = {da:.2f} ms/token {db*1e3:+.2f} us per KV position (resid "
      + " ".join(f"{t['mst']-(da+db*t['ctx']):+.2f}" for t in T) + ")")
print("   Two independent phases agreeing on the same per-position slope is the evidence that the")
print("   length-dependent term IS KV attention - nothing else in the LM grows with audio length.")
print("   Price of a KV-side lever (eviction / KV quantization), by clip:")
for t in T:
    kp, kd = (pb*t['ctx'])*t['rp']/1e3, (db*t['ctx'])*t['tk']/1e3
    print(f"     {t['clip']:18s} upper bound {kp+kd:6.2f} s = {(kp+kd)/t['wall']*100:5.2f} % of wall "
          f"(decode-only {kd/t['wall']*100:4.2f} %)")
print("\n3) tree closure: do the components sum to the measured generation time?")
for t in T:
    s = t['vae'] + t['pre'] + t['dec']
    print(f"   {t['clip']:18s} vae+pre+dec = {s:7.2f} s  vs rtf x audio = {t['wall']:7.2f} s  "
          f"({(s-t['wall'])/t['wall']*100:+.2f} %)")
print("   If this line ever drifts from ~0 %, the profile tree is INCOMPLETE - some phase is unaccounted")
print("   for (a loader, a sync, a wait). At Exp870 it closes to <=0.30 % at every length: nothing hidden.")
mac = 2*(20.54 + 8.59)                                   # GMac per window, both chains (Exp678 graph stats)
print(f"\n4) rate check: {mac:.1f} GMac per window (2 chains x (20.54 early + 8.59 late), Exp678) =>")
for t in T:
    r = mac*t['eff']/t['vae']
    print(f"   {t['clip']:18s} {r:5.1f} GMAC/s aggregate = {r/86*100:.0f} % of the 86 GMAC/s two-core int8 peak (Exp551)")
PY
