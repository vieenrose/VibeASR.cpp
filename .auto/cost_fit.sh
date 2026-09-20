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
#   ./.auto/cost_fit.sh --predict <seconds> [tokens]     what the model says a clip costs (no device time)
#
# VALIDATED OUT OF SAMPLE (Exp871, predictions written to .auto/cost_pred871.txt BEFORE measuring):
# on three clips absent from the fit the VAE term came out -0.07 % / +0.13 % / -0.21 % (including a 155 s
# clip, 15 % beyond the longest fitted length) and total wall to +0.06 % when the token count is given.
# So the RATES are content-independent; the token COUNT is the one content-driven input and must be
# measured - the 6.3 tok/s density constant holds for conversational audio (976 predicted vs 978 measured
# at 155 s) and fails elsewhere (hotwords 109 predicted vs 63, guard slice 63 vs 55).
#
# The flush makes the LAST window partial, so the model is fitted against EFFECTIVE windows
# (N minus the fraction the flush removed); that is the whole difference between this fit and Exp828's,
# and it is why the old intercept was +0.21 s where the honest one is ~0.
set -u
cd "$(dirname "$0")/.."
LOG=.auto/cost_fit.txt
if [ "${1:-}" = "--predict" ]; then
    SEC_ARG=${2:?usage: cost_fit.sh --predict <audio-seconds> [tokens] [--vae-rate S] [--lm-scale F]}
    shift 2
    TOK_ARG=""
    if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then TOK_ARG=$1; shift; fi
    VAE_RATE=""; LM_SCALE=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --vae-rate) VAE_RATE=${2:?--vae-rate needs a value (s per effective window)}; shift 2;;
        --lm-scale) LM_SCALE=${2:?--lm-scale needs a factor}; shift 2;;
        *) echo "ERROR: unknown argument '$1' for --predict (valid: --vae-rate S, --lm-scale F)" >&2; exit 2;;
      esac
    done
    python3 - "$SEC_ARG" "$TOK_ARG" "$VAE_RATE" "$LM_SCALE" <<'PY'
import sys
SEC, WIN, HOP = 24000, 83200, 70400
A, B = 0.39, 3.343                    # vae_s = A + B x effective_windows   (Exp870 fit, LONG-UPTIME regime)
PM, PB = 29.82, 9.11e-3               # prefill ms/row, ms per KV position
DM, DB = 90.31, 10.67e-3              # decode  ms/token, ms per KV position
dur = float(sys.argv[1]); tok = float(sys.argv[2]) if sys.argv[2] else 6.3 * dur
vae_rate = float(sys.argv[3]) if sys.argv[3] else 0.0
lm_scale = float(sys.argv[4]) if sys.argv[4] else 1.0
N = (int(round(dur * SEC)) - 1) // HOP + 1
eff = N - (1.0 - min(WIN, dur * SEC - (N - 1) * HOP) / WIN)
vae = (vae_rate * eff) if vae_rate else (A + B * eff)
ctx = 13.0 * (N - 1) + tok / 2         # mean KV positions a processed row attends over
pre = lm_scale * 28 * N * (PM + PB * ctx) / 1e3
dec = lm_scale * tok * (DM + DB * ctx) / 1e3
print(f"dur={dur:.2f}s windows={N} effective={eff:.2f} tokens={tok:.0f}"
      f"{' (density 6.3/s - only valid for conversational audio)' if not sys.argv[2] else ''}")
print(f"  vae {vae:.1f}s  prefill {pre:.1f}s ({28*N} rows)  decode {dec:.1f}s  -> wall {vae+pre+dec:.1f}s  rtf {(vae+pre+dec)/dur:.4f}")
print(f"  shares vae/pre/dec {vae/(vae+pre+dec)*100:.1f}/{pre/(vae+pre+dec)*100:.1f}/{dec/(vae+pre+dec)*100:.1f} %")
if vae_rate or sys.argv[4]:
    print(f"  regime: USER-OVERRIDDEN constants (vae_rate={vae_rate if vae_rate else 'fitted'}, lm_scale={lm_scale:g})")
    print("    fresh-regime reference (Exp888/891): VAE 2.10 s per effective window at uptime < ~24 h.")
else:
    print("  regime: LONG-UPTIME (Exp870 fit, calibrated PRE-REBOOT; residuals vae +-0.2 %, wall +-0.1 %")
    print("    given tokens, Exp871). WARNING (Exp916): a fresh boot runs the same clip ~1.5x faster, so these")
    print("    defaults over-predict it (250.8 s clip: VAE +49 %, wall +45 %). Fresh-regime VAE rate is")
    print("    2.10 s per effective window (Exp888/891) - pass --vae-rate 2.10 for fresh pricing; the LM legs")
    print("    need measured phase scalars (fresh protocol clip: prefill 1.8 s, decode 3.0 s).")
print("  CAVEAT the model is a RATE model: it prices work, not behaviour. A change that alters what the LM")
print("  emits (a precision, a decode policy) changes tok, which this line takes as an input.")
PY
    exit 0
fi
if [ "${1:-}" = "--selftest" ]; then
    # Exp660 rule: prove the knobs FIRE (an output change), not merely that the flags are accepted, and
    # prove a bad argument is REFUSED (the Exp877 vacuous-run class, in a pricing tool).
    rc=0
    def=$(./.auto/cost_fit.sh --predict 100 200 | awk '{for(i=1;i<=NF;i++) if($i=="vae"){print $(i+1); exit}}' | tr -d 's')
    knb=$(./.auto/cost_fit.sh --predict 100 200 --vae-rate 2.0 | awk '{for(i=1;i<=NF;i++) if($i=="vae"){print $(i+1); exit}}' | tr -d 's')
    pd=$(./.auto/cost_fit.sh --predict 100 200 | awk '{for(i=1;i<=NF;i++) if($i=="prefill"){print $(i+1); exit}}' | tr -d 's')
    ph=$(./.auto/cost_fit.sh --predict 100 200 --lm-scale 0.5 | awk '{for(i=1;i<=NF;i++) if($i=="prefill"){print $(i+1); exit}}' | tr -d 's')
    ./.auto/cost_fit.sh --predict 100 200 | grep -q '^  regime: LONG-UPTIME' || { echo "FAIL: default has no regime line"; rc=1; }
    ./.auto/cost_fit.sh --predict 100 200 --vae-rate 2.0 | grep -q '^  regime: USER-OVERRIDDEN' || { echo "FAIL: override has no regime line"; rc=1; }
    python3 - "$def" "$knb" "$pd" "$ph" <<'PY' || rc=1
import sys
d, k, pd, ph = (float(x) for x in sys.argv[1:5])
N = (2400000 - 1) // 70400 + 1
eff = N - (1.0 - min(83200, 2400000 - (N - 1) * 70400) / 83200)
exp_k = 2.0 * eff
ok = True
if abs(k - exp_k) > 0.11:    print(f"FAIL: --vae-rate did not fire: {k} != {exp_k:.2f}"); ok = False
if abs(d - k) < 1.0:         print("FAIL: --vae-rate inert (default == override)"); ok = False
if abs(ph - pd / 2) > 0.11:  print(f"FAIL: --lm-scale did not fire: {ph} != {pd/2:.2f}"); ok = False
sys.exit(0 if ok else 1)
PY
    if ./.auto/cost_fit.sh --predict 100 200 --bogus >/dev/null 2>&1; then echo "FAIL: unknown flag accepted"; rc=1; fi
    if [ "$rc" = 0 ]; then echo "cost_fit --selftest: OK (regime lines present, both knobs fire, unknown flag refused)"; else echo "cost_fit --selftest: FAILED"; fi
    exit "$rc"
fi
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
