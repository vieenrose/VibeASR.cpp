#!/usr/bin/env bash
# Exp867h: is the LONG clip a sharper instrument than the protocol clip? (the Nano loop reports 0.07%
# within-session repeatability at 240 s vs +-2% across sessions; if that transfers, long-clip bracketing
# becomes the way to resolve sub-0.2% questions, where Exp841's +-0.19% floor on the 10 s clip cannot.)
# Interleaved: one long rep, one short rep, alternating, so both see the same device state.
cd "$(dirname "$0")/.."
LOG=.auto/repeat_cal.txt; : > "$LOG"
for i in 1 2 3; do
  for clip in chat138.wav stream_10s_24k.wav; do
    out=$(AUDIO=$clip ./.auto/measure.sh --skip-build 2>&1)
    r=$(echo "$out" | grep -aoE "METRIC rtf=[0-9.]+" | head -1 | cut -d= -f2)
    t=$(echo "$out" | grep -aoE "METRIC tokens=[0-9]+" | head -1 | cut -d= -f2)
    echo "rep=$i clip=$clip rtf=$r tokens=$t" | tee -a "$LOG"
  done
done
python3 - <<'PY'
import statistics as st
rows=[l.split() for l in open('.auto/repeat_cal.txt')]
d={}
for r in rows:
    k=r[1].split('=')[1]; d.setdefault(k,[]).append(float(r[2].split('=')[1]))
print()
for k,v in d.items():
    m=st.mean(v); sd=st.stdev(v) if len(v)>1 else 0.0
    rng=(max(v)-min(v))/m*100
    print(f'{k:22s} n={len(v)} mean={m:.4f} sd={sd/m*100:.3f}% rel  range={rng:.3f}%  reps={" ".join(f"{x:.4f}" for x in v)}')
if 'stream_10s_24k.wav' in d and 'chat138.wav' in d:
    s=[x for r in rows if r[1].endswith('stream_10s_24k.wav') for x in [float(r[2].split('=')[1])]]
    l=[x for r in rows if r[1].endswith('chat138.wav') for x in [float(r[2].split('=')[1])]]
    if len(s)>1 and len(l)>1:
        print(f'  resolution ratio (sd_short / sd_long) = {st.stdev(s)/st.mean(s)/(st.stdev(l)/st.mean(l)):.2f}x'
              '  >1 means the long clip is the sharper instrument')
PY
