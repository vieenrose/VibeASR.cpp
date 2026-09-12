#!/usr/bin/env bash
# Paired multi-run A/B harness (restored in Exp659 - the previous copy was untracked and
# silently disappeared from .auto during an auto-revert; config/codegen sweeps from
# Exp595-607 could not be re-run for that reason. LESSON: commit harness scripts.)
#
# Runs each ARM REPS times, interleaved per rep so thermal drift hits all arms equally,
# then prints per-arm means and the delta vs the first arm.
#
# Usage: .auto/run_rtf_multi.sh REPS "label|audio|threads|pieces|extra-env" [...]
#   extra-env is prefixed to bench_device.sh, e.g. "MASK=0-7".
#   Defaults (overridable per arm): shipped files, MASK=C0 -> cpu6-7, the two A78 primes.
set -u
DEV=${DEV:-$(grep -m1 '^DEV=' .auto/measure.sh | cut -d= -f2)}   # single source: measure.sh
[ -n "$DEV" ] || { echo "ERROR: no device id (set DEV= or fix .auto/measure.sh)" >&2; exit 2; }
RDIR=/data/local/tmp/vibeasr
REPS=${1:-1}; shift
DEF_ENV="LM_FILE=lm-q8head.gguf VAE_FILE=vae-encoder-q4x4ffn.gguf MASK=C0"
TSV=.auto/multi-$$.tsv
: > "$TSV"
adb -s $DEV push .auto/bench_device.sh $RDIR/ >/dev/null 2>&1
B=$(adb -s "$DEV" shell "dumpsys battery" | grep -oE 'temperature: [0-9]+' | grep -oE '[0-9]+' | head -1 | tr -d '\r')
echo "note: batt_temp_c=$(python3 -c "print(round(${B:-0}/10,1))")"

for r in $(seq 1 "$REPS"); do
  for arm in "$@"; do
    label=${arm%%|*}; rest=${arm#*|}
    IFS='|' read -r audio threads pieces envs <<< "$rest"
    tag=$(echo "$label" | tr -cd 'A-Za-z0-9')$r
    adb -s $DEV shell "$DEF_ENV $envs sh $RDIR/bench_device.sh $audio $threads $pieces $tag" \
      > .auto/multi-run-$tag.txt 2>&1
    adb -s $DEV pull $RDIR/err-$tag.log .auto/multi-err-$tag.txt >/dev/null 2>&1
    rtf=$(grep -oE 'RTF: [0-9.]+' .auto/multi-err-$tag.txt | head -1 | awk '{print $2}')
    tok=$(grep -oiE 'tokens: [0-9]+' .auto/multi-err-$tag.txt | head -1 | awk '{print $2}')
    rss=$(grep -oE 'hwm_kb=[0-9]+' .auto/multi-run-$tag.txt | head -1 | cut -d= -f2)
    maj=$(grep -oE 'majflt_delta=-?[0-9]+' .auto/multi-run-$tag.txt | head -1 | cut -d= -f2)
    echo "$label"$'\t'"$rtf"$'\t'"$tok"$'\t'"${rss:-0}" >> "$TSV"
    echo "ARM $label | rep=$r | rtf=$rtf | tokens=$tok | rss_kb=${rss:-?} | majflt=${maj:-?}"
  done
done

echo "=== summary ==="
python3 - "$TSV" "$REPS" <<'PYEOF'
import statistics, sys
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.strip()]
order = []
for l in rows:
    if l[0] not in order:
        order.append(l[0])
base = None
print(f"{'arm':34s} {'rtf mean':>9} {'rtf all reps':>26} {'tokens':>7} {'RSS MB':>8} {'vs first':>9}")
for a in order:
    v = [float(r[1]) for r in rows if r[0] == a and r[1]]
    tok = [r[2] for r in rows if r[0] == a]
    rss = [float(r[3]) for r in rows if r[0] == a and float(r[3]) > 0]
    if not v:
        print(f"{a:34s} FAILED"); continue
    m = statistics.mean(v)
    base = base if base is not None else m
    print(f"{a[:34]:34s} {m:9.4f} {', '.join(f'{x:.3f}' for x in v)[:26]:>26} "
          f"{tok[0]:>7} {(statistics.mean(rss)/1024 if rss else 0):8.1f} {100*(m-base)/base:+8.1f}%")
PYEOF
rm -f "$TSV"
