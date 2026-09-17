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
# Tier defaults come from measure.sh, NOT from a copy of them here. The previous literal
# (VAE_FILE=vae-encoder-q4x4ffn.gguf) was the pre-Exp690 F16-conv reference and silently made every
# sweep through this runner measure a tier that no longer ships - same class as the stale-lib bug
# check 5 guards against, but in the sweep harness instead of measure.sh.
DEF_LM=$(grep -m1 '^LM_FILE=' .auto/measure.sh | sed -E 's/^[^:]*=\$\{LM_FILE:-([^}]*)\}.*/\1/')
DEF_VAE=$(grep -m1 '^VAE_FILE=' .auto/measure.sh | sed -E 's/^[^:]*=\$\{VAE_FILE:-([^}]*)\}.*/\1/')
[ -n "$DEF_LM" ] && [ -n "$DEF_VAE" ] || { echo "ERROR: could not parse model defaults from .auto/measure.sh" >&2; exit 2; }
DEF_ENV="LM_FILE=$DEF_LM VAE_FILE=$DEF_VAE MASK=C0"
# Freshness guard: this runner does NOT build, so sweeping after editing src/ silently
# measures the old binary (Exp662 burned 12 runs that way). Refuse unless the binary is newer
# than every source file.
stale=$(find src demo 3rdparty/llama.cpp/ggml/src 3rdparty/llama.cpp/src -name '*.cpp' -newer build-android/bin/asr_streaming 2>/dev/null | head -3)
if [ -n "$stale" ]; then
  echo "ERROR: build-android/bin/asr_streaming is older than:" >&2; echo "$stale" >&2
  echo "Rebuild first (bash .auto/measure.sh, without --skip-build)." >&2
  exit 2
fi

TSV=.auto/multi-$$.tsv
: > "$TSV"
adb -s $DEV push .auto/bench_device.sh $RDIR/ >/dev/null 2>&1
B=$(adb -s "$DEV" shell "dumpsys battery" | grep -oE 'temperature: [0-9]+' | grep -oE '[0-9]+' | head -1 | tr -d '\r')
echo "note: batt_temp_c=$(python3 -c "print(round(${B:-0}/10,1))")"
echo "note: tier from measure.sh = $DEF_LM + $DEF_VAE"

for r in $(seq 1 "$REPS"); do
  for arm in "$@"; do
    label=${arm%%|*}; rest=${arm#*|}
    IFS='|' read -r audio threads pieces envs <<< "$rest"
    # Exp837: this field is a DEVICE-side path, so a host clip (e.g. .auto/assets/slice10b_24k.wav) used to
    # fail loudly with "Failed to open WAV file". Accept host paths by pushing them once - the guard-clip
    # rotation must be one command, or it gets skipped and the anti-overfit board goes stale.
    if [[ "$audio" == */* && -f "$audio" ]]; then
      adb -s $DEV push "$audio" "$RDIR/$(basename "$audio")" >/dev/null 2>&1 || { echo "ERROR: push failed: $audio" >&2; exit 2; }
      audio=$(basename "$audio")
    fi
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
