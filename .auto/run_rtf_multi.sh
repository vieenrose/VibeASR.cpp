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
DEF_LM=$(grep -m1 '^LM_FILE=' .auto/measure.sh | sed -E 's/^[^:]*=\$\{LM_FILE:-([^}]*)\}.*/\1/')
DEF_VAE=$(grep -m1 '^VAE_FILE=' .auto/measure.sh | sed -E 's/^[^:]*=\$\{VAE_FILE:-([^}]*)\}.*/\1/')
# Exp865c: threads/pieces defaults come from the tier declaration - never a literal, never empty.
DEF_THREADS=$(grep -m1 '^THREADS=' .auto/measure.sh | sed -E 's/^[^:]*=\$\{THREADS:-([^}]*)\}.*/\1/'); DEF_THREADS=${DEF_THREADS:-2}
DEF_PIECES=$(grep -m1 '^PIECES=' .auto/tier.env 2>/dev/null | cut -d= -f2); DEF_PIECES=${DEF_PIECES:-1}
# Exp865c: --dry prints each arm's RESOLVED command line and runs nothing, so the audit can assert that the
# sweep targets the shipping tier. Empty threads/pieces fields used to expand into NOTHING, shifting
# bench_device.sh's positionals (the run tag landed in the thread-count argument) - eight guard rotations
# measured a config that is not the shipping one. Same class as Exp694/Exp816.
if [ "${1:-}" = "--dry" ]; then
  shift
  for arm in "$@"; do
    label=${arm%%|*}; rest=${arm#*|}
    IFS='|' read -r audio threads pieces envs <<< "$rest"
    threads=${threads:-$DEF_THREADS}; pieces=${pieces:-$DEF_PIECES}
    case "$threads" in (*[!0-9]*|'') echo "ERROR: arm '$arm' threads='$threads' is not a number" >&2; exit 2;; esac
    case "$pieces"  in (1|2|13|26) ;; *) echo "ERROR: arm '$arm' pieces='$pieces' must be 1, 2, 13 or 26" >&2; exit 2;; esac
    echo "DRY $label | vae=$DEF_VAE lm=$DEF_LM mask=C0 threads=$threads pieces=$pieces env=$envs clip=$audio"
  done
  exit 0
fi

REPS=${1:-1}
# Exp877: REPS used to be `${1:-1}` with no validation, so calling the tool as
#   run_rtf_multi.sh "a|x.wav" "b|y.wav"        (arms first, no REPS)
# made REPS="a|x.wav", `seq` failed, the loop ran ZERO times, and the script still printed a summary
# header and exited 0 - a vacuous run in 0.2 s (the Exp863 class, and 0.2 s is arithmetically impossible).
case "$REPS" in
  ''|*[!0-9]*) echo "ERROR: first argument must be REPS (a positive integer), got '$REPS'. Usage: $0 REPS \"label|audio|threads|pieces|env\" [...]" >&2; exit 2;;
esac
if [ "$REPS" -lt 1 ]; then echo "ERROR: REPS must be >= 1, got $REPS" >&2; exit 2; fi
shift
if [ "$#" -lt 1 ]; then echo "ERROR: no arms given (usage: $0 REPS \"label|audio|threads|pieces|env\" [...])" >&2; exit 2; fi

# Exp879: the sweep does not pull out-loop.log, so .auto/last_out.txt KEEPS whatever an earlier run left
# there. A protocol hash taken right after a sweep therefore described a different clip (it happened for
# real: 86 window lines from a 250 s run, read as a protocol-output change). Move it aside so the mistake
# fails loudly instead of silently.
# NB it sits AFTER the argument validation on purpose - Exp879b: when it was first, audit check 12's own
# "REPS with no arms" misuse probe moved the protocol capture aside before erroring out.
if [ -f .auto/last_out.txt ]; then mv .auto/last_out.txt .auto/last_out.prev; fi
# Tier defaults come from measure.sh, NOT from a copy of them here. The previous literal
# (VAE_FILE=vae-encoder-q4x4ffn.gguf) was the pre-Exp690 F16-conv reference and silently made every
# sweep through this runner measure a tier that no longer ships - same class as the stale-lib bug
# check 5 guards against, but in the sweep harness instead of measure.sh.
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
# Exp880: the pre-flight the tool always lacked. Tonight an adb outage left EVERY arm with empty
# rtf and the sweep still printed a summary and exited 0 - the Exp877 vacuous-run class, in the
# sweep instead of the runner. Also check for a co-runner first: against a live second asr_streaming
# every arm would read ~2x slow and look like a real (terrible) result, the Exp679 class.
LIVE=$( { adb -s $DEV shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -c asr_streaming; } || true )
if [ "${LIVE:-0}" -gt 0 ] 2>/dev/null; then
  echo "WARNING: $LIVE asr_streaming already running on device - kill it first; arms run now are PROVISIONAL" >&2
fi
adb -s $DEV get-state >/dev/null 2>&1 || { echo "ERROR: device $DEV unreachable - refusing to emit a vacuous sweep" >&2; exit 2; }
B=$(adb -s "$DEV" shell "dumpsys battery" | grep -oE 'temperature: [0-9]+' | grep -oE '[0-9]+' | head -1 | tr -d '\r')
echo "note: batt_temp_c=$(python3 -c "print(round(${B:-0}/10,1))")"
# Exp844: device STATE next to every measurement. Exp841/843 showed the same binary spans ~1 % of RTF across
# boost/thermal states hours apart, which makes cross-session comparisons meaningless unless the state is
# recorded. thermal_zone* is not readable on this device, so the observable is the big-core clock.
cpufreq() { adb -s "$DEV" shell "cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq" 2>/dev/null | tr -d '\r'; }
echo "note: cpu7_khz_start=$(cpufreq)"
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
    threads=${threads:-$DEF_THREADS}
    pieces=${pieces:-$DEF_PIECES}
    tag=$(echo "$label" | tr -cd 'A-Za-z0-9')$r
    for f in "$audio" "$threads" "$pieces" "$tag"; do
      [ -n "$f" ] || { echo "ERROR: empty positional in arm '$arm' (label|audio|threads|pieces|env)" >&2; exit 2; }
    done
    case "$threads" in (*[!0-9]*|'') echo "ERROR: arm '$arm' threads='$threads' is not a number" >&2; exit 2;; esac
    case "$pieces"  in (1|2|13|26) ;; *) echo "ERROR: arm '$arm' pieces='$pieces' must be 1,2,13,26 (env = field 5)" >&2; exit 2;; esac
    # Exp902 heat pacing: the 4th consecutive sweep arm heat-trips (Exp897/901: +8-9 % step while
    # arms 1-3 stay clean). Short arms are individually fine; the accumulation is not. Gate each arm
    # on battery temperature: above 38.0 C wait 60 s and re-check (max 3 waits), then run anyway with
    # a HOT flag so a heat-contaminated arm is visible, not silent. Light touch by design (the
    # rollback ladder's 37.0 C gate would stall a quick A/B); retune if the 4th-arm step survives it.
    HOT=""
    for ((w = 1; w <= 3; w++)); do
      BT=$(adb -s $DEV shell "dumpsys battery 2>/dev/null | grep 'temperature:'" 2>/dev/null | grep -oE '[0-9]+' | head -1 | tr -d '\r')
      if [ -z "${BT:-}" ] || [ "$BT" -le 380 ] 2>/dev/null; then break; fi
      echo "note: arm $label waiting on heat (batt ${BT})" >&2
      sleep 60
      if [ "$w" = 3 ]; then HOT=" HOT"; fi
    done
    # Exp987: ARM THE SWEEP. This tool - the loop's primary A/B runner, used for all 21 guard rotations -
    # never armed, so its arms ran in the UNARMED device state (Pproto 1.85 vs the armed protocol's
    # 1.25). Ratios survive (every arm shared the state), which is why the rotations stayed comparable,
    # but absolute ARM-line numbers were not. The same Exp979 recipe as measure.sh now spans each arm:
    # burst to ARM, then a KEYCODE_WAKEUP stream to HOLD, stopped after the arm's device run.
    # NO_ARM=1 restores the old unarmed behaviour. The witness printed below is read from the SHARED
    # .auto/deliv_share.py (all three numbers), so the two harnesses cannot drift apart (Exp869 class).
    A_SENT=""
    if [ "${NO_ARM:-0}" != 1 ]; then
      adb -s $DEV shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
      for _i in 1 2 3; do
        adb -s $DEV shell "input keyevent KEYCODE_VOLUME_DOWN" >/dev/null 2>&1 || true
        adb -s $DEV shell "input keyevent KEYCODE_VOLUME_UP" >/dev/null 2>&1 || true
        sleep 1
      done
      A_SENT=$(mktemp /tmp/asr_sweep_arm.XXXXXX 2>/dev/null || echo "")
      if [ -n "$A_SENT" ]; then
        ( _arm_t0=$(date +%s); while [ -f "$A_SENT" ]; do
            adb -s $DEV shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
            # Exp993: adaptive period - dense for the first ARM_DENSE_S seconds (what a SHORT run needs),
            # then ARM_SPARSE if set (cheaper for long runs: each injected event costs ~44 ms of device CPU,
            # and the 69 s cell holds boost at 10 s spacing - Exp988). Unset ARM_SPARSE = dense throughout.
            if [ -n "${ARM_SPARSE:-}" ] && [ $(( $(date +%s) - ${_arm_t0:-0} )) -ge ${ARM_DENSE_S:-20} ]; then
              sleep "$ARM_SPARSE"
            else
              sleep ${ARM_PERIOD:-3}
            fi
          done ) >/dev/null 2>&1 &
        A_PID=$!
        sleep 1
      fi
    fi
    # Exp947: per-arm DELIVERED state (Exp942's column, extracted into .auto/deliv_share.py). The
    # request-based clock fields below come from bench_device.sh's own sampling; this is the
    # request-independent one, and it is what certifies that every arm ran in the same state.
    adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" > /tmp/tis0.$$ 2>/dev/null || true
    adb -s $DEV shell "$DEF_ENV $envs THREADS=$threads sh $RDIR/bench_device.sh $audio $threads $pieces $tag" \
      > .auto/multi-run-$tag.txt 2>&1
    adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" > /tmp/tis1.$$ 2>/dev/null || true
    if [ -n "${A_SENT:-}" ]; then rm -f "$A_SENT"; A_SENT=""; fi
    if [ -n "${A_PID:-}" ]; then wait "$A_PID" 2>/dev/null || true; A_PID=""; fi
    _w=$( { python3 .auto/deliv_share.py /tmp/tis0.$$ /tmp/tis1.$$ 2>/dev/null; } || true )
    dv=$(printf '%s' "${_w:-}" | awk '{print $1}')
    mz=$(printf '%s' "${_w:-}" | awk '{print $2}')
    g2=$(printf '%s' "${_w:-}" | awk '{print $3}')
    rm -f /tmp/tis0.$$ /tmp/tis1.$$
    adb -s $DEV pull $RDIR/err-$tag.log .auto/multi-err-$tag.txt >/dev/null 2>&1
    # Exp919: also pull the arm's STDOUT (where the [n/m] window lines are). Until now the transcript
    # identity had to be pulled by hand after the sweep (Exp909), because bench_device.sh sends the app's
    # stdout to out-<tag>.log and the runner only fetched err-<tag>.log. Identity is the guard board's
    # contract, so the tool now measures it. Control: on the Exp919 sweep the P arms must read the
    # protocol hash and the G arms a common different hash, matching Exp909's manual pulls.
    adb -s $DEV pull $RDIR/out-$tag.log .auto/multi-out-$tag.txt >/dev/null 2>&1
    # Exp919b: hash the WHOLE pulled stdout file, matching the loop's canonical protocol hash
    # (1a095c8496b4 = md5 of the 4-window capture). The first version hashed only the '^[n/m]' lines,
    # which is a different (also stable) oracle - the known-value control caught the mismatch.
    tx=$(md5sum .auto/multi-out-$tag.txt 2>/dev/null | cut -c1-12 || true)
    # Exp935: report the arm's DURING-RUN clock profile, not this script's own post-arm sample (which reads
    # the idle governor target, 1430000, and cannot see the 2400000 -> 2000000 step of Exp931-933).
    kmn=$(grep -oE 'cpu7_khz_min=[0-9?]+' .auto/multi-run-$tag.txt 2>/dev/null | cut -d= -f2 || true)
    kmd=$(grep -oE 'cpu7_khz_med=[0-9?]+' .auto/multi-run-$tag.txt 2>/dev/null | cut -d= -f2 || true)
    kmx=$(grep -oE 'cpu7_khz_max=[0-9?]+' .auto/multi-run-$tag.txt 2>/dev/null | cut -d= -f2 || true)
    rtf=$(grep -oE 'RTF: [0-9.]+' .auto/multi-err-$tag.txt | head -1 | awk '{print $2}')
    tok=$(grep -oiE 'tokens: [0-9]+' .auto/multi-err-$tag.txt | head -1 | awk '{print $2}')
    rss=$(grep -oE 'hwm_kb=[0-9]+' .auto/multi-run-$tag.txt | head -1 | cut -d= -f2)
    maj=$(grep -oE 'majflt_delta=-?[0-9]+' .auto/multi-run-$tag.txt | head -1 | cut -d= -f2)
    echo "$label"$'\t'"$rtf"$'\t'"$tok"$'\t'"${rss:-0}"$'\t'"${tx:-none}"$'\t'"${kmd:-?}"$'\t'"${dv:-?}" >> "$TSV"
    echo "ARM $label | rep=$r | rtf=$rtf | tokens=$tok | rss_kb=${rss:-?} | majflt=${maj:-?} | tx=${tx:-none} | clock=${kmn:-?}/${kmd:-?}/${kmx:-?}kHz | deliv2400=${dv:-?}% mean_mhz=${mz:-?} ge2000=${g2:-?}% arm=$( [ "${NO_ARM:-0}" = 1 ] && echo off || echo wake )$HOT"
    if [ "${NO_ARM:-0}" != 1 ] && [ -n "${mz:-}" ] && [ "$mz" != "-1" ] && awk -v m="$mz" -v t="${ARM_MIN_MHZ:-2000}" 'BEGIN{exit !(m < t)}'; then
      echo "WARNING: sweep arm '$label' ran armed but mean_mhz=$mz (< 2000) - an UNBOOSTED-state arm" >&2
    fi
  done
done

echo "=== summary ==="
python3 - "$TSV" "$REPS" <<'PYEOF'
import statistics, sys
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.strip()]
# Exp880: a sweep with no measurements must FAIL, not print a header and exit 0. Tonight an adb
# outage produced exactly that (every arm FAILED, rc 0) and it took a manual re-read to notice.
missed = [r for r in rows if len(r) < 2 or not r[1]]
if missed or not rows:
    print(f"FAILED: {len(missed)} of {len(rows)} runs produced no RTF - no summary, investigate before re-running")
    sys.exit(1)
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
