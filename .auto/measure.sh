#!/bin/bash
# Phone RTF benchmark: build -> push -> run 10 s slice pinned to big cores.
# Emits METRIC lines. Exits nonzero on build/run failure.
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=AYBY6HQCMBF6B6KZ
RDIR=/data/local/tmp/vibeasr
LM_FILE=${LM_FILE:-lm-q8head.gguf}
# Default VAE: conv weights on the blocked-int8 path (-3% RTF, -124 MB, 40-utt gate paired-equivalent
# with McNemar p=1.0 - Exp690). Rebuild with `.auto/conv_int8.py --device`; the --device half is not
# cosmetic: ggml_quantize_chunk(Q4_0_4_4) on x86 writes ZERO SCALES, so a host-built file is silent
# garbage (Exp688). vae-encoder-q4x4ffn.gguf stays as the F16-conv reference and the probe's source.
VAE_FILE=${VAE_FILE:-vae-encoder-convint8.gguf}
MODELS_DIR=${MODELS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/models-streaming}
AUDIO=${AUDIO:-stream_10s_24k.wav}      # device-side name, unless --clip pushes one
SKIP_BUILD=${SKIP_BUILD:-0}
# --parse-only: run the METRIC parser against the EXISTING .auto/last_*.txt and print PARSER lines
# instead of METRIC lines, so the parse path is testable without device time and can NEVER be logged as
# a measurement (Exp876: an absent-but-legal log line used to abort the whole metric block under pipefail).
PARSE_ONLY=${PARSE_ONLY:-0}

# Strict argument parsing: an unrecognised flag used to be ignored silently, which
# made `measure.sh --clip something.wav` measure the DEFAULT 10 s clip and report it
# as the new one. Anything unknown is now a hard error.
while [ $# -gt 0 ]; do
  case "$1" in
    --clip)       CLIP_LOCAL="${2:-}"; [ -f "$CLIP_LOCAL" ] || { echo "ERROR: --clip file not found: $CLIP_LOCAL" >&2; exit 1; }
                  AUDIO=$(basename "$CLIP_LOCAL"); CLIP_PUSH=1; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --parse-only) PARSE_ONLY=1; shift ;;
    --env)        EXTRA_ENV="${2:-}"; shift 2 ;;
    *)            echo "ERROR: unknown argument '$1' (valid: --clip PATH, --skip-build, --parse-only, --env 'K=V ...')" >&2; exit 2 ;;
  esac
done

[ "${CLIP_PUSH:-0}" = 1 ] && { adb -s $DEV push "$CLIP_LOCAL" $RDIR/ > /dev/null 2>&1 || exit 1; }
# Model-file freshness (Exp689: three int8 runs silently measured a STALE device copy, because this
# script pushed only the binary and libs and never VAE_FILE - so an experiment could report old bytes
# as a new result). Compare hashes and push only on difference; a missing model file is an error, not a
# plausible wrong number.
for _f in "$VAE_FILE" "$LM_FILE"; do
  _hf="$MODELS_DIR/$_f"
  [ -f "$_hf" ] || { echo "ERROR: model file not found on host: $_hf" >&2; exit 1; }
  _hh=$(md5sum "$_hf" | cut -d' ' -f1)
  _dh=$(adb -s $DEV shell "md5sum $RDIR/$_f" 2>/dev/null | tr -d '\r' | cut -c1-32)
  if [ "$_hh" != "$_dh" ]; then
    { echo "note: pushing $_f (device copy was $(if [ -n "$_dh" ]; then echo stale; else echo missing; fi))" >&2; } || true
    adb -s $DEV push "$_hf" "$RDIR/" > /dev/null 2>&1 || { echo "ERROR: push failed for $_f" >&2; exit 1; }
  fi
done
echo "note: audio=$AUDIO skip_build=$SKIP_BUILD parse_only=$PARSE_ONLY" >&2
# Co-runner guard (Exp679): a host-side `timeout` leaves the DEVICE-side run alive, and a live
# second asr_streaming halves throughput (two pinned 2-thread runs on two A78s, Exp533). That
# produced plausible-but-2x-slow numbers here, so say it out loud instead of measuring it.
# NOTE the two set -e traps here, both hit in the first version: `grep -c` EXITS 1 when the count
# is 0 (the normal idle case), and a bare `[ ... ] && echo` returns 1 when the test is false -
# either one aborts this script before it measures anything.
if command -v adb >/dev/null 2>&1; then
  BUSY=$( { adb -s $DEV shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -c asr_streaming; } || true )
  if [ "${BUSY:-0}" -gt 0 ] 2>/dev/null; then
    echo "WARNING: $BUSY asr_streaming already running on device - timings will be inflated (kill them first)" >&2
  fi
  # Exp862, second half of the same class: the device's OWN background work contaminates a run and no
  # process-name grep can see it. Caught +9 % then +29 % on two consecutive reps while Play Store
  # AOT-compiled an app (dex2oat32 -j4, nice 10, 377 % CPU). CALIBRATION: /proc/loadavg is USELESS as a
  # threshold here - it reads ~21 when the device is provably idle (busy 0.9 %) and barely decays, so
  # the signal is the busy-jiffies delta over a 1 s window, with loadavg printed for context only.
  # NOTE the awk trap in the first version: joining two /proc/stat lines with '|' shifts the second
  # line's field indices, which printed busy=-28641 %. Sum each line separately (Exp676 class).
  CPU1=$(adb -s $DEV shell "head -1 /proc/stat" 2>/dev/null | tr -d '\r' | awk '{print $2+$3+$4+$7+$8+$9, $2+$3+$4+$5+$6+$7+$8+$9}')
  sleep 1
  CPU2=$(adb -s $DEV shell "head -1 /proc/stat" 2>/dev/null | tr -d '\r' | awk '{print $2+$3+$4+$7+$8+$9, $2+$3+$4+$5+$6+$7+$8+$9}')
  LOAD=$(adb -s $DEV shell "head -1 /proc/loadavg" 2>/dev/null | tr -d '\r' | awk '{print $1}')
  # Exp880: DEVICE STATE is now sampled next to every measurement, because a phone that was REBOOTED
  # minutes ago runs this byte-identical binary ~1/3 faster than one that has been up for days (the
  # same run that read 1.85 for 40 eras of this loop read 1.22 at uptime 444 s). Every band cell in the
  # ledger is a LONG-UPTIME number; a short-uptime number is not comparable to them, so say the state
  # out loud rather than arguing about it later (the Exp841/844 "two-state" saga was this axis, unseen).
  UPT=$( { adb -s $DEV shell "cat /proc/uptime" 2>/dev/null | tr -d '\r' | awk '{print int($1)}'; } || true )
  NPROC=$( { adb -s $DEV shell "ps -A -o NAME" 2>/dev/null | wc -l; } || true )
  MEMAV=$( { adb -s $DEV shell "grep MemAvailable /proc/meminfo" 2>/dev/null | tr -d '\r' | awk '{print int($2/1024)}'; } || true )
  KHZ=$( { adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq" 2>/dev/null | tr -d '\r'; } || true )
  echo "note: device uptime_s=${UPT:-?} procs=${NPROC:-?} mem_avail_mb=${MEMAV:-?} cpu7_khz=${KHZ:-?} (band cells are long-uptime; see Exp880)" >&2
  # Exp880: the OS build is part of device state too (a patch applied at a reboot is a confound for any
  # "the band moved after the reboot" argument). One getprop per run; the reference copy lives in
  # .auto/device_fingerprint.txt.
  FP=$( { adb -s $DEV shell "getprop ro.build.fingerprint" 2>/dev/null | tr -d '\r'; } || true )
  echo "note: device fingerprint=${FP:-?}" >&2
  # Exp972: SCREEN / BOOST-ARM state. Exp968-971 established that the protocol metric has THREE device
  # states and that they are causally settable: state 3 (delivered 94-97 %) = screen ON + a recent
  # user-activity event, which HOLDS while the screen stays on; state 2 (delivered ~18 %) = screen ON
  # with no recent activity (what a bare KEYCODE_WAKEUP leaves you in); state 1 (delivered 0 %) = screen
  # OFF. Protocol rtf reads ~1.22 / ~1.70 / ~1.85 respectively on the SAME binary. Every "mystery cap"
  # of Exp954-967 was state 2: the loop woke the device but never armed it, and never recorded the
  # display state at all. So: read mScreenState into the record, and ARM explicitly below.
  # NB mScreenState lives in `dumpsys display`, NOT `dumpsys power` (Exp972: the first version grepped
  # power, matched nothing, and silently recorded UNKNOWN - a witness that could not witness. Read-back
  # of a self-check's own value is the only thing that catches this class.)
  SCR=$( { adb -s $DEV shell "dumpsys display 2>/dev/null" | tr -d '\r' | grep -m1 -oE 'mScreenState=[A-Z]+'; } || true )
  SCR=${SCR#mScreenState=}
  echo "note: device screen_before=${SCR:-UNKNOWN} (Exp972: state 3 = screen ON + activity arm)" >&2
  if [ "${UPT:-99999}" -lt 900 ] 2>/dev/null; then
    echo "WARNING: device uptime is ${UPT}s - POST-REBOOT regime. This is NOT the band: the same binary reads materially faster here (Exp880). Do not keep, compare, or report a number taken in this state without the long-uptime control run alongside it." >&2
  fi
  BUSYFRAC=$(echo "$CPU1 $CPU2" | awk '{db=$3-$1; dt=$4-$2; if (dt>0) printf "%.0f", 100*db/dt; else print 0}')
  echo "note: device other_busy=${BUSYFRAC}% load1=${LOAD:-?} (loadavg is context only on this device)" >&2
  if [ "${BUSYFRAC:-0}" -gt 25 ] 2>/dev/null; then
    echo "WARNING: device is burning ${BUSYFRAC}% of its cores on OTHER work - this run is PROVISIONAL, not a result. Typical culprit: Play Store / dex2oat / backup. Wait for it to settle or re-take 3 reps." >&2
  fi
fi

# Exp759: the research tool's discard-revert runs `git checkout -- .` at the SESSION workDir, which
# is the parent of this repo and not a git repo, and it never checks git's exit code - so a discard
# prints "reverted" while leaving the discarded edit in place, and every later number would be of
# the discarded variant. A dirty tracked tree at measurement time is that bug having fired.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIRTY=$( { git -C "$REPO" status --porcelain -- . ':(exclude).auto' 2>/dev/null; } || true )
if [ -n "$DIRTY" ]; then
  echo "WARNING: repo has uncommitted source edits - a discarded experiment may have survived (tool auto-revert cannot reach $REPO). Revert manually unless intentional:" >&2
  printf '%s\n' "$DIRTY" | head -3 >&2
fi

if [ "$SKIP_BUILD" != 1 ]; then
cmake --build build-android --target asr_streaming -j20 > .auto/last_build.log 2>&1 || { tail -n 20 .auto/last_build.log; exit 1; }
fi
# Exp816 FIX - the binary push used to live INSIDE the SKIP_BUILD gate above, which made SKIP_BUILD=1 mean
# "do not build AND do not deploy". The asymmetry hid it: the libs below are pushed on md5 difference
# regardless, so a change split across ggml and demo/src reached the phone only in its library half and
# measured the PREVIOUS binary for everything compiled into the executable. That is exactly how the Exp816
# phase census printed 100% phase=idle: the setter existed in libggml.so, the calls never left the host.
# Now: always sync the binary, by md5 like the libs, so a no-op sync costs one hash comparison.
BINM=$(md5sum build-android/bin/asr_streaming | awk '{print $1}')
BIND=$(adb -s $DEV shell "md5sum $RDIR/asr_streaming 2>/dev/null" | awk '{print $1}' | tr -d '\r')
if [ "$BIND" != "$BINM" ]; then
  adb -s $DEV push build-android/bin/asr_streaming $RDIR/ > /dev/null 2>&1 || { echo "ERROR: push failed" >&2; exit 1; }
  # Verify, do not trust: adb reports success for a push that a busy-text file silently refused.
  G=$(adb -s $DEV shell "md5sum $RDIR/asr_streaming 2>/dev/null" | awk '{print $1}' | tr -d '\r')
  [ "$G" = "$BINM" ] || { echo "ERROR: binary still differs after push ($G != $BINM) - a running process may hold it" >&2; exit 1; }
fi
# The binary is dynamically linked against libggml/libllama: pushing only the
# executable silently leaves a stale ggml on the device, so any ggml-side change
# (ARCH_FLAGS, quant kernels) measures the OLD kernels. Push libs when they differ.
for L in 3rdparty/llama.cpp/ggml/src/libggml.so 3rdparty/llama.cpp/src/libllama.so; do
  B=build-android/$L; [ -f "$B" ] || continue
  M=$(md5sum "$B" | awk '{print $1}')
  D=$(adb -s $DEV shell "md5sum $RDIR/$(basename $B) 2>/dev/null" | awk '{print $1}' | tr -d '\r')
  [ "$D" = "$M" ] || { echo "pushing $(basename $B) (was stale)"; adb -s $DEV push "$B" $RDIR/ > /dev/null 2>&1 || exit 1; }
done
adb -s $DEV push .auto/bench_device.sh $RDIR/ > /dev/null 2>&1 || exit 1

[ -n "${EXTRA_ENV:-}" ] && echo "note: forwarding EXTRA_ENV='$EXTRA_ENV' to the device"
# FIX (Exp798, extended Exp808): these assignments used to be unconditional, so a knob passed through
# EXTRA_ENV with the same name (LM_FILE=..., THREADS=..., MASK=...) was silently overridden by this line -
# both A/B arms then ran the SAME configuration and the sweep's "parity" meant the knob never fired (the
# Exp764 failure class). Honour an EXTRA_ENV assignment by dropping our copy of that variable; bench_device.sh
# still defaults them.
_e=${EXTRA_ENV:-}; _lm=""; _vf=""; _mk=""; _th=""; _ar=""
case " $_e " in *" LM_FILE="*) ;; *) _lm="LM_FILE=${LM_FILE:-lm-q8head.gguf} ";; esac
case " $_e " in *" VAE_FILE="*) ;; *) _vf="VAE_FILE=$VAE_FILE ";; esac
case " $_e " in *" MASK="*)    ;; *) _mk="MASK=${MASK:-C0} ";; esac
case " $_e " in *" THREADS="*) ;; *) _th="THREADS=${THREADS:-2} ";; esac
# Exp948: ARGS= is a CLI passthrough that bench_device.sh appends to the binary's argv, but this script
# never forwarded it, so `ARGS="-c 8192" ./.auto/measure.sh` was a SILENT NO-OP - the device shell has no
# host environment, so the variable never arrived and the arm measured the default (found by running a
# -c 8192 arm that produced byte-identical output to -c 4096; the Exp764/Exp832 class). Two fixes: (a)
# forward ARGS like the four knobs above, (b) if EXTRA_ENV already carries ARGS=, keep THAT copy so the
# two paths cannot disagree.
case " $_e " in *" ARGS="*) ;; *) _ar="ARGS='${ARGS:-}' ";; esac
if [ "$PARSE_ONLY" != 1 ]; then
# Exp972: ARM step - put the device into state 3 before measuring, or the number is state 2 (see the
# screen-state note above). A wake plus a short burst of key events is exactly what Exp969-971 showed to
# be causal; the values chosen are deliberately UI-indifferent (volume up/down pairs, net-zero drift, no
# navigation possible) so nothing about the benchmark or the app changes - only the device state.
# NO_ARM=1 disables it, so state 2 can still be measured deliberately (Exp970/971 used exactly that
# protocol by hand). This is a MEASUREMENT-PROTOCOL fix, not a speedup: state 3 is the state every
# historical ladder cell was taken in, so it is the state the ladder's numbers are comparable in.
if [ "${NO_ARM:-0}" != 1 ]; then
  { adb -s $DEV shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1; } || true
  for _i in 1 2 3; do
    { adb -s $DEV shell "input keyevent KEYCODE_VOLUME_DOWN" >/dev/null 2>&1; } || true
    { adb -s $DEV shell "input keyevent KEYCODE_VOLUME_UP" >/dev/null 2>&1; } || true
    sleep 1
  done
  ARMED=1
else
  ARMED=0
fi
# Post-arm display state: this is what the TSV records, because it describes the state the RUN is in
# (the pre-arm read above is kept only as the before-witness on the note line).
SCR2=$( { adb -s $DEV shell "dumpsys display 2>/dev/null" | tr -d '\r' | grep -m1 -oE 'mScreenState=[A-Z]+'; } || true )
SCR2=${SCR2#mScreenState=}
echo "note: device screen_after=${SCR2:-UNKNOWN} arm_ran=${ARMED}" >&2
# Exp942: DELIVERED-frequency histogram. cpu7's scaling_cur_freq (the 5th TSV column and the med/min/max
# from bench_device.sh) is the governor's REQUEST; this is what the core actually delivered. Read
# cpufreq stats/time_in_state before and after the run and difference it: the unit is a 10 ms jiffy
# (calibrated: 100.3 units per second of wall), and the 2.4 GHz share is the state indicator. Needed
# because Exp940's LM transient had the request pinned at max while the run was 10 % slower.
TIS0=$( { adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" 2>/dev/null; } || true )
adb -s $DEV shell "$_e $_lm$_vf$_mk$_th$_ar sh $RDIR/bench_device.sh ${AUDIO} ${THREADS:-2} ${PIECES:-1} loop" > .auto/last_run.txt 2>&1 || exit 1
TIS1=$( { adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" 2>/dev/null; } || true )
cat .auto/last_run.txt | tail -n 2
adb -s $DEV pull $RDIR/out-loop.log .auto/last_out.txt > /dev/null 2>&1
adb -s $DEV pull $RDIR/err-loop.log .auto/last_err.txt > /dev/null 2>&1
else
  echo "note: PARSE-ONLY - parsing the existing .auto/last_err.txt; this is NOT a measurement" >&2
fi

TAG=$([ "$PARSE_ONLY" = 1 ] && echo "PARSER" || echo "METRIC")
RTF=$( grep -oE 'RTF: [0-9.]+' .auto/last_err.txt | head -n 1 | awk '{print $2}' || true )   # pipefail-safe: absent line != failed run (Exp876)
VAE=$( grep -oE 'VAE: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/VAE: //;s/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
LMS=$( grep -oE 'LM: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/LM: //;s/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
TOK=$( grep -oE 'tokens: [0-9]+' .auto/last_err.txt | head -n 1 | awk '{print $2}' || true )   # pipefail-safe: absent line != failed run (Exp876)
ACS=$( grep -oE 'ac [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
SES=$( grep -oE 'sem [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
PRE=$( grep -oE 'prefill [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
DEC=$( grep -oE 'decode [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
LOAD=$( grep -oE 'load: [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//' || true )   # pipefail-safe: absent line != failed run (Exp876)
PEAKKB=$( grep -oE 'peak_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2 || true )   # pipefail-safe: absent line != failed run (Exp876)
HWMKB=$( grep -oE 'hwm_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2 || true )   # pipefail-safe: absent line != failed run (Exp876)
MAJFLT=$( grep -oE 'majflt_delta=-?[0-9]+' .auto/last_run.txt | cut -d= -f2 || true )   # pipefail-safe: absent line != failed run (Exp876)
# Exp934: the big core's MEDIAN governor request DURING the run (bench_device.sh emits min/med/max).
# The old cpu7_khz column is a single PRE-RUN sample and cannot describe a run that crosses the
# 2400000 -> 2000000 step (Exp931-933). Optional: absent on an old binary, so never required.
KHMED=$( grep -oE 'cpu7_khz_med=[0-9?]+' .auto/last_run.txt | cut -d= -f2 || true )
BATTT=$( adb -s $DEV shell "dumpsys battery 2>/dev/null | grep 'temperature:'" 2>/dev/null | grep -oE '[0-9]+' | head -n 1 || true )   # pipefail-safe: absent line != failed run (Exp876)
[ -z "${RTF:-}" ] && { echo "FAILED: no RTF parsed"; tail -n 5 .auto/last_err.txt; exit 1; }
# Exp876: a pipefail-safe parse must not turn a MISSING optional line into a silently missing metric.
# rtf is the only required number; everything else may legitimately be absent, but say so out loud.
_missing=""
for _v in VAE LMS TOK PEAKKB HWMKB; do
  eval "_val=\${$_v:-}"
  # NB an `[ -z ... ] && ...` here would ABORT: when the test is false the && list is the loop body's
  # last command, the for loop returns 1, and set -e exits - the same silent-abort class, from the inside.
  if [ -z "$_val" ]; then _missing="$_missing $_v"; fi
done
if [ -n "$_missing" ]; then echo "note: not reported (line absent from the run log):$_missing" >&2; fi

echo "$TAG rtf=$RTF"
if [ -n "${VAE:-}" ]; then echo "$TAG vae_s=$VAE"; fi
if [ -n "${LMS:-}" ]; then echo "$TAG lm_s=$LMS"; fi
if [ -n "${TOK:-}" ]; then echo "$TAG tokens=$TOK"; fi
echo "$TAG peak_rss_mb=$(python3 -c "print(round(${HWMKB:-$PEAKKB}/1024,1))")"
echo "$TAG majflt=${MAJFLT:-0}"
[ -n "${ACS:-}" ] && echo "$TAG ac_s=$ACS"
[ -n "${SES:-}" ] && echo "$TAG sem_s=$SES"
[ -n "${PRE:-}" ] && echo "$TAG prefill_s=$PRE"
[ -n "${DEC:-}" ] && echo "$TAG decode_s=$DEC"
[ -n "${LOAD:-}" ] && echo "$TAG load_s=$LOAD"
[ -n "${KHMED:-}" ] && echo "$TAG cpu7_khz_med=$KHMED"
# Exp942: delivered 2.4 GHz share (percent of the run's counted big-core time). -1/absent = the stats
# file is unreadable on this device, said out loud rather than silently blank.
if [ -n "${TIS0:-}" ] && [ -n "${TIS1:-}" ]; then
  printf '%s\n' "$TIS0" > /tmp/tis0.$$
  printf '%s\n' "$TIS1" > /tmp/tis1.$$
  DELIV=$( { python3 -c "
import sys
def rd(p):
    return {l.split()[0]: int(l.split()[1]) for l in open(p) if len(l.split()) == 2}
a, b = rd('/tmp/tis0.$$'), rd('/tmp/tis1.$$')
d = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
tot = sum(v for v in d.values() if v > 0)
print(round(100 * d.get('2400000', 0) / tot) if tot > 0 else -1)
" 2>/dev/null; } || true )
  rm -f /tmp/tis0.$$ /tmp/tis1.$$
  [ -n "${DELIV:-}" ] && echo "$TAG cpu7_deliv2400_pct=$DELIV"
fi
[ -n "${BATTT:-}" ] && echo "$TAG batt_temp_c=$(python3 -c "print(round(${BATTT}/10,1))")"
# Exp894: device-state telemetry used to be PRINTED (note: lines) but never SAVED - so the only
# per-run state record was batt_temp_c, and any uptime/memory/process correlation the loop might
# ever want died with the terminal scrollback (discovered while trying to retrospectively test
# whether MemAvailable explains the fresh-regime excursions). Append one TSV line per run; the
# write must never break a measurement, and PARSE_ONLY must never write a measurement row.
if [ "$PARSE_ONLY" != 1 ] && [ -n "${RTF:-}" ]; then
  _tsv=.auto/device_state.tsv
  # Exp907: a 12th column `extra_env` (EXTRA_ENV, empty = default config). The Exp906 ladder
  # proved rows without config confound every raw correlation (30 hatch arms flipped batt-r and
  # faked uptime significance), and EXTRA_ENV is exactly the confound source for measure.sh rows.
  # Appended at END so columns 0-10 are stable; legacy headers migrate one-time, data rows are
  # never rewritten. Env assignment strings cannot contain tabs/newlines, so the TSV stays clean.
  [ -f "$_tsv" ] || printf 'ts\tuptime_s\tprocs\tmem_avail_mb\tcpu7_khz\tbatt_c\trtf\tvae_s\tlm_s\ttokens\tfingerprint\textra_env\tcpu7_khz_med\tcpu7_deliv2400_pct\tscreen\n' > "$_tsv"
  if ! head -1 "$_tsv" | grep -q 'extra_env'; then
    sed -i '1s/$/\textra_env/' "$_tsv"
  fi
  # Exp934: 13th column `cpu7_khz_med` = the big core's median REQUEST during the run (the 5th column
  # `cpu7_khz` is a single PRE-RUN sample, which reads 1430000 while a run actually sits at 2400000 then
  # 2000000). Same append-only rule as Exp907: history is never rewritten, the header migrates one-time.
  if ! head -1 "$_tsv" | grep -q 'cpu7_khz_med'; then
    sed -i '1s/$/\tcpu7_khz_med/' "$_tsv"
  fi
  # Exp942: 14th column `cpu7_deliv2400_pct` = percent of the run's counted big-core time actually
  # DELIVERED at 2.4 GHz (request-independent). Append-only, same rule.
  if ! head -1 "$_tsv" | grep -q 'cpu7_deliv2400_pct'; then
    sed -i '1s/$/\tcpu7_deliv2400_pct/' "$_tsv"
  fi
  # Exp972: 15th column `screen` = the display state read before the run (ON/OFF/UNKNOWN) plus whether
  # the boost arm ran, as "ON:arm=1". Exp968-971 showed screen/arm state moves the metric 40 % on the
  # SAME binary, and the loop had no record of it - so every pre-Exp972 row is state-ambiguous unless
  # its delivered share says otherwise. Append-only, same rule as the columns above.
  if [ "$(head -1 "$_tsv" | awk -F'\t' '{print $NF}')" != screen ]; then
    sed -i '1s/$/\tscreen/' "$_tsv"
  fi
  _battc=$(python3 -c "print(round(${BATTT:-0}/10,1))" 2>/dev/null || echo "?")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "${UPT:-?}" "${NPROC:-?}" "${MEMAV:-?}" "${KHZ:-?}" "$_battc" "$RTF" "${VAE:-?}" "${LMS:-?}" "${TOK:-?}" "${FP:-?}" "${EXTRA_ENV:-}" "${KHMED:-?}" "${DELIV:--1}" "${SCR2:-UNKNOWN}:arm=${ARMED:-?}" >> "$_tsv" 2>/dev/null || true
fi
