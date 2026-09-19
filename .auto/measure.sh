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
_e=${EXTRA_ENV:-}; _lm=""; _vf=""; _mk=""; _th=""
case " $_e " in *" LM_FILE="*) ;; *) _lm="LM_FILE=${LM_FILE:-lm-q8head.gguf} ";; esac
case " $_e " in *" VAE_FILE="*) ;; *) _vf="VAE_FILE=$VAE_FILE ";; esac
case " $_e " in *" MASK="*)    ;; *) _mk="MASK=${MASK:-C0} ";; esac
case " $_e " in *" THREADS="*) ;; *) _th="THREADS=${THREADS:-2} ";; esac
if [ "$PARSE_ONLY" != 1 ]; then
adb -s $DEV shell "$_e $_lm$_vf$_mk$_th sh $RDIR/bench_device.sh ${AUDIO} ${THREADS:-2} ${PIECES:-1} loop" > .auto/last_run.txt 2>&1 || exit 1
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
[ -n "${BATTT:-}" ] && echo "$TAG batt_temp_c=$(python3 -c "print(round(${BATTT}/10,1))")"
