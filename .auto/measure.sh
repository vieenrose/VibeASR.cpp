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

# Strict argument parsing: an unrecognised flag used to be ignored silently, which
# made `measure.sh --clip something.wav` measure the DEFAULT 10 s clip and report it
# as the new one. Anything unknown is now a hard error.
while [ $# -gt 0 ]; do
  case "$1" in
    --clip)       CLIP_LOCAL="${2:-}"; [ -f "$CLIP_LOCAL" ] || { echo "ERROR: --clip file not found: $CLIP_LOCAL" >&2; exit 1; }
                  AUDIO=$(basename "$CLIP_LOCAL"); CLIP_PUSH=1; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --env)        EXTRA_ENV="${2:-}"; shift 2 ;;
    *)            echo "ERROR: unknown argument '$1' (valid: --clip PATH, --skip-build, --env 'K=V ...')" >&2; exit 2 ;;
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
echo "note: audio=$AUDIO skip_build=$SKIP_BUILD" >&2
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
fi

if [ "$SKIP_BUILD" != 1 ]; then
cmake --build build-android --target asr_streaming -j20 > .auto/last_build.log 2>&1 || { tail -n 20 .auto/last_build.log; exit 1; }
adb -s $DEV push build-android/bin/asr_streaming $RDIR/ > /dev/null 2>&1 || exit 1
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
adb -s $DEV shell "${EXTRA_ENV:-} LM_FILE=${LM_FILE:-lm-q8head.gguf} VAE_FILE=${VAE_FILE:-vae-encoder-q4x4ffn.gguf} MASK=${MASK:-C0} THREADS=${THREADS:-2} sh $RDIR/bench_device.sh ${AUDIO} ${THREADS:-2} ${PIECES:-2} loop" > .auto/last_run.txt 2>&1 || exit 1
cat .auto/last_run.txt | tail -n 2
adb -s $DEV pull $RDIR/out-loop.log .auto/last_out.txt > /dev/null 2>&1
adb -s $DEV pull $RDIR/err-loop.log .auto/last_err.txt > /dev/null 2>&1

RTF=$(grep -oE 'RTF: [0-9.]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
VAE=$(grep -oE 'VAE: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/VAE: //;s/s//')
LMS=$(grep -oE 'LM: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/LM: //;s/s//')
TOK=$(grep -oE 'tokens: [0-9]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
ACS=$(grep -oE 'ac [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//')
SES=$(grep -oE 'sem [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//')
PRE=$(grep -oE 'prefill [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//')
DEC=$(grep -oE 'decode [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//')
LOAD=$(grep -oE 'load: [0-9.]+s' .auto/last_err.txt | head -n 1 | awk '{print $2}' | sed 's/s//')
PEAKKB=$(grep -oE 'peak_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2)
HWMKB=$(grep -oE 'hwm_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2)
MAJFLT=$(grep -oE 'majflt_delta=-?[0-9]+' .auto/last_run.txt | cut -d= -f2)
BATTT=$(adb -s $DEV shell "dumpsys battery 2>/dev/null | grep 'temperature:'" 2>/dev/null | grep -oE '[0-9]+' | head -n 1)
[ -z "${RTF:-}" ] && { echo "FAILED: no RTF parsed"; tail -n 5 .auto/last_err.txt; exit 1; }

echo "METRIC rtf=$RTF"
echo "METRIC vae_s=$VAE"
echo "METRIC lm_s=$LMS"
echo "METRIC tokens=$TOK"
echo "METRIC peak_rss_mb=$(python3 -c "print(round(${HWMKB:-$PEAKKB}/1024,1))")"
echo "METRIC majflt=${MAJFLT:-0}"
[ -n "${ACS:-}" ] && echo "METRIC ac_s=$ACS"
[ -n "${SES:-}" ] && echo "METRIC sem_s=$SES"
[ -n "${PRE:-}" ] && echo "METRIC prefill_s=$PRE"
[ -n "${DEC:-}" ] && echo "METRIC decode_s=$DEC"
[ -n "${LOAD:-}" ] && echo "METRIC load_s=$LOAD"
[ -n "${BATTT:-}" ] && echo "METRIC batt_temp_c=$(python3 -c "print(round(${BATTT}/10,1))")"
