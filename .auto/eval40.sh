#!/bin/bash
# 40-utt LibriSpeech accuracy gate ON THE PHONE (the codegen gate must run on the
# target ISA, unlike the historical desktop-run hyp-* sets).
#   .auto/eval40.sh <hyp-tag> <start> <count>     e.g. .auto/eval40.sh a78 0 20
#   RESUME=1 .auto/eval40.sh <hyp-tag> <start> <count>  # continue an interrupted gate
# A hyp dir never silently mixes configs: re-running an existing non-empty tag
# without RESUME=1 fails loudly, and resume verifies the stored run-info matches.
# Emits METRIC lines (mean rtf over the utterances actually run).
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=AYBY6HQCMBF6B6KZ
RDIR=/data/local/tmp/vibeasr
# Exp1037: VALIDATE arguments (the Exp1035/Exp1036 class, queued in Exp1035 - this gate run is its proof).
# Nothing used to reject anything: `eval40.sh --quick` became TAG=--quick START=0 COUNT=40 and ran a FULL ~9 min
# gate writing into hyp---quick, and START/COUNT go straight into `for ((i=START; ...))`, where a flag evaluates
# as ARITHMETIC (prefix-decrement of an unset variable = 0) so the gate runs ZERO utterances and still reaches
# its summary. Its switches are ENV VARS (RESUME=1, VAE_FILE=, LM_FILE=, PIECES=, EXTRA_ENV=, NO_ARM=1).
for _a in "$@"; do
  case "$_a" in
    -*) echo "ERROR: eval40.sh takes POSITIONAL args only: <hyp-tag> <start> <count>; its switches are env vars (RESUME=1, VAE_FILE=, PIECES=, ...). Got '$_a'" >&2; exit 2 ;;
  esac
done
TAG=${1:-a78}; START=${2:-0}; COUNT=${3:-40}
case "$TAG" in ''|*[!A-Za-z0-9._-]*) echo "ERROR: hyp tag must be a simple name (letters/digits/._-), got '$TAG'" >&2; exit 2 ;; esac
case "$START" in ''|*[!0-9]*) echo "ERROR: <start> must be a non-negative integer, got '$START'" >&2; exit 2 ;; esac
case "$COUNT" in ''|*[!0-9]*) echo "ERROR: <count> must be a positive integer, got '$COUNT'" >&2; exit 2 ;; esac
[ "$COUNT" -ge 1 ] || { echo "ERROR: <count> must be >= 1 - zero utterances is not a gate" >&2; exit 2; }
# The start bound comes from the SET ITSELF, not a hardcoded 40, so the guard cannot drift if the gate set grows.
NWAVS=$( { ls ../eval-librispeech/wav24k/*.wav 2>/dev/null | wc -l; } || echo 0 )
[ "${NWAVS:-0}" -gt 0 ] || NWAVS=40
[ "$START" -lt "$NWAVS" ] || { echo "ERROR: <start>=$START is past the end of the $NWAVS-utt set" >&2; exit 2; }
# THESE THREE DEFAULTS MUST EQUAL THE SHIPPED TIER. Exp694: they did not (VAE defaulted to
# vae-encoder-q8_0mixed, LM to the accuracy-first Q4_K_M, PIECES to the historical 13), so a gate run
# without explicit env measured a DIFFERENT SYSTEM - two variables off the tier being shipped and ~31 %
# slower - which first looked like a speed regression. The run-info.log stamp is what exposed it, so
# always read it. When the tier recipe changes: change these AND the prose in .auto/prompt.md.
VAE_FILE=${VAE_FILE:-vae-encoder-convint8.gguf}
LM_FILE=${LM_FILE:-lm-q8head.gguf}
THREADS=${THREADS:-2}; MASK=${MASK:-C0}
PIECES=${PIECES:-1}
MODELS_DIR=${MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/../models-streaming}
OUT=../eval-librispeech/hyp-$TAG
mkdir -p "$OUT"
# Provenance + resume guard (Exp617): a hyp dir must never silently mix transcripts
# from different configs. Fresh tag => stamp run-info (.log suffix so the *.txt
# scorer glob never sees it). Existing non-empty dir => require RESUME=1 AND a
# matching run-info, else fail loudly.
RESUME=${RESUME:-0}
RUNINFO="$OUT/run-info.log"
cur_info="VAE_FILE=$VAE_FILE LM_FILE=$LM_FILE THREADS=$THREADS MASK=$MASK PIECES=$PIECES EXTRA_ENV=${EXTRA_ENV:-} BIN=$(md5sum build-android/bin/asr_streaming 2>/dev/null | awk '{print $1}')"
if [ -n "$(ls -A "$OUT" 2>/dev/null | grep -v '^run-info.log$')" ]; then
  if [ "$RESUME" != "1" ]; then
    echo "REFUSING: $OUT already holds transcripts; re-running without RESUME=1 would silently mix or skip them. Use a fresh hyp tag, or RESUME=1 to continue the same config (provenance is checked)." >&2
    exit 1
  fi
  if [ -f "$RUNINFO" ]; then
    old_info=$(cat "$RUNINFO")
    if [ "$old_info" != "$cur_info" ]; then
      echo "REFUSING resume: config mismatch." >&2
      echo "  stored: $old_info" >&2
      echo "  now:    $cur_info" >&2
      exit 1
    fi
  fi
else
  echo "$cur_info" > "$RUNINFO"
fi
adb -s $DEV shell "mkdir -p $RDIR/ls40" >/dev/null 2>&1
# same freshness guard as measure.sh - the binary alone is not the artifact.
# NOTE: the executable itself is included: a gate must run the current tree's
# binary, not whatever a previous session left on device.
for L in bin/asr_streaming 3rdparty/llama.cpp/ggml/src/libggml.so 3rdparty/llama.cpp/src/libllama.so; do
  B=build-android/$L; [ -f "$B" ] || continue
  M=$(md5sum "$B" | awk '{print $1}')
  D=$(adb -s $DEV shell "md5sum $RDIR/$(basename $B) 2>/dev/null" | awk '{print $1}' | tr -d '\r')
  [ "$D" = "$M" ] || { echo "pushing $(basename $B)"; adb -s $DEV push "$B" $RDIR/ >/dev/null 2>&1 || exit 1; }
done
# Model freshness - same rule as measure.sh (Exp694). This script pushed only wavs, so a gate could run
# against stale device weights and report them as the tier's accuracy.
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
# Exp983: ARM THE GATE. eval40 invokes the binary directly, so it never got measure.sh's arm and the
# gate ran in the UNARMED device state - its mean rtf read 1.892 where historical gates read ~1.34, i.e.
# the gate's SPEED column silently changed state while its WER column did not (WER is state-independent,
# 13 identical gates). So: the same Exp979 recipe (burst, then a KEYCODE_WAKEUP stream) now spans the
# utterance loop, and the gate's own witness (mean delivered big-core MHz over the whole gate) is written
# to $OUT/gate-state.log - NOT to run-info.log, whose exact text is the resume guard's comparison key.
# NO_ARM=1 keeps the old unarmed behaviour for anyone who wants that state deliberately.
ARM_SENTINEL=""; ARM_PID=""
if [ "${NO_ARM:-0}" != 1 ]; then
  adb -s $DEV shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
  for _i in 1 2 3; do
    adb -s $DEV shell "input keyevent KEYCODE_VOLUME_DOWN" >/dev/null 2>&1 || true
    adb -s $DEV shell "input keyevent KEYCODE_VOLUME_UP" >/dev/null 2>&1 || true
    sleep 1
  done
  ARM_SENTINEL=$(mktemp /tmp/asr_gate_arm.XXXXXX 2>/dev/null || echo "")
  if [ -n "$ARM_SENTINEL" ]; then
    ( _arm_t0=$(date +%s); while [ -f "$ARM_SENTINEL" ]; do
        adb -s $DEV shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
        # Exp993: adaptive period (dense first ARM_DENSE_S seconds, then ARM_SPARSE if set). A gate is
        # long, so the sparse tail is where the saving is; unset ARM_SPARSE = dense throughout.
        if [ -n "${ARM_SPARSE:-}" ] && [ $(( $(date +%s) - ${_arm_t0:-0} )) -ge ${ARM_DENSE_S:-20} ]; then
          sleep "$ARM_SPARSE"
        else
          sleep ${ARM_PERIOD:-3}
        fi
      done ) >/dev/null 2>&1 &
    ARM_PID=$!
    sleep 1
  fi
fi
TIS0=$( { adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" 2>/dev/null; } || true )
mapfile -t WAVS < <(ls ../eval-librispeech/wav24k/*.wav | sort)
SUM=0; N=0; SKIPPED=0
for ((i=START; i<START+COUNT && i<${#WAVS[@]}; i++)); do
  W=${WAVS[$i]}; B=$(basename "$W"); K=${B%.wav}
  [ -s "$OUT/$K.txt" ] && { echo "skip $K"; SKIPPED=$((SKIPPED+1)); continue; }
  adb -s $DEV push "$W" $RDIR/ls40/ >/dev/null 2>&1
  # Exp941: per-utterance TIMESTAMP. The first attempt at state telemetry here sampled the big core's
  # governor request before/after each utterance - and it is INERT: 35 of 40 samples read 1430000, the
  # idle target, because a between-runs sample cannot see the run (the same trap as the old measure.sh
  # column). The working pattern is a single CONCURRENT sampler for the whole gate:
  #     timeout 900 ./.auto/batt_sampler.sh .auto/gate_<tag>_state.txt 5 &
  # then align its samples to these timestamps. Two adb calls per utterance bought nothing and cost ~16 s.
  TS=$(date +%s)
  # The measurement itself. (Exp941b: an edit that added the timestamp ABOVE accidentally deleted this
  # line, and the gate then "ran" 4 utterances in 10.5 s by parsing the PREVIOUS run's stale err file -
  # four identical rtf/tok values and identical timestamps were the tell. Keep this line directly under
  # TS so the two move together.)
  adb -s $DEV shell "cd $RDIR && ${EXTRA_ENV:-} LD_LIBRARY_PATH=. taskset $MASK ./asr_streaming --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio ls40/$B -t $THREADS --vae-pieces $PIECES" > .auto/eval40_out.txt 2> .auto/eval40_err.txt || { echo "FAILED $K"; exit 1; }
  python3 - "$OUT/$K.txt" <<'PY'
import sys
s=open('.auto/eval40_out.txt',encoding='utf-8',errors='replace').read()
t=s.split('--- Transcription ---')[1].strip() if '--- Transcription ---' in s else ''
open(sys.argv[1],'w',encoding='utf-8').write(t+'\n')
PY
  R=$( grep -oE 'RTF: [0-9.]+' .auto/eval40_err.txt | head -1 | awk '{print $2}' || true )   # pipefail-safe: absent line != failed run (Exp876)
  T=$( grep -oE 'tokens: [0-9]+' .auto/eval40_err.txt | head -1 | awk '{print $2}' || true )   # pipefail-safe: absent line != failed run (Exp876)
  [ -z "${R:-}" ] && { echo "FAILED parse $K"; exit 1; }
  echo "$i $K rtf=$R tok=$T t=${TS:-?}"
  # persist per-utterance RTF (Exp711: the mean was previously stdout-only and lost on
  # interrupted/resumed runs - a gate's mean is a ladder cell, not a throwaway line)
  echo "$i $K rtf=$R tok=$T t=${TS:-?}" >> "$OUT/rtf.log" 2>/dev/null || true
  SUM=$(python3 -c "print($SUM+$R)"); N=$((N+1))
done
adb -s $DEV shell "rm -rf $RDIR/ls40" >/dev/null 2>&1
# Exp983: stop the arm stream, then record the gate's own P-state witness next to the transcripts.
TIS1=$( { adb -s $DEV shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" 2>/dev/null; } || true )
if [ -n "${ARM_SENTINEL:-}" ]; then rm -f "$ARM_SENTINEL"; ARM_SENTINEL=""; fi
if [ -n "${ARM_PID:-}" ]; then wait "$ARM_PID" 2>/dev/null || true; ARM_PID=""; fi
if [ -n "${TIS0:-}" ] && [ -n "${TIS1:-}" ]; then
  printf '%s\n' "$TIS0" > /tmp/gtis0.$$
  printf '%s\n' "$TIS1" > /tmp/gtis1.$$
  GW=$( { python3 -c "
def rd(p):
    return {int(l.split()[0]): int(l.split()[1]) for l in open(p) if len(l.split()) == 2}
a, b = rd('/tmp/gtis0.$$'), rd('/tmp/gtis1.$$')
d = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
d = {k: v for k, v in d.items() if v > 0}
tot = sum(d.values())
print(round(sum(f * v for f, v in d.items()) / tot / 1000, 1) if tot > 0 else -1)
" 2>/dev/null; } || true )
  rm -f /tmp/gtis0.$$ /tmp/gtis1.$$
  if [ -n "${GW:-}" ] && [ "$GW" != "-1" ]; then
    printf 'gate_state arm=%s mean_mhz=%s utts=%s\n' "$([ "${NO_ARM:-0}" = 1 ] && echo off || echo wake)" "$GW" "$N" >> "$OUT/gate-state.log" 2>/dev/null || true
    echo "METRIC gate_mean_mhz=$GW"
    if [ "${NO_ARM:-0}" != 1 ] && awk -v m="$GW" -v t="${ARM_MIN_MHZ:-2000}" 'BEGIN{exit !(m < t)}'; then
      echo "WARNING: gate ran armed but gate_mean_mhz=$GW (< 2000) - the gate mean is an UNBOOSTED-state number" >&2
    fi
  fi
fi
# Exp1037: a gate must COVER the range it was asked for. The loop legitimately skips existing transcripts (that
# is how RESUME works), so compare against run+skipped, not against COUNT alone - otherwise a resumed gate looks
# partial. The rtf line is an `if` rather than `[ $N -gt 0 ] && ...` for DETERMINISM, not because the AND-list
# aborts: measured on this host, a bare `[ c ] && cmd` does NOT abort mid-script or inside a loop (it continues,
# exit 0); it only propagates a 1 when it is the last statement of a FUNCTION body or of the script itself, which
# would set -e. Corrects the wording in this file's Exp679-era notes and in audit_harness.py - see ideas.md Exp1037.
EXPECTED=$(( COUNT < ${#WAVS[@]} - START ? COUNT : ${#WAVS[@]} - START ))
if [ $((N + SKIPPED)) -ne "$EXPECTED" ]; then
  echo "ERROR: gate covered $((N + SKIPPED)) of $EXPECTED utterances in range [$START,$((START + COUNT))) ($N run, $SKIPPED skipped) - refusing to report a partial gate as a gate" >&2
  exit 1
fi
if [ "$N" -gt 0 ]; then echo "METRIC rtf=$(python3 -c "print(round($SUM/$N,4))")"; fi
echo "METRIC utts=$N"
echo "done tag=$TAG n=$N"
