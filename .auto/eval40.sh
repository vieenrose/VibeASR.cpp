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
TAG=${1:-a78}; START=${2:-0}; COUNT=${3:-40}
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
mapfile -t WAVS < <(ls ../eval-librispeech/wav24k/*.wav | sort)
SUM=0; N=0
for ((i=START; i<START+COUNT && i<${#WAVS[@]}; i++)); do
  W=${WAVS[$i]}; B=$(basename "$W"); K=${B%.wav}
  [ -s "$OUT/$K.txt" ] && { echo "skip $K"; continue; }
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
[ $N -gt 0 ] && echo "METRIC rtf=$(python3 -c "print(round($SUM/$N,4))")"
echo "METRIC utts=$N"
echo "done tag=$TAG n=$N"
