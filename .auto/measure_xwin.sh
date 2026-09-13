#!/bin/bash
# Cross-window (continuous-carry) A/B against the shipped windowed protocol, on any device clip.
#   .auto/measure_xwin.sh clip1.wav [clip2.wav ...]        # default vs --xwin, interleaved
#
# Why this exists: the backlog records --xwin as "-8.5 % on short clips AND better WER, declined as a
# loop protocol". What was actually declined was AUTO-SELECTING it by file length, which a streaming
# engine cannot know. The mechanism itself (do not reset the encoder cache at window boundaries) is
# length-agnostic, so it deserves a measurement on the current stack instead of a two-year-old note.
#
# This bypasses bench_device.sh (which passes a fixed flag list and has no --xwin), so it re-implements
# the two guards the main harness has: model-file freshness and the co-runner check.
set -euo pipefail
cd "$(dirname "$0")/.."
HERE=$(cd "$(dirname "$0")" && pwd)

# DEV's single source of truth is measure.sh (the audit enforces this; config.json has no DEV).
DEV=$(grep -oE '^DEV=\S+' "$HERE/measure.sh" | head -1 | cut -d= -f2)
[ -n "$DEV" ] || { echo "ERROR: could not parse DEV from measure.sh" >&2; exit 1; }
RDIR=/data/local/tmp/vibeasr
MODELS_DIR=$(cd "$HERE/.." && pwd)/../models-streaming
VAE_FILE=${VAE_FILE:-vae-encoder-convint8.gguf}
LM_FILE=${LM_FILE:-lm-q8head.gguf}
MASK=${MASK:-C0}; THREADS=${THREADS:-2}; PIECES=${PIECES:-2}
TIER=("$HERE/tier.env")
if [ -f "$HERE/tier.env" ]; then               # trust tier.env for the shipped tier, like the audit
  while IFS='=' read -r k v; do
    case "$k" in VAE_FILE) VAE_FILE=$v;; LM_FILE) LM_FILE=$v;; PIECES) PIECES=$v;; esac
  done < <(grep -v '^#' "$HERE/tier.env")
fi

# co-runner guard (Exp679/680): a live second asr_streaming doubles every number, and "ps found
# nothing" must never be read as "device idle".
PSOUT=$(adb -s "$DEV" shell "ps -o PID,RSS,NAME" 2>/dev/null | tr -d '\r') || PSOUT=""
if [ -z "$(echo "$PSOUT" | tail -n +2 | head -1)" ]; then
  echo "ERROR: ps returned no rows - the probe is broken, cannot tell whether the device is idle" >&2; exit 1
fi
CO=$(echo "$PSOUT" | awk '$3=="asr_streaming" && $2>200000 {print $1" "$2}' | head -1)
[ -z "$CO" ] || { echo "ERROR: co-runner alive on device (pid rss: $CO) - timings would be ~2x inflated" >&2; exit 1; }

for f in "$VAE_FILE" "$LM_FILE"; do            # freshness, same rule as measure.sh
  hf="$MODELS_DIR/$f"; [ -f "$hf" ] || { echo "ERROR: missing $hf" >&2; exit 1; }
  hh=$(md5sum "$hf" | cut -d' ' -f1)
  dh=$(adb -s "$DEV" shell "md5sum $RDIR/$f" 2>/dev/null | tr -d '\r' | cut -c1-32)
  [ "$hh" = "$dh" ] || { echo "note: pushing $f (device copy stale)" >&2
                         adb -s "$DEV" push "$hf" "$RDIR/" >/dev/null 2>&1 || exit 1; }
done

run() {   # $1=clip $2=extra-flag
  adb -s "$DEV" shell "cd $RDIR && LD_LIBRARY_PATH=. taskset $MASK ./asr_streaming \
    --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio $1 -t $THREADS --vae-pieces $PIECES $2" \
    > /tmp/xw-out.txt 2> /tmp/xw-err.txt
  R=$(grep -oE 'RTF: [0-9.]+' /tmp/xw-err.txt | head -1 | awk '{print $2}')
  T=$(grep -oE 'tokens: [0-9]+' /tmp/xw-err.txt | head -1 | awk '{print $2}')
  L=$(grep -oE 'load: [0-9.]+' /tmp/xw-err.txt | head -1 | awk '{print $2}')
  [ -n "${R:-}" ] || { echo "ERROR: no RTF parsed for $1 $2 (see /tmp/xw-err.txt)" >&2; exit 1; }
  H=$(python3 -c "
import re,sys,hashlib
s=open('/tmp/xw-out.txt',errors='ignore').read().split('--- Transcription ---')[0]
print(hashlib.md5(re.findall(r\"[a-z][a-z' ]+\",s.lower()) and ' '.join(re.findall(r\"[a-z][a-z' ]+\",s.lower())).encode()).hexdigest()[:12])")
  printf "METRIC rtf=%s tokens=%s load=%s hash=%s\n" "$R" "${T:-?}" "${L:-?}" "$H"
}

for clip in "$@"; do
  echo "=== clip=$clip vae=$VAE_FILE lm=$LM_FILE pieces=$PIECES"
  for arm in "windowed|" "xwin|--xwin" "windowed-b|" "xwin-b|--xwin"; do
    n=${arm%%|*}; fl=${arm#*|}
    printf "%-10s %s\n" "$n" "$(run "$clip" "$fl")"
  done
done
