#!/bin/bash
# 40-utt LibriSpeech accuracy gate ON THE PHONE (the codegen gate must run on the
# target ISA, unlike the historical desktop-run hyp-* sets).
#   .auto/eval40.sh <hyp-tag> <start> <count>     e.g. .auto/eval40.sh a78 0 20
# Emits METRIC lines (mean rtf over the utterances actually run).
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=AYBY6HQCMBF6B6KZ
RDIR=/data/local/tmp/vibeasr
TAG=${1:-a78}; START=${2:-0}; COUNT=${3:-40}
VAE_FILE=${VAE_FILE:-vae-encoder-q8_0mixed.gguf}
LM_FILE=${LM_FILE:-streaming-lm-q4_k_m.gguf}
THREADS=${THREADS:-2}; MASK=${MASK:-C0}
OUT=../eval-librispeech/hyp-$TAG
mkdir -p "$OUT"
adb -s $DEV shell "mkdir -p $RDIR/ls40" >/dev/null 2>&1
# same lib-freshness guard as measure.sh - the binary alone is not the artifact
for L in 3rdparty/llama.cpp/ggml/src/libggml.so 3rdparty/llama.cpp/src/libllama.so; do
  B=build-android/$L; [ -f "$B" ] || continue
  M=$(md5sum "$B" | awk '{print $1}')
  D=$(adb -s $DEV shell "md5sum $RDIR/$(basename $B) 2>/dev/null" | awk '{print $1}' | tr -d '\r')
  [ "$D" = "$M" ] || { echo "pushing $(basename $B)"; adb -s $DEV push "$B" $RDIR/ >/dev/null 2>&1 || exit 1; }
done
mapfile -t WAVS < <(ls ../eval-librispeech/wav24k/*.wav | sort)
SUM=0; N=0
for ((i=START; i<START+COUNT && i<${#WAVS[@]}; i++)); do
  W=${WAVS[$i]}; B=$(basename "$W"); K=${B%.wav}
  [ -s "$OUT/$K.txt" ] && { echo "skip $K"; continue; }
  adb -s $DEV push "$W" $RDIR/ls40/ >/dev/null 2>&1
  adb -s $DEV shell "cd $RDIR && LD_LIBRARY_PATH=. taskset $MASK ./asr_streaming --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio ls40/$B -t $THREADS --vae-pieces 13" > .auto/eval40_out.txt 2> .auto/eval40_err.txt || { echo "FAILED $K"; exit 1; }
  python3 - "$OUT/$K.txt" <<'PY'
import sys
s=open('.auto/eval40_out.txt',encoding='utf-8',errors='replace').read()
t=s.split('--- Transcription ---')[1].strip() if '--- Transcription ---' in s else ''
open(sys.argv[1],'w',encoding='utf-8').write(t+'\n')
PY
  R=$(grep -oE 'RTF: [0-9.]+' .auto/eval40_err.txt | head -1 | awk '{print $2}')
  T=$(grep -oE 'tokens: [0-9]+' .auto/eval40_err.txt | head -1 | awk '{print $2}')
  [ -z "${R:-}" ] && { echo "FAILED parse $K"; exit 1; }
  echo "$i $K rtf=$R tok=$T"
  SUM=$(python3 -c "print($SUM+$R)"); N=$((N+1))
done
adb -s $DEV shell "rm -rf $RDIR/ls40" >/dev/null 2>&1
[ $N -gt 0 ] && echo "METRIC rtf=$(python3 -c "print(round($SUM/$N,4))")"
echo "METRIC utts=$N"
echo "done tag=$TAG n=$N"
