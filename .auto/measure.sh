#!/bin/bash
# Phone RTF benchmark: build -> push -> run 10 s slice pinned to big cores.
# Emits METRIC lines. Exits nonzero on build/run failure.
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=AYBY6HQCMBF6B6KZ
RDIR=/data/local/tmp/vibeasr
LM_FILE=${LM_FILE:-streaming-lm-q4_k_m.gguf}

cmake --build build-android --target asr_streaming -j20 > .auto/last_build.log 2>&1 || { tail -n 20 .auto/last_build.log; exit 1; }
adb -s $DEV push build-android/bin/asr_streaming $RDIR/ > /dev/null 2>&1 || exit 1
adb -s $DEV push .auto/bench_device.sh $RDIR/ > /dev/null 2>&1 || exit 1

adb -s $DEV shell "LM_FILE=${LM_FILE:-streaming-lm-q4_k_m.gguf} VAE_FILE=${VAE_FILE:-vae-encoder-f16.gguf} MASK=${MASK:-F0} THREADS=${THREADS:-4} LM_THREADS=${LM_THREADS:-} sh $RDIR/bench_device.sh stream_10s_24k.wav ${THREADS:-4} 13 loop" > .auto/last_run.txt 2>&1 || exit 1
cat .auto/last_run.txt | tail -n 2
adb -s $DEV pull $RDIR/out-loop.log .auto/last_out.txt > /dev/null 2>&1
adb -s $DEV pull $RDIR/err-loop.log .auto/last_err.txt > /dev/null 2>&1

RTF=$(grep -oE 'RTF: [0-9.]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
VAE=$(grep -oE 'VAE: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/VAE: //;s/s//')
LMS=$(grep -oE 'LM: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/LM: //;s/s//')
TOK=$(grep -oE 'tokens: [0-9]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
PEAKKB=$(grep -oE 'peak_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2)
HWMKB=$(grep -oE 'hwm_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2)
MAJFLT=$(grep -oE 'majflt_delta=-?[0-9]+' .auto/last_run.txt | cut -d= -f2)
[ -z "${RTF:-}" ] && { echo "FAILED: no RTF parsed"; tail -n 5 .auto/last_err.txt; exit 1; }

echo "METRIC rtf=$RTF"
echo "METRIC vae_s=$VAE"
echo "METRIC lm_s=$LMS"
echo "METRIC tokens=$TOK"
echo "METRIC peak_rss_mb=$(python3 -c "print(round(${HWMKB:-$PEAKKB}/1024,1))")"
echo "METRIC majflt=${MAJFLT:-0}"
