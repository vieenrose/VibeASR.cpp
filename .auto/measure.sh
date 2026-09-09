#!/bin/bash
# Phone RTF benchmark: build -> push -> run 10 s slice pinned to big cores.
# Emits METRIC lines. Exits nonzero on build/run failure.
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=AYBY6HQCMBF6B6KZ
RDIR=/data/local/tmp/vibeasr
AUDIO=stream_10s_24k.wav
DUR=10.00

cmake --build build-android --target asr_streaming -j20 > .auto/last_build.log 2>&1 || { tail -n 20 .auto/last_build.log; exit 1; }
adb -s $DEV push build-android/bin/asr_streaming $RDIR/ > /dev/null 2>&1 || exit 1

adb -s $DEV shell "cd $RDIR && export LD_LIBRARY_PATH=. && taskset F0 ./asr_streaming --vae-model ./vae-encoder-f16.gguf --lm-model ./streaming-lm-q4_k_m.gguf --audio $AUDIO -t 4 --vae-pieces 13 > out-loop.log 2> err-loop.log & echo \$! > pid.txt; PID=\$(cat pid.txt); PEAK=0; while kill -0 \$PID 2>/dev/null; do RSS=\$(grep VmRSS /proc/\$PID/status 2>/dev/null | awk '{print \$2}'); if [ -n \"\$RSS\" ] && [ \"\$RSS\" -gt \"\$PEAK\" ]; then PEAK=\$RSS; fi; sleep 0.5; done; wait \$PID; echo \"exit=\$? peak_kb=\$PEAK\"" > .auto/last_run.txt 2>&1 || exit 1
cat .auto/last_run.txt | tail -n 2
adb -s $DEV pull $RDIR/out-loop.log .auto/last_out.txt > /dev/null 2>&1
adb -s $DEV pull $RDIR/err-loop.log .auto/last_err.txt > /dev/null 2>&1

RTF=$(grep -oE 'RTF: [0-9.]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
VAE=$(grep -oE 'VAE: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/VAE: //;s/s//')
LMS=$(grep -oE 'LM: [0-9.]+s' .auto/last_err.txt | head -n 1 | sed 's/LM: //;s/s//')
TOK=$(grep -oE 'tokens: [0-9]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
PEAKKB=$(grep -oE 'peak_kb=[0-9]+' .auto/last_run.txt | cut -d= -f2)
[ -z "${RTF:-}" ] && { echo "FAILED: no RTF parsed"; tail -n 5 .auto/last_err.txt; exit 1; }

echo "METRIC rtf=$RTF"
echo "METRIC vae_s=$VAE"
echo "METRIC lm_s=$LMS"
echo "METRIC tokens=$TOK"
echo "METRIC peak_rss_mb=$(python3 -c "print(round($PEAKKB/1024,1))")"
