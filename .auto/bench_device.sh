#!/system/bin/sh
# Phone bench worker: $1=audio $2=threads $3=pieces $4=tag
cd /data/local/tmp/vibeasr
export LD_LIBRARY_PATH=.
LM_FILE=${LM_FILE:-streaming-lm-q4_k_m.gguf}
VAE_FILE=${VAE_FILE:-vae-encoder-f16.gguf}
taskset F0 ./asr_streaming --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio "$1" -t "$2" --vae-pieces "$3" --xwin > "out-$4.log" 2> "err-$4.log" &
PID=$!
echo $PID > "pid-$4.txt"
PEAK=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(grep VmRSS /proc/$PID/status 2>/dev/null | awk '{print $2}')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ]; then PEAK=$RSS; fi
  sleep 0.5
done
wait $PID; EC=$?
echo "exit=$EC peak_kb=$PEAK"
