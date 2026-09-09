#!/system/bin/sh
# Phone bench worker: $1=audio $2=threads $3=pieces $4=tag
cd /data/local/tmp/vibeasr
export LD_LIBRARY_PATH=.
LM_FILE=${LM_FILE:-streaming-lm-q4_k_m.gguf}
VAE_FILE=${VAE_FILE:-vae-encoder-f16.gguf}
taskset ${MASK:-F0} ./asr_streaming --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio "$1" -t "$2" --vae-pieces "$3" ${LM_THREADS:+--lm-threads $LM_THREADS} > "out-$4.log" 2> "err-$4.log" &
PID=$!
echo $PID > "pid-$4.txt"
MAJ0=$(awk '{print $12}' /proc/$PID/stat 2>/dev/null)
PEAK=0; HWM=0
while kill -0 $PID 2>/dev/null; do
  ST=$(cat /proc/$PID/status 2>/dev/null)
  RSS=$(echo "$ST" | grep VmRSS | awk '{print $2}')
  HW=$(echo "$ST" | grep VmHWM | awk '{print $2}')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ]; then PEAK=$RSS; fi
  if [ -n "$HW" ] && [ "$HW" -gt "$HWM" ]; then HWM=$HW; fi
  sleep 0.5
done
wait $PID; EC=$?
MAJ1=$(awk '{print $12}' /proc/$PID/stat 2>/dev/null)
echo "exit=$EC peak_kb=$PEAK hwm_kb=$HWM majflt_delta=$(( ${MAJ1:-0} - ${MAJ0:-0} ))"
