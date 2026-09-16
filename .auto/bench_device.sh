#!/system/bin/sh
# Phone bench worker: $1=audio $2=threads $3=pieces $4=tag
cd /data/local/tmp/vibeasr
export LD_LIBRARY_PATH=.
LM_FILE=${LM_FILE:-streaming-lm-q4_k_m.gguf}
VAE_FILE=${VAE_FILE:-vae-encoder-f16.gguf}
# $2 (threads) may be overridden by THREADS in the environment, which is how measure.sh forwards a
# deliberate A/B (Exp808): otherwise the positional argument silently won and a -t sweep measured the
# same configuration in both arms. Same convention as LM_FILE/VAE_FILE/MASK above.
TH=${THREADS:-$2}
taskset ${MASK:-C0} ./asr_streaming --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio "$1" -t "$TH" --vae-pieces "$3" > "out-$4.log" 2> "err-$4.log" &
PID=$!
echo $PID > "pid-$4.txt"
MAJ0=$(awk '{print $12}' /proc/$PID/stat 2>/dev/null)
MIN0=$(awk '{print $10}' /proc/$PID/stat 2>/dev/null)   # Exp815: minor faults = pages first-touched
PEAK=0; HWM=0
MIN=$MIN0; MAJ=$MAJ0   # Exp815 FIX: sampled INSIDE the loop. Both counters used to be read AFTER `wait`,
                       # by which time /proc/$PID is gone, awk prints nothing, ${VAR:-0} makes it 0, and the
                       # delta becomes 0 - MAJ0 (or a negative MIN0, which is how I caught it: minflt_delta=-1598).
                       # So `majflt 0` in every historical log meant UNREADABLE, not "no major faults". The
                       # long-run claims that quote majflt (Exp679/797) need re-verification with this build.
while kill -0 $PID 2>/dev/null; do
  ST=$(cat /proc/$PID/status 2>/dev/null)
  S=$(cat /proc/$PID/stat 2>/dev/null)
  RSS=$(echo "$ST" | grep VmRSS | awk '{print $2}')
  HW=$(echo "$ST" | grep VmHWM | awk '{print $2}')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ]; then PEAK=$RSS; fi
  if [ -n "$HW" ] && [ "$HW" -gt "$HWM" ]; then HWM=$HW; fi
  if [ -n "$S" ]; then MIN=$(echo "$S" | awk '{print $10}'); MAJ=$(echo "$S" | awk '{print $12}'); fi
  sleep 0.5
done
wait $PID; EC=$?
echo "exit=$EC peak_kb=$PEAK hwm_kb=$HWM majflt_delta=$(( ${MAJ:-0} - ${MAJ0:-0} )) minflt_delta=$(( ${MIN:-0} - ${MIN0:-0} ))"
