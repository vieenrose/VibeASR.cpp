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
taskset ${MASK:-C0} ./asr_streaming --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio "$1" -t "$TH" --vae-pieces "$3" ${ARGS:-} > "out-$4.log" 2> "err-$4.log" &
PID=$!
echo $PID > "pid-$4.txt"
MAJ0=$(awk '{print $12}' /proc/$PID/stat 2>/dev/null)
MIN0=$(awk '{print $10}' /proc/$PID/stat 2>/dev/null)   # Exp815: minor faults = pages first-touched
PEAK=0; HWM=0
# Exp934: the run's own clock profile. measure.sh samples cpu7 scaling_cur_freq ONCE, pre-run, and it
# reads 1430000 in every historical row while a run actually sits at 2400000 (boost) and then 2000000
# (settled) - Exp931/932/933 showed that step is the dominant short-term driver of the metric
# (prefill x1.187, VAE x1.142, decode x1.028), so a pre-run sample cannot describe the run. This loop
# already polls every 0.5 s for RSS/faults, so one extra file read is free.
: > "khz-$4.txt"
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
  K=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null)
  [ -n "$K" ] && echo "$K" >> "khz-$4.txt"
  sleep 0.5
done
wait $PID; EC=$?
# min / median / max of the big core's governor request DURING the run (tiny sample set; sort is cheap)
if [ -s "khz-$4.txt" ]; then
  KMIN=$(sort -n "khz-$4.txt" | head -1)
  KMAX=$(sort -n "khz-$4.txt" | tail -1)
  KMED=$(sort -n "khz-$4.txt" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
else
  KMIN="?"; KMED="?"; KMAX="?"
fi
rm -f "khz-$4.txt"
echo "exit=$EC peak_kb=$PEAK hwm_kb=$HWM majflt_delta=$(( ${MAJ:-0} - ${MAJ0:-0} )) minflt_delta=$(( ${MIN:-0} - ${MIN0:-0} )) cpu7_khz_min=${KMIN:-?} cpu7_khz_med=${KMED:-?} cpu7_khz_max=${KMAX:-?}"
