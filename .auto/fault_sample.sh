#!/system/bin/sh
# Fault-vs-time sampler (Exp815). Runs ON THE DEVICE next to the running binary and streams
# "elapsed_s minflt majflt rss_kb" once per interval, so the page-in that the per-window trace shows as a
# window-1 premium can be attributed to actual first-touches instead of being called "page-in" by name.
# It is a file rather than an adb argv string on purpose: adb joins argv into one device-shell command and
# `$(...)` / `$f` inside such a string get expanded by the HOST shell (Exp679 - it blessed empty files).
PID=$1
INT=${2:-0.05}
T0=$(date +%s.%N)
while kill -0 "$PID" 2>/dev/null; do
  T=$(date +%s.%N)
  S=$(cat /proc/$PID/stat 2>/dev/null)
  R=$(grep VmRSS /proc/$PID/status 2>/dev/null | awk '{print $2}')
  echo "$T $(echo "$S" | awk '{print $10}') $(echo "$S" | awk '{print $12}') ${R:-0}"
  sleep "$INT"
done
