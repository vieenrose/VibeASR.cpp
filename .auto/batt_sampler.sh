#!/bin/bash
# Concurrent device-state sampler (Exp931, extended Exp931b with CPU frequencies).
#
# Why: measure.sh reports batt_temp_c ONCE, after the run, so no existing instrument can tell whether a
# long clip's per-window rate changes BECAUSE the device state changes during the run. LATENCY_TRACE gives
# per-window times; this gives a device-state time series to correlate them against, including the two big
# cores' current clock - the first hypothesis for a step change in compute rate.
#
#   .auto/batt_sampler.sh <outfile> [interval_s] [max_samples]
#
# Columns: epoch_s <TAB> batt_temp_deciC <TAB> cpu7_khz <TAB> cpu0_khz
#   ("?" in any column = that query failed - loud, never silently blank)
# Each sample is two short shell queries: ~0.3 s of little-core work, so it is compatible with a timed run
# pinned to the two big cores - but it IS a perturbation, so never quote that run's RTF as a settled cell.
set -u
OUT=${1:?usage: batt_sampler.sh <outfile> [interval_s] [max_samples]}
INT=${2:-15}
N=${3:-400}
HERE=$(cd "$(dirname "$0")" && pwd)
DEV=${DEV:-$(grep -m1 '^DEV=' "$HERE/measure.sh" | cut -d= -f2)}
: > "$OUT"
for ((i = 1; i <= N; i++)); do
  T=$(adb -s "$DEV" shell "dumpsys battery 2>/dev/null | grep 'temperature:'" 2>/dev/null \
      | grep -oE '[0-9]+' | head -1 | tr -d '\r')
  F=$(adb -s "$DEV" shell "cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq \
      /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null" 2>/dev/null \
      | tr -d '\r' | grep -oE '[0-9]+' | tr '\n' ',' | sed 's/,$//')
  F7=${F%%,*}; F0=${F##*,}
  printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${T:-?}" "${F7:-?}" "${F0:-?}" >> "$OUT"
  sleep "$INT"
done
