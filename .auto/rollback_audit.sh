#!/usr/bin/env bash
# Rollback-cost audit (Exp677, refreshed by Exp814).
#
# Every escape hatch is exercised IN ONE BINARY, interleaved so thermal drift hits all arms equally, and
# each arm is checked for BOTH:
#   cost  - rtf vs the default arm (what a field runbook must quote when it says "revert X, it costs Y"), and
#   identity - the per-window transcript hash, which is the actual contract: a hatch that changes text is
#              not a rollback path, it is a different system.
# The Exp677 numbers went stale twice because nobody re-ran this after new fusions shipped, and two hatches
# (GGML_GELU_BATCH_OFF, VAE_NORM_FUSE_OFF) had never been exercised at all. Run it whenever a fusion ships.
#
# Usage: .auto/rollback_audit.sh [reps]
set -uo pipefail
# Exp1035: VALIDATE reps. It is read straight into `for ((r = 1; r <= REPS; r++))`, so an argument like
# `--quick` was parsed as ARITHMETIC (prefix-decrement of an unset variable = 0) and the ladder silently ran
# ZERO arms and printed its summary - a silent, not loud, failure of exactly the class Exp1015/Exp1034 fixed
# elsewhere. Reject anything that is not a positive integer.
if [ $# -gt 0 ]; then
  case "$1" in
    -*) echo "ERROR: unknown argument '$1' - usage: rollback_audit.sh [reps]" >&2; exit 2 ;;
    ''|*[!0-9]*) echo "ERROR: reps must be a positive integer, got '$1' (a flag would evaluate as 0 in the arm loop and run NO arms)" >&2; exit 2 ;;
  esac
  [ "$1" -ge 1 ] || { echo "ERROR: reps must be >= 1" >&2; exit 2; }
fi
REPS=${1:-2}
cd "$(dirname "$0")/.." || exit 1

ARMS=(
  "default|"
  "dw_conv1d|VAE_DW_CONV1D_OFF=1"      # Exp670 fused depthwise conv1d kernel -> tap chain
  "gelu_bias|VAE_GELU_BIAS_OFF=1"      # Exp673/674 gelu+bias fusion -> separate add + gelu
  "norm_fuse|VAE_NORM_FUSE_OFF=1"      # Exp784 gamma fold into rms_norm -> rms_norm + separate mul
  "gelu_batch|GGML_GELU_BATCH_OFF=1"   # Exp781 one-pass gelu+bias table -> two-pass loop
  "ls_fuse|VAE_LS_FUSE_OFF=1"          # Exp671 layer-scale+residual add_scaled -> mul then add
  "dw_lpad|VAE_DW_LPAD_OFF=1"         # Exp821 left pad inside the dw conv1d kernel -> explicit splice node
  "dw_axpy|VAE_DW_AXPY_OFF=1"          # Exp664 fused dw-tap axpy -> per-tap mul+add chain
  "mm_m2|GGML_MM_M2_OFF=1"             # Exp765 two-column GEMV tail -> one column per weight pass
  "ct_block|VAE_CT_BLOCK_OFF=1"        # Exp586 channels-first block layout -> [T,C] legacy path
  "bound_batch|BOUND_BATCH_OFF=1"                                                    # Exp835: one 28-row prefill batch -> three decodes per window (isolated 1-row = full weight stream). Exp1035: this element used to END WITH A COMMA inside the array word, so the arm exported BOUND_BATCH_OFF=1, - harmless while every knob tests PRESENCE, but a latent trap the day one compares the value.                                                       # Exp835: one 28-row prefill batch -> three decodes per window (isolated 1-row = full weight stream)
  "flush_off|FLUSH_TAIL_OFF=1"                                                       # Exp830: a PROTOCOL switch, not a fusion - must reproduce the pre-Exp829 hash 55ac39b635cb
  "cont_tile_off|GGML_CONT_TILE_OFF=1"  # Exp864 (v4.8): the blocked-transpose cont fast path (CONT site 7)
  "stack_off|VAE_DW_CONV1D_OFF=1 VAE_GELU_BIAS_OFF=1 VAE_NORM_FUSE_OFF=1 GGML_GELU_BATCH_OFF=1 VAE_LS_FUSE_OFF=1 VAE_DW_AXPY_OFF=1 GGML_MM_M2_OFF=1 VAE_DW_LPAD_OFF=1 BOUND_BATCH_OFF=1 GGML_CONT_TILE_OFF=1"  # Exp824: the honest "what does the fusion stack buy" arm - everything EXCEPT the layout revert, so it stays output-preserving
  "ALL_OFF|VAE_DW_CONV1D_OFF=1 VAE_GELU_BIAS_OFF=1 VAE_NORM_FUSE_OFF=1 GGML_GELU_BATCH_OFF=1 VAE_LS_FUSE_OFF=1 VAE_DW_AXPY_OFF=1 GGML_MM_M2_OFF=1 VAE_DW_LPAD_OFF=1 BOUND_BATCH_OFF=1 GGML_CONT_TILE_OFF=1 VAE_CT_BLOCK_OFF=1"
)

declare -A SUM N HASH TOK KMD MZW
REF=""
DEV=${DEV:-$(grep -m1 '^DEV=' .auto/measure.sh | cut -d= -f2)}   # single source: measure.sh
# Exp883 thermal gate: each arm runs only when the battery reads settled (<= 370 = 37.0 C).
# The fresh regime shows +18 % heat excursions (Exp882), and this round's first ladder attempt
# proved the failure mode: the default drifted 1.19 -> 1.34 across 32 back-to-back runs and every
# hatch cost inflated (stack_off +42.9 % vs +27.1 %, dw_axpy +5.4 % vs +0.0 %) - heat amplifies the
# price of extra passes, so a soaked ladder misleads the runbook it serves. The rep-reversal
# (also Exp883) cancels linear drift; this gate bounds the nonlinear throttle remainder.
# Always echoes its verdict (empty = settled) and always returns 0, so a blind/expired gate is
# visible on the arm line instead of stalling or aborting the ladder.
cool_down() {
  local empty=0 T=""
  for ((w = 1; w <= 20; w++)); do
    T=$(adb -s $DEV shell "dumpsys battery 2>/dev/null | grep 'temperature:'" 2>/dev/null | grep -oE '[0-9]+' | head -n 1 | tr -d '\r')
    if [ -z "${T:-}" ]; then empty=$((empty+1))
      if [ "$empty" -ge 2 ]; then echo " [gate-blind]"; return 0; fi
      sleep 10; continue
    fi
    if [ "$T" -le 370 ] 2>/dev/null; then return 0; fi
    sleep 30
  done
  echo " [HOT batt ${T:-?}]"
  return 0
}
# Exp883 heat-bias control: 16 sequential runs heat-soak the phone, and the fresh regime shows +18 %
# heat excursions (Exp882). A fixed arm order charges all of that drift to the later arms while the
# default, always first, looks artificially good. Even reps run the list REVERSED so a linear drift
# cancels in the mean at zero time cost; the default arm's rep1-vs-rep2 spread is the residual check
# (P4 in cost_pred883.txt). The summary accumulates per-arm, so the order is transparent to it.
for ((r = 1; r <= REPS; r++)); do
  if (( r % 2 == 0 )); then ORDER=(); for ((i=${#ARMS[@]}-1; i>=0; i--)); do ORDER+=("${ARMS[$i]}"); done
  else ORDER=("${ARMS[@]}"); fi
  for a in "${ORDER[@]}"; do
    name=${a%%|*}; env=${a#*|}
    gate=$(cool_down)
    out=$(SKIP_BUILD=1 EXTRA_ENV="$env" .auto/measure.sh 2>/dev/null)
    rtf=$(echo "$out" | grep -o 'METRIC rtf=[0-9.]*' | cut -d= -f2)
    tok=$(echo "$out"  | grep -o 'METRIC tokens=[0-9]*' | cut -d= -f2)
    # The transcript, not the run summary: .auto/multi-run-*.txt holds only the summary tail and would
    # compare "exit=0" strings and always pass (Exp671 trap).
    h=$(md5sum .auto/last_out.txt | cut -c1-12)
    [ "$name" = default ] && [ -z "$REF" ] && REF=$h
    SUM[$name]=$(python3 -c "print(${SUM[$name]:-0} + ${rtf:-0})")
    N[$name]=$(( ${N[$name]:-0} + 1 ))
    HASH[$name]="${HASH[$name]:-}$h "
    TOK[$name]="$tok"
    # Exp937: the arm's DURING-RUN clock median (measure.sh records it since Exp934). A 15-arm ladder
    # is ~10-40 min of intermittent load, so unlike a 4-arm sweep it can plausibly cross the
    # 2400000 -> 2000000 step - and a state mix charges the step to the later arms exactly like the
    # heat drift the rep-reversal exists to cancel. Print it per arm and carry it into the summary.
    kmd=$(echo "$out" | grep -o 'METRIC cpu7_khz_med=[0-9]*' | cut -d= -f2)
    KMD[$name]="${KMD[$name]:-}${kmd:-?} "
    # Exp996: the arm's P-state WITNESS (mean delivered MHz, Exp975/987). Costs only belong in a runbook
    # if they were measured in the state the runbook's numbers describe, and Exp962's ladder was run
    # entirely in the capped state - those costs were quarantined for exactly this reason. Record the
    # witness per arm so a future reader can tell which state a cost was measured in.
    mzw=$(echo "$out" | grep -o 'METRIC cpu7_deliv_mhz=[0-9.]*' | cut -d= -f2)
    MZW[$name]="${MZW[$name]:-}${mzw:-?} "
    echo "rep$r $name rtf=$rtf tok=$tok hash=$h clock=${kmd:-?} mhz=${mzw:-?}$gate"
  done
done

echo
echo "arm           mean_rtf   vs_default   tokens   clock       mean_mhz  transcript_identity"
base=$(python3 -c "print(${SUM[default]:-1}/${N[default]:-1})")   # MEAN, not the raw sum (Exp814: the
                                                                                                                   # first version divided by the sum and printed -50% for every arm)
for a in "${ARMS[@]}"; do
  name=${a%%|*}
  mean=$(python3 -c "print(f'{${SUM[$name]}/${N[$name]}:.4f}')")
  d=$(python3 -c "print(f'{100*($mean/${base:-1}-1):+.1f}%')")
  uniq=$(echo "${HASH[$name]}" | tr ' ' '\n' | grep -c . )
  distinct=$(echo "${HASH[$name]}" | tr ' ' '\n' | sort -u | grep -c .)
  if [ "$distinct" = 1 ] && [ "${HASH[$name]%% *}" = "$REF" ]; then id="identical to default"
  elif [ "$distinct" = 1 ]; then id="deterministic but DIFFERS from default"
  else id="UNSTABLE across reps"; fi
  printf "%-13s %-10s %-12s %-8s %-11s %-9s %s\n" "$name" "$mean" "$d" "${TOK[$name]:-?}" "${KMD[$name]:-?}" "${MZW[$name]:-?}" "$id"
done
echo
echo "ref transcript hash: $REF  (all arms must match it to count as a rollback path)"
