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
  "bound_batch|BOUND_BATCH_OFF=1",                                                       # Exp835: one 28-row prefill batch -> three decodes per window (isolated 1-row = full weight stream)
  "flush_off|FLUSH_TAIL_OFF=1"                                                       # Exp830: a PROTOCOL switch, not a fusion - must reproduce the pre-Exp829 hash 55ac39b635cb
  "stack_off|VAE_DW_CONV1D_OFF=1 VAE_GELU_BIAS_OFF=1 VAE_NORM_FUSE_OFF=1 GGML_GELU_BATCH_OFF=1 VAE_LS_FUSE_OFF=1 VAE_DW_AXPY_OFF=1 GGML_MM_M2_OFF=1 VAE_DW_LPAD_OFF=1 BOUND_BATCH_OFF=1"  # Exp824: the honest "what does the fusion stack buy" arm - everything EXCEPT the layout revert, so it stays output-preserving
  "ALL_OFF|VAE_DW_CONV1D_OFF=1 VAE_GELU_BIAS_OFF=1 VAE_NORM_FUSE_OFF=1 GGML_GELU_BATCH_OFF=1 VAE_LS_FUSE_OFF=1 VAE_DW_AXPY_OFF=1 GGML_MM_M2_OFF=1 VAE_DW_LPAD_OFF=1 VAE_CT_BLOCK_OFF=1"
)

declare -A SUM N HASH TOK
REF=""
for ((r = 1; r <= REPS; r++)); do
  for a in "${ARMS[@]}"; do
    name=${a%%|*}; env=${a#*|}
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
    echo "rep$r $name rtf=$rtf tok=$tok hash=$h"
  done
done

echo
echo "arm           mean_rtf   vs_default   tokens   transcript_identity"
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
  printf "%-13s %-10s %-12s %-8s %s\n" "$name" "$mean" "$d" "${TOK[$name]:-?}" "$id"
done
echo
echo "ref transcript hash: $REF  (all arms must match it to count as a rollback path)"
