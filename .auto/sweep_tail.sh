#!/bin/bash
# Exp763 paired sweep: price blocked-int8 mul_mat GEMV-tail work (extra full weight passes) by
# toggling ONE env var per arm. Discriminants are the vae/prefill sums from LATENCY_TRACE - both
# LM-output-independent (VAE work is audio-driven, prefill is a fixed 26 frames) - NOT rtf: an arm
# that perturbs features changes the token count and rtf would then measure the LM on garbage
# (Exp674 trap). The transcript hash is printed so a null arm (env var that does nothing) cannot
# masquerade as an effect (Exp714 trap).
# Usage: ARM0="env a" ARM1="env b" REPS=n SKIP_BUILD=0|1 .auto/sweep_tail.sh  (arm0 = control)
cd "$(dirname "$0")/.."
REPS=${REPS:-2}
ARMS=("${ARM0:-LATENCY_TRACE=1}" "${ARM1:-LATENCY_TRACE=1 GGML_MM_SKIP_TAIL=1}")
lt() { awk '/^LT /{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="vae")v+=a[2]; if(a[1]=="prefill")p+=a[2]; if(a[1]=="decode")d+=a[2]; if(a[1]=="tok")t+=a[2]}} END{printf "vae_ms=%.0f prefill_ms=%.0f decode_ms=%.0f tok=%.0f", v,p,d,t}' .auto/last_err.txt; }
for i in $(seq 1 "$REPS"); do
  for j in 0 1; do
    RTF=$(SKIP_BUILD="${SKIP_BUILD:-0}" .auto/measure.sh --env "${ARMS[$j]}" 2>/dev/null | { grep -E 'METRIC rtf' || true; })
    RTF=${RTF#METRIC rtf=}
    H=$(grep -E '^\[[0-9]+/[0-9]+\]' .auto/last_out.txt | md5sum | cut -c1-12)
    echo "rep$i arm$j rtf=$RTF $(lt) tx=$H"
    SKIP_BUILD=1
  done
done
