#!/bin/bash
# Exp763 paired sweep: price the per-column GEMV tail of the blocked-int8 mul_mat path.
# Arms differ by ONE env var (GGML_MM_SKIP_TAIL). Discriminants are the vae/prefill sums from
# LATENCY_TRACE (both LM-output-independent: VAE work is audio-driven, prefill is a fixed 26
# frames), NOT rtf - skipping the tail makes the last frames wrong, which changes the token
# count, and rtf would then measure the LM on garbage (Exp674 trap).
cd "$(dirname "$0")/.."
REPS=${REPS:-3}
lt() { awk '/^LT /{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="vae")v+=a[2]; if(a[1]=="prefill")p+=a[2]; if(a[1]=="decode")d+=a[2]; if(a[1]=="tok")t+=a[2]}} END{printf "vae_ms=%.0f prefill_ms=%.0f decode_ms=%.0f tok=%.0f", v,p,d,t}' .auto/last_err.txt; }
for i in $(seq 1 "$REPS"); do
  for arm in control skiptail; do
    if [ "$arm" = skiptail ]; then E="LATENCY_TRACE=1 GGML_MM_SKIP_TAIL=1"; else E="LATENCY_TRACE=1"; fi
    RTF=$(SKIP_BUILD="${SKIP_BUILD:-0}" .auto/measure.sh --env "$E" 2>/dev/null | { grep -E 'METRIC rtf' || true; })
    RTF=${RTF#METRIC rtf=}
    H=$(grep -E '^\[[0-9]+/[0-9]+\]' .auto/last_out.txt | md5sum | cut -c1-12)
    echo "rep$i $arm rtf=$RTF $(lt) tx=$H"
  done
done
