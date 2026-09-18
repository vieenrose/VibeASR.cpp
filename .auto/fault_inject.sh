#!/bin/bash
# fault_inject.sh - model-file corruption regression board (Exp850).
#
# Why: a truncated gguf used to load "successfully" and feed ZEROS as weights - fluent, confident, wrong
# transcripts with exit 0 and normal timing (found by truncate -s -8MB / -64MB on the VAE; the LM path was
# already guarded). The fix is a short-read check in src/vae.cpp. This script keeps it enforced: it rebuilds
# the fixtures on the device, asserts each one FAILS LOUDLY, and asserts the healthy model still transcribes.
#
# Usage: .auto/fault_inject.sh          # full board (~90 s: 3 failure probes + 1 healthy control run)
#        FAULT_HEALTHY=0 .auto/fault_inject.sh   # skip the healthy control (fails-fast probes only, ~15 s)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEV=$(grep -oE '^DEV=\S+' "$HERE/measure.sh" | head -1 | cut -d= -f2)
[ -n "$DEV" ] || { echo "ERROR: could not parse DEV from measure.sh" >&2; exit 2; }
RDIR=/data/local/tmp/vibeasr
VAE=vae-encoder-convint8.gguf
LM=lm-q8head.gguf
CLIP=stream_10s_24k.wav
HEALTHY=${FAULT_HEALTHY:-1}
PASS=0; FAIL=0

# $1 = vae file, $2 = lm file, $3 = expect (ok|err), $4 = label
probe() {   # $1=vae $2=lm $3=expect(ok|err) $4=label   - no shell constructs inside the adb string (Exp832)
  local v=$1 l=$2 exp=$3 label=$4 rc err out code sig
  rc=$(adb -s "$DEV" shell "cd $RDIR && LD_LIBRARY_PATH=. taskset C0 ./asr_streaming --vae-model ./$v --lm-model ./$l \
        --audio $CLIP -t 2 --vae-pieces 1 >./fi.out 2>./fi.err; echo EXIT=\$?" 2>&1 | tr -d '\r' | grep -oE 'EXIT=[0-9]+' | head -1)
  # analyse on the HOST: the device shell is too limited for alternation/|| in argv form
  err=$(adb -s "$DEV" shell "cat $RDIR/fi.err" 2>/dev/null | tr -d '\r')
  out=$(adb -s "$DEV" shell "cat $RDIR/fi.out" 2>/dev/null | tr -d '\r')
  sig=$(printf '%s' "$err" | grep -m1 -oE 'truncated or corrupt[^\n]{0,70}|failed to load model' || true)
  [ -n "$sig" ] || sig=$(printf '%s\n%s' "$err" "$out" | grep -m1 -oE 'tokens: [0-9]+' || true)
  code=$rc
  if [ "$exp" = err ]; then
    if [ "$code" = "EXIT=1" ] && [ -n "$sig" ]; then PASS=$((PASS+1)); printf "%-26s PASS  %s\n" "$label" "$sig";
    elif [ "$code" = "EXIT=1" ]; then PASS=$((PASS+1)); printf "%-26s PASS  exit 1 (no signature matched - check messages)\n" "$label";
    else FAIL=$((FAIL+1)); printf "%-26s FAIL  expected exit 1, got %s  sig=%s\n" "$label" "$code" "${sig:0:60}"; fi
  else
    if [ "$code" = "EXIT=0" ] && printf '%s' "$sig" | grep -qE 'tokens: [1-9]'; then PASS=$((PASS+1)); printf "%-26s PASS  %s (exit 0, real output)\n" "$label" "$sig";
    else FAIL=$((FAIL+1)); printf "%-26s FAIL  %s sig=%s\n" "$label" "$code" "${sig:0:60}"; fi
  fi
}

echo "fault_inject: preparing fixtures on device"
adb -s "$DEV" shell "cd $RDIR && cp -f $LM lm_trunc.gguf && truncate -s -8388608 lm_trunc.gguf \
   && cp -f $VAE vae_trunc.gguf && truncate -s -8388608 vae_trunc.gguf \
   && cp -f $VAE vae_trunc64.gguf && truncate -s -67108864 vae_trunc64.gguf \
   && head -c 1048576 $LM > lm_head.gguf" >/dev/null 2>&1

if [ "$HEALTHY" = 1 ]; then probe "$VAE" "$LM" ok "healthy control"; fi
probe "$VAE" "lm_trunc.gguf"    err "LM tail -8MB"
probe "$VAE" "lm_head.gguf"     err "LM header-only 1MB"
probe "vae_trunc.gguf"   "$LM"  err "VAE tail -8MB"
probe "vae_trunc64.gguf" "$LM"  err "VAE tail -64MB"
echo "---- fault_inject: $PASS pass, $FAIL fail"
[ "$FAIL" = 0 ] || exit 1
echo "no silent-corruption path: every damaged model file fails loudly"
