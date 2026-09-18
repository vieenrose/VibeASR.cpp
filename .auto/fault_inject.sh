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
# --- audio-input probes (Exp850b): the RTF denominator must come from DECODED samples, not the RIFF header ---
AF=$HERE/assets/faults; mkdir -p "$AF"
python3 - "$HERE/assets/slice10b_24k.wav" "$AF" <<'PYGEN'
import struct, sys, os
src, out = sys.argv[1], sys.argv[2]
b = open(src, 'rb').read()
open(os.path.join(out, 'audio_trunc_half.wav'), 'wb').write(b[:44 + (len(b) - 44) // 2])  # header says 10 s, data 5 s
lie = bytearray(b); struct.pack_into('<I', lie, 40, struct.unpack('<I', b[40:44])[0] * 4)   # header says 40 s
open(os.path.join(out, 'audio_lie_dur.wav'), 'wb').write(bytes(lie))
open(os.path.join(out, 'audio_header_only.wav'), 'wb').write(b[:44])
open(os.path.join(out, 'audio_empty.wav'), 'wb').write(b'')
PYGEN

# $1 = fixture path, $2 = expect (run|err), $3 = label. For run: rtf must stay in a plausible band, because a
# metric that trusted the header field would report a FALSE speedup here (lie_dur would read ~0.5, not ~2.0).
aprobe() {
  local clip=$1 exp=$2 label=$3 out rtf
  out=$(SKIP_BUILD=1 "$HERE/measure.sh" --clip "$clip" 2>&1 | grep -oE "METRIC (rtf|tokens)=[0-9.]+" | tr '\n' ' ')
  rtf=$(printf '%s' "$out" | grep -oE "rtf=[0-9.]+" | head -1 | cut -d= -f2)
  if [ "$exp" = err ]; then
    if [ -z "$rtf" ]; then PASS=$((PASS+1)); printf "%-26s PASS  refused to load (no metric produced)\n" "$label";
    else FAIL=$((FAIL+1)); printf "%-26s FAIL  expected a clean refusal, got rtf=%s\n" "$label" "$rtf"; fi
  else
    if [ -n "$rtf" ] && python3 -c "import sys;sys.exit(0 if 1.5 <= float('$rtf') <= 4.0 else 1)"; then
      PASS=$((PASS+1)); printf "%-26s PASS  rtf=%s (content-derived denominator, no header-driven speedup)\n" "$label" "$rtf";
    else FAIL=$((FAIL+1)); printf "%-26s FAIL  rtf=%s out of [1.5,4.0] - check what sets the duration\n" "$label" "${rtf:-none}"; fi
  fi
}
aprobe "$AF/audio_trunc_half.wav"   run "WAV truncated (hdr 10s/5s)"
aprobe "$AF/audio_lie_dur.wav"      run "WAV header lies 4x duration"
aprobe "$AF/audio_header_only.wav"  err "WAV header only (44 B)"
aprobe "$AF/audio_empty.wav"        err "WAV empty (0 B)"

# --- config-edge probes (Exp851): invalid or degenerate configurations must be refused loudly, not silently
cprobe() {  # $1=extra argv $2=expect(exit code) $3=label $4=grep pattern (optional)
  local extra=$1 want=$2 label=$3 pat=$4 rc
  rc=$(adb -s "$DEV" shell "cd $RDIR && timeout 60 sh -c 'LD_LIBRARY_PATH=. taskset $MASK ./asr_streaming \
      --vae-model ./$VAE --lm-model ./$LM --audio $CLIP -t 2 $extra' >/dev/null 2>./ce.err; echo EXIT=\$?" \
      2>&1 | tr -d '\r' | grep -oE 'EXIT=[0-9]+' | head -1)
  local msg; msg=$(adb -s "$DEV" shell "grep -m1 -oE '$4' $RDIR/ce.err" 2>/dev/null | tr -d '\r')
  if [ "$rc" = "EXIT=$want" ]; then PASS=$((PASS+1)); printf "%-26s PASS  %s\n" "$label" "${msg:-exit $want as expected}";
  else FAIL=$((FAIL+1)); printf "%-26s FAIL  wanted EXIT=$want, got %s\n" "$label" "$rc"; fi
}
MASK=${MASK:-C0}
cprobe "-c 16"            1 "n_ctx below one window" 'frames failed|decode failed'
cprobe "--vae-pieces 7"   1 "pieces not a divisor of 26" 'must divide 26'
cprobe "--vae-pieces 0"   1 "pieces = 0"               'must divide 26|Invalid|invalid'

echo "---- fault_inject: $PASS pass, $FAIL fail"
[ "$FAIL" = 0 ] || exit 1
echo "no silent-corruption path: damaged model files and audio inputs fail loudly; RTF denominator stays content-derived"
