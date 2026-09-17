#!/bin/bash
# Behavioral contract watchdog - PRODUCT CONTRACTS, NOT SPEED METRICS.
#
# Why this exists: silence/noise/music labelling, non-24kHz input, diarization tagging and the sub-piece
# short-clip path are contracts the loop promised not to break, but until now they were checked ad hoc per
# era (Exp676/800) with nothing committed, so nothing enforced them. Exp821 is exactly the kind of change
# that could break them silently: the depthwise conv now does its own CAUSAL LEFT PAD, i.e. its behaviour
# at tensor EDGES changed.
#
#   .auto/behavior_watch.sh              # run the set
#   .auto/behavior_watch.sh --selftest   # plant a false expectation; MUST report FAILs (Exp660 rule: a
#                                        # self-check is untrustworthy until a planted fault makes it fail)
#
# Expectations are deliberately about LABELS and TAGS (stable semantics), never about wording, and the
# ladder clips are checked by TOKEN COUNT only, as a drift canary - this script must never become a speed
# measurement, because it runs unpinned-free arms with no interleaving.
#
# KNOWN CLOSED BEHAVIORS (do not "fix" these into failures):
#   * twospk.wav (sequential two voices, 300 ms gaps) collapses to ONE Speaker tag. That is the measured,
#     closed Exp650 finding, not a regression - and cannot be one, since every shipped fusion is
#     output-identical by the 40-utt gate. Only the OVERLAPPED probe is required to split.
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(cd "$(dirname "$0")" && pwd)

DEV=$(grep -oE '^DEV=\S+' "$HERE/measure.sh" | head -1 | cut -d= -f2)
[ -n "$DEV" ] || { echo "ERROR: could not parse DEV from measure.sh" >&2; exit 2; }
RDIR=/data/local/tmp/vibeasr
MODELS_DIR=$(cd "$HERE/.." && pwd)/../models-streaming
VAE_FILE=vae-encoder-convint8.gguf; LM_FILE=lm-q8head.gguf; PIECES=1
[ -f "$HERE/tier.env" ] && while IFS='=' read -r k v; do
  case $k in VAE_FILE) VAE_FILE=$v;; LM_FILE) LM_FILE=$v;; PIECES) PIECES=$v;; esac
done < <(grep -v '^#' "$HERE/tier.env")
MASK=${MASK:-C0}; THREADS=${THREADS:-2}
EXTRA_ENV=${EXTRA_ENV:-}   # injected INSIDE the adb string (env is not inherited by adb shell)

SELFTEST=0
[ "${1:-}" = "--selftest" ] && SELFTEST=1

# --- guards (shared origin: measure.sh / measure_xwin.sh) -------------------------------------------
PSOUT=$(adb -s "$DEV" shell "ps -o PID,RSS,NAME" 2>/dev/null | tr -d '\r') || PSOUT=""
if [ -z "$(echo "$PSOUT" | tail -n +2 | head -1)" ]; then
  echo "ERROR: ps returned no rows - cannot tell whether the device is idle" >&2; exit 2; fi
CO=$(echo "$PSOUT" | awk '$3=="asr_streaming" && $2>200000 {print $1" "$2}' | head -1)
[ -z "$CO" ] || { echo "ERROR: co-runner alive (pid rss: $CO)" >&2; exit 2; }

bh=$(md5sum build-android/bin/asr_streaming | cut -d' ' -f1)
dh=$(adb -s "$DEV" shell "md5sum $RDIR/asr_streaming" 2>/dev/null | tr -d '\r' | cut -c1-32)
if [ "$bh" != "$dh" ]; then                       # Exp816/825: never measure a stale binary
  echo "note: pushing asr_streaming ($dh != $bh)" >&2
  adb -s "$DEV" push build-android/bin/asr_streaming "$RDIR/" >/dev/null 2>&1 || { echo "ERROR: push failed" >&2; exit 2; }
  dh2=$(adb -s "$DEV" shell "md5sum $RDIR/asr_streaming" 2>/dev/null | tr -d '\r' | cut -c1-32)
  [ "$bh" = "$dh2" ] || { echo "ERROR: binary still stale after push" >&2; exit 2; }
fi
for f in "$VAE_FILE" "$LM_FILE"; do
  dhf=$(adb -s "$DEV" shell "md5sum $RDIR/$f" 2>/dev/null | tr -d '\r' | cut -c1-32)
  hhf=$(md5sum "$MODELS_DIR/$f" | cut -d' ' -f1)
  [ "$dhf" = "$hhf" ] || { echo "note: pushing $f" >&2; adb -s "$DEV" push "$MODELS_DIR/$f" "$RDIR/" >/dev/null 2>&1; }
done

echo "behavior_watch: binary $bh  vae=$VAE_FILE lm=$LM_FILE pieces=$PIECES  selftest=$SELFTEST"
PASS=0; FAIL=0

check() {   # $1=clip $2=kind $3=expectation $4=note
  local clip=$1 kind=$2 want=$3 note=$4 out err rc tok
  out=$(adb -s "$DEV" shell "cd $RDIR && LD_LIBRARY_PATH=. ${EXTRA_ENV:-} taskset $MASK ./asr_streaming \
      --vae-model ./$VAE_FILE --lm-model ./$LM_FILE --audio $clip -t $THREADS --vae-pieces $PIECES" \
      2>/tmp/bw-err.txt); rc=$?
  tok=$(grep -oE 'tokens: [0-9]+' /tmp/bw-err.txt | head -1 | awk '{print $2}')
  local sz; sz=$(adb -s "$DEV" shell "stat -c %s $RDIR/$clip" 2>/dev/null | tr -d '\r')
  [ -n "$sz" ] || { printf "%-20s SKIP  clip missing\n" "$clip"; FAIL=$((FAIL+1)); return; }

  # --selftest replaces the expectation with one that CANNOT hold, to prove the checker can fail
  if [ "$SELFTEST" = "1" ]; then kind=tokens; want=999999; fi

  local ok=0 detail=""
  case $kind in
    contains) case "$out" in *"$want"*) ok=1;; esac; detail="looking for [$want]" ;;
    cjk)      if printf '%s' "$out" | grep -qP '[\x{4e00}-\x{9fff}]'; then ok=1; fi
              detail="at least one CJK char" ;;
    words)    n=$(printf '%s' "$out" | tr -cd '[:alpha:]' | wc -c); [ "$n" -ge 8 ] && ok=1
              detail=">=8 letter chars (got $n)" ;;
    tokens)   [ "${tok:-x}" = "$want" ] && ok=1; detail="tokens=$tok expected $want" ;;
    split)    m=$(printf '%s' "$out" | grep -ocE 'Speaker [1-9]'); [ "${m:-0}" -ge 1 ] && ok=1
              detail="a Speaker tag other than 0 (found $m)" ;;
    nofail)   ok=1 ;;   # documented-closed behavior: record only
  esac
  if [ "$ok" = "1" ]; then PASS=$((PASS+1)); printf "%-20s PASS  %-34s tok=%-4s %s\n" "$clip" "$note" "${tok:-?}" "($sz B)"
  else                    FAIL=$((FAIL+1)); printf "%-20s FAIL  %-34s tok=%-4s %s | %s\n" "$clip" "$note" "${tok:-?}" "rc=$rc" "$detail"; fi
}

# --- the contract set -------------------------------------------------------------------------------
check silence5s.wav      contains  "[Silence]"       "silence must be labelled, not hallucinated"
check noise5s.wav        contains  "[Noise]"         "noise must be labelled"
check song20s.wav        contains  "[Music]"         "music must be labelled (Exp532 out-of-domain)"
check proto48k_stereo.wav cjk     ""                "48 kHz STEREO input must resample+transcribe (Exp539)"
check short36.wav        words     ""                "sub-piece clip: graceful output, no crash"
check twospk_overlap.wav split     ""                "overlapped 2 voices must split speakers (Exp609)"
check twospk.wav         nofail    ""                "sequential voices: one tag = CLOSED Exp650, not a bug"
check stream_10s_24k.wav tokens    39                "protocol clip token canary (ladder 10 s)"
check chat17.wav         tokens    108               "ladder 17 s canary"
check chat69.wav         tokens    446               "ladder 69 s canary"
check chat138.wav        tokens    877               "ladder 138 s canary (marginal class: +/-1)"

echo "---- behavior_watch: $PASS pass, $FAIL fail  (selftest=$SELFTEST)"
if [ "$SELFTEST" = "1" ]; then
  [ "$FAIL" -ge 1 ] && { echo "SELFTEST OK: the watchdog detected the planted fault"; exit 0; }
  echo "SELFTEST FAILED: a planted fault produced no failure - this checker proves nothing"; exit 1
fi
[ "$FAIL" = "0" ] || exit 1
echo "all behavioral contracts hold"
