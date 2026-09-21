#!/usr/bin/env bash
# Hard-audio watchdog (Exp1015). WHAT IT IS: one command that re-measures the shipped tier on the three
# hard sets and prints their accuracy/attribution next to the run's state witness.
#
# WHAT IT IS NOT: a gate, and NOT a target. These sets exist to DETECT changes in hard-audio behaviour, so
# they must never be optimized against - no tuning, no prompt/clip selection, no "try X until the number
# moves". Run it when a change touches decode structure, boundary handling, quantization or the protocol;
# read the WER drift and attribute it with the documented hatches if it matters.
#
# Why it exists: Exp1010-1014 hand-ran exactly this sequence four times and found (a) the shipped stack is
# 8 pp better on overlapped speech than the v4.1 transcript, (b) that gain is entirely the v4.7 boundary
# batch, (c) the same batch costs 0.64 pp on zh-TW, and (d) consumer-mic English is token-identical across
# six versions. Each of those was a one-variable discovery, and each would have been missed by the 40-utt
# read-speech gate, which has been output-identical for fourteen consecutive runs. One command makes the
# check routine instead of heroic.
#
# Usage:  .auto/hardaudio_watch.sh              # all three sets
#         .auto/hardaudio_watch.sh --quick      # overlap probe only (~1 min)
#         EXTRA_ENV=... .auto/hardaudio_watch.sh  # any hatch, e.g. BOUND_BATCH_OFF=1
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(cd "$(dirname "$0")" && pwd)
EVAL=$(cd "$HERE/../.." && pwd)/eval-bilingual   # repo root is two levels up from .auto
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

if [ "$QUICK" = 1 ]; then SETS="gate_ms_v2:manifest_ms_v2"; else
  SETS="gate_ms_v2:manifest_ms_v2 holdout_en:manifest_holdout_en holdout_zh:manifest_holdout_zh"; fi

echo "hardaudio_watch: WATCHDOG SETS - measure and report only, never optimize against these"
echo "  env='${EXTRA_ENV:-}'  $(date -u +%Y-%m-%dT%H:%MZ)"
printf '%-14s %-8s %-7s %-9s %-11s %s\n' set rtf tokens WER attribution state
for spec in $SETS; do
  clip=${spec%%:*}; man=${spec##*:}
  out=$(SKIP_BUILD=1 AUDIO="$clip.wav" .auto/measure.sh --skip-build 2>&1)
  rtf=$(printf '%s' "$out" | grep -oE 'METRIC rtf=[0-9.]+' | cut -d= -f2)
  tok=$(printf '%s' "$out" | grep -oE 'METRIC tokens=[0-9]+' | cut -d= -f2)
  mhz=$(printf '%s' "$out" | grep -oE 'METRIC cpu7_deliv_mhz=[0-9.]+' | cut -d= -f2)
  cp .auto/last_out.txt /tmp/haw_$clip.txt
  sc=$(python3 .auto/score_stream.py /tmp/haw_$clip.txt "$EVAL/$man.json" 2>/dev/null || true)
  wer=$(printf '%s' "$sc" | grep -oE 'WER +0\.[0-9]+' | head -1 | awk '{print $2}')
  att=$(printf '%s' "$sc" | grep -oE 'attribution [0-9.]+' | head -1 | awk '{print $2}')
  printf '%-14s %-8s %-7s %-9s %-11s %s\n' "$clip" "${rtf:-?}" "${tok:-?}" "${wer:-?}" "${att:--}" "${mhz:-?}MHz"
done
echo
echo "reference (v4.8, 2026-09-21, current manifests - Exp1010-1015):"
echo "  gate_ms_v2    WER 0.1765  attr 0.4235  tags 4/4"
echo "  holdout_en    WER 0.2636  attr 0.4907  (WER token-identical to the archived v4.1 transcript)"
echo "  holdout_zh    WER 0.1538  attr 0.6674  (WER 0.1474 with BOUND_BATCH_OFF=1)"
echo "REMINDER: a WER on these sets is an observation, not a corpus claim, and any change measured here must"
echo "also be checked for the reason it moved - use the archived hatches (Exp1011's method), not more tuning."
