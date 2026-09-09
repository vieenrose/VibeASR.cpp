#!/bin/bash
# Correctness gate for phone-RTF loop: rejects degenerate transcripts.
# Generic anti-collapse guards (NOT content matching): empty output, character
# repetition runs, single-word dominance (e.g. "Bye bye" loops), token-count
# band vs the 10 s clip baseline (44 tokens; legitimate jitter is +-a few).
set -euo pipefail
cd "$(dirname "$0")/.."
T=.auto/last_out.txt
[ -f "$T" ] || { echo "checks: missing transcript"; exit 1; }
TEXT=$(sed -n '/--- Transcription ---/,$p' "$T" | tail -n +2)
[ -n "$(echo "$TEXT" | tr -d '[:space:]')" ] || { echo "checks: empty transcript"; exit 1; }
if echo "$TEXT" | grep -qE '(.)\1{19,}'; then echo "checks: char-repetition collapse"; exit 1; fi
NWORDS=$(echo "$TEXT" | wc -w)
TOP1=$(echo "$TEXT" | tr '[:upper:]' '[:lower:]' | grep -oE "[a-z0-9']+" | sort | uniq -c | sort -rn | head -n 1 | awk '{print $1}')
if [ "$TOP1" -gt 0 ] && [ "$(( TOP1 * 100 / NWORDS ))" -gt 40 ]; then echo "checks: word-dominance collapse ($TOP1/$NWORDS)"; exit 1; fi
NTOK=$(grep -oE 'tokens: [0-9]+' .auto/last_err.txt | head -n 1 | awk '{print $2}')
if [ "${NTOK:-0}" -lt 20 ] || [ "${NTOK:-0}" -gt 100 ]; then echo "checks: token count out of band ($NTOK)"; exit 1; fi
echo "checks: OK (words=$NWORDS tokens=$NTOK)"
