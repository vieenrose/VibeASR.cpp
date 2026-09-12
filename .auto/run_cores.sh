#!/usr/bin/env bash
# CPU-topology experiment (Exp659): the loop has always pinned to the two A78 primes
# (taskset C0 -> cpu6-7, verified via /proc/<pid>/status). Six A55 little cores (cpu0-5)
# have been idle in every one of 658 runs. Do they add throughput?
cd "$(dirname "$0")"
BIN="./asr_streaming --vae-model ./vae-encoder-q4x4ffn.gguf --lm-model ./lm-q8head.gguf --audio stream_10s_24k.wav"
../venv-vibe/bin/python .auto/run_rtf_multi.sh 2 \
  "2x A78 primes (shipped)|$BIN -t 2 C0" \
  "4 threads over all 8 cpus|$BIN -t 4 0-7" \
  "8 threads over all 8 cpus|$BIN -t 8 0-7" \
  "2 threads on A55 little cores only|$BIN -t 2 0-5"
