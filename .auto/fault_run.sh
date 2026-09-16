#!/system/bin/sh
# Fault-profile driver (Exp815). Pushed and executed ON THE DEVICE: starting the binary from an adb argv
# string is how the first attempt failed silently - adb joins argv into one device-shell command, so `$(...)`
# and `$!` inside it were expanded by the HOST shell and the sampler ran with an empty PID (Exp679 trap,
# second occurrence). Keeping the whole sequence in a file removes the quoting question entirely.
cd /data/local/tmp/vibeasr || exit 1
export LD_LIBRARY_PATH=.
# Shipped tier, exactly as measure.sh configures it (bench_device.sh's own defaults are the F16/Q4_K_M
# reference tier, so they must not be trusted here - Exp772 found a sweep measuring a non-shipping tier
# through the same kind of default).
LATENCY_TRACE=1 taskset ${MASK:-C0} ./asr_streaming \
  --vae-model ./vae-encoder-convint8.gguf --lm-model ./lm-q8head.gguf \
  --audio stream_10s_24k.wav -t 2 --vae-pieces 1 > o-fault.txt 2> e-fault.txt &
PID=$!
sleep 0.4
sh ./fs.sh "$PID" 0.05 > faults.txt 2>&1
wait $PID
# DEVICE_PATHS: vae-encoder-convint8.gguf lm-q8head.gguf fs.sh stream_10s_24k.wav
