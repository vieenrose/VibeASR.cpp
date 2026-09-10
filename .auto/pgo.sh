#!/bin/bash
# PGO train/use cycle for the Android loop build.
# Exp26's PGO measure was invalid twice over: (1) the rebuilt libggml.so never
# reached the device (same push bug as Exp1/Exp14), (2) the profile was trained
# under --xwin + -t4/F0, not the protocol config (-t2/C0, no xwin).
# Usage: .auto/pgo.sh <train-vae-file> [train-lm-file]
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=AYBY6HQCMBF6B6KZ
RDIR=/data/local/tmp/vibeasr
NDK_DIR=${NDK_DIR:-/tmp/ndk/android-ndk-r26d}
LLVMPROFDATA=$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-profdata
PROFRAW=$PWD/.auto/pgo.profraw
PROFDATA=$PWD/.auto/pgo.profdata
VAE=${1:-vae-encoder-f16.gguf}
LM=${2:-streaming-lm-q4_k_m.gguf}
MFLAGS=${MACHINE_FLAGS:--mcpu=cortex-a78}

echo "== 1/5 instrumented build ($MFLAGS + profile-instr-generate)"
cmake -B build-android -DCMAKE_C_FLAGS="$MFLAGS -fprofile-instr-generate" \
      -DCMAKE_CXX_FLAGS="$MFLAGS -fprofile-instr-generate" >/dev/null
cmake --build build-android --target asr_streaming -j20 > .auto/last_build.log 2>&1 || { tail -20 .auto/last_build.log; exit 1; }
adb -s $DEV push build-android/bin/asr_streaming $RDIR/ >/dev/null
for L in 3rdparty/llama.cpp/ggml/src/libggml.so 3rdparty/llama.cpp/src/libllama.so; do
  adb -s $DEV push build-android/$L $RDIR/ >/dev/null
done
for F in $VAE $LM; do
  # models live outside the repo (../models-streaming); push only if missing
  if ! adb -s $DEV shell "test -f $RDIR/$(basename $F)"; then
    SRC=$F; [ -f "$SRC" ] || SRC=../models-streaming/$F
    adb -s $DEV push "$SRC" $RDIR/ >/dev/null
  fi
done

echo "== 2/5 train run on device (protocol config: -t2, C0, no xwin)"
adb -s $DEV shell "cd $RDIR && rm -f pgo.profraw && LD_LIBRARY_PATH=. LLVM_PROFILE_FILE=$RDIR/pgo.profraw taskset C0 ./asr_streaming --vae-model ./$VAE --lm-model ./$LM --audio stream_10s_24k.wav -t 2 --vae-pieces 13" > .auto/pgo_train.log 2>&1
grep -E "RTF:" .auto/pgo_train.log | tail -1
adb -s $DEV pull $RDIR/pgo.profraw "$PROFRAW" >/dev/null || { echo "no profraw - train run did not profile"; exit 1; }
ls -la "$PROFRAW"

echo "== 3/5 merge profile"
"$LLVMPROFDATA" merge -output="$PROFDATA" "$PROFRAW"
ls -la "$PROFDATA"

echo "== 4/5 PGO-use build (clean rebuild)"
cmake --build build-android --target clean >/dev/null
cmake -B build-android -DCMAKE_C_FLAGS="$MFLAGS -fprofile-instr-use=$PROFDATA" \
      -DCMAKE_CXX_FLAGS="$MFLAGS -fprofile-instr-use=$PROFDATA" >/dev/null
cmake --build build-android --target asr_streaming -j20 > .auto/last_build.log 2>&1 || { tail -20 .auto/last_build.log; exit 1; }

echo "== 5/5 ready (measure.sh will push the rebuilt libs)"
ls -la build-android/3rdparty/llama.cpp/ggml/src/libggml.so
