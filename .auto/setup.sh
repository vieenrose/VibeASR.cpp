#!/bin/bash
# One-time Android build setup for the phone-RTF loop.
# Requires: Android NDK r26d extracted at $NDK_DIR (default /tmp/ndk/android-ndk-r26d).
# Produces build-android/; afterwards ./.auto/measure.sh handles build+push+run.
set -euo pipefail
cd "$(dirname "$0")/.."

NDK_DIR=${NDK_DIR:-/tmp/ndk/android-ndk-r26d}
[ -d "$NDK_DIR" ] || { echo "NDK not found at $NDK_DIR (set NDK_DIR=...)"; exit 1; }

# Device build protocol (loop-level, like -t2/C0 pinning): tune codegen for the
# test SoC's big cores (Dimensity 1300 = Cortex-A78). Re-tested after discovering
# measure.sh used to leave a stale libggml.so on the device (Exp1's discard was
# therefore invalid). Requires an armv8.2-a part w/ dotprod -> NOT a portable
# default; the in-tree CMakeLists stays arm64-v8a baseline.
MACHINE_FLAGS=${MACHINE_FLAGS:-"-mcpu=cortex-a78"}
cmake -B build-android -S . \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_DIR/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_ARM_DOTPROD=ON \
  -DCMAKE_C_FLAGS="$MACHINE_FLAGS" \
  -DCMAKE_CXX_FLAGS="$MACHINE_FLAGS"
cmake --build build-android --target asr_streaming -j20
echo "setup done: build-android/bin/asr_streaming"
