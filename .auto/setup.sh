#!/bin/bash
# One-time Android build setup for the phone-RTF loop.
# Requires: Android NDK r26d extracted at $NDK_DIR (default /tmp/ndk/android-ndk-r26d).
# Produces build-android/; afterwards ./.auto/measure.sh handles build+push+run.
set -euo pipefail
cd "$(dirname "$0")/.."

NDK_DIR=${NDK_DIR:-/tmp/ndk/android-ndk-r26d}
[ -d "$NDK_DIR" ] || { echo "NDK not found at $NDK_DIR (set NDK_DIR=...)"; exit 1; }

cmake -B build-android -S . \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_DIR/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_ARM_DOTPROD=ON
cmake --build build-android --target asr_streaming -j20
echo "setup done: build-android/bin/asr_streaming"
