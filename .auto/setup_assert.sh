#!/bin/bash
# Exp722: assert-armed device build (Debug => NO -DNDEBUG, so every GGML_ASSERT in the tree is
# live on-device). Purpose: shared-code changes (e.g. the Exp720 im2col builder/dispatch edits)
# are gated by runs of a Release binary where asserts are compiled out (Exp667 lesson) - an
# assert-violating-but-lucky default path would be invisible. This builds build-android-assert
# (the Release tree is untouched; measure.sh BIN checks still refer to build-android).
#   usage: .auto/setup_assert.sh   then push manually, e.g.
#     adb push build-android-assert/bin/asr_streaming /data/local/tmp/vibeasr/asr_dbg
#   and run asr_dbg directly (measure.sh has no BIN override and always pushes the
#   Release binary). Timing from a Debug build is NOT comparable to Release - this
#   binary exists only to make GGML_ASSERT failures loud (Exp667 lesson).
set -euo pipefail
cd "$(dirname "$0")/.."
NDK_DIR=${NDK_DIR:-/tmp/ndk/android-ndk-r26d}
MACHINE_FLAGS=${MACHINE_FLAGS:-"-mcpu=cortex-a78"}
cmake -B build-android-assert -S . \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_DIR/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_ARM_DOTPROD=ON \
  -DGGML_OPENMP=OFF \
  -DCMAKE_C_FLAGS="$MACHINE_FLAGS" \
  -DCMAKE_CXX_FLAGS="$MACHINE_FLAGS"
cmake --build build-android-assert --target asr_streaming -j20
echo "setup done: build-android-assert/bin/asr_streaming (assertions live)"
