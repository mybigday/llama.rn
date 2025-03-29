#!/bin/bash -e

# Set Android SDK and NDK paths
export ANDROID_HOME=~/android-sdk
export NDK_VERSION=26.3.11579264

# Add command line tools to PATH
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin

# Set compiler paths
export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"

# Check if NDK exists
if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
    echo "NDK not found at $ANDROID_HOME/ndk/$NDK_VERSION"
    exit 1
fi

# Check if CMake exists
if [ ! -d "$ANDROID_HOME/cmake" ]; then
    echo "CMake not found at $ANDROID_HOME/cmake"
    exit 1
fi

# Check if compilers are executable
if [ ! -x "$CC" ]; then
    echo "C compiler not executable at $CC"
    exit 1
fi

if [ ! -x "$CXX" ]; then
    echo "C++ compiler not executable at $CXX"
    exit 1
fi

# Store the root directory
ROOT_DIR=$(pwd)

# Build for arm64-v8a
mkdir -p android/build/intermediates/cmake/release/obj/arm64-v8a
cd android/build/intermediates/cmake/release/obj/arm64-v8a
cmake "$ROOT_DIR/android/src/main" \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release
make -j4
cd "$ROOT_DIR"

# Build for x86_64
mkdir -p android/build/intermediates/cmake/release/obj/x86_64
cd android/build/intermediates/cmake/release/obj/x86_64
cmake "$ROOT_DIR/android/src/main" \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=x86_64 \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release
make -j4
cd "$ROOT_DIR"

# Copy the shared libraries to the appropriate directories
mkdir -p android/src/main/jniLibs/arm64-v8a
mkdir -p android/src/main/jniLibs/x86_64
cp android/build/intermediates/cmake/release/obj/arm64-v8a/librnllama*.so android/src/main/jniLibs/arm64-v8a/
cp android/build/intermediates/cmake/release/obj/x86_64/librnllama*.so android/src/main/jniLibs/x86_64/
