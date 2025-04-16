#!/bin/bash -e

# Set Android SDK and NDK paths
if [ -z "$ANDROID_HOME" ]; then
    # Try common Android SDK locations
    if [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_HOME="$HOME/Android/Sdk"
    elif [ -d "/usr/local/lib/android/sdk" ]; then
        export ANDROID_HOME="/usr/local/lib/android/sdk"
    elif [ -d "$HOME/android-sdk" ]; then
        export ANDROID_HOME="$HOME/android-sdk"
    else
        echo "Error: ANDROID_HOME is not set and could not find Android SDK"
        echo "Please set ANDROID_HOME environment variable to your Android SDK path"
        exit 1
    fi
fi

export NDK_VERSION=26.3.11579264

# Add command line tools to PATH
if [ -d "$ANDROID_HOME/cmdline-tools/latest" ]; then
    export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
elif [ -d "$ANDROID_HOME/cmdline-tools/tools/bin" ]; then
    export PATH=$PATH:$ANDROID_HOME/cmdline-tools/tools/bin
fi

echo "Using Android SDK at: $ANDROID_HOME"

# Detect if running in WSL
if [[ $(uname -r) =~ [Mm]icrosoft ]]; then
    # Running in WSL
    # Check if using Windows NDK path or Linux NDK path
    if [[ "$ANDROID_HOME" == /mnt/* ]]; then
        # Path is mounted from Windows, use Windows paths
        export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
        export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
    else
        # Path is in Linux filesystem, use Linux paths
        export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
        export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
    fi
else
    # Not running in WSL, use Linux paths
    export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
    export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
fi

# Check if NDK exists
if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
    echo "NDK not found at $ANDROID_HOME/ndk/$NDK_VERSION"
    echo "Please install NDK version $NDK_VERSION through Android Studio"
    exit 1
fi

# Check if CMake exists
if [ ! -d "$ANDROID_HOME/cmake" ]; then
    echo "CMake not found at $ANDROID_HOME/cmake"
    echo "Please install CMake through Android Studio"
    exit 1
fi

# Check if compilers are executable
if [ ! -f "$CC" ]; then
    echo "C compiler not found at $CC"
    exit 1
fi

if [ ! -f "$CXX" ]; then
    echo "C++ compiler not found at $CXX"
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
