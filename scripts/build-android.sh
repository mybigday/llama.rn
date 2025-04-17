#!/bin/bash -e

if [ -z "$ANDROID_HOME" ]; then
    if [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_HOME="$HOME/Android/Sdk"
    elif [ -d "/usr/local/lib/android/sdk" ]; then
        export ANDROID_HOME="/usr/local/lib/android/sdk"
    elif [ -d "$HOME/android-sdk" ]; then
        export ANDROID_HOME="$HOME/android-sdk"
    elif [ -d "$HOME/Library/Android/sdk" ]; then
        export ANDROID_HOME="$HOME/Library/Android/sdk"
    else
        echo "Error: ANDROID_HOME is not set and could not find Android SDK"
        echo "Please set ANDROID_HOME environment variable to your Android SDK path"
        exit 1
    fi
fi

export NDK_VERSION=26.3.11579264

if [ -d "$ANDROID_HOME/cmdline-tools/latest" ]; then
    export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
elif [ -d "$ANDROID_HOME/cmdline-tools/tools/bin" ]; then
    export PATH=$PATH:$ANDROID_HOME/cmdline-tools/tools/bin
fi

echo "Using Android SDK at: $ANDROID_HOME"

if [[ $(uname -r) =~ [Mm]icrosoft ]]; then
    if [[ "$ANDROID_HOME" == /mnt/* ]]; then
        export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
        export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
    else
        export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
        export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
    fi
elif [[ $(uname) == "Darwin" ]]; then
    export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang"
    export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++"
else
    export CC="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
    export CXX="$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
fi

if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
    echo "NDK not found at $ANDROID_HOME/ndk/$NDK_VERSION"
    echo "Please install NDK version $NDK_VERSION through Android Studio"
    exit 1
fi

if [ ! -d "$ANDROID_HOME/cmake" ]; then
    echo "CMake not found at $ANDROID_HOME/cmake"
    echo "Please install CMake through Android Studio"
    exit 1
fi

if [ ! -f "$CC" ]; then
    echo "C compiler not found at $CC"
    exit 1
fi

if [ ! -f "$CXX" ]; then
    echo "C++ compiler not found at $CXX"
    exit 1
fi

ROOT_DIR=$(pwd)
echo "Root directory: $ROOT_DIR"

mkdir -p "$ROOT_DIR/android/src/main/jniLibs/arm64-v8a"
mkdir -p "$ROOT_DIR/android/src/main/jniLibs/x86_64"

if [ ! -d "$ROOT_DIR/android" ]; then
    echo "Error: android directory not found at $ROOT_DIR/android"
    exit 1
fi

if [ ! -d "$ROOT_DIR/android/src/main" ]; then
    echo "Error: android/src/main directory not found"
    exit 1
fi

echo "Building for x86_64..."
mkdir -p "$ROOT_DIR/android/build/intermediates/cmake/release/obj/x86_64"
cd "$ROOT_DIR/android/build/intermediates/cmake/release/obj/x86_64"
cmake "$ROOT_DIR/android/src/main" \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=x86_64 \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release
make -j4

if [ $? -eq 0 ] && [ -f "$ROOT_DIR/android/build/intermediates/cmake/release/obj/x86_64/librnllama.so" ]; then
    echo "x86_64 build successful"
    cp "$ROOT_DIR/android/build/intermediates/cmake/release/obj/x86_64/librnllama"*.so "$ROOT_DIR/android/src/main/jniLibs/x86_64/"
else
    echo "x86_64 build failed"
fi

cd "$ROOT_DIR"

# Build for arm64-v8a
echo "Building for arm64-v8a..."
mkdir -p "$ROOT_DIR/android/build/intermediates/cmake/release/obj/arm64-v8a"
cd "$ROOT_DIR/android/build/intermediates/cmake/release/obj/arm64-v8a"
cmake "$ROOT_DIR/android/src/main" \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release
make -j4

if [ $? -eq 0 ] && [ -f "$ROOT_DIR/android/build/intermediates/cmake/release/obj/arm64-v8a/librnllama.so" ]; then
    echo "arm64-v8a build successful"
    cp "$ROOT_DIR/android/build/intermediates/cmake/release/obj/arm64-v8a/librnllama"*.so "$ROOT_DIR/android/src/main/jniLibs/arm64-v8a/"
else
    echo "arm64-v8a build failed"
    echo "Using x86_64 libraries as fallback for arm64-v8a"
    if [ -f "$ROOT_DIR/android/build/intermediates/cmake/release/obj/x86_64/librnllama.so" ]; then
        cp "$ROOT_DIR/android/build/intermediates/cmake/release/obj/x86_64/librnllama"*.so "$ROOT_DIR/android/src/main/jniLibs/arm64-v8a/"
    else
        echo "No libraries available for fallback. Build may fail."
    fi
fi

cd "$ROOT_DIR"
echo "Android build completed"
