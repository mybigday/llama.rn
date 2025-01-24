#! /bin/bash

NDK_VERSION=26.1.10909125
CMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake
ANDROID_PLATFORM=android-21
CMAKE_BUILD_TYPE=Release

cd android/src/main

# Build the Android library (arm64-v8a)
cmake -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -B build-arm64

cmake --build build-arm64 --config Release

mkdir -p jniLibs/arm64-v8a

# Copy the library to the example app
cp build-arm64/*.so jniLibs/arm64-v8a/

rm -rf build-arm64

# Build the Android library (x86_64)
cmake -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DANDROID_ABI=x86_64 \
  -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -B build-x86_64

cmake --build build-x86_64 --config Release

mkdir -p jniLibs/x86_64

# Copy the library to the example app
cp build-x86_64/*.so jniLibs/x86_64/

rm -rf build-x86_64
