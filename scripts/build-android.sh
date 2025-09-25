#!/bin/bash -e

NDK_VERSION=27.3.13750724
CMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake
ANDROID_PLATFORM=android-21
CMAKE_BUILD_TYPE=Release

if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
  echo "NDK $NDK_VERSION not found, available versions: $(ls $ANDROID_HOME/ndk)"
  echo "Run \$ANDROID_HOME/tools/bin/sdkmanager \"ndk;$NDK_VERSION\""
  exit 1
fi

CMAKE_PATH=$(which cmake)

# check cmake
if ! command -v $CMAKE_PATH &> /dev/null; then
  if [ -d "$ANDROID_HOME/cmake" ]; then
    echo "trying to find cmake in $ANDROID_HOME/cmake"
    VERSION=$(ls $ANDROID_HOME/cmake | grep -E "3\.[0-9]+\.[0-9]+" | sort -V | tail -n 1)
    if [ -n "$VERSION" ]; then
      CMAKE_PATH="$ANDROID_HOME/cmake/$VERSION/bin/cmake"
    else
      echo "cmake could not be found, please install it"
      echo "run \$ANDROID_HOME/tools/bin/sdkmanager \"cmake;3.10.2.4988404\""
      exit 1
    fi
  fi
  if ! command -v $CMAKE_PATH &> /dev/null; then
    echo "cmake could not be found, please install it"
    exit 1
  fi
fi

n_cpu=1
if uname -a | grep -q "Darwin"; then
  n_cpu=$(sysctl -n hw.logicalcpu)
elif uname -a | grep -q "Linux"; then
  n_cpu=$(nproc)
fi

t0=$(date +%s)

cd android/src/main

# Build the Android library (arm64-v8a)
echo "Building arm64-v8a with flexible page sizes support..."
$CMAKE_PATH -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -B build-arm64

$CMAKE_PATH --build build-arm64 --config Release -j $n_cpu

mkdir -p jniLibs/arm64-v8a

# Copy the library to the example app
cp build-arm64/*.so jniLibs/arm64-v8a/

rm -rf build-arm64

# Build the Android library (x86_64)
echo "Building x86_64 with flexible page sizes support..."
$CMAKE_PATH -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DANDROID_ABI=x86_64 \
  -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -B build-x86_64

$CMAKE_PATH --build build-x86_64 --config Release -j $n_cpu

mkdir -p jniLibs/x86_64

# Copy the library to the example app
cp build-x86_64/*.so jniLibs/x86_64/

rm -rf build-x86_64

t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
