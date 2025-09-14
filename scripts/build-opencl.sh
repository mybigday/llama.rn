#!/bin/bash -e

NDK_VERSION=26.3.11579264
CMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake
ANDROID_PLATFORM=android-21
CMAKE_BUILD_TYPE=Release

if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
  echo "NDK $NDK_VERSION not found, available versions: $(ls $ANDROID_HOME/ndk)"
  echo "Run \$ANDROID_HOME/tools/bin/sdkmanager \"ndk;$NDK_VERSION\""
  exit 1
fi

if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it"
  exit 1
fi

n_cpu=1
if uname -a | grep -q "Darwin"; then
  n_cpu=$(sysctl -n hw.logicalcpu)
elif uname -a | grep -q "Linux"; then
  n_cpu=$(nproc)
fi

t0=$(date +%s)

cd OpenCL-ICD-Loader

# Function to build for a given ABI
build_opencl() {
  ABI=$1
  BUILD_DIR=build/$ABI

  rm -rf $BUILD_DIR
  mkdir -p $BUILD_DIR && cd $BUILD_DIR
  echo $PWD
  
  cmake ../.. \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
    -DANDROID_ABI=$ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DANDROID_STL=c++_shared \
    -DOpenCLHeaders_DIR=$PWD/../../../OpenCL-Headers \

  cmake --build . --config Release -j $n_cpu

  #cp libOpenCL.so ../../../android/src/main/jniLibs/$ABI/

  #cd ../..
  #rm -rf $BUILD_DIR
}

# Build for arm64-v8a
build_opencl arm64-v8a

# Build for x86_64
build_opencl x86_64

t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
