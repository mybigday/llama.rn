#!/bin/bash

function cp_headers() {
  mkdir -p ../ios/rnllama.xcframework/$1/rnllama.framework/Headers
  cp ../cpp/*.h ../ios/rnllama.xcframework/$1/rnllama.framework/Headers/
}

function build_framework() {
  # Parameters:
  # $1: system_name (iOS/tvOS)
  # $2: architectures
  # $3: sysroot
  # $4: output_path
  # $5: build_dir

  cd $5

  # Configure CMake
  cmake ../ios \
    -GXcode \
    -DCMAKE_SYSTEM_NAME=$1 \
    -DCMAKE_OSX_ARCHITECTURES="$2" \
    -DCMAKE_OSX_SYSROOT=$3 \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DCMAKE_IOS_INSTALL_COMBINED=YES

  # Build
  cmake --build . --config Release

  # Setup framework directory
  rm -rf ../ios/rnllama.xcframework/$4
  mkdir -p ../ios/rnllama.xcframework/$4
  mv Release-$3/rnllama.framework ../ios/rnllama.xcframework/$4/rnllama.framework
  mkdir -p ../ios/rnllama.xcframework/$4/rnllama.framework/Headers

  # Copy headers and metallib
  cp_headers $4

  # TODO: May need to re-build metallib for tvOS
  cp ../cpp/ggml-llama.metallib ../ios/rnllama.xcframework/$4/rnllama.framework/ggml-llama.metallib

  rm -rf ./*
}

rm -rf build-ios
mkdir -p build-ios

# Build iOS frameworks
build_framework "iOS" "arm64;x86_64" "iphonesimulator" "ios-arm64_x86_64-simulator" "build-ios"
build_framework "iOS" "arm64" "iphoneos" "ios-arm64" "build-ios"

cd ..
rm -rf build-ios

rm -rf build-tvos
mkdir -p build-tvos

# Build tvOS frameworks
build_framework "tvOS" "arm64;x86_64" "appletvsimulator" "tvos-arm64_x86_64-simulator" "build-tvos"
build_framework "tvOS" "arm64" "appletvos" "tvos-arm64" "build-tvos"

cd ..
rm -rf build-tvos
