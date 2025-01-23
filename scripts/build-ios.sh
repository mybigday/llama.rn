#!/bin/bash

# Clean up the build directory
rm -rf build-ios
rm -rf build-tvos

function cp_headers() {
    mkdir -p ../ios/rnllama.xcframework/$1/rnllama.framework/Headers
    cp ../cpp/*.h ../ios/rnllama.xcframework/$1/rnllama.framework/Headers/
}

# Create build directory
mkdir -p build-ios
cd build-ios

# Configure CMake for iOS simulator
cmake ../ios \
    -GXcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DCMAKE_IOS_INSTALL_COMBINED=YES

# Build
cmake --build . --config Release

rm -rf ../ios/rnllama.xcframework/ios-arm64_x86_64-simulator/rnllama.framework
mv Release-iphoneos/rnllama.framework ../ios/rnllama.xcframework/ios-arm64_x86_64-simulator/
mkdir -p ../ios/rnllama.xcframework/ios-arm64_x86_64-simulator/rnllama.framework/Headers
cp_headers ios-arm64_x86_64-simulator

rm -rf ./*

cmake ../ios \
    -GXcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DCMAKE_IOS_INSTALL_COMBINED=YES

cmake --build . --config Release

rm -rf ../ios/rnllama.xcframework/ios-arm64/rnllama.framework
mv Release-iphoneos/rnllama.framework ../ios/rnllama.xcframework/ios-arm64/
mkdir -p ../ios/rnllama.xcframework/ios-arm64/rnllama.framework/Headers
cp_headers ios-arm64

cd ..

mkdir -p build-tvos
cd build-tvos

# Configure CMake for tvOS
cmake ../ios \
    -GXcode \
    -DCMAKE_SYSTEM_NAME=tvOS \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DCMAKE_IOS_INSTALL_COMBINED=YES

cmake --build . --config Release

rm -rf ../ios/rnllama.xcframework/tvos-arm64_x86_64-simulator/rnllama.framework
mv Release-appletvos/rnllama.framework ../ios/rnllama.xcframework/tvos-arm64_x86_64-simulator/
mkdir -p ../ios/rnllama.xcframework/tvos-arm64_x86_64-simulator/rnllama.framework/Headers
cp_headers tvos-arm64_x86_64-simulator

rm -rf ./*

cmake ../ios \
    -GXcode \
    -DCMAKE_SYSTEM_NAME=tvOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DCMAKE_IOS_INSTALL_COMBINED=YES

cmake --build . --config Release

rm -rf ../ios/rnllama.xcframework/tvos-arm64/rnllama.framework
mv Release-appletvos/rnllama.framework ../ios/rnllama.xcframework/tvos-arm64/
mkdir -p ../ios/rnllama.xcframework/tvos-arm64/rnllama.framework/Headers
cp_headers tvos-arm64
