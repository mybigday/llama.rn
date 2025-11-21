#!/bin/bash -e

# Build configuration
NDK_VERSION=27.3.13750724
ANDROID_PLATFORM=android-21
CMAKE_BUILD_TYPE=Release

# Hexagon SDK configuration
HEXAGON_SDK_VERSION="6.4.0.2"
HEXAGON_TOOLS_VERSION="19.0.04"
HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"

# Auto-detect Android SDK/NDK location if not set
if [ -z "$ANDROID_HOME" ]; then
  # Try common locations (Docker uses /opt/android-ndk-r28b)
  for location in "/opt/android-sdk" "/opt/android" "/android-sdk" "$HOME/Android/Sdk" "$HOME/Library/Android/sdk"; do
    if [ -d "$location" ]; then
      export ANDROID_HOME="$location"
      echo "Auto-detected ANDROID_HOME: $ANDROID_HOME"
      break
    fi
  done

  if [ -z "$ANDROID_HOME" ]; then
    echo "Error: ANDROID_HOME not set and could not auto-detect Android SDK/NDK"
    exit 1
  fi
fi

# Find NDK - handle both standalone NDK and SDK with NDK subdirectory
if [ -f "$ANDROID_HOME/build/cmake/android.toolchain.cmake" ]; then
  # ANDROID_HOME points directly to NDK (e.g., /opt/android-ndk-r28b)
  CMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/build/cmake/android.toolchain.cmake"
  echo "Using standalone Android NDK: $ANDROID_HOME"
elif [ -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
  # ANDROID_HOME points to SDK with NDK subdirectory
  CMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake"
  echo "Using Android NDK: $ANDROID_HOME/ndk/$NDK_VERSION"
else
  # Try to find any available NDK version
  echo "NDK $NDK_VERSION not found at $ANDROID_HOME/ndk/$NDK_VERSION"

  if [ -d "$ANDROID_HOME/ndk" ]; then
    AVAILABLE_NDKS=$(ls "$ANDROID_HOME/ndk" 2>/dev/null | head -5)
    if [ -n "$AVAILABLE_NDKS" ]; then
      echo "Available NDK versions:"
      echo "$AVAILABLE_NDKS"
      # Use the latest available NDK
      NDK_VERSION=$(ls "$ANDROID_HOME/ndk" | sort -V | tail -n 1)
      echo "Using NDK version: $NDK_VERSION"
      CMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake"
    else
      echo "No NDK found in $ANDROID_HOME/ndk"
      exit 1
    fi
  else
    echo "NDK not found. Expected either:"
    echo "  - Standalone NDK: $ANDROID_HOME/build/cmake/android.toolchain.cmake"
    echo "  - SDK with NDK: $ANDROID_HOME/ndk/$NDK_VERSION/"
    exit 1
  fi
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
OS_TYPE="unknown"
if uname -a | grep -q "Darwin"; then
  n_cpu=$(sysctl -n hw.logicalcpu)
  OS_TYPE="Darwin"
elif uname -a | grep -q "Linux"; then
  n_cpu=$(nproc)
  OS_TYPE="Linux"
fi

# Auto-detect Hexagon SDK on all platforms
if [ -z "$HEXAGON_SDK_ROOT" ]; then
  # Try to auto-detect SDK installation at default location
  if [ -d "$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION" ]; then
    echo "Auto-detected Hexagon SDK at $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
    export HEXAGON_SDK_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
    export HEXAGON_TOOLS_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION"
  fi
fi

# Set default build variables if not already set
export DEFAULT_HLOS_ARCH="${DEFAULT_HLOS_ARCH:-64}"
export DEFAULT_TOOLS_VARIANT="${DEFAULT_TOOLS_VARIANT:-toolv19}"
export DEFAULT_NO_QURT_INC="${DEFAULT_NO_QURT_INC:-0}"

# Display platform-specific build info
if [ -n "$HEXAGON_SDK_ROOT" ] && [ -d "$HEXAGON_SDK_ROOT" ]; then
  echo "Building on $OS_TYPE with Hexagon SDK - Hexagon backend enabled"
  echo "SDK location: $HEXAGON_SDK_ROOT"
else
  echo "Building on $OS_TYPE - Hexagon SDK not detected"
  echo "Hexagon backend will be skipped"
  echo ""
  echo "To enable Hexagon:"
  echo "  1. Run 'npm run bootstrap' to install SDK"
  echo "  2. Or set HEXAGON_INSTALL_DIR=/path/to/sdk"
  if [ "$OS_TYPE" = "Darwin" ]; then
    echo "  3. Note: HTP libraries must be built in Docker first"
  fi
fi
echo ""

t0=$(date +%s)

# Find strip tool - handle both standalone NDK and SDK with NDK subdirectory
if [ -f "$ANDROID_HOME/build/cmake/android.toolchain.cmake" ]; then
  # Standalone NDK
  STRIP=$ANDROID_HOME/toolchains/llvm/prebuilt/*/bin/llvm-strip
else
  # SDK with NDK subdirectory
  STRIP=$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/*/bin/llvm-strip
fi

cd android/src/main/rnllama

# Build the Android library (arm64-v8a)
echo "Building arm64-v8a prebuilt shared libraries..."
$CMAKE_PATH -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
  -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT \
  -B build-arm64

$CMAKE_PATH --build build-arm64 --config Release -j $n_cpu

# Strip debug symbols from libraries
for lib in build-arm64/*.so; do
  echo "Stripping $(basename $lib)..."
  $STRIP $lib
done

mkdir -p ../jniLibs/arm64-v8a

# Copy the shared libraries
cp build-arm64/*.so ../jniLibs/arm64-v8a/

rm -rf build-arm64

# Build the Android library (x86_64)
echo "Building x86_64 prebuilt shared libraries..."
$CMAKE_PATH -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DANDROID_ABI=x86_64 \
  -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -B build-x86_64

$CMAKE_PATH --build build-x86_64 --config Release -j $n_cpu

# Strip debug symbols from libraries
STRIP=$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/*/bin/llvm-strip
for lib in build-x86_64/*.so; do
  echo "Stripping $(basename $lib)..."
  $STRIP $lib
done

mkdir -p ../jniLibs/x86_64

# Strip debug symbols from libraries
STRIP=$ANDROID_HOME/ndk/$NDK_VERSION/toolchains/llvm/prebuilt/*/bin/llvm-strip
for lib in build-x86_64/*.so; do
  echo "Stripping $(basename $lib)..."
  $STRIP $lib
done

# Copy the shared libraries
cp build-x86_64/*.so ../jniLibs/x86_64/

rm -rf build-x86_64

t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
echo "Prebuilt shared libraries (rnllama APIs only) are in android/src/main/jniLibs/"
