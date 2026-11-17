#!/bin/bash

set -euo pipefail

if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it"
  exit 1
fi

# Usage: scripts/build-ios.sh [slice...]
#
# Builds ios/rnllama.xcframework from the named slices, all of them by default.
# Each slice compiles the whole library, so CI builds them as separate jobs; an
# xcframework made from a subset only serves the platforms of those slices.
ALL_SLICES=(ios-simulator ios-device tvos-simulator tvos-device)

# <system name>|<architectures>|<sysroot>|<xcframework library identifier>
slice_config() {
  case "$1" in
    ios-simulator)  echo "iOS|arm64;x86_64|iphonesimulator|ios-arm64_x86_64-simulator" ;;
    ios-device)     echo "iOS|arm64|iphoneos|ios-arm64" ;;
    tvos-simulator) echo "tvOS|arm64;x86_64|appletvsimulator|tvos-arm64_x86_64-simulator" ;;
    tvos-device)    echo "tvOS|arm64|appletvos|tvos-arm64" ;;
    *)
      echo "Unknown slice '$1' (expected: ${ALL_SLICES[*]})" >&2
      return 1
      ;;
  esac
}

slice_library_identifier() {
  slice_config "$1" | cut -d'|' -f4
}

if [[ $# -gt 0 ]]; then
  SLICES=("$@")
else
  SLICES=("${ALL_SLICES[@]}")
fi
for slice in "${SLICES[@]}"; do
  slice_config "$slice" > /dev/null
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/ios/rnllama.xcframework"
BUILD_ROOT="$ROOT_DIR/.build-ios-xcframework"
STAGING_DIR="$BUILD_ROOT/staging"

# CMake's Xcode generator ignores compiler launchers, so ccache only works by
# standing in for the compiler: when its compiler symlinks come first on PATH,
# make Xcode resolve the compilers through PATH instead of its toolchain.
XCODEBUILD_ARGS=()
if [[ "$(command -v clang)" == *ccache* ]]; then
  echo "Using ccache via $(command -v clang)"
  XCODEBUILD_ARGS=(-- CC=clang CPLUSPLUS=clang++ LD=clang LDPLUSPLUS=clang++ COMPILER_INDEX_STORE_ENABLE=NO)
fi

cleanup() {
  rm -rf "$BUILD_ROOT"
}

trap cleanup EXIT

copy_headers() {
  local framework_path="$1"
  local headers="$framework_path/Headers"
  local llama_cpp="$ROOT_DIR/vendor/llama.cpp"

  # Flat layout: the JSI glue includes these as <rnllama/name.h>.
  mkdir -p "$headers"
  cp "$ROOT_DIR"/cpp/*.h "$headers/"
  cp "$llama_cpp"/include/*.h "$headers/"
  cp "$llama_cpp"/src/*.h "$headers/"
  cp "$llama_cpp"/ggml/include/*.h "$headers/"
  cp "$llama_cpp"/ggml/src/*.h "$headers/"

  mkdir -p "$headers/jinja"
  cp "$llama_cpp"/common/jinja/*.h "$headers/jinja/"

  mkdir -p "$headers/nlohmann"
  cp "$llama_cpp"/vendor/nlohmann/*.hpp "$headers/nlohmann/"

  # common/ headers the glue needs, at the root (included without a path prefix)
  local h
  for h in chat.h common.h sampling.h speculative.h json.h json-schema-to-grammar.h json-schema.h peg-parser.h; do
    cp "$llama_cpp/common/$h" "$headers/"
  done
}

copy_framework_support_files() {
  local framework_path="$1"

  copy_headers "$framework_path"
  # Metal sources are not shipped as framework resources: each split kernel is
  # embedded via vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-embed-*.s.
}

# ggml/gguf keep upstream names; they must stay internal to this framework so a
# second ggml-based framework in the same app never resolves through it (see
# ios/unexported-symbols.txt).
assert_no_ggml_exports() {
  local binary="$1"
  local leaked

  leaked="$(nm -gU "$binary" | awk '{print $3}' | grep -E '^_(ggml|gguf|quantize|dequantize|iq2xs|iq3xs)_|^__Z[A-Z]*[0-9]+(ggml|gguf)_|^__Z[A-Z]*N4ggml' || true)"
  if [[ -n "$leaked" ]]; then
    echo "ggml symbols exported from $binary:" >&2
    echo "$leaked" | head -20 >&2
    exit 1
  fi
}

assert_matching_dsym() {
  local framework_path="$1"
  local dsym_path="$2"
  local binary_uuids
  local dsym_uuids

  if [[ ! -d "$dsym_path" ]]; then
    echo "Missing dSYM bundle for $framework_path" >&2
    exit 1
  fi

  binary_uuids="$(dwarfdump --uuid "$framework_path/rnllama" | awk '{print $2 ":" $3}' | sort)"
  dsym_uuids="$(dwarfdump --uuid "$dsym_path/Contents/Resources/DWARF/rnllama" | awk '{print $2 ":" $3}' | sort)"

  if [[ "$binary_uuids" != "$dsym_uuids" ]]; then
    echo "dSYM UUID mismatch for $framework_path" >&2
    echo "Framework UUIDs:" >&2
    echo "$binary_uuids" >&2
    echo "dSYM UUIDs:" >&2
    echo "$dsym_uuids" >&2
    exit 1
  fi
}

build_framework_slice() {
  local slice="$1"
  local system_name architectures sysroot library_identifier
  IFS='|' read -r system_name architectures sysroot library_identifier <<< "$(slice_config "$slice")"
  local build_dir="$BUILD_ROOT/$slice"
  local release_dir="$build_dir/Release-$sysroot"
  local framework_path="$release_dir/rnllama.framework"
  local dsym_path="$release_dir/rnllama.framework.dSYM"
  local staged_dir="$STAGING_DIR/$library_identifier"

  rm -rf "$build_dir" "$staged_dir"
  mkdir -p "$build_dir" "$staged_dir"

  (
    cd "$build_dir"

    cmake "$ROOT_DIR/ios" \
      -GXcode \
      -DCMAKE_SYSTEM_NAME="$system_name" \
      -DCMAKE_OSX_ARCHITECTURES="$architectures" \
      -DCMAKE_OSX_SYSROOT="$sysroot" \
      -DCMAKE_INSTALL_PREFIX="$PWD/install" \
      -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
      -DCMAKE_IOS_INSTALL_COMBINED=YES

    # (the expansion is spelled this way because macOS's bash 3.2 treats an
    # empty array as unset under `set -u`)
    cmake --build . --config Release -j "$(sysctl -n hw.logicalcpu)" \
      ${XCODEBUILD_ARGS[@]+"${XCODEBUILD_ARGS[@]}"}
  )

  if [[ ! -d "$framework_path" ]]; then
    echo "Missing framework build output at $framework_path" >&2
    exit 1
  fi

  assert_matching_dsym "$framework_path" "$dsym_path"
  assert_no_ggml_exports "$framework_path/rnllama"

  ditto "$framework_path" "$staged_dir/rnllama.framework"
  copy_framework_support_files "$staged_dir/rnllama.framework"
  ditto "$dsym_path" "$staged_dir/rnllama.framework.dSYM"
}

validate_packaged_dsym() {
  local library_identifier="$1"
  local packaged_framework="$OUTPUT_DIR/$library_identifier/rnllama.framework"
  local packaged_dsym

  packaged_dsym="$(find "$OUTPUT_DIR/$library_identifier" -maxdepth 3 -type d -name 'rnllama.framework.dSYM' -print -quit)"

  if [[ -z "$packaged_dsym" ]]; then
    echo "Missing packaged dSYM for $library_identifier in $OUTPUT_DIR" >&2
    exit 1
  fi

  assert_matching_dsym "$packaged_framework" "$packaged_dsym"
}

t0=$(date +%s)

rm -rf "$OUTPUT_DIR"
mkdir -p "$STAGING_DIR"

create_xcframework_args=()
for slice in "${SLICES[@]}"; do
  build_framework_slice "$slice"

  library_identifier="$(slice_library_identifier "$slice")"
  create_xcframework_args+=(
    -framework "$STAGING_DIR/$library_identifier/rnllama.framework"
    -debug-symbols "$STAGING_DIR/$library_identifier/rnllama.framework.dSYM"
  )
done

xcodebuild -create-xcframework "${create_xcframework_args[@]}" -output "$OUTPUT_DIR"

for slice in "${SLICES[@]}"; do
  validate_packaged_dsym "$(slice_library_identifier "$slice")"
done

t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
