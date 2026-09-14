#!/bin/bash
#
# Re-vendors vendor/ from the pins in vendor/VERSIONS.
#
# For each dependency:
#   1. clone (once) or fetch the upstream repo into $LLAMA_RN_CACHE_DIR
#   2. export the subset llama.rn builds, keeping the upstream directory layout
#      and file contents untouched
#   3. apply scripts/patches/<dep>/*.patch (-p1, paths relative to the tree)
#   4. regenerate the version files upstream normally produces at build time
#
# Everything this script writes is committed. Builds and `npm run bootstrap`
# never run it, so CI always compiles the tree as checked in. Run it after
# editing vendor/VERSIONS or scripts/patches/.
# No `set -u`: empty manifest arrays are expanded indirectly (bash 3.2 compatible).
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor"
PATCHES_DIR="$ROOT_DIR/scripts/patches"
CACHE_DIR="${LLAMA_RN_CACHE_DIR:-$HOME/.cache/llama.rn}"

# shellcheck source=../vendor/VERSIONS
source "$VENDOR_DIR/VERSIONS"

# ---------------------------------------------------------------------------
# Subset manifests. Paths are upstream pathspecs; a directory takes everything
# below it. Prefer listing files where upstream keeps unrelated siblings, so a
# new upstream file is opted in deliberately and a removed one fails loudly.
# ---------------------------------------------------------------------------

LLAMA_CPP_PATHS=(
  LICENSE

  include

  # src/ minus llama-quant.* (see LLAMA_CPP_PRUNE)
  src

  ggml/include
  ggml/src/ggml.c
  ggml/src/ggml-alloc.c
  ggml/src/ggml-backend.cpp
  ggml/src/ggml-backend-dl.cpp
  ggml/src/ggml-backend-dl.h
  ggml/src/ggml-backend-impl.h
  ggml/src/ggml-backend-meta.cpp
  ggml/src/ggml-backend-reg.cpp
  ggml/src/ggml-common.h
  ggml/src/ggml-feats.h
  ggml/src/ggml-impl.h
  ggml/src/ggml-opt.cpp
  ggml/src/ggml-quants.c
  ggml/src/ggml-quants.h
  ggml/src/ggml-threading.cpp
  ggml/src/ggml-threading.h
  ggml/src/ggml-version.h.in
  ggml/src/gguf.cpp

  ggml/src/ggml-cpu/arch-fallback.h
  ggml/src/ggml-cpu/binary-ops.cpp
  ggml/src/ggml-cpu/binary-ops.h
  ggml/src/ggml-cpu/common.h
  ggml/src/ggml-cpu/ggml-cpu-impl.h
  ggml/src/ggml-cpu/ggml-cpu.c
  ggml/src/ggml-cpu/ggml-cpu.cpp
  ggml/src/ggml-cpu/iqp.cpp
  ggml/src/ggml-cpu/iqp.h
  ggml/src/ggml-cpu/ops.cpp
  ggml/src/ggml-cpu/ops.h
  ggml/src/ggml-cpu/quants.c
  ggml/src/ggml-cpu/quants.h
  ggml/src/ggml-cpu/repack.cpp
  ggml/src/ggml-cpu/repack.h
  ggml/src/ggml-cpu/simd-gemm.h
  ggml/src/ggml-cpu/simd-mappings.h
  ggml/src/ggml-cpu/traits.cpp
  ggml/src/ggml-cpu/traits.h
  ggml/src/ggml-cpu/unary-ops.cpp
  ggml/src/ggml-cpu/unary-ops.h
  ggml/src/ggml-cpu/vec.cpp
  ggml/src/ggml-cpu/vec.h
  ggml/src/ggml-cpu/amx
  ggml/src/ggml-cpu/arch/arm
  ggml/src/ggml-cpu/arch/x86

  ggml/src/ggml-blas
  ggml/src/ggml-metal
  ggml/src/ggml-opencl
  ggml/src/ggml-hexagon

  common/build-info.cpp.in
  common/build-info.h
  common/chat-auto-parser-generator.cpp
  common/chat-auto-parser-helpers.cpp
  common/chat-auto-parser-helpers.h
  common/chat-auto-parser.h
  common/chat-diff-analyzer.cpp
  common/chat-peg-parser.cpp
  common/chat-peg-parser.h
  common/chat.cpp
  common/chat.h
  common/common.cpp
  common/common.h
  common/fit.cpp
  common/fit.h
  common/jinja
  common/json-schema-to-grammar.cpp
  common/json-schema-to-grammar.h
  common/json.cpp
  common/json.h
  common/log.cpp
  common/log.h
  common/ngram-cache.cpp
  common/ngram-cache.h
  common/ngram-map.cpp
  common/ngram-map.h
  common/ngram-mod.cpp
  common/ngram-mod.h
  common/peg-parser.cpp
  common/peg-parser.h
  common/reasoning-budget.cpp
  common/reasoning-budget.h
  common/sampling.cpp
  common/sampling.h
  common/speculative.cpp
  common/speculative.h
  common/trie.cpp
  common/trie.h
  common/unicode.cpp
  common/unicode.h

  tools/mtmd/clip-graph.h
  tools/mtmd/clip-impl.h
  tools/mtmd/clip-model.h
  tools/mtmd/clip.cpp
  tools/mtmd/clip.h
  tools/mtmd/debug
  tools/mtmd/models
  tools/mtmd/mtmd-audio.cpp
  tools/mtmd/mtmd-audio.h
  tools/mtmd/mtmd-helper-common.h
  tools/mtmd/mtmd-helper-gen.cpp
  tools/mtmd/mtmd-helper.cpp
  tools/mtmd/mtmd-helper.h
  tools/mtmd/mtmd-image.cpp
  tools/mtmd/mtmd-image.h
  tools/mtmd/mtmd-internal.h
  tools/mtmd/mtmd.cpp
  tools/mtmd/mtmd.h

  vendor/nlohmann
  vendor/miniaudio
  vendor/stb
  # mtmd hashes media with SHA-256 only; skip the unused hash engines
  vendor/hash/hash.cpp
  vendor/hash/hash.h
  vendor/hash/rotate-bits
  vendor/hash/sha256
)

# Exported by a directory pathspec above but not wanted. Upstream build files
# make no sense for a partial tree; ggml-hexagon keeps its own because it
# builds the DSP libraries (scripts/build-hexagon-htp.sh).
LLAMA_CPP_PRUNE=(
  src/llama-quant.cpp
  src/llama-quant.h
  src/CMakeLists.txt
  ggml/src/ggml-blas/CMakeLists.txt
  ggml/src/ggml-metal/CMakeLists.txt
  ggml/src/ggml-opencl/CMakeLists.txt
  vendor/miniaudio/CMakeLists.txt
  vendor/nlohmann/CMakeLists.txt
  vendor/stb/CMakeLists.txt
)

# Files that live inside the tree but are not upstream content: generated by
# this script (version files) or by builds (Metal embeds, HTP stubs). They are
# never deleted or overwritten by the export.
LLAMA_CPP_KEEP=(
  /ggml/src/ggml-version.h
  /src/llama-version.h
  /common/build-info.cpp
  /ggml/src/ggml-metal/ggml-metal-embed-*.s
  /ggml/src/ggml-hexagon/htp/v73/
)

CODEC_CPP_PATHS=(
  LICENSE
  include
  src
  # audio_lm.cpp + codec_common.h; tts_runner*.cpp are excluded by the builds
  common
  # wav/npy helpers used by the host test probes
  examples/utils
)
# codec.cpp's own llama.cpp submodule (exported as an empty directory)
CODEC_CPP_PRUNE=(
  common/third-party
)
CODEC_CPP_KEEP=()

OPENCL_HEADERS_PATHS=(
  LICENSE
  CL
)
OPENCL_HEADERS_PRUNE=()
OPENCL_HEADERS_KEEP=()

# Only what its CMake project needs to build libOpenCL.so (scripts/build-opencl.sh);
# README.md is referenced by its CPack config.
OPENCL_ICD_LOADER_PATHS=(
  LICENSE
  README.md
  CMakeLists.txt
  OpenCL.pc.in
  cmake
  include
  loader
)
OPENCL_ICD_LOADER_PRUNE=()
OPENCL_ICD_LOADER_KEEP=()

# ---------------------------------------------------------------------------

log() { printf '\n==> %s\n' "$*"; }

# Clone once into the cache, then only fetch when the pinned ref is unknown.
# A previous submodule checkout under .git/modules seeds the clone so nothing
# is downloaded twice.
ensure_repo() {
  local name="$1" url="$2" ref="$3"
  local repo="$CACHE_DIR/$name"

  if [ ! -d "$repo/.git" ]; then
    log "Cloning $url into $repo"
    mkdir -p "$CACHE_DIR"
    local seed="$ROOT_DIR/.git/modules/vendor/$name"
    if [ -d "$seed/objects" ]; then
      git clone --quiet --no-checkout --reference "$seed" --dissociate "$url" "$repo"
    else
      git clone --quiet --no-checkout "$url" "$repo"
    fi
  fi

  local commit=""
  if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then
    git -C "$repo" cat-file -e "$ref^{commit}" 2>/dev/null && commit="$ref"
  else
    commit="$(git -C "$repo" rev-parse -q --verify "refs/tags/$ref^{commit}" 2>/dev/null || true)"
  fi
  if [ -z "$commit" ]; then
    log "Fetching $ref from $url"
    git -C "$repo" fetch --quiet origin "$ref"
    commit="$(git -C "$repo" rev-parse --verify FETCH_HEAD^{commit})"
  fi
  RESOLVED_COMMIT="$commit"
}

# Export the pinned subset into vendor/<name>. Files upstream removed
# disappear; entries in KEEP survive.
export_subset() {
  local name="$1" commit="$2" prefix="$3"
  local paths_ref="${prefix}_PATHS[@]" prune_ref="${prefix}_PRUNE[@]" keep_ref="${prefix}_KEEP[@]"
  local paths=("${!paths_ref}") prune=("${!prune_ref}") keep=("${!keep_ref}")
  local repo="$CACHE_DIR/$name"
  local dest="$VENDOR_DIR/$name"
  local tmp
  tmp="$(mktemp -d)"

  git -C "$repo" archive --format=tar "$commit" "${paths[@]}" | tar -xf - -C "$tmp"

  local p
  for p in "${prune[@]}"; do
    rm -rf "${tmp:?}/$p"
  done
  local rsync_args=(-a --delete)
  for p in "${keep[@]}"; do
    rsync_args+=("--exclude=$p")
  done
  mkdir -p "$dest"
  rsync "${rsync_args[@]}" "$tmp/" "$dest/"
  rm -rf "$tmp"
}

apply_patches() {
  local name="$1"
  local dest="$VENDOR_DIR/$name"
  local dir="$PATCHES_DIR/$name"
  [ -d "$dir" ] || return 0

  local patch_file
  for patch_file in "$dir"/*.patch; do
    [ -e "$patch_file" ] || continue
    echo "  patch: $(basename "$patch_file")"
    patch -p1 -d "$dest" < "$patch_file"
  done
  find "$dest" \( -name '*.orig' -o -name '*.rej' \) -delete
}

# Rewrite <PREFIX>_COMMIT in VERSIONS so the pin is reproducible even if the
# ref was a branch or a tag that later moves.
record_commit() {
  local prefix="$1" commit="$2"
  local tmp
  tmp="$(mktemp)"
  sed "s/^${prefix}_COMMIT=.*/${prefix}_COMMIT=${commit}/" "$VENDOR_DIR/VERSIONS" > "$tmp"
  mv "$tmp" "$VENDOR_DIR/VERSIONS"
}

sync_dep() {
  local name="$1" prefix="$2"
  local repo_var="${prefix}_REPO" ref_var="${prefix}_REF"

  log "Syncing $name @ ${!ref_var}"
  ensure_repo "$name" "${!repo_var}" "${!ref_var}"
  echo "  commit: $RESOLVED_COMMIT"
  export_subset "$name" "$RESOLVED_COMMIT" "$prefix"
  apply_patches "$name"
  record_commit "$prefix" "$RESOLVED_COMMIT"
}

# llama.cpp derives these from its own CMake project and git history; llama.rn
# compiles the sources directly, so fill the upstream templates here.
generate_llama_cpp_version_files() {
  local repo="$CACHE_DIR/llama.cpp"
  local dest="$VENDOR_DIR/llama.cpp"
  local commit
  commit="$(sed -n 's/^LLAMA_CPP_COMMIT=//p' "$VENDOR_DIR/VERSIONS")"

  local build_number build_commit
  build_number="$(git -C "$repo" rev-list --count "$commit")"
  build_commit="$(git -C "$repo" rev-parse --short=7 "$commit")"

  cmake_version() {  # <CMakeLists.txt path in upstream> <PREFIX>
    local file="$1" prefix="$2" major minor patch
    local content
    content="$(git -C "$repo" show "$commit:$file")"
    major="$(sed -n "s/^set(${prefix}_VERSION_MAJOR \([0-9][0-9]*\))$/\1/p" <<< "$content")"
    minor="$(sed -n "s/^set(${prefix}_VERSION_MINOR \([0-9][0-9]*\))$/\1/p" <<< "$content")"
    patch="$(sed -n "s/^set(${prefix}_VERSION_PATCH \([0-9][0-9]*\))$/\1/p" <<< "$content")"
    if [ -z "$major" ] || [ -z "$minor" ] || [ -z "$patch" ]; then
      echo "Failed to read ${prefix}_VERSION_* from upstream $file" >&2
      exit 1
    fi
    echo "$major.$minor.$patch-dev"
  }
  local llama_version ggml_version
  llama_version="$(cmake_version CMakeLists.txt LLAMA)"
  ggml_version="$(cmake_version ggml/CMakeLists.txt GGML)"

  log "Generating version files (build $build_number, commit $build_commit)"
  sed -e "s|@LLAMA_VERSION@|$llama_version|g" \
      -e "s|@LLAMA_BUILD_COMMIT@|$build_commit|g" \
      "$dest/src/llama-version.h.in" > "$dest/src/llama-version.h"
  sed -e "s|@GGML_VERSION@|$ggml_version|g" \
      -e "s|@GGML_BUILD_COMMIT@|$build_commit|g" \
      "$dest/ggml/src/ggml-version.h.in" > "$dest/ggml/src/ggml-version.h"
  sed -e "s|@LLAMA_BUILD_NUMBER@|$build_number|g" \
      -e "s|@LLAMA_BUILD_COMMIT@|$build_commit|g" \
      -e "s|@BUILD_COMPILER@|unknown|g" \
      -e "s|@BUILD_TARGET@|unknown|g" \
      "$dest/common/build-info.cpp.in" > "$dest/common/build-info.cpp"

  cat > "$ROOT_DIR/src/version.ts" <<TS
export const BUILD_NUMBER = '$build_number'
export const BUILD_COMMIT = '$build_commit'
TS
}

sync_dep llama.cpp LLAMA_CPP
generate_llama_cpp_version_files
sync_dep codec.cpp CODEC_CPP
sync_dep OpenCL-Headers OPENCL_HEADERS
sync_dep OpenCL-ICD-Loader OPENCL_ICD_LOADER

log "Done. Review with: git status vendor src/version.ts"
