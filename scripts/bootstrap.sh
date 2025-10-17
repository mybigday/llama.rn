#!/bin/bash -e

ROOT_DIR=$(pwd)

LLAMA_DIR=third_party/llama.cpp
CPP_DIR="$ROOT_DIR/cpp"
SRC_DIR="$ROOT_DIR/src"

git submodule init "$LLAMA_DIR"
git submodule update --recursive "$LLAMA_DIR"

# ggml api
cp ./$LLAMA_DIR/ggml/include/ggml.h ./cpp/ggml.h
cp ./$LLAMA_DIR/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./$LLAMA_DIR/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./$LLAMA_DIR/ggml/include/ggml-cpu.h ./cpp/ggml-cpu.h
cp ./$LLAMA_DIR/ggml/include/ggml-cpp.h ./cpp/ggml-cpp.h
cp ./$LLAMA_DIR/ggml/include/ggml-opt.h ./cpp/ggml-opt.h
cp ./$LLAMA_DIR/ggml/include/ggml-metal.h ./cpp/ggml-metal.h
cp ./$LLAMA_DIR/ggml/include/ggml-opencl.h ./cpp/ggml-opencl.h
cp ./$LLAMA_DIR/ggml/include/ggml-blas.h ./cpp/ggml-blas.h
cp ./$LLAMA_DIR/ggml/include/gguf.h ./cpp/gguf.h

cp -r ./$LLAMA_DIR/ggml/src/ggml-metal ./cpp/
rm ./cpp/ggml-metal/CMakeLists.txt
rm ./cpp/ggml-metal/ggml-metal.metal

cp -r ./$LLAMA_DIR/ggml/src/ggml-blas ./cpp/
rm ./cpp/ggml-blas/CMakeLists.txt

cp -r ./$LLAMA_DIR/ggml/src/ggml-opencl ./cpp/
rm ./cpp/ggml-opencl/CMakeLists.txt

cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.c ./cpp/ggml-cpu/ggml-cpu.c
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.cpp ./cpp/ggml-cpu/ggml-cpu.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu-impl.h ./cpp/ggml-cpu/ggml-cpu-impl.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/quants.h ./cpp/ggml-cpu/quants.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/quants.c ./cpp/ggml-cpu/quants.c
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/arch-fallback.h ./cpp/ggml-cpu/arch-fallback.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/repack.cpp ./cpp/ggml-cpu/repack.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/repack.h ./cpp/ggml-cpu/repack.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/traits.h ./cpp/ggml-cpu/traits.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/traits.cpp ./cpp/ggml-cpu/traits.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/common.h ./cpp/ggml-cpu/common.h

cp ./$LLAMA_DIR/ggml/src/ggml-cpu/unary-ops.h ./cpp/ggml-cpu/unary-ops.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/unary-ops.cpp ./cpp/ggml-cpu/unary-ops.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/binary-ops.h ./cpp/ggml-cpu/binary-ops.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/binary-ops.cpp ./cpp/ggml-cpu/binary-ops.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/vec.h ./cpp/ggml-cpu/vec.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/vec.cpp ./cpp/ggml-cpu/vec.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/simd-mappings.h ./cpp/ggml-cpu/simd-mappings.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ops.h ./cpp/ggml-cpu/ops.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ops.cpp ./cpp/ggml-cpu/ops.cpp

cp -r ./$LLAMA_DIR/ggml/src/ggml-cpu/amx ./cpp/ggml-cpu/
mkdir -p ./cpp/ggml-cpu/arch
cp -r ./$LLAMA_DIR/ggml/src/ggml-cpu/arch/arm ./cpp/ggml-cpu/arch/
cp -r ./$LLAMA_DIR/ggml/src/ggml-cpu/arch/x86 ./cpp/ggml-cpu/arch/

cp ./$LLAMA_DIR/ggml/src/ggml.c ./cpp/ggml.c
cp ./$LLAMA_DIR/ggml/src/ggml-impl.h ./cpp/ggml-impl.h
cp ./$LLAMA_DIR/ggml/src/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./$LLAMA_DIR/ggml/src/ggml-backend.cpp ./cpp/ggml-backend.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-backend-impl.h ./cpp/ggml-backend-impl.h
cp ./$LLAMA_DIR/ggml/src/ggml-backend-reg.cpp ./cpp/ggml-backend-reg.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-common.h ./cpp/ggml-common.h
cp ./$LLAMA_DIR/ggml/src/ggml-opt.cpp ./cpp/ggml-opt.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-quants.h ./cpp/ggml-quants.h
cp ./$LLAMA_DIR/ggml/src/ggml-quants.c ./cpp/ggml-quants.c
cp ./$LLAMA_DIR/ggml/src/ggml-threading.cpp ./cpp/ggml-threading.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-threading.h ./cpp/ggml-threading.h
cp ./$LLAMA_DIR/ggml/src/gguf.cpp ./cpp/gguf.cpp

# llama api
cp ./$LLAMA_DIR/include/llama.h ./cpp/llama.h
cp ./$LLAMA_DIR/include/llama-cpp.h ./cpp/llama-cpp.h
cp ./$LLAMA_DIR/src/llama.cpp ./cpp/llama.cpp
cp ./$LLAMA_DIR/src/llama-chat.h ./cpp/llama-chat.h
cp ./$LLAMA_DIR/src/llama-chat.cpp ./cpp/llama-chat.cpp
cp ./$LLAMA_DIR/src/llama-context.h ./cpp/llama-context.h
cp ./$LLAMA_DIR/src/llama-context.cpp ./cpp/llama-context.cpp
cp ./$LLAMA_DIR/src/llama-mmap.h ./cpp/llama-mmap.h
cp ./$LLAMA_DIR/src/llama-mmap.cpp ./cpp/llama-mmap.cpp
cp ./$LLAMA_DIR/src/llama-model-loader.h ./cpp/llama-model-loader.h
cp ./$LLAMA_DIR/src/llama-model-loader.cpp ./cpp/llama-model-loader.cpp
cp ./$LLAMA_DIR/src/llama-model-saver.h ./cpp/llama-model-saver.h
cp ./$LLAMA_DIR/src/llama-model-saver.cpp ./cpp/llama-model-saver.cpp
cp ./$LLAMA_DIR/src/llama-model.h ./cpp/llama-model.h
cp ./$LLAMA_DIR/src/llama-model.cpp ./cpp/llama-model.cpp
cp ./$LLAMA_DIR/src/llama-kv-cells.h ./cpp/llama-kv-cells.h
cp ./$LLAMA_DIR/src/llama-kv-cache.h ./cpp/llama-kv-cache.h
cp ./$LLAMA_DIR/src/llama-kv-cache.cpp ./cpp/llama-kv-cache.cpp
cp ./$LLAMA_DIR/src/llama-kv-cache-iswa.h ./cpp/llama-kv-cache-iswa.h
cp ./$LLAMA_DIR/src/llama-kv-cache-iswa.cpp ./cpp/llama-kv-cache-iswa.cpp
cp ./$LLAMA_DIR/src/llama-memory-hybrid.h ./cpp/llama-memory-hybrid.h
cp ./$LLAMA_DIR/src/llama-memory-hybrid.cpp ./cpp/llama-memory-hybrid.cpp
cp ./$LLAMA_DIR/src/llama-memory-recurrent.h ./cpp/llama-memory-recurrent.h
cp ./$LLAMA_DIR/src/llama-memory-recurrent.cpp ./cpp/llama-memory-recurrent.cpp
cp ./$LLAMA_DIR/src/llama-adapter.h ./cpp/llama-adapter.h
cp ./$LLAMA_DIR/src/llama-adapter.cpp ./cpp/llama-adapter.cpp
cp ./$LLAMA_DIR/src/llama-arch.h ./cpp/llama-arch.h
cp ./$LLAMA_DIR/src/llama-arch.cpp ./cpp/llama-arch.cpp
cp ./$LLAMA_DIR/src/llama-batch.h ./cpp/llama-batch.h
cp ./$LLAMA_DIR/src/llama-batch.cpp ./cpp/llama-batch.cpp
cp ./$LLAMA_DIR/src/llama-cparams.h ./cpp/llama-cparams.h
cp ./$LLAMA_DIR/src/llama-cparams.cpp ./cpp/llama-cparams.cpp
cp ./$LLAMA_DIR/src/llama-hparams.h ./cpp/llama-hparams.h
cp ./$LLAMA_DIR/src/llama-hparams.cpp ./cpp/llama-hparams.cpp
cp ./$LLAMA_DIR/src/llama-impl.h ./cpp/llama-impl.h
cp ./$LLAMA_DIR/src/llama-impl.cpp ./cpp/llama-impl.cpp

cp ./$LLAMA_DIR/src/llama-vocab.h ./cpp/llama-vocab.h
cp ./$LLAMA_DIR/src/llama-vocab.cpp ./cpp/llama-vocab.cpp
cp ./$LLAMA_DIR/src/llama-grammar.h ./cpp/llama-grammar.h
cp ./$LLAMA_DIR/src/llama-grammar.cpp ./cpp/llama-grammar.cpp
cp ./$LLAMA_DIR/src/llama-sampling.h ./cpp/llama-sampling.h
cp ./$LLAMA_DIR/src/llama-sampling.cpp ./cpp/llama-sampling.cpp

cp ./$LLAMA_DIR/src/unicode.h ./cpp/unicode.h
cp ./$LLAMA_DIR/src/unicode.cpp ./cpp/unicode.cpp
cp ./$LLAMA_DIR/src/unicode-data.h ./cpp/unicode-data.h
cp ./$LLAMA_DIR/src/unicode-data.cpp ./cpp/unicode-data.cpp

cp ./$LLAMA_DIR/src/llama-graph.h ./cpp/llama-graph.h
cp ./$LLAMA_DIR/src/llama-graph.cpp ./cpp/llama-graph.cpp
cp ./$LLAMA_DIR/src/llama-io.h ./cpp/llama-io.h
cp ./$LLAMA_DIR/src/llama-io.cpp ./cpp/llama-io.cpp
cp ./$LLAMA_DIR/src/llama-memory.h ./cpp/llama-memory.h
cp ./$LLAMA_DIR/src/llama-memory.cpp ./cpp/llama-memory.cpp

cp ./$LLAMA_DIR/common/log.h ./cpp/log.h
cp ./$LLAMA_DIR/common/log.cpp ./cpp/log.cpp
cp ./$LLAMA_DIR/common/common.h ./cpp/common.h
cp ./$LLAMA_DIR/common/common.cpp ./cpp/common.cpp
cp ./$LLAMA_DIR/common/sampling.h ./cpp/sampling.h
cp ./$LLAMA_DIR/common/sampling.cpp ./cpp/sampling.cpp
cp ./$LLAMA_DIR/common/json-schema-to-grammar.h ./cpp/json-schema-to-grammar.h
cp ./$LLAMA_DIR/common/json-schema-to-grammar.cpp ./cpp/json-schema-to-grammar.cpp
cp ./$LLAMA_DIR/common/json-partial.h ./cpp/json-partial.h
cp ./$LLAMA_DIR/common/json-partial.cpp ./cpp/json-partial.cpp
cp ./$LLAMA_DIR/common/regex-partial.h ./cpp/regex-partial.h
cp ./$LLAMA_DIR/common/regex-partial.cpp ./cpp/regex-partial.cpp
cp ./$LLAMA_DIR/common/chat.h ./cpp/chat.h
cp ./$LLAMA_DIR/common/chat.cpp ./cpp/chat.cpp
cp ./$LLAMA_DIR/common/chat-parser.h ./cpp/chat-parser.h
cp ./$LLAMA_DIR/common/chat-parser.cpp ./cpp/chat-parser.cpp

# Copy multimodal files from tools/mtmd
rm -rf ./cpp/tools/mtmd
mkdir -p ./cpp/tools/mtmd
cp ./$LLAMA_DIR/tools/mtmd/mtmd.h ./cpp/tools/mtmd/mtmd.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd.cpp ./cpp/tools/mtmd/mtmd.cpp
cp ./$LLAMA_DIR/tools/mtmd/clip.h ./cpp/tools/mtmd/clip.h
cp ./$LLAMA_DIR/tools/mtmd/clip.cpp ./cpp/tools/mtmd/clip.cpp
cp ./$LLAMA_DIR/tools/mtmd/clip-impl.h ./cpp/tools/mtmd/clip-impl.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd-helper.cpp ./cpp/tools/mtmd/mtmd-helper.cpp
cp ./$LLAMA_DIR/tools/mtmd/mtmd-helper.h ./cpp/tools/mtmd/mtmd-helper.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd-audio.h ./cpp/tools/mtmd/mtmd-audio.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd-audio.cpp ./cpp/tools/mtmd/mtmd-audio.cpp

rm -rf ./cpp/minja
rm -rf ./cpp/nlohmann
cp -r ./$LLAMA_DIR/vendor/minja ./cpp/minja
cp -r ./$LLAMA_DIR/vendor/nlohmann ./cpp/nlohmann
rm -rf ./cpp/tools/mtmd/miniaudio
rm -rf ./cpp/tools/mtmd/stb
cp -r ./$LLAMA_DIR/vendor/miniaudio ./cpp/tools/mtmd/miniaudio
cp -r ./$LLAMA_DIR/vendor/stb ./cpp/tools/mtmd/stb


# List of files to process
files_add_lm_prefix=(
  # ggml api
  "./cpp/ggml-common.h"
  "./cpp/ggml.h"
  "./cpp/ggml.c"
  "./cpp/gguf.h"
  "./cpp/gguf.cpp"
  "./cpp/ggml-impl.h"
  "./cpp/ggml-cpp.h"
  "./cpp/ggml-opt.h"
  "./cpp/ggml-opt.cpp"
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
  "./cpp/ggml-backend.h"
  "./cpp/ggml-backend.cpp"
  "./cpp/ggml-backend-impl.h"
  "./cpp/ggml-backend-reg.cpp"
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal/ggml-metal.cpp"
  "./cpp/ggml-metal/ggml-metal-impl.h"
  "./cpp/ggml-metal/ggml-metal-common.h"
  "./cpp/ggml-metal/ggml-metal-common.cpp"
  "./cpp/ggml-metal/ggml-metal-context.h"
  "./cpp/ggml-metal/ggml-metal-context.m"
  "./cpp/ggml-metal/ggml-metal-device.h"
  "./cpp/ggml-metal/ggml-metal-device.cpp"
  "./cpp/ggml-metal/ggml-metal-device.m"
  "./cpp/ggml-metal/ggml-metal-ops.h"
  "./cpp/ggml-metal/ggml-metal-ops.cpp"
  "./cpp/ggml-blas.h"
  "./cpp/ggml-blas/ggml-blas.cpp"
  ."/cpp/ggml-opencl.h"
  "./cpp/ggml-opencl/ggml-opencl.cpp"
  "./cpp/ggml-cpu.h"
  "./cpp/ggml-cpu/ggml-cpu-impl.h"
  "./cpp/ggml-cpu/ggml-cpu.c"
  "./cpp/ggml-cpu/ggml-cpu.cpp"
  "./cpp/ggml-cpu/quants.h"
  "./cpp/ggml-cpu/quants.c"
  "./cpp/ggml-cpu/traits.h"
  "./cpp/ggml-cpu/traits.cpp"
  "./cpp/ggml-cpu/arch-fallback.h"
  "./cpp/ggml-cpu/repack.cpp"
  "./cpp/ggml-cpu/repack.h"
  "./cpp/ggml-cpu/common.h"
  "./cpp/ggml-threading.h"
  "./cpp/ggml-threading.cpp"
  "./cpp/ggml-cpu/amx/amx.h"
  "./cpp/ggml-cpu/amx/amx.cpp"
  "./cpp/ggml-cpu/amx/mmq.h"
  "./cpp/ggml-cpu/amx/mmq.cpp"
  "./cpp/ggml-cpu/amx/common.h"
  "./cpp/ggml-cpu/unary-ops.h"
  "./cpp/ggml-cpu/unary-ops.cpp"
  "./cpp/ggml-cpu/binary-ops.h"
  "./cpp/ggml-cpu/binary-ops.cpp"
  "./cpp/ggml-cpu/vec.h"
  "./cpp/ggml-cpu/vec.cpp"
  "./cpp/ggml-cpu/simd-mappings.h"
  "./cpp/ggml-cpu/ops.h"
  "./cpp/ggml-cpu/ops.cpp"
  "./cpp/ggml-cpu/arch/arm/cpu-feats.cpp"
  "./cpp/ggml-cpu/arch/arm/quants.c"
  "./cpp/ggml-cpu/arch/arm/repack.cpp"
  "./cpp/ggml-cpu/arch/x86/cpu-feats.cpp"
  "./cpp/ggml-cpu/arch/x86/quants.c"
  "./cpp/ggml-cpu/arch/x86/repack.cpp"

  # llama api
    "./cpp/llama-impl.h"
  "./cpp/llama-impl.cpp"
  "./cpp/llama-vocab.h"
  "./cpp/llama-vocab.cpp"
  "./cpp/llama-grammar.h"
  "./cpp/llama-grammar.cpp"
  "./cpp/llama-sampling.h"
  "./cpp/llama-sampling.cpp"
  "./cpp/llama-adapter.h"
  "./cpp/llama-adapter.cpp"
  "./cpp/llama-arch.h"
  "./cpp/llama-arch.cpp"
  "./cpp/llama-batch.h"
  "./cpp/llama-batch.cpp"
  "./cpp/llama-chat.h"
  "./cpp/llama-chat.cpp"
  "./cpp/llama-context.h"
  "./cpp/llama-context.cpp"
  "./cpp/llama-model-loader.h"
  "./cpp/llama-model-loader.cpp"
  "./cpp/llama-model-saver.h"
  "./cpp/llama-model-saver.cpp"
  "./cpp/llama-model.h"
  "./cpp/llama-model.cpp"
  "./cpp/llama-kv-cache.h"
  "./cpp/llama-kv-cache.cpp"
  "./cpp/llama-kv-cache-iswa.h"
  "./cpp/llama-kv-cache-iswa.cpp"
  "./cpp/llama-memory-hybrid.h"
  "./cpp/llama-memory-hybrid.cpp"
  "./cpp/llama-memory-recurrent.h"
  "./cpp/llama-memory-recurrent.cpp"
  "./cpp/llama-mmap.h"
  "./cpp/llama-mmap.cpp"
  "./cpp/llama-hparams.h"
  "./cpp/llama-hparams.cpp"
  "./cpp/llama-cparams.h"
  "./cpp/llama-cparams.cpp"
  "./cpp/llama-graph.h"
  "./cpp/llama-graph.cpp"
  "./cpp/llama-io.h"
  "./cpp/llama-io.cpp"
  "./cpp/llama-memory.h"
  "./cpp/llama-memory.cpp"
  "./cpp/log.h"
  "./cpp/log.cpp"
  "./cpp/llama.h"
  "./cpp/llama.cpp"
  "./cpp/sampling.cpp"
  "./cpp/common.h"
  "./cpp/common.cpp"
  "./cpp/chat.h"
  "./cpp/chat.cpp"
  "./cpp/chat-parser.h"
  "./cpp/chat-parser.cpp"
  "./cpp/json-schema-to-grammar.h"
  "./cpp/json-schema-to-grammar.cpp"
  "./cpp/json-partial.h"
  "./cpp/json-partial.cpp"

  # Multimodal files
  "./cpp/tools/mtmd/mtmd.h"
  "./cpp/tools/mtmd/mtmd.cpp"
  "./cpp/tools/mtmd/clip.h"
  "./cpp/tools/mtmd/clip.cpp"
  "./cpp/tools/mtmd/clip-impl.h"
  "./cpp/tools/mtmd/mtmd-helper.cpp"
  "./cpp/tools/mtmd/mtmd-audio.h"
  "./cpp/tools/mtmd/mtmd-audio.cpp"
)

# Loop through each file and run the sed commands
OS=$(uname)
for file in "${files_add_lm_prefix[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/GGML_/LM_GGML_/g' $file
    sed -i '' 's/ggml_/lm_ggml_/g' $file
    sed -i '' 's/GGUF_/LM_GGUF_/g' $file
    sed -i '' 's/gguf_/lm_gguf_/g' $file
    sed -i '' 's/GGMLMetalClass/LMGGMLMetalClass/g' $file

    # <nlohmann/json.hpp> -> "nlohmann/json.hpp"
    sed -i '' 's/<nlohmann\/json.hpp>/"nlohmann\/json.hpp"/g' $file

    # <nlohmann/json_fwd.hpp> -> "nlohmann/json_fwd.hpp"
    sed -i '' 's/<nlohmann\/json_fwd.hpp>/"nlohmann\/json_fwd.hpp"/g' $file
  else
    sed -i 's/GGML_/LM_GGML_/g' $file
    sed -i 's/ggml_/lm_ggml_/g' $file
    sed -i 's/GGUF_/LM_GGUF_/g' $file
    sed -i 's/gguf_/lm_gguf_/g' $file
    sed -i 's/GGMLMetalClass/LMGGMLMetalClass/g' $file

    # <nlohmann/json.hpp> -> "nlohmann/json.hpp"
    sed -i 's/<nlohmann\/json.hpp>/"nlohmann\/json.hpp"/g' $file

    # <nlohmann/json_fwd.hpp> -> "nlohmann/json_fwd.hpp"
    sed -i 's/<nlohmann\/json_fwd.hpp>/"nlohmann\/json_fwd.hpp"/g' $file
  fi
done

files_iq_add_lm_prefix=(
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml.c"
)

for file in "${files_iq_add_lm_prefix[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' $file
    sed -i '' 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' $file
    sed -i '' 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' $file
    sed -i '' 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' $file
  else
    sed -i 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' $file
    sed -i 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' $file
    sed -i 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' $file
    sed -i 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' $file
  fi
done

echo "Replacement completed successfully!"

cd example && npm install && cd ..

# Apply patch
patch -p0 -d ./cpp < ./scripts/patches/common.h.patch
patch -p0 -d ./cpp < ./scripts/patches/common.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/chat.h.patch
patch -p0 -d ./cpp < ./scripts/patches/chat.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/log.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml.c.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml-quants.c.patch
patch -p0 -d ./cpp < ./scripts/patches/llama-mmap.cpp.patch
patch -p0 -d ./cpp/minja < ./scripts/patches/minja.hpp.patch
patch -p0 -d ./cpp/minja < ./scripts/patches/chat-template.hpp.patch
patch -p0 -d ./cpp/ggml-metal < ./scripts/patches/ggml-metal-device.m.patch
rm -rf ./cpp/*.orig
rm -rf ./cpp/**/*.orig

if [ "$OS" = "Darwin" ]; then
  # Build metallib (~2.6MB)
  cd "$LLAMA_DIR/ggml/src/ggml-metal"

  # Create a symbolic link to ggml-common.h in the current directory
  ln -sf ../ggml-common.h .

  xcrun --sdk iphoneos metal -O3 -std=metal3.2 -mios-version-min=16.0 -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_HAS_BF16=1
  xcrun --sdk iphoneos metallib ggml-metal.air -o ggml-llama.metallib
  rm ggml-metal.air
  mv ./ggml-llama.metallib "$CPP_DIR/ggml-metal/ggml-llama.metallib"

  xcrun --sdk iphonesimulator metal -O3 -std=metal3.2 -mios-version-min=16.0 -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_HAS_BF16=1
  xcrun --sdk iphonesimulator metallib ggml-metal.air -o ggml-llama.metallib
  rm ggml-metal.air
  mv ./ggml-llama.metallib "$CPP_DIR/ggml-metal/ggml-llama-sim.metallib"

  # Remove the symbolic link
  rm ggml-common.h

  cd -

  # Generate .xcode.env.local in iOS example
  cd example/ios
  echo export NODE_BINARY=$(command -v node) > .xcode.env.local

  cd -
fi

# Get version info
cd "$LLAMA_DIR"
BUILD_NUMBER=$(git rev-list --count HEAD)
BUILD_COMMIT=$(git rev-parse --short=7 HEAD)

# Put to ../version.ts
# clean up version.ts
rm -f "$SRC_DIR/version.ts"

echo "export const BUILD_NUMBER = '$BUILD_NUMBER';" > "$SRC_DIR/version.ts"
echo "export const BUILD_COMMIT = '$BUILD_COMMIT';" >> "$SRC_DIR/version.ts"

cd "$ROOT_DIR"
