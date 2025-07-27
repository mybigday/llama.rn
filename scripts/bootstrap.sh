#!/bin/bash -e

git submodule init
git submodule update --recursive

# ggml api
cp ./llama.cpp/ggml/include/ggml.h ./cpp/ggml.h
cp ./llama.cpp/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./llama.cpp/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./llama.cpp/ggml/include/ggml-cpu.h ./cpp/ggml-cpu.h
cp ./llama.cpp/ggml/include/ggml-cpp.h ./cpp/ggml-cpp.h
cp ./llama.cpp/ggml/include/ggml-opt.h ./cpp/ggml-opt.h
cp ./llama.cpp/ggml/include/ggml-metal.h ./cpp/ggml-metal.h
cp ./llama.cpp/ggml/include/gguf.h ./cpp/gguf.h

cp ./llama.cpp/ggml/src/ggml-metal/ggml-metal.m ./cpp/ggml-metal.m
cp ./llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h ./cpp/ggml-metal-impl.h

cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c ./cpp/ggml-cpu/ggml-cpu.c
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp ./cpp/ggml-cpu/ggml-cpu.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h ./cpp/ggml-cpu/ggml-cpu-impl.h
cp ./llama.cpp/ggml/src/ggml-cpu/quants.h ./cpp/ggml-cpu/quants.h
cp ./llama.cpp/ggml/src/ggml-cpu/quants.c ./cpp/ggml-cpu/quants.c
cp ./llama.cpp/ggml/src/ggml-cpu/arch-fallback.h ./cpp/ggml-cpu/arch-fallback.h
cp ./llama.cpp/ggml/src/ggml-cpu/repack.cpp ./cpp/ggml-cpu/repack.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/repack.h ./cpp/ggml-cpu/repack.h
cp ./llama.cpp/ggml/src/ggml-cpu/traits.h ./cpp/ggml-cpu/traits.h
cp ./llama.cpp/ggml/src/ggml-cpu/traits.cpp ./cpp/ggml-cpu/traits.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/common.h ./cpp/ggml-cpu/common.h

cp ./llama.cpp/ggml/src/ggml-cpu/unary-ops.h ./cpp/ggml-cpu/unary-ops.h
cp ./llama.cpp/ggml/src/ggml-cpu/unary-ops.cpp ./cpp/ggml-cpu/unary-ops.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/binary-ops.h ./cpp/ggml-cpu/binary-ops.h
cp ./llama.cpp/ggml/src/ggml-cpu/binary-ops.cpp ./cpp/ggml-cpu/binary-ops.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/vec.h ./cpp/ggml-cpu/vec.h
cp ./llama.cpp/ggml/src/ggml-cpu/vec.cpp ./cpp/ggml-cpu/vec.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/simd-mappings.h ./cpp/ggml-cpu/simd-mappings.h
cp ./llama.cpp/ggml/src/ggml-cpu/ops.h ./cpp/ggml-cpu/ops.h
cp ./llama.cpp/ggml/src/ggml-cpu/ops.cpp ./cpp/ggml-cpu/ops.cpp

cp -r ./llama.cpp/ggml/src/ggml-cpu/amx ./cpp/ggml-cpu/
mkdir -p ./cpp/ggml-cpu/arch
cp -r ./llama.cpp/ggml/src/ggml-cpu/arch/arm ./cpp/ggml-cpu/arch/
cp -r ./llama.cpp/ggml/src/ggml-cpu/arch/x86 ./cpp/ggml-cpu/arch/

cp ./llama.cpp/ggml/src/ggml.c ./cpp/ggml.c
cp ./llama.cpp/ggml/src/ggml-impl.h ./cpp/ggml-impl.h
cp ./llama.cpp/ggml/src/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./llama.cpp/ggml/src/ggml-backend.cpp ./cpp/ggml-backend.cpp
cp ./llama.cpp/ggml/src/ggml-backend-impl.h ./cpp/ggml-backend-impl.h
cp ./llama.cpp/ggml/src/ggml-backend-reg.cpp ./cpp/ggml-backend-reg.cpp
cp ./llama.cpp/ggml/src/ggml-common.h ./cpp/ggml-common.h
cp ./llama.cpp/ggml/src/ggml-opt.cpp ./cpp/ggml-opt.cpp
cp ./llama.cpp/ggml/src/ggml-quants.h ./cpp/ggml-quants.h
cp ./llama.cpp/ggml/src/ggml-quants.c ./cpp/ggml-quants.c
cp ./llama.cpp/ggml/src/ggml-threading.cpp ./cpp/ggml-threading.cpp
cp ./llama.cpp/ggml/src/ggml-threading.h ./cpp/ggml-threading.h
cp ./llama.cpp/ggml/src/gguf.cpp ./cpp/gguf.cpp

# llama api
cp ./llama.cpp/include/llama.h ./cpp/llama.h
cp ./llama.cpp/include/llama-cpp.h ./cpp/llama-cpp.h
cp ./llama.cpp/src/llama.cpp ./cpp/llama.cpp
cp ./llama.cpp/src/llama-chat.h ./cpp/llama-chat.h
cp ./llama.cpp/src/llama-chat.cpp ./cpp/llama-chat.cpp
cp ./llama.cpp/src/llama-context.h ./cpp/llama-context.h
cp ./llama.cpp/src/llama-context.cpp ./cpp/llama-context.cpp
cp ./llama.cpp/src/llama-mmap.h ./cpp/llama-mmap.h
cp ./llama.cpp/src/llama-mmap.cpp ./cpp/llama-mmap.cpp
cp ./llama.cpp/src/llama-model-loader.h ./cpp/llama-model-loader.h
cp ./llama.cpp/src/llama-model-loader.cpp ./cpp/llama-model-loader.cpp
cp ./llama.cpp/src/llama-model-saver.h ./cpp/llama-model-saver.h
cp ./llama.cpp/src/llama-model-saver.cpp ./cpp/llama-model-saver.cpp
cp ./llama.cpp/src/llama-model.h ./cpp/llama-model.h
cp ./llama.cpp/src/llama-model.cpp ./cpp/llama-model.cpp
cp ./llama.cpp/src/llama-kv-cells.h ./cpp/llama-kv-cells.h
cp ./llama.cpp/src/llama-kv-cache-unified.h ./cpp/llama-kv-cache-unified.h
cp ./llama.cpp/src/llama-kv-cache-unified.cpp ./cpp/llama-kv-cache-unified.cpp
cp ./llama.cpp/src/llama-kv-cache-unified-iswa.h ./cpp/llama-kv-cache-unified-iswa.h
cp ./llama.cpp/src/llama-kv-cache-unified-iswa.cpp ./cpp/llama-kv-cache-unified-iswa.cpp
cp ./llama.cpp/src/llama-memory-hybrid.h ./cpp/llama-memory-hybrid.h
cp ./llama.cpp/src/llama-memory-hybrid.cpp ./cpp/llama-memory-hybrid.cpp
cp ./llama.cpp/src/llama-memory-recurrent.h ./cpp/llama-memory-recurrent.h
cp ./llama.cpp/src/llama-memory-recurrent.cpp ./cpp/llama-memory-recurrent.cpp
cp ./llama.cpp/src/llama-adapter.h ./cpp/llama-adapter.h
cp ./llama.cpp/src/llama-adapter.cpp ./cpp/llama-adapter.cpp
cp ./llama.cpp/src/llama-arch.h ./cpp/llama-arch.h
cp ./llama.cpp/src/llama-arch.cpp ./cpp/llama-arch.cpp
cp ./llama.cpp/src/llama-batch.h ./cpp/llama-batch.h
cp ./llama.cpp/src/llama-batch.cpp ./cpp/llama-batch.cpp
cp ./llama.cpp/src/llama-cparams.h ./cpp/llama-cparams.h
cp ./llama.cpp/src/llama-cparams.cpp ./cpp/llama-cparams.cpp
cp ./llama.cpp/src/llama-hparams.h ./cpp/llama-hparams.h
cp ./llama.cpp/src/llama-hparams.cpp ./cpp/llama-hparams.cpp
cp ./llama.cpp/src/llama-impl.h ./cpp/llama-impl.h
cp ./llama.cpp/src/llama-impl.cpp ./cpp/llama-impl.cpp

cp ./llama.cpp/src/llama-vocab.h ./cpp/llama-vocab.h
cp ./llama.cpp/src/llama-vocab.cpp ./cpp/llama-vocab.cpp
cp ./llama.cpp/src/llama-grammar.h ./cpp/llama-grammar.h
cp ./llama.cpp/src/llama-grammar.cpp ./cpp/llama-grammar.cpp
cp ./llama.cpp/src/llama-sampling.h ./cpp/llama-sampling.h
cp ./llama.cpp/src/llama-sampling.cpp ./cpp/llama-sampling.cpp

cp ./llama.cpp/src/unicode.h ./cpp/unicode.h
cp ./llama.cpp/src/unicode.cpp ./cpp/unicode.cpp
cp ./llama.cpp/src/unicode-data.h ./cpp/unicode-data.h
cp ./llama.cpp/src/unicode-data.cpp ./cpp/unicode-data.cpp

cp ./llama.cpp/src/llama-graph.h ./cpp/llama-graph.h
cp ./llama.cpp/src/llama-graph.cpp ./cpp/llama-graph.cpp
cp ./llama.cpp/src/llama-io.h ./cpp/llama-io.h
cp ./llama.cpp/src/llama-io.cpp ./cpp/llama-io.cpp
cp ./llama.cpp/src/llama-memory.h ./cpp/llama-memory.h
cp ./llama.cpp/src/llama-memory.cpp ./cpp/llama-memory.cpp

cp ./llama.cpp/common/log.h ./cpp/log.h
cp ./llama.cpp/common/log.cpp ./cpp/log.cpp
cp ./llama.cpp/common/common.h ./cpp/common.h
cp ./llama.cpp/common/common.cpp ./cpp/common.cpp
cp ./llama.cpp/common/sampling.h ./cpp/sampling.h
cp ./llama.cpp/common/sampling.cpp ./cpp/sampling.cpp
cp ./llama.cpp/common/json-schema-to-grammar.h ./cpp/json-schema-to-grammar.h
cp ./llama.cpp/common/json-schema-to-grammar.cpp ./cpp/json-schema-to-grammar.cpp
cp ./llama.cpp/common/json-partial.h ./cpp/json-partial.h
cp ./llama.cpp/common/json-partial.cpp ./cpp/json-partial.cpp
cp ./llama.cpp/common/regex-partial.h ./cpp/regex-partial.h
cp ./llama.cpp/common/regex-partial.cpp ./cpp/regex-partial.cpp
cp ./llama.cpp/common/chat.h ./cpp/chat.h
cp ./llama.cpp/common/chat.cpp ./cpp/chat.cpp
cp ./llama.cpp/common/chat-parser.h ./cpp/chat-parser.h
cp ./llama.cpp/common/chat-parser.cpp ./cpp/chat-parser.cpp

# Copy multimodal files from tools/mtmd
rm -rf ./cpp/tools/mtmd
mkdir -p ./cpp/tools/mtmd
cp ./llama.cpp/tools/mtmd/mtmd.h ./cpp/tools/mtmd/mtmd.h
cp ./llama.cpp/tools/mtmd/mtmd.cpp ./cpp/tools/mtmd/mtmd.cpp
cp ./llama.cpp/tools/mtmd/clip.h ./cpp/tools/mtmd/clip.h
cp ./llama.cpp/tools/mtmd/clip.cpp ./cpp/tools/mtmd/clip.cpp
cp ./llama.cpp/tools/mtmd/clip-impl.h ./cpp/tools/mtmd/clip-impl.h
cp ./llama.cpp/tools/mtmd/mtmd-helper.cpp ./cpp/tools/mtmd/mtmd-helper.cpp
cp ./llama.cpp/tools/mtmd/mtmd-helper.h ./cpp/tools/mtmd/mtmd-helper.h
cp ./llama.cpp/tools/mtmd/mtmd-audio.h ./cpp/tools/mtmd/mtmd-audio.h
cp ./llama.cpp/tools/mtmd/mtmd-audio.cpp ./cpp/tools/mtmd/mtmd-audio.cpp

rm -rf ./cpp/minja
rm -rf ./cpp/nlohmann
cp -r ./llama.cpp/vendor/minja ./cpp/minja
cp -r ./llama.cpp/vendor/nlohmann ./cpp/nlohmann
rm -rf ./cpp/tools/mtmd/miniaudio
rm -rf ./cpp/tools/mtmd/stb
cp -r ./llama.cpp/vendor/miniaudio ./cpp/tools/mtmd/miniaudio
cp -r ./llama.cpp/vendor/stb ./cpp/tools/mtmd/stb

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
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal.m"
  "./cpp/ggml-metal-impl.h"
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
  "./cpp/ggml-backend.h"
  "./cpp/ggml-backend.cpp"
  "./cpp/ggml-backend-impl.h"
  "./cpp/ggml-backend-reg.cpp"
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
  "./cpp/llama-kv-cache-unified.h"
  "./cpp/llama-kv-cache-unified.cpp"
  "./cpp/llama-kv-cache-unified-iswa.h"
  "./cpp/llama-kv-cache-unified-iswa.cpp"
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

yarn example

# Apply patch
patch -p0 -d ./cpp < ./scripts/patches/common.h.patch
patch -p0 -d ./cpp < ./scripts/patches/common.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/chat.h.patch
patch -p0 -d ./cpp < ./scripts/patches/chat.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/log.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml-metal.m.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml.c.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml-quants.c.patch
patch -p0 -d ./cpp < ./scripts/patches/llama-mmap.cpp.patch
patch -p0 -d ./cpp/minja < ./scripts/patches/minja.hpp.patch
patch -p0 -d ./cpp/minja < ./scripts/patches/chat-template.hpp.patch
rm -rf ./cpp/*.orig

if [ "$OS" = "Darwin" ]; then
  # Build metallib (~2.6MB)
  cd llama.cpp/ggml/src/ggml-metal

  # Create a symbolic link to ggml-common.h in the current directory
  ln -sf ../ggml-common.h .

  xcrun --sdk iphoneos metal -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_USE_BF16=1
  xcrun --sdk iphoneos metallib ggml-metal.air   -o ggml-llama.metallib
  rm ggml-metal.air
  mv ./ggml-llama.metallib ../../../../cpp/ggml-llama.metallib

  xcrun --sdk iphonesimulator metal -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_USE_BF16=1
  xcrun --sdk iphonesimulator metallib ggml-metal.air   -o ggml-llama.metallib
  rm ggml-metal.air
  mv ./ggml-llama.metallib ../../../../cpp/ggml-llama-sim.metallib

  # Remove the symbolic link
  rm ggml-common.h

  cd -

  # Generate .xcode.env.local in iOS example
  cd example/ios
  echo export NODE_BINARY=$(command -v node) > .xcode.env.local
fi
