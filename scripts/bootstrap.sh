#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./llama.cpp/include/llama.h ./cpp/llama.h

cp ./llama.cpp/ggml/include/ggml.h ./cpp/ggml.h
cp ./llama.cpp/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./llama.cpp/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./llama.cpp/ggml/include/ggml-cpu.h ./cpp/ggml-cpu.h
cp ./llama.cpp/ggml/include/ggml-cpp.h ./cpp/ggml-cpp.h
cp ./llama.cpp/ggml/include/ggml-opt.h ./cpp/ggml-opt.h
cp ./llama.cpp/ggml/include/ggml-metal.h ./cpp/ggml-metal.h

cp ./llama.cpp/ggml/src/ggml-metal/ggml-metal.m ./cpp/ggml-metal.m
cp ./llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h ./cpp/ggml-metal-impl.h

cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c ./cpp/ggml-cpu.c
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp ./cpp/ggml-cpu.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h ./cpp/ggml-cpu-impl.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-aarch64.h ./cpp/ggml-cpu-aarch64.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ./cpp/ggml-cpu-aarch64.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.h ./cpp/ggml-cpu-quants.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c ./cpp/ggml-cpu-quants.c
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-traits.h ./cpp/ggml-cpu-traits.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-traits.cpp ./cpp/ggml-cpu-traits.cpp

cp -r ./llama.cpp/ggml/src/ggml-cpu/amx ./cpp/

cp ./llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.h ./cpp/sgemm.h
cp ./llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp ./cpp/sgemm.cpp

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

cp ./llama.cpp/src/llama.cpp ./cpp/llama.cpp
cp ./llama.cpp/src/llama-impl.h ./cpp/llama-impl.h

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

cp ./llama.cpp/common/log.h ./cpp/log.h
cp ./llama.cpp/common/log.cpp ./cpp/log.cpp
cp ./llama.cpp/common/common.h ./cpp/common.h
cp ./llama.cpp/common/common.cpp ./cpp/common.cpp
cp ./llama.cpp/common/sampling.h ./cpp/sampling.h
cp ./llama.cpp/common/sampling.cpp ./cpp/sampling.cpp

# List of files to process
files_add_lm_prefix=(
  "./cpp/llama-impl.h"
  "./cpp/llama-vocab.h"
  "./cpp/llama-vocab.cpp"
  "./cpp/llama-grammar.h"
  "./cpp/llama-grammar.cpp"
  "./cpp/llama-sampling.h"
  "./cpp/llama-sampling.cpp"
  "./cpp/log.h"
  "./cpp/log.cpp"
  "./cpp/llama.h"
  "./cpp/llama.cpp"
  "./cpp/sampling.cpp"
  "./cpp/sgemm.h"
  "./cpp/sgemm.cpp"
  "./cpp/common.h"
  "./cpp/common.cpp"
  "./cpp/ggml-common.h"
  "./cpp/ggml.h"
  "./cpp/ggml.c"
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
  "./cpp/ggml-cpu-impl.h"
  "./cpp/ggml-cpu.h"
  "./cpp/ggml-cpu.c"
  "./cpp/ggml-cpu.cpp"
  "./cpp/ggml-cpu-aarch64.h"
  "./cpp/ggml-cpu-aarch64.cpp"
  "./cpp/ggml-cpu-quants.h"
  "./cpp/ggml-cpu-quants.c"
  "./cpp/ggml-cpu-traits.h"
  "./cpp/ggml-cpu-traits.cpp"
  "./cpp/ggml-threading.h"
  "./cpp/ggml-threading.cpp"
  "./cpp/amx/amx.h"
  "./cpp/amx/amx.cpp"
  "./cpp/amx/mmq.h"
  "./cpp/amx/mmq.cpp"
  "./cpp/amx/common.h"
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
  else
    sed -i 's/GGML_/LM_GGML_/g' $file
    sed -i 's/ggml_/lm_ggml_/g' $file
    sed -i 's/GGUF_/LM_GGUF_/g' $file
    sed -i 's/gguf_/lm_gguf_/g' $file
    sed -i 's/GGMLMetalClass/LMGGMLMetalClass/g' $file
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
patch -p0 -d ./cpp < ./scripts/common.h.patch
patch -p0 -d ./cpp < ./scripts/common.cpp.patch
patch -p0 -d ./cpp < ./scripts/log.cpp.patch
patch -p0 -d ./cpp < ./scripts/llama.cpp.patch
patch -p0 -d ./cpp < ./scripts/ggml-metal.m.patch
patch -p0 -d ./cpp < ./scripts/ggml-backend-reg.cpp.patch
patch -p0 -d ./cpp < ./scripts/ggml.c.patch
patch -p0 -d ./cpp < ./scripts/ggml-quants.c.patch

if [ "$OS" = "Darwin" ]; then
  # Build metallib (~1.4MB)
  cd llama.cpp/ggml/src/ggml-metal
  xcrun --sdk iphoneos metal -c ggml-metal.metal -o ggml-metal.air
  xcrun --sdk iphoneos metallib ggml-metal.air   -o ggml-llama.metallib
  rm ggml-metal.air
  cp ./ggml-llama.metallib ../../../../cpp/ggml-llama.metallib

  cd -

  # Generate .xcode.env.local in iOS example
  cd example/ios
  echo export NODE_BINARY=$(command -v node) > .xcode.env.local
fi
