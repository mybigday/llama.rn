#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./llama.cpp/ggml.h ./cpp/ggml.h
cp ./llama.cpp/ggml.c ./cpp/ggml.c
cp ./llama.cpp/ggml-metal.h ./cpp/ggml-metal.h
cp ./llama.cpp/ggml-metal.m ./cpp/ggml-metal.m
cp ./llama.cpp/ggml-metal.metal ./cpp/ggml-metal-llama.metal
cp ./llama.cpp/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./llama.cpp/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./llama.cpp/ggml-backend.h ./cpp/ggml-backend.h
cp ./llama.cpp/ggml-backend.c ./cpp/ggml-backend.c
cp ./llama.cpp/ggml-backend-impl.h ./cpp/ggml-backend-impl.h
cp ./llama.cpp/ggml-impl.h ./cpp/ggml-impl.h
cp ./llama.cpp/llama.h ./cpp/llama.h
cp ./llama.cpp/llama.cpp ./cpp/llama.cpp
cp ./llama.cpp/ggml-quants.h ./cpp/ggml-quants.h
cp ./llama.cpp/ggml-quants.c ./cpp/ggml-quants.c
cp ./llama.cpp/unicode.h ./cpp/unicode.h
cp ./llama.cpp/common/log.h ./cpp/log.h
cp ./llama.cpp/common/common.h ./cpp/common.h
cp ./llama.cpp/common/common.cpp ./cpp/common.cpp
cp ./llama.cpp/common/grammar-parser.h ./cpp/grammar-parser.h
cp ./llama.cpp/common/grammar-parser.cpp ./cpp/grammar-parser.cpp
cp ./llama.cpp/common/sampling.h ./cpp/sampling.h
cp ./llama.cpp/common/sampling.cpp ./cpp/sampling.cpp

# List of files to process
files=(
  "./cpp/ggml.h"
  "./cpp/ggml.c"
  "./cpp/common.h"
  "./cpp/common.cpp"
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal.m"
  "./cpp/llama.h"
  "./cpp/llama.cpp"
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
  "./cpp/ggml-backend.h"
  "./cpp/ggml-backend.c"
  "./cpp/ggml-backend-impl.h"
  "./cpp/ggml-impl.h"
)

# Loop through each file and run the sed commands
OS=$(uname)
for file in "${files[@]}"; do
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

echo "Replacement completed successfully!"

yarn example

# Apply patch
patch -p0 -d ./cpp < ./scripts/common.h.patch
patch -p0 -d ./cpp < ./scripts/common.cpp.patch
patch -p0 -d ./cpp < ./scripts/log.h.patch
patch -p0 -d ./cpp < ./scripts/llama.cpp.patch
patch -p0 -d ./cpp < ./scripts/ggml-metal.m.patch
