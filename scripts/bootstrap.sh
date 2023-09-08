#!/bin/bash -e

git submodule init
git submodule update --recursive

cd llama.cpp
./scripts/build-info.sh > build-info.h
cd -

cp ./llama.cpp/build-info.h ./cpp/build-info.h
cp ./llama.cpp/ggml.h ./cpp/ggml.h
cp ./llama.cpp/ggml.c ./cpp/ggml.c
cp ./llama.cpp/ggml-metal.h ./cpp/ggml-metal.h
cp ./llama.cpp/ggml-metal.m ./cpp/ggml-metal.m
cp ./llama.cpp/ggml-metal.metal ./cpp/ggml-metal.metal
cp ./llama.cpp/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./llama.cpp/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./llama.cpp/llama.h ./cpp/llama.h
cp ./llama.cpp/llama.cpp ./cpp/llama.cpp
cp ./llama.cpp/k_quants.h ./cpp/k_quants.h
cp ./llama.cpp/k_quants.c ./cpp/k_quants.c
cp ./llama.cpp/common/log.h ./cpp/log.h
cp ./llama.cpp/common/common.h ./cpp/common.h
cp ./llama.cpp/common/common.cpp ./cpp/common.cpp
cp ./llama.cpp/common/grammar-parser.h ./cpp/grammar-parser.h
cp ./llama.cpp/common/grammar-parser.cpp ./cpp/grammar-parser.cpp

# List of files to process
files=(
  "./cpp/ggml.h"
  "./cpp/ggml.c"
  "./cpp/common.cpp"
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal.m"
  "./cpp/llama.h"
  "./cpp/llama.cpp"
  "./cpp/k_quants.h"
  "./cpp/k_quants.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
)

# Loop through each file and run the sed commands
OS=$(uname)
for file in "${files[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/GGML_/LM_GGML_/g' $file
    sed -i '' 's/ggml_/lm_ggml_/g' $file
  else
    sed -i 's/GGML_/LM_GGML_/g' $file
    sed -i 's/ggml_/lm_ggml_/g' $file
  fi
done

echo "Replacement completed successfully!"

yarn example

# Apply patch
patch -p0 -d ./cpp < ./scripts/ggml-metal.m.patch
patch -p0 -d ./cpp < ./scripts/llama.cpp.patch
