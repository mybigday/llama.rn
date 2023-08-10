#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./llama.cpp/ggml.h ./cpp/ggml.h
cp ./llama.cpp/ggml.c ./cpp/ggml.c
cp ./llama.cpp/ggml-metal.h ./cpp/ggml-metal.h
cp ./llama.cpp/ggml-metal.m ./cpp/ggml-metal.m
cp ./llama.cpp/ggml-metal.metal ./cpp/ggml-metal.metal
cp ./llama.cpp/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./llama.cpp/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./llama.cpp/llama-util.h ./cpp/llama-util.h 
cp ./llama.cpp/llama.h ./cpp/llama.h
cp ./llama.cpp/llama.cpp ./cpp/llama.cpp
cp ./llama.cpp/k_quants.h ./cpp/k_quants.h
cp ./llama.cpp/k_quants.c ./cpp/k_quants.c
cp ./llama.cpp/examples/common.h ./cpp/common.h
cp ./llama.cpp/examples/common.cpp ./cpp/common.cpp

# Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/ggml.h
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/ggml.h
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/ggml.c
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/ggml.c
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/ggml-metal.h
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/ggml-metal.h
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/ggml-metal.m
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/ggml-metal.m
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/llama.h
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/llama.h
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/llama.cpp
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/llama.cpp
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/llama-util.h
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/llama-util.h
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/k_quants.h
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/k_quants.h
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/k_quants.c
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/k_quants.c
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/ggml-alloc.h
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/ggml-alloc.h
sed -i '' 's/GGML_/LM_GGML_/g' ./cpp/ggml-alloc.c
sed -i '' 's/ggml_/lm_ggml_/g' ./cpp/ggml-alloc.c

yarn example

# Apply patch
patch -p0 -d ./cpp < ./scripts/ggml-metal.m.patch

rm cpp/ggml-metal.m.orig || true
