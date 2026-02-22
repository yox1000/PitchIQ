#!/bin/bash
# compile the cuda kernels into a shared lib for use via ctypes
# requires nvcc (CUDA toolkit >= 11.7) and a compatible GPU driver
#
# tested on:
#   CUDA 12.1, driver 530.30.02, RTX 3090
#   Ubuntu 22.04

set -e

ARCH=${1:-sm_86}   # sm_86 = ampere (3090). change for your gpu:
                   # sm_75 = turing (2080), sm_80 = a100, sm_89 = ada (4090)

echo "compiling for arch $ARCH ..."

nvcc -O3 -use_fast_math \
     -arch=$ARCH \
     --compiler-options '-fPIC' \
     -shared \
     homography_kernel.cu \
     -o cuda_rt.so

echo "done: cuda_rt.so"
echo "run: python bench_latency.py"
