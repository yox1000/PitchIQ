#!/bin/bash
# compile pitchiq cuda kernels → cuda_rt.so
#
# tested environment (RunPod):
#   pod image : runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
#   gpu       : RTX 4090 (sm_89, Ada Lovelace)
#   nvcc      : V12.3.107
#   driver    : 545.23.08
#
# usage:
#   bash build.sh          # defaults to sm_89 (4090)
#   bash build.sh sm_86    # 3090 / 3080
#   bash build.sh sm_80    # A100
#   bash build.sh sm_75    # T4

set -e

ARCH=${1:-sm_89}

echo "[pitchiq] compiling for $ARCH ..."

nvcc -O3 \
     -use_fast_math \
     -arch=$ARCH \
     -Xptxas -v \
     --compiler-options '-fPIC -O3' \
     -shared \
     homography_kernel.cu \
     -o cuda_rt.so \
     2>&1

echo ""
echo "[pitchiq] built: cuda_rt.so"
echo "[pitchiq] run:   python bench_latency.py"
