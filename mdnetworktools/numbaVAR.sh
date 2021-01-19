#!/bin/bash

#/bin/bash

# For python2.7, numba 0.43.1 and CUDA 8.0
TPB=1024
CU_DEVICE=1
export CUDA_HOME=/usr/local/cuda-8.0
export NUMBAPRO_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=$CUDA_HOME/nvvm/libdevice/

# For python 3.7, numba 0.52.1 and CUDA >= 9.0
#TPB=512
#CU_DEVICE=0
#export CUDA_HOME=/usr/local/cuda-9.0 
