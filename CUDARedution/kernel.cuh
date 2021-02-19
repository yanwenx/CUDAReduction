#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace kernel
{
  __device__ void WarpReduce(
    volatile int* shared, const unsigned int tid, const unsigned int tid_global, const unsigned int size);

  __global__ void ReduceSum0(const int* in, int* out, const unsigned int size);

  __global__ void ReduceSum1(const int* in, int* out, const unsigned int size);

  __global__ void ReduceSum2(const int* in, int* out, const unsigned int size);

  __global__ void ReduceSum3(const int* in, int* out, const unsigned int size);

  __global__ void ReduceSum4(const int* in, int* out, const unsigned int size);

  __global__ void ReduceMin(const int* in, int* out, const unsigned int size);

}  // namespace kernel