#include "kernel.cuh"

namespace kernel
{
  __device__ void WarpReduce(
    volatile int* shared, const unsigned int tid, const unsigned int tid_global, const unsigned int size)
  {
    if (tid_global + 32 < size) {
      shared[tid] += shared[tid + 32];
    }
    if (tid_global + 16 < size) {
      shared[tid] += shared[tid + 16];
    }
    if (tid_global + 8 < size) {
      shared[tid] += shared[tid + 8];
    }
    if (tid_global + 4 < size) {
      shared[tid] += shared[tid + 4];
    }
    if (tid_global + 2 < size) {
      shared[tid] += shared[tid + 2];
    }
    if (tid_global + 1 < size) {
      shared[tid] += shared[tid + 1];
    }
  }

  __global__ void ReduceSum0(const int* in, int* out, const unsigned int size)
  {
    extern __shared__ int shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    if (i < size) {
      shared[tid] = in[i];
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2 * s) == 0 && i + s < size) {
        shared[tid] += shared[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      out[blockIdx.x] = shared[0];
    }
  }

  __global__ void ReduceSum1(const int* in, int* out, const unsigned int size)
  {
    extern __shared__ int shared[];
    unsigned int tid = threadIdx.x;
    unsigned int t0 = blockIdx.x * blockDim.x;
    unsigned int i = t0 + tid;
    if (i < size) {
      shared[tid] = in[i];
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      unsigned int index1 = 2 * s * tid;
      unsigned int index2 = index1 + s;
      if (index1 < blockDim.x && t0 + index2 < size) {
        shared[index1] += shared[index2];
      }
      __syncthreads();
    }

    if (tid == 0) {
      out[blockIdx.x] = shared[0];
    }
  }

  __global__ void ReduceSum2(const int* in, int* out, const unsigned int size)
  {
    extern __shared__ int shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    if (i < size) {
      shared[tid] = in[i];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && i + s < size) {
        shared[tid] += shared[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      out[blockIdx.x] = shared[0];
    }
  }

  __global__ void ReduceSum3(const int* in, int* out, const unsigned int size)
  {
    extern __shared__ int shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + tid;
    if (i + blockDim.x < size) {
      shared[tid] = in[i] + in[i + blockDim.x];
    }
    else if (i < size) {
      shared[tid] = in[i];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && i + s < size) {
        shared[tid] += shared[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      out[blockIdx.x] = shared[0];
    }
  }

  __global__ void ReduceSum4(const int* in, int* out, const unsigned int size)
  {
    extern __shared__ int shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + tid;
    if (i + blockDim.x < size) {
      shared[tid] = in[i] + in[i + blockDim.x];
    }
    else if (i < size) {
      shared[tid] = in[i];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
      if (tid < s && i + s < size) {
        shared[tid] += shared[tid + s];
      }
      __syncthreads();
    }

    if (tid < 32) {
      WarpReduce(shared, tid, i, size);
    }

    if (tid == 0) {
      out[blockIdx.x] = shared[0];
    }
  }

  __global__ void ReduceMin(const int* in, int* out, const unsigned int size)
  {
    extern __shared__ int shared[];
    unsigned int tid_local = threadIdx.x;
    unsigned int tid_global = blockIdx.x * (2 * blockDim.x) + tid_local;
    if (tid_global + blockDim.x < size) {
      shared[tid_local] = (in[tid_global] < in[tid_global + blockDim.x]) ? in[tid_global] : in[tid_global + blockDim.x];
    }
    else if (tid_global < size) {
      shared[tid_local] = in[tid_global];
    }
    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (tid_local < i && tid_global + i < size) {
        shared[tid_local] = (shared[tid_local] < shared[tid_local + i]) ? shared[tid_local] : shared[tid_local + i];
      }
      __syncthreads();
    }

    if (tid_local == 0) {
      out[blockIdx.x] = shared[0];
    }
  }

}  // namespace kernel