#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace kernel
{
  __device__ void Swap(int* a, int* b);

  __device__ int Partition(int* arr, int l, int r);

  __device__ void Qsort(int* arr, int l, int r);

  __global__ void QuickSort(int* arr, const unsigned int size);

  __global__ void QuickSort(int* arr, const unsigned int rows, const unsigned int cols);

  __global__ void InsertionSort(int* arr, const unsigned int size);
}