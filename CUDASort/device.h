#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace device
{
  void QuickSort(int* d_arr, const unsigned int size);
  void QuickSort(int* d_arr, const unsigned int rows, const unsigned int cols);
  void InsertionSort(int* d_arr, const unsigned int size);
}
