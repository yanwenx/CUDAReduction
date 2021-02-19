#include <cmath>
#include "device.h"
#include "kernel.cuh"

namespace device
{
  void QuickSort(int* d_arr, const unsigned int size)
  {
    kernel::QuickSort<<<1, 1>>>(d_arr, size);
  }

  void QuickSort(int* d_arr, const unsigned int rows, const unsigned int cols)
  {
    unsigned int block_dim{ 256 };
    dim3 dim_block{ block_dim };
    dim3 dim_grid{ static_cast<unsigned int>(ceil(static_cast<double>(rows) / block_dim)) };
    kernel::QuickSort<<<dim_grid, dim_block>>>(d_arr, rows, cols);
  }

  void InsertionSort(int* d_arr, const unsigned int size)
  {
    kernel::InsertionSort<<<1, 1>>>(d_arr, size);
  }
}