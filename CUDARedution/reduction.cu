#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "kernel.cuh"

const unsigned int size = 1000000;

int main()
{
  /* initialize host array */
  int* h_arr = new int[size];
  srand(time(nullptr));
  for (unsigned int i = 0; i < size; ++i) {
    h_arr[i] = rand() % 100;
  }

  /* initialize device array */
  int* d_arr;
  cudaMalloc((void**)(&d_arr), size * sizeof(int));
  cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

  /* parallel computing */
  auto t0 = std::chrono::high_resolution_clock::now();

  unsigned int block_size{ 256 };
  unsigned int blocks{ size / block_size };
  unsigned int residue{ size % block_size };
  dim3 dim_block{ block_size };
  dim3 dim_grid{ (unsigned int)(ceil((double)size / block_size)) };
  int* d_reduce_arr;
  int* d_sum_arr;
  cudaMalloc((void**)(&d_reduce_arr), size * sizeof(int));
  cudaMemcpy(d_reduce_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToDevice);

  // half the grid size
  if (dim_grid.x > 1) {
    dim_grid.x = (unsigned int)(ceil(dim_grid.x / 2.0));
  }
  cudaMalloc((void**)(&d_sum_arr), dim_grid.x * sizeof(int));
  while (dim_grid.x > 1) {
    kernel::ReduceSum4<<<dim_grid, dim_block, block_size * sizeof(int)>>>(
      d_reduce_arr, d_sum_arr, blocks * block_size + residue);
    cudaDeviceSynchronize();

    cudaMemcpy(d_reduce_arr, d_sum_arr, dim_grid.x * sizeof(int), cudaMemcpyDeviceToDevice);
    blocks = dim_grid.x / block_size;
    residue = dim_grid.x % block_size;
    dim_grid.x = (unsigned int)(ceil((double)dim_grid.x / block_size));
    // half the grid size
    dim_grid.x = (unsigned int)(ceil(dim_grid.x / 2.0));
  }
  kernel::ReduceSum4<<<dim_grid, dim_block, block_size * sizeof(int)>>>(
    d_reduce_arr, d_sum_arr, blocks * block_size + residue);
  int d_sum;
  cudaMemcpy(&d_sum, d_sum_arr, sizeof(int), cudaMemcpyDeviceToHost);

  auto t1 = std::chrono::high_resolution_clock::now();

  /* sequential computing */
  int h_sum = 0;
  for (int i = 0; i < size; ++i) {
    h_sum += h_arr[i];
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  /* compare */
  if (d_sum == h_sum) {
    std::cout << "Same result: " << d_sum << std::endl;
    std::cout << "Device exe time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() <<
      "ms, host exe time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms." << std::endl;
  }
  else {
    std::cout << "Different result: d_sum = " << d_sum << ", h_sum = " << h_sum << "." << std::endl;
  }

  /* release */
  delete[] h_arr;
  cudaFree(d_arr);
  cudaFree(d_reduce_arr);
  cudaFree(d_sum_arr);

  return 0;
}