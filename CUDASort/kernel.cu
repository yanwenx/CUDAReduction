#include "kernel.cuh"

namespace kernel
{
  __device__ void Swap(int* a, int* b)
  {
    int temp = *a;
    *a = *b;
    *b = temp;
  }

  __device__ int Partition(int* arr, int l, int r)
  {
    int i = l - 1;
    int j = r;
    int v = arr[r];
    while (true) {
      while (arr[++i] < v);
      while (v < arr[--j]) {
        if (j == l) {
          break;
        }
      }
      if (i >= j) {
        break;
      }
      Swap(arr + i, arr + j);
    }
    Swap(arr + i, arr + r);
    return i;
  }

  __device__ void Qsort(int* arr, int l, int r)
  {
    if (r <= l) {
      return;
    }
    int i = Partition(arr, l, r);
    Qsort(arr, l, i - 1);
    Qsort(arr, i + 1, r);
  }

  __device__ void Isort(int* arr, int n)
  {
    for (unsigned int i = 1; i < n; ++i) {
      // the ith pass inserts the ith element next to the first element on its left and smaller than it
      for (unsigned int j = i; j > 0; --j) {
        if (arr[j] < arr[j - 1]) {
          Swap(arr + j, arr + j - 1);
        }
      }
    }
  }

  __global__ void QuickSort(int* arr, const unsigned int size)
  {
    int l = 0;
    int r = static_cast<int>(size - 1);
    Qsort(arr, l, r);
  }

  __global__ void QuickSort(int* arr, const unsigned int rows, const unsigned int cols)
  {
    unsigned int tid{ blockIdx.x * blockDim.x + threadIdx.x };
    if (tid < rows) {
      int l = static_cast<int>(tid * cols);
      int r = l + static_cast<int>(cols) - 1;
      Qsort(arr, l, r);
    }
    /*int row = static_cast<int>(blockIdx.x);
    int l = row * static_cast<int>(cols);
    int r = l + static_cast<int>(cols) - 1;
    Qsort(arr, l, r);*/
  }

  __global__ void InsertionSort(int* arr, const unsigned int size)
  {
    Isort(arr, size);
  }
}