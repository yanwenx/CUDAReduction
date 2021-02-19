#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include "device.h"

const unsigned int size = 4000;
const unsigned int rows = 500;
const unsigned int cols = 4000;

void Swap(int *a, int *b)
{
  int temp = *a;
  *a = *b;
  *b = temp;
}

void BubbleSort(int *arr, const unsigned int n)
{
  for (unsigned int i = 0; i < n; ++i) {
    // the ith pass stops at index i and floats the ith "lightest" element to index i
    for (unsigned int j = n - 1; j > i; --j) {
      if (arr[j] < arr[j - 1]) {
        Swap(arr + j, arr + j - 1);
      }
    }
  }
}

void SelectionSort(int* arr, const unsigned int n)
{
  for (unsigned int i = 0; i < n - 1; ++i) {
    // the ith pass selects the smallest element after index i and swaps it with the element at index i
    int min = i;
    for (unsigned int j = i + 1; j < n; ++j) {
      if (arr[j] < arr[min]) {
        min = j;
      }
    }
    Swap(arr + i, arr + min);
  }
}

void InsertionSort(int* arr, const unsigned int n)
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

int partition(int* arr, int l, int r)
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

void qsort(int* arr, int l, int r)
{
  if (r <= l) {
    return;
  }
  int i = partition(arr, l, r);
  qsort(arr, l, i - 1);
  qsort(arr, i + 1, r);
}

void QuickSort(int* arr, const unsigned int n)
{
  int l = 0;
  int r = static_cast<int>(n - 1);
  qsort(arr, l, r);
}

int main()
{
  ///* initialize arrays */
  //std::vector<int> vec(size);
  //int *h_arr = new int[size];
  //int* d_arr;
  //srand(static_cast<unsigned int>(time(nullptr)));
  //for (auto& v : vec) {
  //  v = rand();
  //}
  //cudaMalloc((void**)(&d_arr), size * sizeof(int));
  //cudaMemcpy(d_arr, vec.data(), size * sizeof(int), cudaMemcpyHostToDevice);

  ///* sort */
  //auto t0 = std::chrono::high_resolution_clock::now();
  //std::sort(vec.begin(), vec.end());
  //auto t1 = std::chrono::high_resolution_clock::now();
  ////QuickSort(d_arr, size);
  //device::QuickSort(d_arr, size);
  //auto t2 = std::chrono::high_resolution_clock::now();
  //cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

  ///* compare */
  //bool same = true;
  //unsigned int count = 0;
  //for (const auto &v : vec) {
  //  if (v != h_arr[count++]) {
  //    same = false;
  //    std::cout << "vec[" << count - 1 << "] = " << v << ", h_arr[" << count - 1 << "] = " << h_arr[count - 1] << std::endl;
  //    break;
  //  }
  //}
  //if (same) {
  //  std::cout << "Same result." << std::endl;
  //  std::cout << "STL sort takes " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms, "
  //    << " quick sort takes " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms." << std::endl;
  //}
  //else {
  //  std::cout << "Different result." << std::endl;
  //}

  ///* release */
  //cudaFree(d_arr);
  //delete[] h_arr;

  /* initialize */
  std::vector<std::vector<int>> mat(rows, std::vector<int>(cols));
  srand(static_cast<unsigned int>(time(nullptr)));
  for (unsigned int i = 0; i < rows; ++i) {
    for (unsigned int j = 0; j < cols; ++j) {
      mat[i][j] = rand();
    }
  }
  int* h_mat = new int[rows * cols];
  int* d_mat;
  cudaMalloc((void**)(&d_mat), rows * cols * sizeof(int));
  for (unsigned int i = 0; i < rows; ++i) {
    cudaMemcpy(d_mat + i * cols, mat[i].data(), cols * sizeof(int), cudaMemcpyHostToDevice);
  }

  /* sort */
  auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < rows; ++i) {
    std::sort(mat[i].begin(), mat[i].end());
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  device::QuickSort(d_mat, rows, cols);
  auto t2 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(h_mat, d_mat, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

  /* compare */
  bool same = true;
  for (unsigned int i = 0; i < rows; ++i) {
    for (unsigned int j = 0; j < cols; ++j) {
      if (mat[i][j] != *(h_mat + i * cols + j)) {
        same = false;
        break;
      }
    }
  }
  if (same) {
    std::cout << "Same result." << std::endl;
    std::cout << "Host takes " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms, "
      << "device takes " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms." << std::endl;
  }
  else {
    std::cout << "Different result." << std::endl;
  }

  /* release */
  cudaFree(d_mat);
  delete[] h_mat;

  return 0;
}