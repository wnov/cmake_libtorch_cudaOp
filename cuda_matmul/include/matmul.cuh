#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <cuda_runtime.h>

#define CHECK(error)                                                    \
  {                                                                     \
    if (error != cudaSuccess) {                                         \
      printf("ERROR: %s:%d", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s", error, cudaGetErrorString(error)); \
    }                                                                   \
  }

template <typename T>
__host__ __device__ void printMat(T *mat, size_t h, size_t w) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      printf("%.2f, ", mat[i * w + j]);
    }
    printf("\n");
  }
}

constexpr int blockSize = 32;
extern "C" {}
void matmul(float *matA, float *matB, float *matC, size_t height, size_t width,
            size_t num, int nAlgo);
void matmul_cpu(float *A, float *B, float *C, int m, int n, int k);
void checkResult(float *hostRef, float *gpuRef, const int N);

#endif  //__MATMUL_H__