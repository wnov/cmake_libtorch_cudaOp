#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifndef __MATMUL_H__
#define __MARMUL_H__
#include <cuda_runtime.h>

#define blockSize 32

template <typename T>
__device__ void matTranspose(T *mat, T *matO, int height, int width);

template <typename T>
__global__ void matmul0(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num);

template <typename T>
__global__ void matmul1(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num);

template <typename T>
__global__ void matmul2(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num);

template <typename T>
__global__ void matmul3(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num);

template <typename T>
__global__ void matmul4(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num);

void matmul_cublas(float *A, float *B, float *C, int m, int n, int k,
                   const float alpha, const float beta);
#endif