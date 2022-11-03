#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>

#include "matmul.h"

template <typename T>
__device__ void matTranspose(T *mat, T *matO, int height, int width) {
  __shared__ T s_mat[blockSize][blockSize];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  s_mat[y][x] = mat[y * width + x];
  __syncthreads();
  matO[y * width + x] = s_mat[x][y];
}

template <typename T>
__global__ void matmul0(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num) {
  size_t x_index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_index >= width || y_index >= height) {
    return;
  }

  T res = 0.;
#pragma unroll
  for (int i = 0; i < num; i++) {
    res += matA[y_index * num + i] * matB[x_index + i * width];
  }

  matC[y_index * width + x_index] = res;
}

// 共享内存优化全局内存合并读取，并且降低重复读取的次数
template <typename T>
__global__ void matmul1(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num) {
  __shared__ T blockA[blockSize][blockSize], blockB[blockSize][blockSize + 1];
  size_t tid_x = threadIdx.x, tid_y = threadIdx.y;
  size_t x_index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  int n_iter = (num - 1) / blockSize + 1;

  if (x_index >= width || y_index >= height) {
    return;
  }

  T res = 0.;
#pragma unroll
  for (int i = 0; i < n_iter; i++) {
    blockA[tid_y][tid_x] = matA[y_index * num + (i * blockSize + tid_x)];
    blockB[tid_y][tid_x] = matB[(tid_y + i * blockSize) * width + x_index];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < blockSize && i * blockSize + j < num; j++)
      res += blockA[tid_y][j] * blockB[j][tid_x];
    __syncthreads();
  }

  matC[y_index * width + x_index] = res;
}

#define FETCH_FLOAT4(x) reinterpret_cast<float4 *>(&x)[0]
// float4 提高内存带宽和单个线程计算强度
template <typename T>
__global__ void matmul2(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num) {
  __shared__ float4 blockA[blockSize][blockSize],
      blockB[blockSize][blockSize + 1];
  T *matBT;
  matTranspose(matB, matBT, num, width);
  size_t tid_x = threadIdx.x * 4, tid_y = threadIdx.y * 4;
  size_t x_index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  int n_iter = (num - 1) / (blockSize * 4) + 1;

  if (x_index >= width || y_index >= height) {
    return;
  }

  T res = 0.;
#pragma unroll
  for (int i = 0; i < n_iter; i++) {
    blockA[tid_y][tid_x] =
        FETCH_FLOAT4(matA[y_index * num + (i * blockSize + tid_x)]);
    blockB[tid_x][tid_y] =
        FETCH_FLOAT4(matBT[x_index * num + (i * blockSize + tid_y)]);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < blockSize && i * blockSize + j < num; j++)
      res += blockA[tid_y][j].x * blockB[tid_x][j].x +
             blockA[tid_y][j].y * blockB[tid_x][j].y +
             blockA[tid_y][j].z * blockB[tid_x][j].z +
             blockA[tid_y][j].w * blockB[tid_x][j].w;
    __syncthreads();
  }

  matC[y_index * width + x_index] = res;
}

// 使用数据预取隐藏延迟
template <typename T>
__global__ void matmul3(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num) {
  __shared__ float4 blockA[blockSize][blockSize],
      blockAPre[blockSize][blockSize], blockB[blockSize][blockSize],
      blockBPre[blockSize][blockSize];
  T *matBT;
  matTranspose(matB, matBT, num, width);
  size_t tid_x = threadIdx.x * 4, tid_y = threadIdx.y * 4;
  size_t x_index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  int n_iter = (num - 1) / (blockSize * 4) + 1;

  if (x_index >= width || y_index >= height) {
    return;
  }

  blockAPre[tid_y][tid_x] = FETCH_FLOAT4(matA[y_index * num + tid_x]);
  blockBPre[tid_y][tid_x] = FETCH_FLOAT4(matBT[x_index * num + tid_y]);

  T res = 0.;
#pragma unroll
  for (int i = 1; i < n_iter; i++) {
    blockA[tid_y][tid_x] = blockAPre[tid_y][tid_x];
    blockB[tid_x][tid_y] = blockBPre[tid_y][tid_x];
    __syncthreads();

    blockAPre[tid_y][tid_x] =
        FETCH_FLOAT4(matA[y_index * num + (i * blockSize + tid_x)]);
    blockBPre[tid_x][tid_y] =
        FETCH_FLOAT4(matBT[x_index * num + (i * blockSize + tid_y)]);

#pragma unroll
    for (int j = 0; j < blockSize && i * blockSize * 4 + j * 4 < num; j++)
      res += blockA[tid_y][j].x * blockB[tid_x][j].x +
             blockA[tid_y][j].y * blockB[tid_x][j].y +
             blockA[tid_y][j].z * blockB[tid_x][j].z +
             blockA[tid_y][j].w * blockB[tid_x][j].w;
  }
  __syncthreads();
  matC[y_index * width + x_index] = res;
}

// 使用寄存器替代共享内存
template <typename T>
__global__ void matmul4(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num) {
  __shared__ float4 blockA[blockSize][blockSize], blockB[blockSize][blockSize];
  float4 blockAPre, blockBPre;
  T *matBT;
  matTranspose(matB, matBT, num, width);
  size_t tid_x = threadIdx.x * 4, tid_y = threadIdx.y * 4;
  size_t x_index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  int n_iter = (num - 1) / (blockSize * 4) + 1;

  if (x_index >= width || y_index >= height) {
    return;
  }

  blockAPre = FETCH_FLOAT4(matA[y_index * num + tid_x]);
  blockBPre = FETCH_FLOAT4(matBT[x_index * num + tid_y]);

  T res = 0.;
#pragma unroll
  for (int i = 1; i < n_iter; i++) {
    blockA[tid_y][tid_x] = blockAPre;
    blockB[tid_x][tid_y] = blockBPre;
    __syncthreads();

    blockAPre = FETCH_FLOAT4(matA[y_index * num + (i * blockSize + tid_x)]);
    blockBPre = FETCH_FLOAT4(matBT[x_index * num + (i * blockSize + tid_y)]);

#pragma unroll
    for (int j = 0; j < blockSize / 4 && i * blockSize * 4 + j * 4 < num; j++)
      res += blockA[tid_y][j].x * blockB[tid_x][j].x +
             blockA[tid_y][j].y * blockB[tid_x][j].y +
             blockA[tid_y][j].z * blockB[tid_x][j].z +
             blockA[tid_y][j].w * blockB[tid_x][j].w;
  }
  __syncthreads();
  matC[y_index * width + x_index] = res;
}
