#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "helper/include/utils.h"
#include "matmul.cuh"

template <typename T>
__global__ void matTranspose(T *mat, T *matO, int height, int width) {
  __shared__ T s_mat[blockSize][blockSize + 1];
  int tid_x = threadIdx.x, tid_y = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    s_mat[tid_y][tid_x] = mat[y * width + x];
  }
  __syncthreads();
  if (x < width && y < height) {
    matO[x * height + y] = s_mat[tid_x][tid_y];
  }
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

// reinterpret_cast<float4 *>(&x)[0]
// float4 提高内存带宽和单个线程计算强度
template <typename T>
__global__ void matmul2(T *matA, T *matB, T *matC, size_t height, size_t width,
                        size_t num) {
  __shared__ float blockA[blockSize][blockSize * 4],
      blockB[4 * blockSize][blockSize];
  size_t tid_x = threadIdx.x, tid_y = threadIdx.y;
  size_t x_index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  int n_iter = (num - 1) / (blockSize * 4) + 1;

  if (x_index >= width || y_index >= height) {
    return;
  }

  int offset_x = blockIdx.x * blockDim.x;
  int offset_y = blockIdx.y * blockDim.y;
  int xA = tid_x << 2;
  int yA = tid_y;
  int xB = (tid_x << 2) % blockSize;
  int yB = (tid_x << 2) / blockSize + (tid_y << 2);
  T res = 0.;
#pragma unroll
  for (int i = 0; i < n_iter; i++) {
    if ((xA + i * blockSize * 4) < num)
      FETCH_FLOAT4(blockA[yA][xA]) =
          FETCH_FLOAT4(matA[(xA + i * blockSize * 4) + (offset_y + yA) * num]);
    if ((yB + i * blockSize * 4) < num)
      FETCH_FLOAT4(blockB[yB][xB]) = FETCH_FLOAT4(
          matB[(xB + offset_x) + (yB + i * blockSize * 4) * width]);
    __syncthreads();

#pragma unroll
    for (int j = 0; j < blockSize * 4 && i * blockSize * 4 + j < num; ++j)
      res += blockA[tid_y][j] * blockB[j][tid_x];
    __syncthreads();
  }

  matC[y_index * width + x_index] = res;
}

// // 使用数据预取隐藏延迟
// template <typename T>
// __global__ void matmul3(T *matA, T *matB, T *matC, size_t height, size_t
// width,
//                         size_t num) {
//   __shared__ float blockA[blockSize][blockSize * 4],
//       blockAPre[blockSize][blockSize * 4], blockB[blockSize * 4][blockSize],
//       blockBPre[blockSize * 4][blockSize];
//   size_t tid_x = threadIdx.x * 4, tid_y = threadIdx.y * 4;
//   size_t x_index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
//   size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
//   int n_iter = (num - 1) / (blockSize * 4) + 1;

//   if (x_index >= width || y_index >= height) {
//     return;
//   }

//   int offset_x = blockIdx.x * blockDim.x;
//   int offset_y = blockIdx.y * blockDim.y;
//   int xA = tid_x << 2;
//   int yA = tid_y;
//   int xB = (tid_x << 2) % blockSize;
//   int yB = (tid_x << 2) / blockSize + (tid_y << 2);

//   if (xA < num)
//     FETCH_FLOAT4(blockAPre[yA][xA]) =
//         FETCH_FLOAT4(matA[xA + (offset_y + yA) * num]);
//   if (yB < num)
//     FETCH_FLOAT4(blockBPre[yB][xB]) =
//         FETCH_FLOAT4(matB[(xB + offset_x) + yB * width]);

//   // blockAPre[tid_y][tid_x] = FETCH_FLOAT4(matA[y_index * num + tid_x]);
//   // blockBPre[tid_y][tid_x] = FETCH_FLOAT4(matB[x_index * num + tid_y]);

//   T res = 0.;
// #pragma unroll
//   for (int i = 1; i < n_iter; i++) {
//     if ((xA + i * blockSize * 4) < num)
//       FETCH_FLOAT4(blockA[yA][xA]) =
//           FETCH_FLOAT4(matA[(xA + i * blockSize * 4) + (offset_y + yA) *
//           num]);
//     if ((yB + i * blockSize * 4) < num)
//       FETCH_FLOAT4(blockB[yB][xB]) = FETCH_FLOAT4(
//           matB[(xB + offset_x) + (yB + i * blockSize * 4) * width]);
//     __syncthreads();

//     FETCH_FLOAT4(blockAPre[tid_y][tid_x]) =
//         FETCH_FLOAT4(matA[y_index * num + (i * blockSize + tid_x)]);
//     FETCH_FLOAT4(blockBPre[tid_x][tid_y]) =
//         FETCH_FLOAT4(matB[x_index * num + (i * blockSize + tid_y)]);

// #pragma unroll
//     for (int j = 0; j < blockSize * 4 && (i - 1) * blockSize * 4 + j < num;
//     ++j)
//       res += blockA[tid_y][j] * blockB[j][tid_x];
//     __syncthreads();
//   }

//   for (int j = 0; j < blockSize * 4 && (n_iter - 1) * blockSize * 4 + j <
//   num;
//        ++j)
//     res += blockA[tid_y][j] * blockB[j][tid_x];
//   __syncthreads();
//   matC[y_index * width + x_index] = res;
// }

// // 使用寄存器替代共享内存
// template <typename T>
// __global__ void matmul4(T *matA, T *matB, T *matC, size_t height, size_t
// width,
//                         size_t num) {
//   __shared__ float4 blockA[blockSize][blockSize],
//   blockB[blockSize][blockSize]; float4 blockAPre, blockBPre; T *matB;
//   matTranspose(matB, matB, num, width);
//   size_t tid_x = threadIdx.x * 4, tid_y = threadIdx.y * 4;
//   size_t x_index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
//   size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
//   int n_iter = (num - 1) / (blockSize * 4) + 1;

//   if (x_index >= width || y_index >= height) {
//     return;
//   }

//   blockAPre = FETCH_FLOAT4(matA[y_index * num + tid_x]);
//   blockBPre = FETCH_FLOAT4(matB[x_index * num + tid_y]);

//   T res = 0.;
// #pragma unroll
//   for (int i = 1; i < n_iter; i++) {
//     blockA[tid_y][tid_x] = blockAPre;
//     blockB[tid_x][tid_y] = blockBPre;
//     __syncthreads();

//     blockAPre = FETCH_FLOAT4(matA[y_index * num + (i * blockSize + tid_x)]);
//     blockBPre = FETCH_FLOAT4(matB[x_index * num + (i * blockSize + tid_y)]);

// #pragma unroll
//     for (int j = 0; j < blockSize / 4 && i * blockSize * 4 + j * 4 < num;
//     j++)
//       res += blockA[tid_y][j].x * blockB[tid_x][j].x +
//              blockA[tid_y][j].y * blockB[tid_x][j].y +
//              blockA[tid_y][j].z * blockB[tid_x][j].z +
//              blockA[tid_y][j].w * blockB[tid_x][j].w;
//   }
//   __syncthreads();
//   matC[y_index * width + x_index] = res;
// }

// void matmul_cublas(float *A, float *B, float *C, int m, int n, int k,
//                    const float alpha, const float beta) {
//   cublasHandle_t handle;
//   cublasCreate(&handle);

//   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k,
//               &beta, C, n);
// }

void matmul(float *matA, float *matB, float *matC, size_t height, size_t width,
            size_t num, int nAlgo) {
  dim3 block(blockSize, blockSize);
  dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1);

  switch (nAlgo) {
    case 0:
      matmul0<float><<<grid, block>>>(matA, matB, matC, height, width, num);
      break;
    case 1:
      matmul1<float><<<grid, block>>>(matA, matB, matC, height, width, num);
      break;
    case 2:
      matmul2<float><<<grid, block>>>(matA, matB, matC, height, width, num);
      break;
      // case 3:
      //   matmul3<float><<<grid, block>>>(matA, matB, matC, height, width,
      //   num); break;
      // case 4:
      //   matmul4<float><<<grid, block>>>(matA, matB, matC, height, width,
      //   num); break;
      // case 5:
      //   matmul_cublas(matA, matB, matC, height, width, num, 1.0, 0.);
      //   break;

    default:
      break;
  }
}
