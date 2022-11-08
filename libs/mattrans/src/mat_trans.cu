#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "helper/include/utils.h"
#include "mattrans.cuh"

__global__ void matTranspose(float *mat, float *matO, int height, int width) {
  __shared__ float s_mat[blockSize][blockSize + 1];
  int tid_x = threadIdx.x, tid_y = threadIdx.y;
  int offset_x = blockIdx.x * blockDim.x;
  int offset_y = blockIdx.y * blockDim.y;

  if (offset_x + tid_x < width && offset_y + tid_y < height) {
    s_mat[tid_y][tid_x] = mat[(tid_y + offset_y) * width + offset_x + tid_x];
  }
  __syncthreads();
  if (offset_x + tid_x < width && offset_y + tid_y < height) {
    matO[(offset_x + tid_y) * height + offset_y + tid_x] = s_mat[tid_x][tid_y];
  }
}
