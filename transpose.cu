#ifndef __CUDACC__
#define __CUDACC__
#endif

static const int blockSize = 32;

template <typename T>
__device__ __host__ void matTranspose(T *mat, T *matO, int height, int width) {
  __shared__ T s_mat[blockDim][blockDim];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  s_mat[y][x] = mat[y][x];
  __syncthreads();
  matO[y][x] = s_mat[x][y];
}