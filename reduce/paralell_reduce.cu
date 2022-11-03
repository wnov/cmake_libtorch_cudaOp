#ifndef __CUDACC__
#define __CUDACC__
#endif

// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #include <device_functions.h>

#include "cuda_runtime.h"
#include "paralell_reduce.h"
// // #include "device_launch_parameters.h"

// 交错寻址，分支分化，取模运算慢
__global__ void reduce0(int *g_idate, int *g_odata) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[index] = g_idate[index];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s <<= 1) {
    if (tid % (s * 2) == 0) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

// 交错寻址，优化线程束利用率
__global__ void reduce1(int *g_idate, int *g_odata) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[index] = g_idate[index];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s <<= 1) {
    int id = tid * s * 2;
    if (id < blockDim.x) {
      s_data[id] += s_data[id + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

// 并行规约
__global__ void reduce2(int *g_idate, int *g_odata) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[index] = g_idate[index];
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

// 加载数据预处理，增加block处理的数据范围
__global__ void reduce3(int *g_idate, int *g_odata) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  s_data[index] = g_idate[index] + g_idate[index + blockDim.x];
  __syncthreads();

  for (int s = blockDim.x; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

// 循环展开
__device__ void warpReduce(volatile int *s_data, int tid) {
  s_data[tid] = s_data[tid + 32];
  s_data[tid] = s_data[tid + 16];
  s_data[tid] = s_data[tid + 8];
  s_data[tid] = s_data[tid + 4];
  s_data[tid] = s_data[tid + 2];
  s_data[tid] = s_data[tid + 1];
}

__global__ void reduce4(int *g_idate, int *g_odata) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  s_data[index] = g_idate[index] + g_idate[index + blockDim.x];
  __syncthreads();

  for (int s = blockDim.x; s > 32; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) warpReduce(s_data, tid);

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

// 完全循环展开
template <size_t blockSize>
__device__ void warpReduce5(volatile int *s_data, int tid) {
  if (blockSize >= 64) s_data[tid] = s_data[tid + 32];
  if (blockSize >= 32) s_data[tid] = s_data[tid + 16];
  if (blockSize >= 16) s_data[tid] = s_data[tid + 8];
  if (blockSize >= 8) s_data[tid] = s_data[tid + 4];
  if (blockSize >= 4) s_data[tid] = s_data[tid + 2];
  if (blockSize >= 2) s_data[tid] = s_data[tid + 1];
}

template <size_t blockSize>
__global__ void reduce5(int *g_idate, int *g_odata) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockSize * 2 + threadIdx.x;
  s_data[index] = g_idate[index] + g_idate[index + blockDim.x];
  __syncthreads();

  if (blockSize >= 1024 && tid < 512) s_data[tid] += s_data[tid + 512];
  if (blockSize >= 512 && tid < 256) s_data[tid] += s_data[tid + 256];
  if (blockSize >= 256 && tid < 128) s_data[tid] += s_data[tid + 128];
  if (blockSize >= 128 && tid < 64) s_data[tid] += s_data[tid + 64];

  if (tid < 32) warpReduce5<blockSize>(s_data, tid);

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}

// Multiple Adds / Thread
template <size_t blockSize>
__global__ void reduce6(int *g_idate, int *g_odata, int n) {
  extern __shared__ int s_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockSize * 2 + threadIdx.x;
  size_t gridSize = 2 * gridDim.x * blockDim.x;
  s_data[tid] = 0;

  while (index < n) {
    s_data[index] += g_idate[index] + g_idate[index + blockSize];
    index += gridSize;
  }

  __syncthreads();

  if (blockSize >= 1024 && tid < 512) s_data[tid] += s_data[tid + 512];
  if (blockSize >= 512 && tid < 256) s_data[tid] += s_data[tid + 256];
  if (blockSize >= 256 && tid < 128) s_data[tid] += s_data[tid + 128];
  if (blockSize >= 128 && tid < 64) s_data[tid] += s_data[tid + 64];

  if (tid < 32) warpReduce5<blockSize>(s_data, tid);

  if (tid == 0) g_odata[blockIdx.x] = s_data[0];
}
