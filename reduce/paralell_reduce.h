#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifndef __PARALELL_REDUCE_H__
#define __PARALELL_REDUCE_H__
#include "cuda_runtime.h"

template <size_t blockSize>
__global__ void reduce0(int *g_idate, int *g_odata);

template <size_t blockSize>
__global__ void reduce1(int *g_idate, int *g_odata);

template <size_t blockSize>
__global__ void reduce2(int *g_idate, int *g_odata);

template <size_t blockSize>
__global__ void reduce3(int *g_idate, int *g_odata);

template <size_t blockSize>
__global__ void reduce4(int *g_idate, int *g_odata);

template <size_t blockSize>
__global__ void reduce5(int *g_idate, int *g_odata);

template <size_t blockSize>
__global__ void reduce6(int *g_idate, int *g_odata);

#endif
