#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <cuda_runtime.h>

constexpr int blockSize = 32;
extern "C" {}
void matmul(float *matA, float *matB, float *matC, size_t height, size_t width,
            size_t num, int nAlgo);

#endif  //__MATMUL_H__