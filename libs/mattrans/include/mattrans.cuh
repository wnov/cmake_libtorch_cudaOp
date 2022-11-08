#ifndef __MAT_TRANS_H__
#define __MAT_TRANS_H__

#include <cuda_runtime.h>

constexpr int blockSize = 32;
extern "C" {}
__global__ void matTranspose(float *mat, float *matO, int height, int width);

#endif  //__MATMUL_H__