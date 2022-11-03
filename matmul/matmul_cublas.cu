#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matmul.h"

void matmul_cublas(float *A, float *B, float *C, int m, int n, int k,
                   const float alpha, const float beta) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k,
              &beta, C, n);
}
