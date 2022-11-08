#ifndef __UTILS_H__
#define __UTILS_H__

#define CHECK(error)                                                    \
  {                                                                     \
    if (error != cudaSuccess) {                                         \
      printf("ERROR: %s:%d", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s", error, cudaGetErrorString(error)); \
    }                                                                   \
  }

#define FETCH_FLOAT4(x) *reinterpret_cast<float4 *>(&x)

template <typename T>
void printMat(T *mat, int h, int w);
void matmul_cpu(float *A, float *B, float *C, int m, int n, int k);
void checkResult(float *hostRef, float *gpuRef, const int N);
void initialData(float *ip, int size);

#endif  //__MATMUL_H__