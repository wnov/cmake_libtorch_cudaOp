#include "utils.h"

#include <stdio.h>

#include <cmath>
#include <ctime>

template <typename T>
void printMat(T *mat, int h, int w) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      printf("%.2f, ", mat[i * w + j]);
    }
    printf("\n");
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i],
             i);
      return;
    }
  }
  printf("Check result success!\n");
}

void matmul_cpu(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int h = 0; h < k; h++) {
        C[i * n + j] += A[i * k + h] * B[h * n + j];
      }
    }
  }
}

void initialData(float *ip, int size) {
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xffff) / 1000.0f;
  }
}