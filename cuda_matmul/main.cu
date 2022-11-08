#include <memory.h>

#include <string>

#include "matmul.cuh"

void initialData(float *ip, int size) {
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xffff) / 1000.0f;
  }
}
void initialData_int(int *ip, int size) {
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = int(rand() & 0xff);
  }
}

int main(int argc, char **argv) {
  int nAlgo = 2, h = 4, w = 4, k = 4;
  if (argc > 1) {
    nAlgo = std::stoi(argv[1]);
  }
  if (argc == 5) {
    h = std::stoi(argv[2]);
    w = std::stoi(argv[3]);
    k = std::stoi(argv[4]);
  }

  printf("Initializing host memory...\n");
  float *mA = (float *)malloc(h * k * sizeof(float));
  float *mB = (float *)malloc(k * w * sizeof(float));
  float *mC = (float *)malloc(h * w * sizeof(float));
  float *mCH = (float *)malloc(h * w * sizeof(float));

  initialData(mA, h * k);
  initialData(mB, k * w);
  printf("Initializing host memory finished.\n");

  matmul_cpu(mA, mB, mCH, h, w, k);
  // printMat(mA, h, k);
  // printMat(mB, k, w);

  printf("Initializing device memory...\n");
  float *mAd = nullptr, *mBd = nullptr, *mCd = nullptr;
  CHECK(cudaMalloc((void **)&mAd, h * k * sizeof(float)));
  CHECK(cudaMalloc((void **)&mBd, k * w * sizeof(float)));
  CHECK(cudaMalloc((void **)&mCd, h * w * sizeof(float)));

  CHECK(cudaMemcpy(mAd, mA, h * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(mBd, mB, k * w * sizeof(float), cudaMemcpyHostToDevice));
  printf("Initializing device memory finished.\n");

  cudaEvent_t start, finish;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&finish));
  CHECK(cudaEventRecord(start));

  printf("Start to calculation matrix multiplication...\n");

  matmul(mAd, mBd, mCd, h, w, k, nAlgo);
  CHECK(cudaEventRecord(finish));
  CHECK(cudaDeviceSynchronize());
  printf("after kernel function: %s\n", cudaGetErrorString(cudaGetLastError()));

  float t;
  CHECK(cudaEventElapsedTime(&t, start, finish));
  printf("Matmul computation finished in %f ms.\n", t);

  CHECK(cudaMemcpy(mC, mCd, h * w * sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(mCH, mC, h * w);
  // printMat(mC, h, w);
  // printMat(mCH, h, w);
  printf("Task done!");

  CHECK(cudaFree(mAd));
  CHECK(cudaFree(mBd));
  CHECK(cudaFree(mCd));

  free(mA);
  free(mB);
  free(mC);

  return 0;
}