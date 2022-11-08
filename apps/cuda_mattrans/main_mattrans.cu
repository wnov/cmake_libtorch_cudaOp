#include <memory.h>

#include <string>

#include "helper/include/utils.h"
#include "mattrans/include/mattrans.cuh"

int main(int argc, char **argv) {
  int h = 4, w = 4;
  if (argc == 3) {
    h = std::stoi(argv[1]);
    w = std::stoi(argv[2]);
  }

  printf("Initializing host memory...\n");
  float *mA = (float *)malloc(h * w * sizeof(float));
  float *mB = (float *)malloc(h * w * sizeof(float));

  initialData(mA, h * w);
  printf("Initializing host memory finished.\n");

  printf("Initializing device memory...\n");
  float *mAd = nullptr, *mBd = nullptr;
  CHECK(cudaMalloc((void **)&mAd, h * w * sizeof(float)));
  CHECK(cudaMalloc((void **)&mBd, h * w * sizeof(float)));

  CHECK(cudaMemcpy(mAd, mA, h * w * sizeof(float), cudaMemcpyHostToDevice));
  printf("Initializing device memory finished.\n");

  cudaEvent_t start, finish;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&finish));
  CHECK(cudaEventRecord(start));

  printf("Start to calculation matrix multiplication...\n");

  dim3 block1(blockSize, blockSize);
  dim3 grid1((w - 1) / blockSize, (h - 1) / blockSize);
  matTranspose<<<grid1, block1>>>(mAd, mBd, h, w);
  dim3 block2(blockSize, blockSize);
  dim3 grid2((h - 1) / blockSize, (w - 1) / blockSize);
  matTranspose<<<grid2, block2>>>(mBd, mAd, w, h);
  CHECK(cudaEventRecord(finish));
  CHECK(cudaDeviceSynchronize());
  printf("after kernel function: %s\n", cudaGetErrorString(cudaGetLastError()));

  float t;
  CHECK(cudaEventElapsedTime(&t, start, finish));
  printf("MatTranspose computation finished in %f ms.\n", t);
  CHECK(cudaMemcpy(mB, mAd, h * w * sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(mA, mB, h * w);
  // printMat(mC, h, w);
  // printMat(mCH, h, w);
  printf("Task done!");

  CHECK(cudaFree(mAd));
  CHECK(cudaFree(mBd));

  free(mA);
  free(mB);

  return 0;
}