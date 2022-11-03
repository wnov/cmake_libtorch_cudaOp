#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <iostream>

#include "matmul.h"

int main(int argc, char** argv) {
  int algo_n;
  if (argc > 1)
    algo_n = std::stoi(argv[1]);
  else
    algo_n = 0;
  size_t height = 100, width = 10, num = 100;
  thrust::device_vector<float> matA(10000, 1), matB(1000, 11), matC(1000);
  dim3 block(blockSize, blockSize);
  dim3 grid(width / blockSize, height / blockSize);
  switch (algo_n) {
    case 0:
      matmul0<float><<<grid, block>>>(matA.data().get(), matB.data().get(),
                                      matC.data().get(), height, width, num);
      break;
    case 1:
      matmul1<float><<<grid, block>>>(matA.data().get(), matB.data().get(),
                                      matC.data().get(), height, width, num);
      break;
    case 2:
      matmul2<float><<<grid, block>>>(matA.data().get(), matB.data().get(),
                                      matC.data().get(), height, width, num);
      break;
    case 3:
      matmul3<float><<<grid, block>>>(matA.data().get(), matB.data().get(),
                                      matC.data().get(), height, width, num);
      break;
    case 4:
      matmul4<float><<<grid, block>>>(matA.data().get(), matB.data().get(),
                                      matC.data().get(), height, width, num);
      break;
    case 5:
      matmul_cublas(matA.data().get(), matB.data().get(), matC.data().get(),
                    height, width, num, 1., 0.);
      break;
    default:
      break;
  }
}