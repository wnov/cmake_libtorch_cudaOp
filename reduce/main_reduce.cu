#include "cuda_runtime.h"
#include "paralell_reduce.h"

int main(void) {
  int d_idata[10] = {1, 1, 3, 1, 5, 2, 8}, d_odata[10];
  int threads = 1024;
  size_t dimGrid = 1, dimBlock = threads, smemSize = threads * sizeof(float);

  switch (threads) {
    case 512:
      reduce5<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 256:
      reduce5<256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 128:
      reduce5<128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 64:
      reduce5<64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 32:
      reduce5<32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 16:
      reduce5<16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 8:
      reduce5<8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 4:
      reduce5<4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 2:
      reduce5<2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
    case 1:
      reduce5<1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
      break;
  }
}