#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <cuda_runtime.h>

#define TILE_WIDTH 4

__global__ void matrix_mul_kernel(float* d_M, float* d_N, float* d_P, int width);
__global__ void matrix_mul_kernel_t(float* d_M, float* d_N, float* d_P, int Width);

#endif