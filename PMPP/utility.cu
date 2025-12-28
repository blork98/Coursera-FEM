#include "utility.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

__global__ void matrix_mul_kernel(float* d_M, float* d_N, float* d_P, int width)
{
	// Calculate the row index of the d_P element and d_M
	int Row = blockIdx.y*blockDim.y + threadIdx.y;

	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((Row < width) && (Col < width))
	{
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < width; ++k)
		{
			Pvalue += d_M[Row*width + k] * d_N[k*width + Col];
		}

		d_P[Row*width + Col] = Pvalue;
	}
}

__global__ void matrix_mul_kernel_t(float* d_M, float* d_N, float* d_P, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int  tx = threadIdx.x, ty = threadIdx.y;

	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;

	float Pvalue = 0;

	for (int m = 0; m < Width / TILE_WIDTH; ++m)
	{
		Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];

		__syncthreads();

	}

	d_P[Row*Width+Col] = Pvalue;
}