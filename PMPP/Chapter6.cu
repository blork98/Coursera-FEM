#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>

#include "utility.h"


__global__ void kernel_6_2(float* vec, float* vecOut, int vecLength)
{
	extern __shared__ float partialSum[];													//1

	unsigned int t = threadIdx.x;															//2
	unsigned int globIndex = blockIdx.x*blockDim.x + threadIdx.x;

	//load data into shared memory
	partialSum[t] = vec[globIndex];

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)							//3
	{
		__syncthreads();																	//5

		if (t % (2 * stride) == 0)															//6
			partialSum[t] += partialSum[t + stride];										//7
	}

	vecOut[blockIdx.x] = partialSum[0];
}

__global__ void kernel_6_2_modified(float* vec, float* vecOut, int vecLength)
{
	extern __shared__ float partialSum[];													//1

	unsigned int t = threadIdx.x;															//2
	unsigned int globIndex = blockIdx.x*blockDim.x + threadIdx.x;

	//load data into shared memory
	unsigned int offset = gridDim.x*blockDim.x;
	partialSum[t] = vec[globIndex];
	partialSum[t] += vec[globIndex + offset];

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)							//3
	{
		__syncthreads();																	//5

		if (t % (2 * stride) == 0)															//6
			partialSum[t] += partialSum[t + stride];										//7
	}

	vecOut[blockIdx.x] = partialSum[0];
}

__global__ void kernel_6_4(float* vec, float* vecOut, int vecLength)
{
	extern __shared__ float partialSum[];
	
	unsigned int t = threadIdx.x;															//2
	unsigned int globIndex = blockIdx.x*blockDim.x + threadIdx.x;

	//load data into shared memory
	partialSum[t] = vec[globIndex];

	for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2)
	{
		__syncthreads();

		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}

	vecOut[blockIdx.x] = partialSum[0];
}

__global__ void kernel_6_4_modified(float* vec, float* vecOut, int vecLength)
{
	extern __shared__ float partialSum[];

	unsigned int t = threadIdx.x;															//2
	unsigned int globIndex = 2*blockIdx.x*blockDim.x + threadIdx.x;

	//load data into shared memory
	partialSum[t] = vec[globIndex];
	partialSum[t] += vec[globIndex + blockDim.x];

	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
	{
		__syncthreads();

		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}

	vecOut[blockIdx.x] = partialSum[0];
}

void excercise_6_1()
{

	size_t length = 256;

	//launch params
	int numBlocks = 2;
	int numThreads = ceil(length / numBlocks);

	//allocate host data
	float* h_vec = new float[length];
	float* h_vecOut = new float[numBlocks];

	for (size_t i = 0; i < length; ++i) {
		h_vec[i] = i;
		std::cout << h_vec[i] << std::endl;
	}

	//allocate device data
	cudaError_t cudaStatus;
	size_t sizeDeviceMem = sizeof(float)*length;
	size_t sizeDeviceOut = sizeof(float)*numBlocks;
	float *d_vec, *d_vecOut;

	cudaStatus = cudaMalloc(&d_vec, sizeDeviceMem);
	cudaStatus = cudaMalloc(&d_vecOut, sizeDeviceMem);

	//transfer data to device
	cudaStatus = cudaMemcpy(d_vec, h_vec, sizeDeviceMem, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_vecOut, 0, sizeDeviceOut);

	//kernel launch

	//original 6_2
	//size_t sharedmemsize = length*sizeof(float);
	//kernel_6_2 << <numBlocks, numThreads, sharedmemsize >> >(d_vec,d_vecOut,length);

	//modified 6_2 for launching half the number of threads
	//size_t sharedmemsize = length*sizeof(float) / 2;
	//numBlocks = 2;
	//numThreads = ceil(length / numBlocks)/2;
	//kernel_6_2_modified << <numBlocks, numThreads, sharedmemsize >> >(d_vec, d_vecOut, length);

	//original 6_4
	//size_t sharedmemsize = length*sizeof(float)/2;
	//kernel_6_4 << <numBlocks, numThreads, sharedmemsize >> >(d_vec, d_vecOut, length);

	//modified 6_4
	size_t sharedmemsize = length*sizeof(float) / 2;
	numBlocks = 2;
	numThreads = ceil(length / numBlocks) / 2;
	kernel_6_4_modified << <numBlocks, numThreads, sharedmemsize >> >(d_vec, d_vecOut, length);

	//trasnfer results to host
	cudaStatus = cudaMemcpy(h_vecOut, d_vecOut, sizeDeviceOut, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < numBlocks; ++i) {
		std::cout << h_vecOut[i] << std::endl;
	}

	//release resources
	delete[] h_vec;
	delete[] h_vecOut;
	cudaStatus = cudaFree(d_vec);
	cudaStatus = cudaFree(d_vecOut);
}

#define TILE_WIDTH_C 2

__global__ void matrix_mul_kernel_tiled_comb(float* d_M, float* d_N, float* d_P, int Width)
{
	__shared__ float Mds[TILE_WIDTH_C][TILE_WIDTH_C];
	__shared__ float Nds1[TILE_WIDTH_C][TILE_WIDTH_C];
	__shared__ float Nds2[TILE_WIDTH_C][TILE_WIDTH_C];

	int bx = blockIdx.x, by = blockIdx.y;
	int  tx = threadIdx.x, ty = threadIdx.y;

	int Row = by*TILE_WIDTH_C + ty;
	int Col1 = bx*TILE_WIDTH_C*2 + tx;
	int Col2 = bx*TILE_WIDTH_C*2 + tx + TILE_WIDTH_C;

	float Pvalue1 = 0;
	float Pvalue2 = 0;

	for (int m = 0; m < Width / TILE_WIDTH_C; ++m)
	{
		Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH_C + tx];
		Nds1[ty][tx] = d_N[(m*TILE_WIDTH_C + ty)*Width + Col1];
		Nds2[ty][tx] = d_N[(m*TILE_WIDTH_C + ty)*Width + Col2];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH_C; ++k)
		{
			Pvalue1 += Mds[ty][k] * Nds1[k][tx];
			Pvalue2 += Mds[ty][k] * Nds2[k][tx];
		}

		__syncthreads();

	}

	d_P[Row*Width + Col1] = Pvalue1;
	d_P[Row*Width + Col2] = Pvalue2;
}

void excercise_6_10()
{
	size_t matrixDim = 8;
	size_t matrixSize = matrixDim*matrixDim;

	//allocate host matrices and generate matrix
	float* h_a = new float[matrixSize];
	float* h_b = new float[matrixSize];
	float* h_c = new float[matrixSize];

	for (size_t i = 0; i < matrixSize; ++i) {
		h_a[i] = i;
		h_b[i] = i;
		h_c[i] = 0.f;
		std::cout << h_a[i] << "," << h_b[i] << "," << h_c[i] << std::endl;
	}

	//allocate device matrices
	cudaError_t cudaStatus;
	size_t sizeDeviceMem = sizeof(float)*matrixSize;
	float *d_a, *d_b, *d_c;

	cudaStatus = cudaMalloc(&d_a, sizeDeviceMem);
	cudaStatus = cudaMalloc(&d_b, sizeDeviceMem);
	cudaStatus = cudaMalloc(&d_c, sizeDeviceMem);

	//transfer data from host to device
	cudaStatus = cudaMemcpy(d_a, h_a, sizeDeviceMem, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_b, h_b, sizeDeviceMem, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_c, 0, sizeDeviceMem);

	//launch kernel

	//for original tiled mat-mat mult
	//dim3 grid(2,2);
	//dim3 block(4, 4);
	//matrix_mul_kernel_t<< <grid, block >> >(d_a,d_b,d_c,matrixDim);

	//for combined tiled mat-mat mul
	//dim3 grid(2, 4);
	//dim3 block(2, 2);
	//matrix_mul_kernel_tiled_comb << <grid, block >> >(d_a, d_b, d_c, matrixDim);

	//trasnfer results to host
	cudaStatus = cudaMemcpy(h_c, d_c, sizeDeviceMem, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < matrixSize; ++i) {
		std::cout << h_c[i] << std::endl;
	}

	//release resources
	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	cudaStatus = cudaFree(d_a);
	cudaStatus = cudaFree(d_b);
	cudaStatus = cudaFree(d_c);
}