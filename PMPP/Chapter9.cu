#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#define SECTION_SIZE 8

__global__ void work_inefficient_inclusive_scan_kernel(float* x, float* y, int inputSize)
{
	__shared__ float XY[SECTION_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < inputSize)
		XY[threadIdx.x] = x[i];

	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2)
	{
		__syncthreads();
		XY[threadIdx.x] += XY[threadIdx.x - stride];
	}

	y[i] = XY[threadIdx.x];
}

__global__ void work_inefficient_exclusive_scan_kernel(float* x, float* y, int inputSize)
{
	__shared__ float XY[SECTION_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < inputSize && threadIdx.x != 0)
		XY[threadIdx.x] = x[i - 1];
	else
		XY[threadIdx.x] = 0.0f;

	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2)
	{
		__syncthreads();
		XY[threadIdx.x] += XY[threadIdx.x - stride];
	}

	y[i] = XY[threadIdx.x];
}

void excercise_9_4()
{
	size_t length = 256;

	//allocate host data
	float* h_vecIn = new float[length];
	float* h_vecOut = new float[length];

	for (size_t i = 1; i <= length; ++i) {
		h_vecIn[i-1] = i;
		std::cout << h_vecIn[i-1] << std::endl;
	}

	//allocate device data
	cudaError_t cudaStatus;
	size_t sizeDeviceMem = sizeof(float)*length;
	float *d_vecIn, *d_vecOut;

	cudaStatus = cudaMalloc(&d_vecIn, sizeDeviceMem);
	cudaStatus = cudaMalloc(&d_vecOut, sizeDeviceMem);

	//transfer data to device
	cudaStatus = cudaMemcpy(d_vecIn, h_vecIn, sizeDeviceMem, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_vecOut, 0, sizeDeviceMem);

	//kernel launch
	int numBlocks = 1;
	int numThreads = length;
	work_inefficient_exclusive_scan_kernel << <numBlocks, numThreads >> >(d_vecIn, d_vecOut, length);

	//trasnfer results to host
	cudaStatus = cudaMemcpy(h_vecOut, d_vecOut, sizeDeviceMem, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < length; ++i) {
		std::cout << h_vecOut[i] << std::endl;
	}

	//release resources
	delete[] h_vecIn;
	delete[] h_vecOut;
	cudaStatus = cudaFree(d_vecIn);
	cudaStatus = cudaFree(d_vecOut);
}

__global__ void work_efficient_scan_kernel(float *x, float *y, int inputSize)
{
	__shared__ float XY[SECTION_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < inputSize)
		XY[threadIdx.x] = x[i];

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();

		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < blockDim.x)
			XY[index] += XY[index - stride];
	}

	for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();

		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < blockDim.x) {
			XY[index + stride] += XY[index];
		}
	}

	__syncthreads();

	y[i] = XY[threadIdx.x];
}

__global__ void work_efficient_scan_hierch_kernel(float *x, float *y,float* s, int inputSize, int level)
{
	__shared__ float XY[SECTION_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < inputSize)
		XY[threadIdx.x] = x[i];

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();

		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < blockDim.x)
			XY[index] += XY[index - stride];
	}

	for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();

		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < blockDim.x) {
			XY[index + stride] += XY[index];
		}
	}

	__syncthreads();

	y[i] = XY[threadIdx.x];

	if (threadIdx.x == 0 && level == 1)
		s[blockIdx.x] = XY[SECTION_SIZE-1];
}

__global__ void update_scan_blocks(float *y, float *s, int inputSize)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int offset = SECTION_SIZE;

	y[i + SECTION_SIZE] += s[blockIdx.x];
}

void excercise_9_5()
{
	size_t length = 32;

	//allocate host data
	float* h_vecIn = new float[length];
	float* h_vecOut = new float[length];
	float* h_vecS = new float[length / SECTION_SIZE];

	for (size_t i = 1; i <= length; ++i) {
		h_vecIn[i - 1] = i;
		std::cout << h_vecIn[i - 1] << std::endl;
	}

	//allocate device data
	cudaError_t cudaStatus;
	size_t sizeDeviceMem = sizeof(float)*length;
	size_t sizeDeviceS = sizeof(float)*(length / SECTION_SIZE);
	float *d_vecIn, *d_vecOut, *d_vecS;

	cudaStatus = cudaMalloc(&d_vecIn, sizeDeviceMem);
	cudaStatus = cudaMalloc(&d_vecOut, sizeDeviceMem);
	cudaStatus = cudaMalloc(&d_vecS, sizeDeviceS);

	//transfer data to device
	cudaStatus = cudaMemcpy(d_vecIn, h_vecIn, sizeDeviceMem, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_vecOut, 0, sizeDeviceMem);
	cudaStatus = cudaMemset(d_vecS, 0, sizeDeviceS);

	//kernel launch for step 1
	int numBlocks = length/SECTION_SIZE;
	int numThreads = SECTION_SIZE;
	work_efficient_scan_hierch_kernel << <numBlocks, numThreads >> >(d_vecIn, d_vecOut,d_vecS, length,1);

	/*
	//check step 1 results
	
	cudaStatus = cudaMemcpy(h_vecOut, d_vecOut, sizeDeviceMem, cudaMemcpyDeviceToHost);

	std::cout << "Block by block scan" << std::endl;
	for (size_t i = 0; i < length; ++i) {
		std::cout << h_vecOut[i] << std::endl;
	}

	cudaStatus = cudaMemcpy(h_vecS, d_vecS, sizeDeviceS, cudaMemcpyDeviceToHost);

	std::cout << "S vector"<< std::endl;
	for (size_t i = 0; i < numBlocks; ++i) {
		std::cout << h_vecS[i] << std::endl;
	}
	*/

	//kernel launch step 2
	work_efficient_scan_kernel << <1, numBlocks >> >(d_vecS, d_vecS, numBlocks);

	/*
	//check step 2 results
	cudaStatus = cudaMemcpy(h_vecS, d_vecS, sizeDeviceS, cudaMemcpyDeviceToHost);

	std::cout << "S vector after scan" << std::endl;
	for (size_t i = 0; i < numBlocks; ++i) {
		std::cout << h_vecS[i] << std::endl;
	}
	*/

	//kernel launch step 3
	int numBlocks3 = numBlocks - 1;
	update_scan_blocks << <numBlocks3, numThreads >> >(d_vecOut, d_vecS, length);

	cudaStatus = cudaMemcpy(h_vecOut, d_vecOut, sizeDeviceMem, cudaMemcpyDeviceToHost);

	std::cout << "Final Result" << std::endl;
	for (size_t i = 0; i < length; ++i) {
		std::cout << h_vecOut[i] << std::endl;
	}

	//release resources
	delete[] h_vecIn;
	delete[] h_vecOut;
	delete[] h_vecS;
	cudaStatus = cudaFree(d_vecIn);
	cudaStatus = cudaFree(d_vecOut);
	cudaStatus = cudaFree(d_vecS);

}