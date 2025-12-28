#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>

__global__ void matrix_addition_b(float* a, float* b, float* c, int n)
{
	int i = blockIdx.x* blockDim.x + threadIdx.x;

	if (i < n*n)
		c[i] = a[i] + b[i];
}

__global__ void matrix_addition_c(float* a, float* b, float* c, int n)
{
	int i = (blockIdx.x* blockDim.x + threadIdx.x)*n;

	for (int offset = 0; offset < n; ++offset)
		if (i < n*n)
			c[i+offset] = a[i + offset] + b[i + offset];
}

__global__ void matrix_addition_d(float* a, float* b, float* c, int n)
{
	int i = blockIdx.x* blockDim.x + threadIdx.x;

	for (int offset = 0; offset < n; ++offset)
		if (i < n*n)
			c[i + offset*n] = a[i + offset*n] + b[i + offset*n];
}

void matrix_addition()
{
	size_t matrixDim = 16;
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
	int numBlocks = 0;
	int threadsPerBlock = 0;

	//part b
	//numBlocks = 2;
	//threadsPerBlock = ceil(matrixSize / numBlocks);
	//matrix_addition_b<<<numBlocks, threadsPerBlock>>>(d_a,d_b,d_c, matrixDim);

	//part c
	//numBlocks = 2;
	//threadsPerBlock = ceil(matrixDim / numBlocks);
	//matrix_addition_c<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, matrixDim);

	//part d
	numBlocks = 2;
	threadsPerBlock = ceil(matrixDim / numBlocks);
	matrix_addition_d<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, matrixDim);

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

__global__ void matrix_vector_mult(float* mat, float* vecIn, float* vecOut, int vecDim, int matDim)
{
	int i = blockIdx.x* blockDim.x + threadIdx.x;

	for (int ctr = 0; ctr < matDim; ++ctr)
		vecOut[i] += vecIn[ctr] * mat[ctr + i*matDim];
}

void matrix_vector_mult()
{
	size_t matrixDim = 16;
	size_t matrixSize = matrixDim*matrixDim;
	size_t vectorDim = 16;

	//allocate host matrices and generate matrix
	float* h_mat    = new float[matrixSize];
	float* h_vecIn  = new float[vectorDim];
	float* h_vecOut = new float[vectorDim];

	for (size_t i = 0; i < matrixSize; ++i) {
		h_mat[i] = i;
		std::cout << h_mat[i] << std::endl;
	};
	
	for (size_t i = 0; i < vectorDim; ++i) {
		h_vecIn[i]  = i;
		h_vecOut[i] = 0;
		std::cout << h_vecIn[i] << "," << h_vecOut[i] << std::endl;
	}

	//allocate device matrices
	cudaError_t cudaStatus;
	size_t sizeDeviceMemMatrix = sizeof(float)*matrixSize;
	size_t sizeDeviceMemVector = sizeof(float)*vectorDim;
	float *d_mat, *d_vecIn, *d_vecOut;

	cudaStatus = cudaMalloc(&d_mat, sizeDeviceMemMatrix);
	cudaStatus = cudaMalloc(&d_vecIn, sizeDeviceMemVector);
	cudaStatus = cudaMalloc(&d_vecOut, sizeDeviceMemVector);

	//transfer data from host to device
	cudaStatus = cudaMemcpy(d_mat, h_mat, sizeDeviceMemMatrix, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_vecIn, h_vecIn, sizeDeviceMemVector, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_vecOut, 0, sizeDeviceMemVector);

	//launch kernel
	int numBlocks = 2;
	int threadsPerBlock = ceil(vectorDim / numBlocks);

	matrix_vector_mult << <numBlocks, threadsPerBlock >> >(d_mat,d_vecIn,d_vecOut,vectorDim, matrixDim);

	//trasnfer results to host
	cudaStatus = cudaMemcpy(h_vecOut, d_vecOut, sizeDeviceMemVector, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < vectorDim; ++i) {
		std::cout << h_vecOut[i] << std::endl;
	}

	//release resources
	delete[] h_mat;
	delete[] h_vecOut;
	delete[] h_vecIn;
	cudaStatus = cudaFree(d_mat);
	cudaStatus = cudaFree(d_vecOut);
	cudaStatus = cudaFree(d_vecIn);
}