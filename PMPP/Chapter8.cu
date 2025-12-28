#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdio.h>

#define TILE_SIZE 32
#define MAX_MASK_WIDTH 10
__constant__ float M[MAX_MASK_WIDTH];
__constant__ float M_C[MAX_MASK_WIDTH][MAX_MASK_WIDTH];

__global__ void convulution_1D_basic_kernel(float *N, float* M, float* P,
  int Mask_Width, int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	float Pvalue = 0.0f;
	int N_start_point = i - (Mask_Width / 2);

	for (int j = 0; j < Mask_Width; ++j)
	{
		if ((N_start_point + j >= 0) && (N_start_point + j < Width))
			Pvalue += N[N_start_point+j] * M[j];
	}

	P[i] = Pvalue;
}

__global__ void convulution_1D_basic_kernel(float *N, float* P,
	int Mask_Width, int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	float Pvalue = 0.0f;
	int N_start_point = i - (Mask_Width / 2);

	for (int j = 0; j < Mask_Width; ++j)
	{
		if ((N_start_point + j >= 0) && (N_start_point + j < Width))
			Pvalue += N[N_start_point + j] * M[j];
	}

	P[i] = Pvalue;
}

__global__ void convulution_1D_basic_kernel_tiled(float *N, float* P,
	int Mask_Width, int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];

	int n = Mask_Width / 2;

	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x >= blockDim.x - n) 
	{
		N_ds[threadIdx.x - (blockDim.x - n)] =
			(halo_index_left < 0) ? 0 : N[halo_index_left];
	}

	N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];

	int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x < n) 
	{
		N_ds[n + blockDim.x + threadIdx.x] =
			(halo_index_right >= Width) ? 0 : N[halo_index_right];
	}

	__syncthreads();

	float Pvalue = 0;
	for (int j = 0; j < Mask_Width; j++) 
	{
		Pvalue += N_ds[threadIdx.x + j] * M[j];
	}

	P[i] = Pvalue;
}

__global__ void convolution_1D_basic_kernel(float *N, float *P, 
  int Mask_Width, int Width) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_SIZE];

	N_ds[threadIdx.x] = N[i];

	__syncthreads();

	int This_tile_start_point = blockIdx.x * blockDim.x;
	int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
	int N_start_point = i - (Mask_Width / 2);

	float Pvalue = 0;
	for (int j = 0; j < Mask_Width; j++) 
	{
		int N_index = N_start_point + j;

		if (N_index >= 0 && N_index < Width) 
		{
			if ((N_index >= This_tile_start_point) && (N_index<Next_tile_start_point))
					Pvalue += N_ds[threadIdx.x + j - (Mask_Width / 2)] * M[j];
		}
		else {
			Pvalue += N[N_index] * M[j];
		}
	}

	P[i] = Pvalue;
}

__global__ void convulution_2D_basic(float *N, float* M, float* P,
	int maskWidthX, int maskWidthY, 
	int numRowsN, int numColsN)
{

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int nPos = row*numColsN + col;
	int nPosX = 0, nPosY = 0;
	int offsetX = maskWidthX / 2;
	int offsetY = maskWidthY / 2;

	float Pvalue = 0.0f;
	bool validXPos = false;
	bool validYPos = false;

	for (int i = 0; i < maskWidthY; ++i)
	{
		for (int j = 0; j < maskWidthX; ++j)
		{
			nPosX = col - offsetX + j;
			nPosY = row - offsetY + i;
			validXPos = (nPosX >= 0 && nPosX < numColsN) ? true : false;
			validYPos = (nPosY >= 0 && nPosY < numRowsN) ? true : false;

			if (validXPos && validYPos)
				Pvalue += M[i*maskWidthX + j] * N[nPosY*numColsN + nPosX];

		};
	};

	if (nPos < numRowsN*numColsN)
		P[nPos] = Pvalue;
}

void excercise_8_8()
{
	//initialize inputs
	size_t matrixDim = 8;
	size_t matrixSize = matrixDim*matrixDim;

	size_t maskDim = 3;
	size_t maskSize = maskDim*maskDim;

	//allocate host matrices and generate matrix
	float* h_n = new float[matrixSize];
	float* h_m = new float[maskSize];
	float* h_p = new float[matrixSize];

	for (int i = 0; i < matrixSize; ++i)
	{
		h_n[i] = i;
		h_p[i] = 0.0f;
	}

	for (int i = 0; i < maskSize; ++i)
	{
		h_m[i] = i;
	}

	//allocate device matrices
	cudaError_t cudaStatus;
	size_t sizeDeviceN = sizeof(float)*matrixSize;
	size_t sizeDeviceM = sizeof(float)*maskSize;
	float *d_n, *d_p, *d_m;

	cudaStatus = cudaMalloc(&d_n, sizeDeviceN);
	cudaStatus = cudaMalloc(&d_p, sizeDeviceN);
	cudaStatus = cudaMalloc(&d_m, sizeDeviceM);

	//transfer data from host to device
	cudaStatus = cudaMemcpy(d_n, h_n, sizeDeviceN, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_m, h_m, sizeDeviceM, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_p, 0, sizeDeviceN);

	//launch kernel
	dim3 grid(2, 2);
	dim3 block(4, 4);

	convulution_2D_basic << <grid, block >> >(d_n,d_m,d_p,maskDim,maskDim,matrixDim,matrixDim);

	//copy results
	cudaStatus = cudaMemcpy(h_p, d_p, sizeDeviceN, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < matrixSize; ++i) {
		std::cout << h_p[i] << ",";
		if( (i+1) % matrixDim == 0)
			std::cout<<std::endl;
	}

	//release resources
	delete[] h_n;
	delete[] h_m;
	delete[] h_p;
	cudaStatus = cudaFree(d_n);
	cudaStatus = cudaFree(d_m);
	cudaStatus = cudaFree(d_p);

}

__global__ void convulution_2D_const(float *N, float* P,
	int maskWidthX, int maskWidthY,
	int numRowsN, int numColsN)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int nPos = row*numColsN + col;
	int nPosX = 0, nPosY = 0;
	int offsetX = maskWidthX / 2;
	int offsetY = maskWidthY / 2;

	float Pvalue = 0.0f;
	bool validXPos = false;
	bool validYPos = false;

	for (int i = 0; i < maskWidthY; ++i)
	{
		for (int j = 0; j < maskWidthX; ++j)
		{
			nPosX = col - offsetX + j;
			nPosY = row - offsetY + i;
			validXPos = (nPosX >= 0 && nPosX < numColsN) ? true : false;
			validYPos = (nPosY >= 0 && nPosY < numRowsN) ? true : false;

			if (validXPos && validYPos)
				Pvalue += M[i*maskWidthX + j] * N[nPosY*numColsN + nPosX];

		};
	};

	if (nPos < numRowsN*numColsN)
		P[nPos] = Pvalue;
}

void excercise_8_9()
{
	//initialize inputs
	size_t matrixDim = 8;
	size_t matrixSize = matrixDim*matrixDim;

	size_t maskDim = 3;
	size_t maskSize = maskDim*maskDim;

	//allocate host matrices and generate matrix
	float* h_n = new float[matrixSize];
	float* h_m = new float[maskSize];
	float* h_p = new float[matrixSize];

	for (int i = 0; i < matrixSize; ++i)
	{
		h_n[i] = i;
		h_p[i] = 0.0f;
	}

	for (int i = 0; i < maskSize; ++i)
	{
		h_m[i] = i;
		std::cout << h_m[i] << std::endl;
	}
	

	//allocate device matrices
	cudaError_t cudaStatus;
	size_t sizeDeviceN = sizeof(float)*matrixSize;
	size_t sizeDeviceM = sizeof(float)*maskSize;
	float *d_n, *d_p;

	cudaStatus = cudaMalloc(&d_n, sizeDeviceN);
	cudaStatus = cudaMalloc(&d_p, sizeDeviceN);

	//transfer data from host to device
	cudaStatus = cudaMemcpy(d_n, h_n, sizeDeviceN, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_p, 0, sizeDeviceN);

	//transfer mask to constant memory
	cudaMemcpyToSymbol(M, h_m, sizeDeviceM);

	//launch kernel
	dim3 grid(2, 2);
	dim3 block(4, 4);

	convulution_2D_const << <grid, block >> >(d_n, d_p, maskDim, maskDim, matrixDim, matrixDim);

	//copy results
	cudaStatus = cudaMemcpy(h_p, d_p, sizeDeviceN, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < matrixSize; ++i) {
		std::cout << h_p[i] << ",";
		if ((i + 1) % matrixDim == 0)
			std::cout << std::endl;
	}

	//release resources
	delete[] h_n;
	delete[] h_p;
	delete[] h_m;
	cudaStatus = cudaFree(d_n);
	cudaStatus = cudaFree(d_p);

}

template < int widthX, int widthY>
__global__ void convulution_2D_const_tiled_shared(
	float *N, float* P,
	int maskWidthX, int maskWidthY,
	int numRowsN, int numColsN)
{
	__shared__ float N_ds[widthY][widthX];

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int nPos = row*numColsN + col;
	int nPosX = 0, nPosY = 0;
	int offsetX = maskWidthX / 2;
	int offsetY = maskWidthY / 2;

	//load data in shared memory
	int sharedCol = threadIdx.x;
	int sharedRow = threadIdx.y;

	//middle elements
	N_ds[sharedRow + offsetY][sharedCol + offsetX] = N[nPos];

	//for extra elements on left
	if (threadIdx.x < offsetX)
		N_ds[sharedRow + offsetY][sharedCol] = (col - offsetX >= 0) ? 
			N[row*numColsN + (col - offsetX)] : 0.0f;
		
	//for extra elements on right
	if (threadIdx.x >= (blockDim.x - offsetX))
		N_ds[sharedRow + offsetY][sharedCol + 2 * offsetX] = (col + offsetX < numColsN) ? 
			N[row*numColsN + (col + offsetX)] : 0.0f;

	//for extra elements on top
	if (threadIdx.y < offsetY)
		N_ds[sharedRow][sharedCol + offsetX] = (row - offsetY >= 0) ? 
			N[(row-offsetY)*numColsN + col] : 0.0f;

	//for extra elements on bottomw
	if (threadIdx.y >= (blockDim.y - offsetY))
		N_ds[sharedRow + 2 * offsetX][sharedCol + offsetX] = (row + offsetY < numRowsN) ? 
			N[(row + offsetY)*numColsN + col] : 0.0f;

	//for extra elements on corners
	if (threadIdx.y < offsetY && threadIdx.x < offsetX)
	{
		//top left
		N_ds[sharedCol][sharedRow] = ((col - offsetX >= 0) && (row - offsetY >= 0)) ? 
			N[(row - offsetY)*numColsN + col - offsetX] : 0.0f;
		//top right
		N_ds[sharedRow][sharedCol + blockDim.x + offsetX] = ((col + offsetX + blockDim.x < numColsN) && (row - offsetY >= 0)) ?
			N[(row - offsetY)*numColsN + (col + blockDim.x)] : 0.0f;
		//bottom right
		N_ds[sharedRow + blockDim.y + offsetY][sharedCol] = ((row + offsetY + blockDim.y < numRowsN) && (col - offsetX >= 0)) ?
			N[(row + blockDim.y)*numColsN + col - offsetX]: 0.0f;
		//bottom left
		N_ds[sharedRow + blockDim.y + offsetY][sharedCol + blockDim.x + offsetX] =
			((row + offsetY + blockDim.y < numRowsN) && (col + offsetX + blockDim.x < numColsN)) ?
			N[(row + blockDim.y)*numColsN + (col + blockDim.x)] : 0.0f;
	}

	__syncthreads();

	//compute convulution
	float Pvalue = 0.0f;
	bool validXPos = false;
	bool validYPos = false;

	for (int i = 0; i < maskWidthY; ++i)
	{
		for (int j = 0; j < maskWidthX; ++j)
		{
			//nPosX = col - offsetX + j;
			//nPosY = row - offsetY + i;
			//validXPos = (nPosX >= 0 && nPosX < numColsN) ? true : false;
			//validYPos = (nPosY >= 0 && nPosY < numRowsN) ? true : false;
			Pvalue += M[i*maskWidthX + j] * N_ds[threadIdx.y+i][threadIdx.x+j];

		};
	};

	if (nPos < numRowsN*numColsN)
		P[nPos] = Pvalue;
}

void excercise_8_10()
{
	//initialize inputs
	size_t matrixDim = 8;
	size_t matrixSize = matrixDim*matrixDim;

	size_t maskDim = 3;
	size_t maskSize = maskDim*maskDim;

	//allocate host matrices and generate matrix
	float* h_n = new float[matrixSize];
	float* h_m = new float[maskSize];
	float* h_p = new float[matrixSize];

	for (int i = 0; i < matrixSize; ++i)
	{
		h_n[i] = i;
		h_p[i] = 0.0f;
	}

	for (int i = 0; i < maskSize; ++i)
	{
		h_m[i] = i;
		std::cout << h_m[i] << std::endl;
	}


	//allocate device matrices
	cudaError_t cudaStatus;
	size_t sizeDeviceN = sizeof(float)*matrixSize;
	size_t sizeDeviceM = sizeof(float)*maskSize;
	float *d_n, *d_p;

	cudaStatus = cudaMalloc(&d_n, sizeDeviceN);
	cudaStatus = cudaMalloc(&d_p, sizeDeviceN);

	//transfer data from host to device
	cudaStatus = cudaMemcpy(d_n, h_n, sizeDeviceN, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(d_p, 0, sizeDeviceN);

	//transfer mask to constant memory
	cudaMemcpyToSymbol(M, h_m, sizeDeviceM);

	//launch kernel
	dim3 grid(2, 2);
	dim3 block(4, 4);

	convulution_2D_const_tiled_shared<6,6> << <grid, block >> >(d_n, d_p, maskDim, maskDim, matrixDim, matrixDim);

	//copy results
	cudaStatus = cudaMemcpy(h_p, d_p, sizeDeviceN, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < matrixSize; ++i) {
		std::cout << h_p[i] << ",";
		if ((i + 1) % matrixDim == 0)
			std::cout << std::endl;
	}

	//release resources
	delete[] h_n;
	delete[] h_p;
	delete[] h_m;
	cudaStatus = cudaFree(d_n);
	cudaStatus = cudaFree(d_p);
}