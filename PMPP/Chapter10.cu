#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

__global__ void SpMV_CSR(int num_rows, float *data, int *col_index,
	int *row_ptr, float *x, float *y) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < num_rows) 
	{
		float dot = 0;
		int row_start = row_ptr[row];
		int row_end = row_ptr[row + 1];
		
		for (int elem = row_start; elem < row_end; elem++) 
			dot += data[elem] * x[col_index[elem]];

		y[row] = dot;
	}
}

__global__ void SpMV_ELL(int num_rows, float *data, int *col_index,
	int num_elem, float *x, float *y) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < num_rows) 
	{
		float dot = 0; float data1 = 0.0; float x1 = 0.0;
		for (int i = 0; i < num_elem; i++)
		{
			data1 = data[row + i*num_rows];
			x1 = x[col_index[row + i*num_rows]];
			dot += data[row + i*num_rows] * x[col_index[row + i*num_rows]];
		}
		
		y[row] = dot;
	}
}

void create_ELL_COO_from_CSR(float* dataCSR, int* colIndexCSR, int* rowPtrCSR,
	float*& dataELL, int*& colIndexELL,
	float*& dataCOO, int*& colIndexCOO, int*& rowIndexCOO,
	int matrixDim, int numElems,
	int& numElemsPerRow, int& numCOOElems)
{
	//calculate number of elements in each row
	std::vector<int> elemsPerRow(matrixDim, 0);

	for (int i = 0; i < matrixDim; ++i)
		elemsPerRow[i] = rowPtrCSR[i + 1] - rowPtrCSR[i];

	size_t numElements = std::accumulate(elemsPerRow.begin(), elemsPerRow.end(), 0.0);
	size_t maxLength = *(std::max_element(elemsPerRow.begin(), elemsPerRow.end()));
	int cutoffSize = floor(numElements / elemsPerRow.size());

	//put all elements <= cutoffSize in ELL, everything else in COO

	//initialize COO
	//determine how many elements in COO
	size_t numElementsCOO = numElements - cutoffSize*matrixDim;
	dataCOO = new float[numElementsCOO];
	colIndexCOO = new int[numElementsCOO];
	rowIndexCOO = new int[numElementsCOO];
	numCOOElems = numElementsCOO;

	int start = 0, end = 0, cooCtr = 0;
	for (int rowPtr = 0; rowPtr < matrixDim; ++rowPtr)
	{
		if (elemsPerRow[rowPtr] > cutoffSize)
		{
			start = rowPtrCSR[rowPtr] + cutoffSize;
			end = rowPtrCSR[rowPtr + 1];
			for (int i = start; i < end; ++i)
			{
				dataCOO[cooCtr] = dataCSR[i];
				colIndexCOO[cooCtr] = colIndexCSR[i];
				rowIndexCOO[cooCtr] = rowPtr;
				++cooCtr;
			}
		}
	}

	//check COO elements
	/*
	for (int i = 0; i < numElementsCOO; ++i)
	{
		std::cout << dataCOO[i] << "," << colIndexCOO[i] << "," << rowIndexCOO[i] << std::endl;
	}
	*/

	//initialize ELL
	//determine how many elements in COO
	size_t numElementsELL = cutoffSize*matrixDim;
	dataELL = new float[numElementsELL]; 
	colIndexELL = new int[numElementsELL];
	memset(dataELL, 0.0, numElementsELL*sizeof(float));
	memset(colIndexELL, 0, numElementsELL*sizeof(int));

	int ellCtr = 0;
	int csrCtr = 0;
	
	for (int rowCtr = 0; rowCtr < matrixDim; ++rowCtr)
	{
		ellCtr = rowCtr;
		csrCtr = rowPtrCSR[rowCtr];
		for (int colCtr = 0; colCtr < cutoffSize; ++colCtr)
		{			
			dataELL[ellCtr] = dataCSR[csrCtr];
			colIndexELL[ellCtr] = colIndexCSR[csrCtr];
			++csrCtr;
			ellCtr += matrixDim;
		}
	}

	for (int i = 0; i < numElementsELL; ++i)
		std::cout << dataELL[i] << "," << colIndexELL[i] << std::endl;
	
	numElemsPerRow = cutoffSize;
}

void excercise_10_1()
{
	size_t matrixDim = 6;
	size_t vectorDim = 6;
	int numElems = 16;

	//initialize vectors x,y
	float* h_x = new float[vectorDim];
	float* h_y = new float[vectorDim];

	for (int i = 0; i < vectorDim; ++i)
	{
		h_x[i] = i;
		h_y[i] = 0.0;
		std::cout << h_x[i] << " , " << h_y[i] << std::endl;
	}

	//create sparse matrix

	//CSR Representation
	float* h_data = new float[numElems];
	int* h_colIndex = new int[numElems];
	int* h_rowPtr = new int[matrixDim + 1];

	h_data[0] = 1.0f; h_data[1] = 5.0f; 
	h_data[2] = 6.0f; h_data[3] = 10.0f;
	h_data[4] = 13.0f; h_data[5] = 17.0f;
	h_data[6] = 21.0f; h_data[7] = 22.0f;
	h_data[8] = 25.0f; h_data[9] = 26.0f; h_data[10] = 28.0f; h_data[11] = 29.0f;
	h_data[12] = 30.0f; h_data[13] = 32.0f; h_data[14] = 33.0f; h_data[15] = 35.0f;

	h_colIndex[0] = 1; h_colIndex[1] = 5;
	h_colIndex[2] = 0; h_colIndex[3] = 4;
	h_colIndex[4] = 1; h_colIndex[5] = 5;
	h_colIndex[6] = 3; h_colIndex[7] = 4;
	h_colIndex[8] = 1; h_colIndex[9] = 2; h_colIndex[10] = 4; h_colIndex[11] = 5;
	h_colIndex[12] = 0; h_colIndex[13] = 2; h_colIndex[14] = 3; h_colIndex[15] = 5;

	h_rowPtr[0] = 0; h_rowPtr[1] = 2; h_rowPtr[2] = 4; h_rowPtr[3] = 6; 
	h_rowPtr[4] = 8; h_rowPtr[5] = 12; h_rowPtr[6] = 16;

	//ELL
	float* h_dataELL = nullptr;
	int* h_colIndexELL = nullptr;

	//COO
	float* h_dataCOO = nullptr;
	int* h_colIndexCOO = nullptr;
	int* h_rowIndexCOO = nullptr;

	//transform CSR into ELL and COO
	int numElemsELL = 0;
	int numElemsCOO = 0;
	create_ELL_COO_from_CSR(h_data, h_colIndex, h_rowPtr, 
		h_dataELL, h_colIndexELL, 
		h_dataCOO, h_colIndexCOO, h_rowIndexCOO, 
		matrixDim, numElems, numElemsELL, numElemsCOO);

	//check
	for (int i = 0; i < numElemsELL*matrixDim; ++i)
		std::cout << h_dataELL[i] << "," << h_colIndexELL[i] << std::endl;

	//allocate device matrices
	cudaError_t cudaStatus;
	size_t sizeDeviceELLData = sizeof(float)*numElemsELL*matrixDim;
	size_t sizeDeviceELLCols = sizeof(int)*numElemsELL*matrixDim;
	size_t sizeDeviceMemVector = sizeof(float)*vectorDim;
	float *d_dataELL, *d_y, *d_x;
	int *d_colIndexELL;

	cudaStatus = cudaMalloc(&d_dataELL, sizeDeviceELLData);
	cudaStatus = cudaMalloc(&d_colIndexELL, sizeDeviceELLCols);
	cudaStatus = cudaMalloc(&d_y, sizeDeviceMemVector);
	cudaStatus = cudaMalloc(&d_x, sizeDeviceMemVector);

	//transfer data to device
	cudaStatus = cudaMemset(d_y, 0, sizeDeviceMemVector);
	cudaStatus = cudaMemcpy(d_x, h_x, sizeDeviceMemVector, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_dataELL, h_dataELL, sizeDeviceELLData, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_colIndexELL, h_colIndexELL, sizeDeviceELLCols, cudaMemcpyHostToDevice);

	//launch ELL kernel
	int numThreads = matrixDim;
	int numBlocks = 1;

	SpMV_ELL << <numBlocks, numThreads >> >(matrixDim, d_dataELL, d_colIndexELL, numElemsELL, d_x, d_y);

	//transfer results to host
	cudaStatus = cudaMemcpy(h_y, d_y, sizeDeviceMemVector, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < vectorDim; ++i) {
		std::cout << h_y[i] << std::endl;
	}

	//perform COO
	for (int i = 0; i < numElemsCOO; i++)
		h_y[h_rowIndexCOO[i]] += h_dataCOO[i] * h_x[h_colIndexCOO[i]];

	//check 
	for (size_t i = 0; i < vectorDim; ++i) {
		std::cout << h_y[i] << std::endl;
	}

	//release resources
	delete[] h_x;
	delete[] h_y;
	delete[] h_data;
	delete[] h_colIndex;
	delete[] h_rowPtr;
	delete[] h_dataCOO;
	delete[] h_colIndexCOO;
	delete[] h_rowIndexCOO;
	delete[] h_dataELL;
	delete[] h_colIndexELL;
	cudaStatus = cudaFree(d_dataELL);
	cudaStatus = cudaFree(d_y);
	cudaStatus = cudaFree(d_x);
	cudaStatus = cudaFree(d_colIndexELL);
}