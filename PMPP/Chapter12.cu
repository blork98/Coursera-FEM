#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <math_functions.h>

#define BLOCK_SIZE 16

struct atominfo
{
	float x, y, z, w;
};

__global__ void cenergy_1(float* energyGrid, float gridspacing, 
	atominfo* atomInfo, size_t numatoms, dim3 gridDims )
{
	size_t xindex = blockIdx.x*gridDim.x + threadIdx.x;
	size_t yindex = blockIdx.y*gridDim.y + threadIdx.y;
	size_t outaddr = yindex*gridDims.x + xindex;

	float curenergy = energyGrid[outaddr];
	float coorx = gridspacing*xindex;
	float coory = gridspacing*yindex;
	float energyval = 0.0f;

	for (size_t atomid = 0; atomid < numatoms; ++atomid)
	{
		float dx = coorx - atomInfo[atomid].x;
		float dy = coory - atomInfo[atomid].y;
		energyval += atomInfo[atomid].w*rsqrtf(dx*dx + dy*dy + atomInfo[atomid].z);
	}

	energyGrid[outaddr] = energyval + curenergy;
}

__global__ void cenergy_3(float* energyGrid, float gridspacing,
	atominfo* atomInfo, size_t numatoms, dim3 gridDims)
{
	size_t xindex = blockIdx.x*gridDim.x + threadIdx.x;
	size_t yindex = blockIdx.y*gridDim.y + threadIdx.y;
	size_t outaddr = yindex*gridDims.x + xindex;

	float coorx = gridspacing*xindex;
	float coory = gridspacing*yindex;
	float gridspacingCoalesce = gridspacing*BLOCK_SIZE;
	float energyval1 = 0.0f, energyval2 = 0.0f, energyval3 = 0.0f, energyval4 = 0.0f;

	for (size_t atomid = 0; atomid < numatoms; ++atomid)
	{
		float dy = coory - atomInfo[atomid].y;
		float dyz2 = (dy*dy) + atomInfo[atomid].z;

		float dx1 = coorx - atomInfo[atomid].x;
		float dx2 = dx1 + gridspacingCoalesce;
		float dx3 = dx2 + gridspacingCoalesce;
		float dx4 = dx3 + gridspacingCoalesce;

		energyval1 += atomInfo[atomid].w*rsqrtf(dx1*dx1 + dyz2);
		energyval2 += atomInfo[atomid].w*rsqrtf(dx2*dx2 + dyz2);
		energyval3 += atomInfo[atomid].w*rsqrtf(dx3*dx3 + dyz2);
		energyval4 += atomInfo[atomid].w*rsqrtf(dx4*dx4 + dyz2);
	}

	energyGrid[outaddr] += energyval1;
	energyGrid[outaddr + BLOCK_SIZE] += energyval2;
	energyGrid[outaddr + 2*BLOCK_SIZE] += energyval3;
	energyGrid[outaddr + 3*BLOCK_SIZE] += energyval4;
}