#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHUNK_SIZE 2048

#include <cmath>

struct kdata 
{
	float x, y, z;
};

__constant__ struct kdata k_c[CHUNK_SIZE];

__global__ void cmpMu(float* rPhi, float* iPhi, float*  rD, 
	float*  iD, float* rMu, float* iMu)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];
	iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
}

__global__ void cmpFHd(float* rFhD, float* iFhD, float* phiMag,
	float* kx, float* ky, float* kz, float*  x, float* y, 
	float*  z, float* rMu, float* iMu, int M)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	float xn_r = x[n]; float yn_r = y[n]; float zn_r = z[n];
	float rFhD_r = rFhD[n]; float iFhD_r = iFhD[n];

	for (int m = 0; m < M; m++) 
	{
		float expFhD = 2 * M_PI*(kx[m] * xn_r + ky[m] * yn_r + kz[m] * zn_r);

		float cArg = cos(expFhD);
		float sArg = sin(expFhD);

		rFhD_r += rMu[m] * cArg - iMu[m] * sArg;
		iFhD_r += iMu[m] * cArg + rMu[m] * sArg;
	}

	rFhD[n] = rFhD_r; 
	iFhD[n] = iFhD_r;
}

__global__ void cmpFHd_ConstMem(float* rFhD, float* iFhD, float* phiMag,
	float*  x, float* y, float*  z, 
	float* rMu, float* iMu, int M)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	float xn_r = x[n]; float yn_r = y[n]; float zn_r = z[n];
	float rFhD_r = rFhD[n]; float iFhD_r = iFhD[n];

	for (int m = 0; m < M; m++)
	{
		float expFhD = 2 * M_PI*(k_c[m].x * xn_r + k_c[m].y * yn_r + k_c[m].z * zn_r);

		float cArg = cos(expFhD);
		float sArg = sin(expFhD);

		rFhD_r += rMu[m] * cArg - iMu[m] * sArg;
		iFhD_r += iMu[m] * cArg + rMu[m] * sArg;
	}

	rFhD[n] = rFhD_r;
	iFhD[n] = iFhD_r;
}