#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "defines.h"
#include "EquationF.cuh"
using namespace std;

__global__ void computeNextF(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L)
{
	int r = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int phi = blockIdx.z*blockDim.z + threadIdx.z;
}