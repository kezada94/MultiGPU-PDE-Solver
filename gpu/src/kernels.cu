#include "kernels.cuh"
#include "defines.h"
#include <cuda.h>


#include "EquationAlfa.cuh"
#include "EquationF.cuh"
#include "EquationG.cuh"



__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0){

	int r = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int phi = blockIdx.z*blockDim.z + threadIdx.z;

	a[(l)*M*N*O + (r)*N*O + (theta)*O + phi] = a_0[r] + PI_1;
	F[(l)*M*N*O + (r)*N*O + (theta)*O + phi] = q*(dtheta*theta) + PI_2;
	G[(l)*M*N*O + (r)*N*O + (theta)*O + phi] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
} 


