#include "kernels.cuh"
#include "defines.h"
#include <cuda.h>


#include "EquationAlfa.cuh"
#include "EquationF.cuh"
#include "EquationG.cuh"



__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0){

	int gid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int gid_y = blockIdx.y*blockDim.y + threadIdx.y;
	int gid_z = blockIdx.z*blockDim.z + threadIdx.z;

	a[(l)*M*N*O + (gid_x)*N*O + (gid_y)*O + gid_z] = a_0[gid_x];
	F[(l)*M*N*O + (gid_x)*N*O + (gid_y)*O + gid_z] = q*(dtheta*gid_y);
	G[(l)*M*N*O + (gid_x)*N*O + (gid_y)*O + gid_z] = p*((dt*l)/L - dphi*gid_z);
} 


