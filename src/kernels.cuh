#pragma once

#include "Equation.cuh"
#include <cuda.h>

__global__ void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q){

	int m = blockIdx.x*blockDim.x + threadIdx.x;
	int n = blockIdx.y*blockDim.y + threadIdx.y;
	int o = blockIdx.z*blockDim.z + threadIdx.z;
	if (m == 0 || m == M-1 || m == M-2 || m == 1 ){
		a[(t)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
		F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
		G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
		return;
	}
	if (n == 0 || n == N-1 || n == N-2 || n == 1 ){
		a[(t)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
		F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
		G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
		return;
	}
	if (o == 0 || o == O-1 || o == O-2 || o == 1 ){
		a[(t)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
		F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
		G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
		return;
	}
	a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNexta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
	F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
	G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);

} 

__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, size_t epa, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int offsetX, int offsetY, int offsetZ){

	int gid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int gid_y = blockIdx.y*blockDim.y + threadIdx.y;
	int gid_z = blockIdx.z*blockDim.z + threadIdx.z;

	int x = blockIdx.x*blockDim.x + threadIdx.x + offsetX;
	int y = blockIdx.y*blockDim.y + threadIdx.y + offsetY;
	int z = blockIdx.z*blockDim.z + threadIdx.z + offsetZ;

	if (x > M || y > N || z > O){
		return;
	}

	a[(l)*epa*epa*epa + (gid_x)*epa*epa + (gid_y)*epa + gid_z] = dr*x;
	F[(l)*epa*epa*epa + (gid_x)*epa*epa + (gid_y)*epa + gid_z] = q*(dtheta*y);
	G[(l)*epa*epa*epa + (gid_x)*epa*epa + (gid_y)*epa + gid_z] = p*((dt*l)/bigl - dphi*z);

} 
