#pragma once

#include "Equation.cuh"
#include <cuda.h>

__global__ void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q){

	for (int m=threadIdx.x; m<M; m+=blockDim.x){
		for (int n=threadIdx.y; n<N; n+=blockDim.y){
			for (int o=threadIdx.z; o<O; o+=blockDim.z){
				if (m == 0 || m == M-1 || m == M-2 || m == 1 ){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
				}
				if (n == 0 || n == N-1 || n == N-2 || n == 1 ){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
				}
				if (o == 0 || o == O-1 || o == O-2 || o == 1 ){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
				}
				if (m<2 || m>M-3 || n<2 || n>N-3 || o<2 || o>O-3){
					continue;
				}
                a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNexta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
			}
		}
	}

} 

__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q){

	for (int m=threadIdx.z; m<M; m+=blockDim.z){
		for (int n=threadIdx.y; n<N; n+=blockDim.y){
			for (int o=threadIdx.x; o<O; o+=blockDim.x){
				a[(l)*M*N*O + (m)*N*O + (n)*O + o] = dr*m;
				F[(l)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
				G[(l)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - dphi*o);
			}
		}
	}

} 
