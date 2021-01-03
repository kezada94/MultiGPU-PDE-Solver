#pragma once

#include "equations.cuh"

__global___ void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl){
	size_t tx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t ty = blockIdx.y*blockDim.y + threadIdx.y;
	size_t tz = blockIdx.z*blockDim.z + threadIdx.z;

	for (int m=threadIdx.x; m<M; m+=blockDim.x){
		for (int n=threadIdx.y; n<N; n+=blockDim.y){
			for (int o=threadIdx.z; o<O; o+=blockDim.z){
                    if (m == 0 || m == M-1 || m == M-2 || m == 1 ){
                            a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = (a->axes[0][m]);
                            F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                            G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - a->axes[2][o]);
                    }
                    if (n == 0 || n == N-1 || n == N-2 || n == 1 ){
                            a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = a->axes[0][m];
                            F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                            G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - a->axes[2][o]);
                    }
                    if (o == 0 || o == O-1 || o == O-2 || o == 1 ){
                            a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = a->axes[0][m];
                            F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                            G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - a->axes[2][o]);
                    }
                    a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNexta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                    F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                    G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
			}
		}
	}
} 
