#include "kernels.cuh"
#include "defines.h"
#include <cuda.h>


#include "EquationAlfa.cuh"
#include "EquationF.cuh"
#include "EquationG.cuh"



__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0){

	int r = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int phi = blockIdx.z*blockDim.z + threadIdx.z;


	if (r<M && theta<N && phi<O){
		int global_phi = phi + phi_offset;

		a[I(l, phi, theta, r)] = a_0[r] + PI_1;
		F[I(l, phi, theta, r)] = q*(dtheta*theta) + PI_2;
		G[I(l, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)global_phi) + PI_3;
	}
} 



void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0, dim3 b, dim3 g, size_t sharedMemorySizeb){

    computeNexta<<<g, b, sharedMemorySizeb>>>(a, F, G, l, tp1, t, tm1, tm2, M, N, O, phi_offset, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
    cucheck(cudaDeviceSynchronize());

    computeNextF<<<g, b, sharedMemorySizeb>>>(a, F, G, l, tp1, t, tm1, tm2, M, N, O, phi_offset, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
    cucheck(cudaDeviceSynchronize());

    computeNextG<<<g, b, sharedMemorySizeb>>>(a, F, G, l, tp1, t, tm1, tm2, M, N, O, phi_offset, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
    cucheck(cudaDeviceSynchronize());
}

void computeFirstIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0, dim3 b, dim3 g, size_t sharedMemorySizeb){
/*
	computeFirsta<<<g, b, sharedMemorySizeb>>>(a, F, G, l, tp1, t, tm1, tm2, M, N, O, phi_offset, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
    cucheck(cudaDeviceSynchronize());

	computeFirstF<<<g, b, sharedMemorySizeb>>>(a, F, G, l, tp1, t, tm1, tm2, M, N, O, phi_offset, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
    cucheck(cudaDeviceSynchronize());

	computeFirstG<<<g, b, sharedMemorySizeb>>>(a, F, G, l, tp1, t, tm1, tm2, M, N, O, phi_offset, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
    cucheck(cudaDeviceSynchronize());*/
}

__global__ void fillTemporalGhostVolume(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dphi, REAL dtheta, REAL dr, REAL p){
	int r = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int phi = blockIdx.z*blockDim.z + threadIdx.z;
	int global_phi = phi+phi_offset;

	if (r<M && theta<N && phi<O){
		a[I(tm1, phi, theta, r)] = a[I(t, phi, theta, r)] - 2*dt*PI_4;
		F[I(tm1, phi, theta, r)] = F[I(t, phi, theta, r)] - 2*dt*PI_5;
		G[I(tm1, phi, theta, r)] = G[I(t, phi, theta, r)] - 2*dt*p*PI_6;
	}
}

__global__ void fillGhostPoints(REAL* a, size_t t, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth){
	int r = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int phi = blockIdx.z*blockDim.z + threadIdx.z;
	int global_phi = phi + phi_offset;

    if (r >= M || theta >= N || phi >= O){
        return;
    }
    
    // Left
    if (r == 0){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta, r+2)];
    }
    // Right
    if (r == M-1){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta, r-2)];
    }
    // Front
    if (theta == 0){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta+2, r)];
    }
    // Back
    if (theta == N-1){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta-2, r)];
    }
    // Top
    if (global_phi == 0){
        a[E(t, phi, theta, r)] = a[E(t, phi+2, theta, r)];
    }
    // Bottom
    if (global_phi == globalWidth-1){
        a[E(t, phi, theta, r)] = a[E(t, phi-2, theta, r)];
    }

    // Corners
    // Top-Left
    if (r == 0 && theta == 0){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta+2, r+2)];
    }
    // Top-Right
    if (r == M-1 && theta == 0){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta+2, r-2)];
    }
    // Bottom-Left
    if (r == 0 && theta == N-1){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta-2, r+2)];
    }
    //Bottom-Right
    if (r == M-1 && theta == N-1){
        a[E(t, phi, theta, r)] = a[E(t, phi, theta-2, r-2)];
    }

    // Now border of halo in Z[-1]
    if (global_phi == 0){
        // Left
        if (r == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta, r+2)];
        }
        // Right
        if (r == M-1){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta, r-2)];
        }
        // Front
        if (theta == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta+2, r)];
        }
        // Back
        if (theta == N-1){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta-2, r)];
        }

        // Corners
        // Top-Left
        if (r == 0 && theta == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta+2, r+2)];
        }
        // Top-Right
        if (r == M-1 && theta == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta+2, r-2)];
        }
        // Bottom-Left
        if (r == 0 && theta == N-1){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta-2, r+2)];
        }
        //Bottom-Right
        if (r == M-1 && theta == N-1){
            a[E(t, phi, theta, r)] = a[E(t, phi+2, theta-2, r-2)];
        }
    }

    // Now border of halo in Z[+1]
    if (global_phi == globalWidth-1){
        // Left
        if (r == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta, r+2)];
        }
        // Right
        if (r == M-1){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta, r-2)];
        }
        // Front
        if (theta == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta+2, r)];
        }
        // Back
        if (theta == N-1){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta-2, r)];
        }
        // Corners
        // Top-Left
        if (r == 0 && theta == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta+2, r+2)];
        }
        // Top-Right
        if (r == M-1 && theta == 0){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta+2, r-2)];
        }
        // Bottom-Left
        if (r == 0 && theta == N-1){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta-2, r+2)];
        }
        //Bottom-Right
        if (r == M-1 && theta == N-1){
            a[E(t, phi, theta, r)] = a[E(t, phi-2, theta-2, r-2)];
        }
    }
	
}

__global__ void fillDirichletBoundary(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){

	int r = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int phi = blockIdx.z*blockDim.z + threadIdx.z;

	if (r<M && theta<N && phi<O){
	    int global_phi = phi + phi_offset;

        if (r == 0 || r == M-1 ){
            a[I(t, phi, theta, r)] = a_0[r] + PI_1;
            F[I(t, phi, theta, r)] = q*(dtheta*theta) + PI_2;
            G[I(t, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)global_phi) + PI_3;
        } else if (theta == 0 || theta == N-1 ){
            a[I(t, phi, theta, r)] = a_0[r] + PI_1;
            F[I(t, phi, theta, r)] = q*(dtheta*theta) + PI_2;
            G[I(t, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)global_phi) + PI_3;
        } else if (global_phi == 0 || global_phi == globalWidth-1 ){
            a[I(t, phi, theta, r)] = a_0[r] + PI_1;
            F[I(t, phi, theta, r)] = q*(dtheta*theta) + PI_2;
            G[I(t, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)global_phi) + PI_3;
        }
	}
}

