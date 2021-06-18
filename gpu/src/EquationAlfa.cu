//### THIS IS A GENERATED FILE - DO NOT MODIFY ### 
 #include "EquationAlfa.cuh"
 #include "SharedMem.cuh"

__global__ void computeNexta(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int theta = blockIdx.y*blockDim.y + threadIdx.y;
    int phi = blockIdx.z*blockDim.z + threadIdx.z;
    int global_phi = phi + phi_offset;
    if (r>=M && theta>=N && phi>=O){
        return;
    }
    extern __shared__ REAL shmem[];
    
    fillSharedMemory(shmem, a, F, G, M, N, O, global_phi, r, theta, phi);
    __syncthreads();
    if (r<M && theta<N && phi<O){
        
 		shmem[CI(0, tp1, threadIdx.z, threadIdx.y, threadIdx.x)] = 1;
	}
 	    copySharedMemoryToGlobal(shmem, a, F, G, M, N, O, global_phi, r, theta, phi, tp1);
}
__global__ void computeSeconda(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int theta = blockIdx.y*blockDim.y + threadIdx.y;
    int phi = blockIdx.z*blockDim.z + threadIdx.z;
    int global_phi = phi + phi_offset;
    if (r>=M && theta>=N && phi>=O){
        return;
    }
    extern __shared__ REAL shmem[];
    
    fillSharedMemory(shmem, a, F, G, M, N, O, global_phi, r, theta, phi);
    __syncthreads();
    if (r<M && theta<N && phi<O){
        
 		shmem[CI(0, tp1, threadIdx.z, threadIdx.y, threadIdx.x)] = 1;
	}
 	    copySharedMemoryToGlobal(shmem, a, F, G, M, N, O, global_phi, r, theta, phi, tp1);
}
__global__ void computeFirsta(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int theta = blockIdx.y*blockDim.y + threadIdx.y;
    int phi = blockIdx.z*blockDim.z + threadIdx.z;
    int global_phi = phi + phi_offset;
    if (r>=M && theta>=N && phi>=O){
        return;
    }
    extern __shared__ REAL shmem[];
    
    fillSharedMemory(shmem, a, F, G, M, N, O, global_phi, r, theta, phi);
    __syncthreads();
    if (r<M && theta<N && phi<O){
        
 		shmem[CI(0, tp1, threadIdx.z, threadIdx.y, threadIdx.x)] = 1;
	}
 	    copySharedMemoryToGlobal(shmem, a, F, G, M, N, O, global_phi, r, theta, phi, tp1);
}