//### THIS IS A GENERATED FILE - DO NOT MODIFY ### 
 #include "EquationG.cuh"
 #include "SharedMem.cuh"

__global__ void computeNextG(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int theta = blockIdx.y*blockDim.y + threadIdx.y;
    int phi = blockIdx.z*blockDim.z + threadIdx.z;
    int global_phi = phi + phi_offset;
    
    extern __shared__ REAL shmem[];
    if (r<M && theta<N && phi<O){
        fillSharedMemory(shmem, a, F, G, M, N, O, global_phi, r, theta, phi);
    }
    __syncthreads();
    if (r<M && theta<N && phi<O){
        int tidxp = threadIdx.x+1;
        int tidxm = threadIdx.x-1;
        
        int tidyp = threadIdx.y+1;
        int tidym = threadIdx.y-1;        
        
        int tidzp = threadIdx.z+1;
        int tidzm = threadIdx.z-1;
        
        
 		shmem[CI(2, tp1, threadIdx.z, threadIdx.y, threadIdx.x)] =  ((dr*dr)*(dt*dt)*(shmem[CI(2,(t),(threadIdx.z),(tidym),(threadIdx.x))] - 2*shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(threadIdx.x))] + shmem[CI(2,(t),(threadIdx.z),(tidyp),(threadIdx.x))]) - (dtheta*dtheta)*((dr*dr)*(shmem[CI(2,(tm1),(threadIdx.z),(threadIdx.y),(threadIdx.x))] - 2*shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(threadIdx.x))]) - (dt*dt)*(shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(tidxm))] - 2*shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(threadIdx.x))] + shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(tidxp))])))/((dr*dr)*(dtheta*dtheta));
	}
     __syncthreads();
    if (r<M && theta<N && phi<O){        copySharedMemoryToGlobalG(shmem, a, F, G, M, N, O, global_phi, r, theta, phi, tp1);
    }
}
__global__ void computeFirstG(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int theta = blockIdx.y*blockDim.y + threadIdx.y;
    int phi = blockIdx.z*blockDim.z + threadIdx.z;
    int global_phi = phi + phi_offset;

    extern __shared__ REAL shmem[];
    if (r<M && theta<N && phi<O){
        fillSharedMemory(shmem, a, F, G, M, N, O, global_phi, r, theta, phi);
    }
    __syncthreads();
    if (r<M && theta<N && phi<O){
        int tidxp = threadIdx.x+1;
        int tidxm = threadIdx.x-1;
        
        int tidyp = threadIdx.y+1;
        int tidym = threadIdx.y-1;
        
        int tidzp = threadIdx.z+1;
        int tidzm = threadIdx.z-1;
        
 		shmem[CI(2, tp1, threadIdx.z, threadIdx.y, threadIdx.x)] =  (1.0/2.0)*((dr*dr)*(dt*dt)*(shmem[CI(2,(t),(threadIdx.z),(tidym),(threadIdx.x))] - 2*shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(threadIdx.x))] + shmem[CI(2,(t),(threadIdx.z),(tidyp),(threadIdx.x))]) + (dtheta*dtheta)*(2*shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(threadIdx.x))]*(dr*dr) + (dt*dt)*(shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(tidxm))] - 2*shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(threadIdx.x))] + shmem[CI(2,(t),(threadIdx.z),(threadIdx.y),(tidxp))])) + (theta*theta)*((dr*dr*dr*dr)*dt*(dtheta*dtheta*dtheta*dtheta)*(r*r) - (dr*dr*dr)*dt*(dtheta*dtheta*dtheta*dtheta)*r) - theta*((dr*dr*dr*dr)*dt*(dtheta*dtheta*dtheta)*(r*r) - (dr*dr*dr)*dt*(dtheta*dtheta*dtheta)*r))/((dr*dr)*(dtheta*dtheta));
	}
     if (r<M && theta<N && phi<O){        copySharedMemoryToGlobalG(shmem, a, F, G, M, N, O, global_phi, r, theta, phi, tp1);
    }
}