#include "defines.h"

__device__ fillSharedMemory(REAL* sh_a, REAL* sh_F, REAL* sh_G, REAL *a, REAL *F, REAL *G, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi){


    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    for (size_t i = 0; i < 4 /*buffSize*/; i++) {
        sh_a[CI(i, threadIdx.z, threadIdx.y, threadIdx.x)] = a[I(i, global_phi, theta, r)];
        sh_F[CI(i, threadIdx.z, threadIdx.y, threadIdx.x)] = F[I(i, global_phi, theta, r)];
        sh_G[CI(i, threadIdx.z, threadIdx.y, threadIdx.x)] = G[I(i, global_phi, theta, r)];
    }

    // Ahora se llena el halo en shmem
    // Left
    if (r == 0){
        for (size_t i = 0; i < 4 /*buffSize*/; i++) {
            sh_a[CI(i, threadIdx.z, threadIdx.y, threadIdx.x-1)] = a[I(i, global_phi, theta, r-1)];
            sh_F[CI(i, threadIdx.z, threadIdx.y, threadIdx.x-1)] = F[I(i, global_phi, theta, r-1)];
            sh_G[CI(i, threadIdx.z, threadIdx.y, threadIdx.x-1)] = G[I(i, global_phi, theta, r-1)];
        }
    }
    // Right
    if (r == M-1){
        for (size_t i = 0; i < 4 /*buffSize*/; i++) {
            sh_a[CI(i, threadIdx.z, threadIdx.y, threadIdx.x+blockDim.x)] = a[I(i, global_phi, theta, r+blockDim.x)];
            sh_F[CI(i, threadIdx.z, threadIdx.y, threadIdx.x+blockDim.x)] = F[I(i, global_phi, theta, r+blockDim.x)];
            sh_G[CI(i, threadIdx.z, threadIdx.y, threadIdx.x+blockDim.x)] = G[I(i, global_phi, theta, r+blockDim.x)];
        }
    }

    // Front
    if (theta == 0){
        for (size_t i = 0; i < 4 /*buffSize*/; i++) {
            sh_a[CI(i, threadIdx.z, threadIdx.y-1, threadIdx.x)] = a[I(i, global_phi, theta-1, r)];
            sh_F[CI(i, threadIdx.z, threadIdx.y-1, threadIdx.x)] = F[I(i, global_phi, theta-1, r)];
            sh_G[CI(i, threadIdx.z, threadIdx.y-1, threadIdx.x)] = G[I(i, global_phi, theta-1, r)];
        }
    }
    // Back
    if (theta == N-1){
        for (size_t i = 0; i < 4 /*buffSize*/; i++) {
            sh_a[CI(i, threadIdx.z, threadIdx.y+blockDim.y, threadIdx.x)] = a[I(i, global_phi, theta+blockDim.y, r)];
            sh_F[CI(i, threadIdx.z, threadIdx.y+blockDim.y, threadIdx.x)] = F[I(i, global_phi, theta+blockDim.y, r)];
            sh_G[CI(i, threadIdx.z, threadIdx.y+blockDim.y, threadIdx.x)] = G[I(i, global_phi, theta+blockDim.y, r)];
        }
    }

    // Top
    if (phi == 0){
        for (size_t i = 0; i < 4 /*buffSize*/; i++) {
            sh_a[CI(i, threadIdx.z-1, threadIdx.y, threadIdx.x)] = a[I(i, global_phi-1, theta, r)];
            sh_F[CI(i, threadIdx.z-1, threadIdx.y, threadIdx.x)] = F[I(i, global_phi-1, theta, r)];
            sh_G[CI(i, threadIdx.z-1, threadIdx.y, threadIdx.x)] = G[I(i, global_phi-1, theta, r)];
        }
    }
    // Bottom
    if (phi == O){
        for (size_t i = 0; i < 4 /*buffSize*/; i++) {
            sh_a[CI(i, threadIdx.z+blockDim.z, threadIdx.y, threadIdx.x)] = a[I(i, global_phi+blockDim.z, theta, r)];
            sh_F[CI(i, threadIdx.z+blockDim.z, threadIdx.y, threadIdx.x)] = F[I(i, global_phi+blockDim.z, theta, r)];
            sh_G[CI(i, threadIdx.z+blockDim.z, threadIdx.y, threadIdx.x)] = G[I(i, global_phi+blockDim.z, theta, r)];
        }
    }
}