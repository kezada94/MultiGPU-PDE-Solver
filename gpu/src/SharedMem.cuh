#pragma once

#include "defines.h"

__device__ inline void fillSharedMemory(REAL* sh, REAL *a, REAL *F, REAL *G, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi){

    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    for (size_t i = 0; i < 4; i++) {
        sh[CI(0, i, threadIdx.z, threadIdx.y, threadIdx.x)] = a[E(i, phi, theta, r)];
        sh[CI(1, i, threadIdx.z, threadIdx.y, threadIdx.x)] = F[E(i, phi, theta, r)];
        sh[CI(2, i, threadIdx.z, threadIdx.y, threadIdx.x)] = G[E(i, phi, theta, r)];
    }
}

__device__ inline void copySharedMemoryToGlobal(REAL* sh, REAL *func, int funcIndex, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi, size_t tp1){
    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    func[E(tp1, phi, theta, r)] = sh[CI(funcIndex, tp1, threadIdx.z, threadIdx.y, threadIdx.x)];
}
