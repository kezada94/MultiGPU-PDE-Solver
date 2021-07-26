#pragma once

#include "defines.h"

__device__ inline void fillSharedMemory(REAL* sh, REAL *a, REAL *F, REAL *G, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi){


    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    for (size_t i = 0; i < 4; i++) {
        sh[CI(0, i, threadIdx.z, threadIdx.y, threadIdx.x)] = a[I(i, phi, theta, r)];
        sh[CI(1, i, threadIdx.z, threadIdx.y, threadIdx.x)] = F[I(i, phi, theta, r)];
        sh[CI(2, i, threadIdx.z, threadIdx.y, threadIdx.x)] = G[I(i, phi, theta, r)];
    }

    // Ahora se llena el halo en shmem
    // Left
    if (threadIdx.x == 0){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, threadIdx.y, -1)] = a[I(i, phi, theta, r-1)];
            sh[CI(1, i, threadIdx.z, threadIdx.y, -1)] = F[I(i, phi, theta, r-1)];
            sh[CI(2, i, threadIdx.z, threadIdx.y, -1)] = G[I(i, phi, theta, r-1)];
        }
    }
    // Right
    if (threadIdx.x == blockDim.x-1 || r == M-1){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, threadIdx.y, threadIdx.x+1)] = a[I(i, phi, theta, r+1)];
            sh[CI(1, i, threadIdx.z, threadIdx.y, threadIdx.x+1)] = F[I(i, phi, theta, r+1)];
            sh[CI(2, i, threadIdx.z, threadIdx.y, threadIdx.x+1)] = G[I(i, phi, theta, r+1)];
        }
    }
    // Front
    if (threadIdx.y == 0){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, -1, threadIdx.x)] = a[I(i, phi, theta-1, r)];
            sh[CI(1, i, threadIdx.z, -1, threadIdx.x)] = F[I(i, phi, theta-1, r)];
            sh[CI(2, i, threadIdx.z, -1, threadIdx.x)] = G[I(i, phi, theta-1, r)];
        }
    }
    // Back
    if (threadIdx.y == blockDim.y-1 || theta == N-1){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, threadIdx.y+1, threadIdx.x)] = a[I(i, phi, theta+1, r)];
            sh[CI(1, i, threadIdx.z, threadIdx.y+1, threadIdx.x)] = F[I(i, phi, theta+1, r)];
            sh[CI(2, i, threadIdx.z, threadIdx.y+1, threadIdx.x)] = G[I(i, phi, theta+1, r)];
        }
    }
    // Top
    if (threadIdx.z == 0){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, -1, threadIdx.y, threadIdx.x)] = a[I(i, phi-1, theta, r)];
            sh[CI(1, i, -1, threadIdx.y, threadIdx.x)] = F[I(i, phi-1, theta, r)];
            sh[CI(2, i, -1, threadIdx.y, threadIdx.x)] = G[I(i, phi-1, theta, r)];
        }
    }
    // Bottom
    if (threadIdx.z == blockDim.z-1 || phi == O-1){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z+1, threadIdx.y, threadIdx.x)] = a[I(i, phi+1, theta, r)];
            sh[CI(1, i, threadIdx.z+1, threadIdx.y, threadIdx.x)] = F[I(i, phi+1, theta, r)];
            sh[CI(2, i, threadIdx.z+1, threadIdx.y, threadIdx.x)] = G[I(i, phi+1, theta, r)];
        }
    }

    // Corners
    // Top-Left
    if (threadIdx.x == 0 && threadIdx.y == 0){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, -1, -1)] = a[I(i, phi, theta-1, r-1)];
            sh[CI(1, i, threadIdx.z, -1, -1)] = F[I(i, phi, theta-1, r-1)];
            sh[CI(2, i, threadIdx.z, -1, -1)] = G[I(i, phi, theta-1, r-1)];
        }
    }
    // Top-Right
    if ((threadIdx.x == blockDim.x-1 || threadIdx.x == M-1) && threadIdx.y == 0){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, -1, threadIdx.x+1)] = a[I(i, phi, theta-1, r+1)];
            sh[CI(1, i, threadIdx.z, -1, threadIdx.x+1)] = F[I(i, phi, theta-1, r+1)];
            sh[CI(2, i, threadIdx.z, -1, threadIdx.x+1)] = G[I(i, phi, theta-1, r+1)];
        }
    }
    // Bottom-Left
    if (threadIdx.x == 0 && (threadIdx.y == blockDim.y-1 || threadIdx.y == N-1)){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, threadIdx.y+1, -1)] = a[I(i, phi, theta+1, r-1)];
            sh[CI(1, i, threadIdx.z, threadIdx.y+1, -1)] = F[I(i, phi, theta+1, r-1)];
            sh[CI(2, i, threadIdx.z, threadIdx.y+1, -1)] = G[I(i, phi, theta+1, r-1)];
        }
    }
    //Bottom-Right
    if ((threadIdx.x == blockDim.x-1 || threadIdx.x == M-1) && (threadIdx.y == blockDim.y-1 || threadIdx.y == N-1)){
        for (size_t i = 0; i < 4; i++) {
            sh[CI(0, i, threadIdx.z, threadIdx.y+1, threadIdx.x+1)] = a[I(i, phi, theta+1, r+1)];
            sh[CI(1, i, threadIdx.z, threadIdx.y+1, threadIdx.x+1)] = F[I(i, phi, theta+1, r+1)];
            sh[CI(2, i, threadIdx.z, threadIdx.y+1, threadIdx.x+1)] = G[I(i, phi, theta+1, r+1)];
        }
    }


    // Now border of halo in Z[-1]
    if (threadIdx.z == 0){
        // Left
        if (threadIdx.x == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, threadIdx.y, -1)] = a[I(i, phi-1, theta, r-1)];
                sh[CI(1, i, -1, threadIdx.y, -1)] = F[I(i, phi-1, theta, r-1)];
                sh[CI(2, i, -1, threadIdx.y, -1)] = G[I(i, phi-1, theta, r-1)];
            }
        }
        // Right
        if (threadIdx.x == blockDim.x-1 || r == M-1){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, threadIdx.y, threadIdx.x+1)] = a[I(i, phi-1, theta, r+1)];
                sh[CI(1, i, -1, threadIdx.y, threadIdx.x+1)] = F[I(i, phi-1, theta, r+1)];
                sh[CI(2, i, -1, threadIdx.y, threadIdx.x+1)] = G[I(i, phi-1, theta, r+1)];
            }
        }
        // Front
        if (threadIdx.y == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, -1, threadIdx.x)] = a[I(i, phi-1, theta-1, r)];
                sh[CI(1, i, -1, -1, threadIdx.x)] = F[I(i, phi-1, theta-1, r)];
                sh[CI(2, i, -1, -1, threadIdx.x)] = G[I(i, phi-1, theta-1, r)];
            }
        }
        // Back
        if (threadIdx.y == blockDim.y-1 || theta == N-1){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, threadIdx.y+1, threadIdx.x)] = a[I(i, phi-1, theta+1, r)];
                sh[CI(1, i, -1, threadIdx.y+1, threadIdx.x)] = F[I(i, phi-1, theta+1, r)];
                sh[CI(2, i, -1, threadIdx.y+1, threadIdx.x)] = G[I(i, phi-1, theta+1, r)];
            }
        }

        // Corners
        // Top-Left
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, -1, -1)] = a[I(i, phi-1, theta-1, r-1)];
                sh[CI(1, i, -1, -1, -1)] = F[I(i, phi-1, theta-1, r-1)];
                sh[CI(2, i, -1, -1, -1)] = G[I(i, phi-1, theta-1, r-1)];
            }
        }
        // Top-Right
        if ((threadIdx.x == blockDim.x-1 || threadIdx.x == M-1) && threadIdx.y == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, -1, threadIdx.x+1)] = a[I(i, phi-1, theta-1, r+1)];
                sh[CI(1, i, -1, -1, threadIdx.x+1)] = F[I(i, phi-1, theta-1, r+1)];
                sh[CI(2, i, -1, -1, threadIdx.x+1)] = G[I(i, phi-1, theta-1, r+1)];
            }
        }
        // Bottom-Left
        if (threadIdx.x == 0 && (threadIdx.y == blockDim.y-1 || threadIdx.y == N-1)){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, threadIdx.y+1, -1)] = a[I(i, phi-1, theta+1, r-1)];
                sh[CI(1, i, -1, threadIdx.y+1, -1)] = F[I(i, phi-1, theta+1, r-1)];
                sh[CI(2, i, -1, threadIdx.y+1, -1)] = G[I(i, phi-1, theta+1, r-1)];
            }
        }
        //Bottom-Right
        if ((threadIdx.x == blockDim.x-1 || threadIdx.x == M-1) && (threadIdx.y == blockDim.y-1 || threadIdx.y == N-1)){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, -1, threadIdx.y+1, threadIdx.x+1)] = a[I(i, phi-1, theta+1, r+1)];
                sh[CI(1, i, -1, threadIdx.y+1, threadIdx.x+1)] = F[I(i, phi-1, theta+1, r+1)];
                sh[CI(2, i, -1, threadIdx.y+1, threadIdx.x+1)] = G[I(i, phi-1, theta+1, r+1)];
            }
        }
    }

    // Now border of halo in Z[+1]
    if (threadIdx.z == blockDim.z-1 || phi == O-1){
        // Left
        if (threadIdx.x == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, threadIdx.y, -1)] = a[I(i, phi+1, theta, r-1)];
                sh[CI(1, i, threadIdx.z+1, threadIdx.y, -1)] = F[I(i, phi+1, theta, r-1)];
                sh[CI(2, i, threadIdx.z+1, threadIdx.y, -1)] = G[I(i, phi+1, theta, r-1)];
            }
        }
        // Right
        if (threadIdx.x == blockDim.x-1 || r == M-1){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, threadIdx.y, threadIdx.x+1)] = a[I(i, phi+1, theta, r+1)];
                sh[CI(1, i, threadIdx.z+1, threadIdx.y, threadIdx.x+1)] = F[I(i, phi+1, theta, r+1)];
                sh[CI(2, i, threadIdx.z+1, threadIdx.y, threadIdx.x+1)] = G[I(i, phi+1, theta, r+1)];
            }
        }
        // Front
        if (threadIdx.y == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, -1, threadIdx.x)] = a[I(i, phi+1, theta-1, r)];
                sh[CI(1, i, threadIdx.z+1, -1, threadIdx.x)] = F[I(i, phi+1, theta-1, r)];
                sh[CI(2, i, threadIdx.z+1, -1, threadIdx.x)] = G[I(i, phi+1, theta-1, r)];
            }
        }
        // Back
        if (threadIdx.y == blockDim.y-1 || theta == N-1){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, threadIdx.y+1, threadIdx.x)] = a[I(i, phi+1, theta+1, r)];
                sh[CI(1, i, threadIdx.z+1, threadIdx.y+1, threadIdx.x)] = F[I(i, phi+1, theta+1, r)];
                sh[CI(2, i, threadIdx.z+1, threadIdx.y+1, threadIdx.x)] = G[I(i, phi+1, theta+1, r)];
            }
        }
        // Corners
        // Top-Left
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, -1, -1)] = a[I(i, phi+1, theta-1, r-1)];
                sh[CI(1, i, threadIdx.z+1, -1, -1)] = F[I(i, phi+1, theta-1, r-1)];
                sh[CI(2, i, threadIdx.z+1, -1, -1)] = G[I(i, phi+1, theta-1, r-1)];
            }
        }
        // Top-Right
        if ((threadIdx.x == blockDim.x-1 || threadIdx.x == M-1) && threadIdx.y == 0){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, -1, threadIdx.x+1)] = a[I(i, phi+1, theta-1, r+1)];
                sh[CI(1, i, threadIdx.z+1, -1, threadIdx.x+1)] = F[I(i, phi+1, theta-1, r+1)];
                sh[CI(2, i, threadIdx.z+1, -1, threadIdx.x+1)] = G[I(i, phi+1, theta-1, r+1)];
            }
        }
        // Bottom-Left
        if (threadIdx.x == 0 && (threadIdx.y == blockDim.y-1 || threadIdx.y == N-1)){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, threadIdx.y+1, -1)] = a[I(i, phi+1, theta+1, r-1)];
                sh[CI(1, i, threadIdx.z+1, threadIdx.y+1, -1)] = F[I(i, phi+1, theta+1, r-1)];
                sh[CI(2, i, threadIdx.z+1, threadIdx.y+1, -1)] = G[I(i, phi+1, theta+1, r-1)];
            }
        }
        //Bottom-Right
        if ((threadIdx.x == blockDim.x-1 || threadIdx.x == M-1) && (threadIdx.y == blockDim.y-1 || threadIdx.y == N-1)){
            for (size_t i = 0; i < 4; i++) {
                sh[CI(0, i, threadIdx.z+1, threadIdx.y+1, threadIdx.x+1)] = a[I(i, phi+1, theta+1, r+1)];
                sh[CI(1, i, threadIdx.z+1, threadIdx.y+1, threadIdx.x+1)] = F[I(i, phi+1, theta+1, r+1)];
                sh[CI(2, i, threadIdx.z+1, threadIdx.y+1, threadIdx.x+1)] = G[I(i, phi+1, theta+1, r+1)];
            }
        }
    }
}

__device__ inline void copySharedMemoryToGlobala(REAL* sh, REAL *a, REAL *F, REAL *G, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi, size_t tp1){
    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    a[I(tp1, phi, theta, r)] = sh[CI(0, tp1, threadIdx.z, threadIdx.y, threadIdx.x)];
}
__device__ inline void copySharedMemoryToGlobalF(REAL* sh, REAL *a, REAL *F, REAL *G, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi, size_t tp1){
    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    F[I(tp1, phi, theta, r)] = sh[CI(1, tp1, threadIdx.z, threadIdx.y, threadIdx.x)];
}
__device__ inline void copySharedMemoryToGlobalG(REAL* sh, REAL *a, REAL *F, REAL *G, size_t M, size_t N, size_t O, size_t global_phi, int r, int theta, int phi, size_t tp1){
    // Cada hilo copia el valor que le corresponde dentro del volumen por cada uno de los 4 tiempos
    G[I(tp1, phi, theta, r)] = sh[CI(2, tp1, threadIdx.z, threadIdx.y, threadIdx.x)];
}
