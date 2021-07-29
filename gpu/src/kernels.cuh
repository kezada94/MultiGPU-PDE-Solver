#pragma once
#include <cuda.h>
#include "defines.h"



void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0, dim3 b, dim3 g, size_t sharedMemorySizeb);
void computeFirstIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0, dim3 b, dim3 g, size_t sharedMemorySizeb);

__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, size_t phi_offset, size_t GPUWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0);
__global__ void fillGhostPoints(REAL* a, size_t t, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth);
__global__ void fillTemporalGhostVolume(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dphi, REAL dtheta, REAL dr, REAL p);
__global__ void fillDirichletBoundary(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t M, size_t N, size_t O, size_t phi_offset, size_t GPUWidth, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0);
