#pragma once
#include <cuda.h>
#include "defines.h"


__global__ void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q);

__global__ void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q);