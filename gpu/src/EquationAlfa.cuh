//### THIS IS A GENERATED FILE - DO NOT MODIFY ### 
 #pragma once
 #include <iostream>
 #include <unistd.h>
 #include <vector>
 #include <string>
 #include <fstream>
 #include <cmath>
 #include "defines.h"
 #include "SharedMem.cuh"

 using namespace std;

__global__ void computeNexta(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L);
__global__ void computeSeconda(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L);
__global__ void computeFirsta(REAL *a, REAL *F, REAL *G, size_t l, size_t tp1, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, size_t phi_offset, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L);