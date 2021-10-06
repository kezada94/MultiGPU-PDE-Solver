//### THIS IS A GENERATED FILE - DO NOT MODIFY ### 
 #pragma once
 #include <iostream>
 #include <unistd.h>
 #include <vector>
 #include <string>
 #include <fstream>
 #include <cmath>
 #include "defines.h"
 using namespace std;

REAL computeNextF(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L);
REAL computeFirstF(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L);