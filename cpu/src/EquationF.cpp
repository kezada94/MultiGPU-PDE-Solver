//### THIS IS A GENERATED FILE - DO NOT MODIFY ### 
 #include <iostream>
 #include <unistd.h>
 #include <vector>
 #include <string>
 #include <fstream>
 #include <cmath>
 #include "defines.h"
 #include "EquationF.h"

REAL computeNextF(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    return  (36*(dr*dr)*(dt*dt)*(F[I((t),(phi),(theta-1),(r))] - 2*F[I((t),(phi),(theta),(r))] + F[I((t),(phi),(theta+1),(r))]) - (dtheta*dtheta)*((dr*dr)*(F[I((tm1),(phi),(theta),(r))] - 2*F[I((t),(phi),(theta),(r))]) - 36*(dt*dt)*(F[I((t),(phi),(theta),(r-1))] - 2*F[I((t),(phi),(theta),(r))] + F[I((t),(phi),(theta),(r+1))])))/((dr*dr)*(dtheta*dtheta));
}
REAL computeFirstF(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    return  (36*(dr*dr)*(dt*dt)*(F[I((t),(phi+1),(theta-1),(r))] - 2*F[I((t),(phi+1),(theta),(r))] + F[I((t),(phi+1),(theta+1),(r))]) + (dtheta*dtheta)*(F[I((t),(phi+1),(theta),(r))]*(dr*dr) + 36*(dt*dt)*(F[I((t),(phi+1),(theta),(r-1))] - 2*F[I((t),(phi+1),(theta),(r))] + F[I((t),(phi+1),(theta),(r+1))])))/((dr*dr)*(dtheta*dtheta));
}