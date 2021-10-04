//### THIS IS A GENERATED FILE - DO NOT MODIFY ### 
 #include <iostream>
 #include <unistd.h>
 #include <vector>
 #include <string>
 #include <fstream>
 #include <cmath>
 #include "defines.h"
 #include "EquationAlfa.h"

REAL computeNexta(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L){
    return  (a[I((tm1),(phi),(theta),(r))]*(dr*dr)*(dtheta*dtheta) + 2*p*((dr*dr)*dt*(a[I((t),(phi),(theta-1),(r))] - 2*a[I((t),(phi),(theta),(r))] + a[I((t),(phi),(theta+1),(r))]) + dt*(dtheta*dtheta)*(a[I((t),(phi),(theta),(r-1))] - 2*a[I((t),(phi),(theta),(r))] + a[I((t),(phi),(theta),(r+1))])))/((dr*dr)*(dtheta*dtheta));
}