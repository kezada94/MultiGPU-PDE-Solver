#pragma once

#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "defines.h"
using namespace std;

__device__ void computeNextF(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L)
{
	
	return (1.0/4.0)*(pow(dt, 2)*(16*pow(dphi, 2)*pow(dr, 2)*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) + pow(dphi, 2)*(L*a[(t)*M*N*O+(r-1)*N*O+(theta-1)*O+(phi)]*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) + 4*L*pow(a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)], 2)*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) - L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*a[(t)*M*N*O+(r-1)*N*O+(theta+1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) - L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*a[(t)*M*N*O+(r+1)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) + 4*L*pow(a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)], 2)*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) - a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]*(L*a[(t)*M*N*O+(r-1)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) + 8*L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) - L*a[(t)*M*N*O+(r-1)*N*O+(theta+1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) - L*a[(t)*M*N*O+(r+1)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)])) + a[(t)*M*N*O+(r+1)*N*O+(theta+1)*O+(phi)]*(L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) - L*a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)])))) - 4*pow(dtheta, 2)*(4*pow(dphi, 2)*pow(dr, 2)*(F[(tm1)*M*N*O+(r)*N*O+(theta)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) + pow(dphi, 2)*(L*pow(a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)], 2)*(F[(tm1)*M*N*O+(r)*N*O+(theta)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) - 2*L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]*(F[(tm1)*M*N*O+(r)*N*O+(theta)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) + L*pow(a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)], 2)*(F[(tm1)*M*N*O+(r)*N*O+(theta)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)])) - pow(dt, 2)*(L*pow(a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)], 2)*(F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi-1)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi+1)]) - 2*L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi-1)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi+1)]) + L*pow(a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)], 2)*(F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi-1)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi+1)]) + 4*pow(dphi, 2)*(F[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]) + 4*pow(dr, 2)*(F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi-1)] - 2*F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + F[(t)*M*N*O+(r)*N*O+(theta)*O+(phi+1)]))))/(pow(dtheta, 2)*(4*pow(dphi, 2)*pow(dr, 2) + pow(dphi, 2)*(L*pow(a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)], 2) - 2*L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)] + L*pow(a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)], 2))));
}
