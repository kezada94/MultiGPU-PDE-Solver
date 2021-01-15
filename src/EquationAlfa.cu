#include "EquationAlfa.cuh"


__global__ void computeNexta(REAL *a, REAL *F, REAL *G, size_t t, size_t tm1, size_t tm2, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L)
{
    int m = blockIdx.x*blockDim.x + threadIdx.x;
	int n = blockIdx.y*blockDim.y + threadIdx.y;
	int o = blockIdx.z*blockDim.z + threadIdx.z;
	a[(t)*M*N*O + (m)*N*O + (n)*O + o] = 
	-1.0/4.0*(4*pow(dphi, 2)*pow(dr, 2)*(L*a[(tm1)*M*N*O+(r)*N*O+(theta)*O+(phi)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2)) - 2*L*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2)))*pow(sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]), 2) - pow(dt, 2)*(pow(dphi, 2)*(L*pow(a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)], 2)*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2))*sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)])*cos(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) + L*pow(a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)], 2)*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2))*sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)])*cos(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) - a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]*(2*L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2))*sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)])*cos(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) + L*(-4*pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) + F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r-1)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r-1)*N*O+(theta+1)*O+(phi)]) - 4*pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2) - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]*(F[(t)*M*N*O+(r-1)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r-1)*N*O+(theta+1)*O+(phi)] - 8*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]) - F[(t)*M*N*O+(r+1)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) + F[(t)*M*N*O+(r+1)*N*O+(theta+1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]))*pow(sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]), 2)) + (L*a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)]*(4*pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) + F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r-1)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r-1)*N*O+(theta+1)*O+(phi)]) + 4*pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2) - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]*(F[(t)*M*N*O+(r-1)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r-1)*N*O+(theta+1)*O+(phi)] + 8*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]) - F[(t)*M*N*O+(r+1)*N*O+(theta-1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)]) + F[(t)*M*N*O+(r+1)*N*O+(theta+1)*O+(phi)]*(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] - F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)])) - 8*L*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2)))*pow(sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]), 2)) + 4*pow(dr, 2)*(L*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi+1)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2))*pow(sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]), 2) - pow(dphi, 2)*(-4*a[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)] + 8*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] - 4*a[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + (pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2))*sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)])*cos(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)])) + (L*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi-1)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2)) - 2*L*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2)))*pow(sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]), 2))) + 16*pow(dtheta, 2)*(pow(dphi, 2)*pow(dr, 2)*(a[(tm1)*M*N*O+(r)*N*O+(theta)*O+(phi)] - 2*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]) - pow(dt, 2)*(pow(dphi, 2)*(a[(t)*M*N*O+(r-1)*N*O+(theta)*O+(phi)] - 2*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + a[(t)*M*N*O+(r+1)*N*O+(theta)*O+(phi)]) + pow(dr, 2)*(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi-1)] - 2*a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)] + a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi+1)]))))/(L*pow(dphi, 2)*pow(dr, 2)*(pow(F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)], 2) - 2*F[(t)*M*N*O+(r)*N*O+(theta-1)*O+(phi)]*F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)] + pow(F[(t)*M*N*O+(r)*N*O+(theta+1)*O+(phi)], 2))*pow(sin(a[(t)*M*N*O+(r)*N*O+(theta)*O+(phi)]), 2) + 4*pow(dphi, 2)*pow(dr, 2)*pow(dtheta, 2));
}