#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void fillGhostPoints(REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O){
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t n=0; n<N+2; n++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, 0, n, o)] = a[E(t, 2, n, o)];
			F[E(t, 0, n, o)] = F[E(t, 2, n, o)];
			G[E(t, 0, n, o)] = G[E(t, 2, n, o)];
		}
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t m=0; m<M+2; m++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, m, 0, o)] = a[E(t, m, 2, o)];
			F[E(t, m, 0, o)] = F[E(t, m, 2, o)];
			G[E(t, m, 0, o)] = G[E(t, m, 2, o)];
		}
	}	
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t m=0; m<M+2; m++){
		for(size_t n=0; n<N+2; n++){
			a[E(t, m, n, 0)] = a[E(t, m, n, 2)];
			F[E(t, m, n, 0)] = F[E(t, m, n, 2)];
			G[E(t, m, n, 0)] = G[E(t, m, n, 2)];
		}
	}

	// Boundary m=L
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t n=0; n<N+2; n++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, M+1, n, o)] = a[E(t, M-1, n, o)];
			F[E(t, M+1, n, o)] = F[E(t, M-1, n, o)];
			G[E(t, M+1, n, o)] = G[E(t, M-1, n, o)];
		}
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t m=0; m<M+2; m++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, m, N+1, o)] = a[E(t, m, N-1, o)];
			F[E(t, m, N+1, o)] = F[E(t, m, N-1, o)];
			G[E(t, m, N+1, o)] = G[E(t, m, N-1, o)];
		}
	}	
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t m=0; m<M+2; m++){
		for(size_t n=0; n<N+2; n++){
			a[E(t, m, n, O+1)] = a[E(t, m, n, O-1)];
			F[E(t, m, n, O+1)] = F[E(t, m, n, O-1)];
			G[E(t, m, n, O+1)] = G[E(t, m, n, O-1)];
		}
	}
}

void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<M; phi++){
		for(size_t theta=0; theta<N; theta++){
			for(size_t r=0; r<O; r++){
				a[I(t, phi, theta, r)] = computeNexta(a, F, G, tm1, tm2, tm3, r, theta, phi, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				F[I(t, phi, theta, r)] = computeNextF(a, F, G, tm1, tm2, tm3, r, theta, phi, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				G[I(t, phi, theta, r)] = computeNextG(a, F, G, tm1, tm2, tm3, r, theta, phi, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			}
		}
	}
	
} 


std::fstream& gotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O; phi++){
		for(size_t theta=0; theta<N; theta++){
			for(size_t r=0; r<M; r++){
				a[I(l, phi, theta, r)] = a_0[r] + PI_1;
				F[I(l, phi, theta, r)] = (REAL)q*(dtheta*(REAL)theta) + PI_2;
				G[I(l, phi, theta, r)] = p*((dt*l)/L - dphi*phi) + PI_3;
			}
		}
	}
	
} 

void fillDirichletBoundary(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O; phi++){
		for(size_t theta=0; theta<N; theta++){
			for(size_t r=0; r<M; r++){
				if (r == 0 || r == M-1 ){
					a[I(t, phi, theta, r)] = a_0[r] + PI_1;
					F[I(t, phi, theta, r)] = q*(dtheta*theta) + PI_2;
					G[I(t, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
				} else if (theta == 0 || theta == N-1 ){
					a[I(t, phi, theta, r)] = a_0[r] + PI_1;
					F[I(t, phi, theta, r)] = q*(dtheta*theta) + PI_2;
					G[I(t, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
				} else if (phi == 0 || phi == O-1 ){
					a[I(t, phi, theta, r)] = a_0[r] + PI_1;
					F[I(t, phi, theta, r)] = q*(dtheta*theta) + PI_2;
					G[I(t, phi, theta, r)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
				}
			}
		}
	}
}
