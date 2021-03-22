#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void fillGhostPoints(REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O){
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t n=0; n<N+2; n++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, 0, n, o)] = a[E(t, 2, n, o)];
			F[E(t, 0, n, o)] = F[E(t, 2, n, o)];
			G[E(t, 0, n, o)] = G[E(t, 2, n, o)];
		}
	}
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, m, 0, o)] = a[E(t, m, 2, o)];
			F[E(t, m, 0, o)] = F[E(t, m, 2, o)];
			G[E(t, m, 0, o)] = G[E(t, m, 2, o)];
		}
	}	
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t n=0; n<N+2; n++){
			a[E(t, m, n, 0)] = a[E(t, m, n, 2)];
			F[E(t, m, n, 0)] = F[E(t, m, n, 2)];
			G[E(t, m, n, 0)] = G[E(t, m, n, 2)];
		}
	}

	// Boundary m=L
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t n=0; n<N+2; n++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, M+1, n, o)] = a[E(t, M-1, n, o)];
			F[E(t, M+1, n, o)] = F[E(t, M-1, n, o)];
			G[E(t, M+1, n, o)] = G[E(t, M-1, n, o)];
		}
	}
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t o=0; o<O+2; o++){
			a[E(t, m, N+1, o)] = a[E(t, m, N-1, o)];
			F[E(t, m, N+1, o)] = F[E(t, m, N-1, o)];
			G[E(t, m, N+1, o)] = G[E(t, m, N-1, o)];
		}
	}	
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t n=0; n<N+2; n++){
			a[E(t, m, n, O+1)] = a[E(t, m, n, O-1)];
			F[E(t, m, n, O+1)] = F[E(t, m, n, O-1)];
			G[E(t, m, n, O+1)] = G[E(t, m, n, O-1)];
		}
	}
}

void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M; m++){
		cout << m << endl;
		for(size_t n=0; n<N; n++){
			for(size_t o=0; o<O; o++){
				a[I(t, m, n, o)] = computeNexta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				F[I(t, m, n, o)] = computeNextF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				G[I(t, m, n, o)] = computeNextG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			}
		}
	}
	
} 

void computeFirstIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M; m++){
		cout << m << endl;
		for(size_t n=0; n<N; n++){
			for(size_t o=0; o<O; o++){
				a[I(t, m, n, o)] = computeFirsta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				F[I(t, m, n, o)] = computeFirstF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				G[I(t, m, n, o)] = computeFirstG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
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
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t n=0; n<N; n++){
			for(size_t o=0; o<O; o++){
				a[I(l, m, n, o)] = a_0[m] + PI_1;
				F[I(l, m, n, o)] = (REAL)q*(dtheta*(REAL)n) + PI_2;
				G[I(l, m, n, o)] = p*((dt*l)/L - dphi*o) + PI_3;
			}
		}
	}
	
} 

void fillDirichletBoundary(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(dynamic) num_threads(10)
	for(size_t r=0; r<M; r++){
		for(size_t theta=0; theta<N; theta++){
			for(size_t phi=0; phi<O; phi++){
				if (r == 0 || r == M-1 ){
					a[I(l, r, theta, phi)] = a_0[r] + PI_1;
					F[I(l, r, theta, phi)] = q*(dtheta*theta) + PI_2;
					G[I(l, r, theta, phi)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
				} else if (theta == 0 || theta == N-1 ){
					a[I(l, r, theta, phi)] = a_0[r] + PI_1;
					F[I(l, r, theta, phi)] = q*(dtheta*theta) + PI_2;
					G[I(l, r, theta, phi)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
				} else if (phi == 0 || phi == O-1 ){
					a[I(l, r, theta, phi)] = a_0[r] + PI_1;
					F[I(l, r, theta, phi)] = q*(dtheta*theta) + PI_2;
					G[I(l, r, theta, phi)] = p*((dt*(REAL)l)/(REAL)L - dphi*(REAL)phi) + PI_3;
				}
			}
		}
	}
}
