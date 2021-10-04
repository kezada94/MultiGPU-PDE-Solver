#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void fillGhostPoints(REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O){
 
	
	// bottom
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		for(size_t phi=0; phi<N+2; phi++){
			a[E(t, phi, N+1, r)] = a[E(t, phi, N-1, r)];
			F[E(t, phi, N+1, r)] = F[E(t, phi, N-1, r)];
			G[E(t, phi, N+1, r)] = G[E(t, phi, N-1, r)];
		}
	}

}

void fillTemporalGhostVolume(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dphi, REAL dtheta, REAL dr, REAL p){
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O; phi++){
		for(size_t theta=0; theta<N; theta++){
			for(size_t r=0; r<M; r++){
				a[I(tm1, phi, theta, r)] = a[I(t, phi, theta, r)] - 2.0*dt*PI_4;
				F[I(tm1, phi, theta, r)] = F[I(t, phi, theta, r)] - 2.0*dt*PI_5;
				G[I(tm1, phi, theta, r)] = G[I(t, phi, theta, r)] - 2.0*p*dt*PI_6;
			}
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
				//a[I(l, phi, theta, r)] = a_0[r] + PI_1;
				//F[I(l, phi, theta, r)] = (REAL)q*(dtheta*(REAL)theta) + PI_2;
				//G[I(l, phi, theta, r)] = p*((dt*l)/L - dphi*phi) + PI_3;
                a[I(l, phi, theta, r)] = 100;
				F[I(l, phi, theta, r)] = 0;
				G[I(l, phi, theta, r)] = 0;
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
					a[I(t, phi, theta, r)] = 0;
					F[I(t, phi, theta, r)] = 0;
					G[I(t, phi, theta, r)] = 0;
				} else if (theta == 0 ){
					a[I(t, phi, theta, r)] = 0;
					F[I(t, phi, theta, r)] = 0;
					G[I(t, phi, theta, r)] = 0;
				} else if (phi == 0 || phi == O-1 ){
					a[I(t, phi, theta, r)] = 0;
					F[I(t, phi, theta, r)] = 0;
					G[I(t, phi, theta, r)] = 0;
				}
			}
		}
	}
}
