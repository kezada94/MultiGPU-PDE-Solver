#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void fillGhostPoints(REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O){
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t n=0; n<N+2; n++){
		for(size_t o=0; o<O+2; o++){
			a[(t)*M*N*O + (0)*N*O + (n)*O + o] = a[(t)*M*N*O + (2)*N*O + (n)*O + o];
			F[(t)*M*N*O + (0)*N*O + (n)*O + o] = F[(t)*M*N*O + (2)*N*O + (n)*O + o];
			G[(t)*M*N*O + (0)*N*O + (n)*O + o] = G[(t)*M*N*O + (2)*N*O + (n)*O + o];
		}
	}
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t o=0; o<O+2; o++){
			a[(t)*M*N*O + (m)*N*O + (0)*O + o] = a[(t)*M*N*O + (m)*N*O + (2)*O + o];
			F[(t)*M*N*O + (m)*N*O + (0)*O + o] = F[(t)*M*N*O + (m)*N*O + (2)*O + o];
			G[(t)*M*N*O + (m)*N*O + (0)*O + o] = G[(t)*M*N*O + (m)*N*O + (2)*O + o];
		}
	}	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t n=0; n<N+2; n++){
			a[(t)*M*N*O + (m)*N*O + (n)*O + 0] = a[(t)*M*N*O + (m)*N*O + (n)*O + 2];
			F[(t)*M*N*O + (m)*N*O + (n)*O + 0] = a[(t)*M*N*O + (m)*N*O + (n)*O + 2];
			G[(t)*M*N*O + (m)*N*O + (n)*O + 0] = a[(t)*M*N*O + (m)*N*O + (n)*O + 2];
		}
	}

	// Boundary m=L
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t n=0; n<N+2; n++){
		for(size_t o=0; o<O+2; o++){
			a[(t)*M*N*O + (M+1)*N*O + (n)*O + o] = a[(t)*M*N*O + (M-1)*N*O + (n)*O + o];
			F[(t)*M*N*O + (M+1)*N*O + (n)*O + o] = a[(t)*M*N*O + (M-1)*N*O + (n)*O + o];
			G[(t)*M*N*O + (M+1)*N*O + (n)*O + o] = a[(t)*M*N*O + (M-1)*N*O + (n)*O + o];
		}
	}
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t o=0; o<O+2; o++){
			a[(t)*M*N*O + (m)*N*O + (N+1)*O + o] = a[(t)*M*N*O + (m)*N*O + (N-1)*O + o];
			F[(t)*M*N*O + (m)*N*O + (N+1)*O + o] = a[(t)*M*N*O + (m)*N*O + (N-1)*O + o];
			G[(t)*M*N*O + (m)*N*O + (N+1)*O + o] = a[(t)*M*N*O + (m)*N*O + (N-1)*O + o];
		}
	}	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M+2; m++){
		for(size_t n=0; n<N+2; n++){
			a[(t)*M*N*O + (m)*N*O + (n)*O + O+1] = a[(t)*M*N*O + (m)*N*O + (n)*O + O-1];
			F[(t)*M*N*O + (m)*N*O + (n)*O + O+1] = a[(t)*M*N*O + (m)*N*O + (n)*O + O-1];
			G[(t)*M*N*O + (m)*N*O + (n)*O + O+1] = a[(t)*M*N*O + (m)*N*O + (n)*O + O-1];
		}
	}
}

void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(static) num_threads(10)
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
	#pragma omp parallel for schedule(static) num_threads(10)
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
	#pragma omp parallel for schedule(static) num_threads(10)
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


