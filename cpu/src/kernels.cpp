#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=1; m<M-1; m++){
		cout << m << endl;
		for(size_t n=1; n<N-1; n++){
			for(size_t o=1; o<O-1; o++){
				a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNexta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			}
		}
	}
	
	// Boundary m=0
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t n=0; n<N; n++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (0)*N*O + (n)*O + o] = computeBoundaryAr0(a, F, G, tm1, tm2, tm3, 0, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (0)*N*O + (n)*O + o] = computeBoundaryFr0(a, F, G, tm1, tm2, tm3, 0, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (0)*N*O + (n)*O + o] = computeBoundaryGr0(a, F, G, tm1, tm2, tm3, 0, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (m)*N*O + (0)*O + o] = computeBoundaryAtheta0(a, F, G, tm1, tm2, tm3, m, 0, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (0)*O + o] = computeBoundaryFtheta0(a, F, G, tm1, tm2, tm3, m, 0, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (0)*O + o] = computeBoundaryGtheta0(a, F, G, tm1, tm2, tm3, m, 0, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t n=0; n<N; n++){
			a[(t)*M*N*O + (m)*N*O + (n)*O + 0] = computeBoundaryAphi0(a, F, G, tm1, tm2, tm3, m, n, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (n)*O + 0] = computeBoundaryFphi0(a, F, G, tm1, tm2, tm3, m, n, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (n)*O + 0] = computeBoundaryGphi0(a, F, G, tm1, tm2, tm3, m, n, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}

	// Boundary m=0
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t n=0; n<N; n++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (M-1)*N*O + (n)*O + o] = computeBoundaryAr0(a, F, G, tm1, tm2, tm3, M-1, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (M-1)*N*O + (n)*O + o] = computeBoundaryFr0(a, F, G, tm1, tm2, tm3, M-1, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (M-1)*N*O + (n)*O + o] = computeBoundaryGr0(a, F, G, tm1, tm2, tm3, M-1, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (m)*N*O + (N-1)*O + o] = computeBoundaryAtheta0(a, F, G, tm1, tm2, tm3, m, N-1, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (N-1)*O + o] = computeBoundaryFtheta0(a, F, G, tm1, tm2, tm3, m, N-1, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (N-1)*O + o] = computeBoundaryGtheta0(a, F, G, tm1, tm2, tm3, m, N-1, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t n=0; n<N; n++){
			a[(t)*M*N*O + (m)*N*O + (n)*O + O-1] = computeBoundaryAphi0(a, F, G, tm1, tm2, tm3, m, n, O-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (n)*O + O-1] = computeBoundaryFphi0(a, F, G, tm1, tm2, tm3, m, n, O-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (n)*O + O-1] = computeBoundaryGphi0(a, F, G, tm1, tm2, tm3, m, n, O-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}
	
} 

void computeFirstIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=1; m<M-1; m++){
		cout << m << endl;
		for(size_t n=1; n<N-1; n++){
			for(size_t o=1; o<O-1; o++){
				a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeFirsta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeFirstF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
				G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeFirstG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			}
		}
	}



	// Boundary m=0
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t n=0; n<N; n++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (0)*N*O + (n)*O + o] = computeBoundaryAr0(a, F, G, tm1, tm2, tm3, 0, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (0)*N*O + (n)*O + o] = computeBoundaryFr0(a, F, G, tm1, tm2, tm3, 0, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (0)*N*O + (n)*O + o] = computeBoundaryGr0(a, F, G, tm1, tm2, tm3, 0, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (m)*N*O + (0)*O + o] = computeBoundaryAtheta0(a, F, G, tm1, tm2, tm3, m, 0, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (0)*O + o] = computeBoundaryFtheta0(a, F, G, tm1, tm2, tm3, m, 0, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (0)*O + o] = computeBoundaryGtheta0(a, F, G, tm1, tm2, tm3, m, 0, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t n=0; n<N; n++){
			a[(t)*M*N*O + (m)*N*O + (n)*O + 0] = computeBoundaryAphi0(a, F, G, tm1, tm2, tm3, m, n, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (n)*O + 0] = computeBoundaryFphi0(a, F, G, tm1, tm2, tm3, m, n, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (n)*O + 0] = computeBoundaryGphi0(a, F, G, tm1, tm2, tm3, m, n, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}

	// Boundary m=0
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t n=0; n<N; n++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (M-1)*N*O + (n)*O + o] = computeBoundaryAr0(a, F, G, tm1, tm2, tm3, M-1, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (M-1)*N*O + (n)*O + o] = computeBoundaryFr0(a, F, G, tm1, tm2, tm3, M-1, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (M-1)*N*O + (n)*O + o] = computeBoundaryGr0(a, F, G, tm1, tm2, tm3, M-1, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t o=0; o<O; o++){
			a[(t)*M*N*O + (m)*N*O + (N-1)*O + o] = computeBoundaryAtheta0(a, F, G, tm1, tm2, tm3, m, N-1, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (N-1)*O + o] = computeBoundaryFtheta0(a, F, G, tm1, tm2, tm3, m, N-1, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (N-1)*O + o] = computeBoundaryGtheta0(a, F, G, tm1, tm2, tm3, m, N-1, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
		}
	}	
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t n=0; n<N; n++){
			a[(t)*M*N*O + (m)*N*O + (n)*O + O-1] = computeBoundaryAphi0(a, F, G, tm1, tm2, tm3, m, n, O-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			F[(t)*M*N*O + (m)*N*O + (n)*O + O-1] = computeBoundaryFphi0(a, F, G, tm1, tm2, tm3, m, n, O-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
			G[(t)*M*N*O + (m)*N*O + (n)*O + O-1] = computeBoundaryGphi0(a, F, G, tm1, tm2, tm3, m, n, O-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, L);
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
				a[(l)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m] + PI_1;
				F[(l)*M*N*O + (m)*N*O + (n)*O + o] = (REAL)q*(dtheta*(REAL)n) + PI_2;
				G[(l)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o) + PI_3;
			}
		}
	}
	
} 


