#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void computeNextIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		cout << m << endl;
		for(size_t n=0; n<N; n++){
			for(size_t o=0; o<O; o++){
				if (m == 0 || m == M-1){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
				}
				else if (n == 0 || n == N-1){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
				}
				else if (o == 0 || o == O-1){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
				} else {
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNexta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl, a_0);
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl, a_0);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeNextG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl, a_0);
				}

			}
		}
	}
	
} 

void computeFirstIteration(REAL* a, REAL* F, REAL *G, size_t l, size_t t, size_t tm1, size_t tm2, size_t tm3, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		cout << m << endl;
		for(size_t n=0; n<N; n++){
			for(size_t o=0; o<O; o++){
				if (m == 0 || m == M-1 ){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
				}
				else if (n == 0 || n == N-1){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
				}
				else if (o == 0 || o == O-1){
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(dtheta*n);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
				} else {
					a[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeFirsta(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl, a_0);
					F[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeFirstF(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl, a_0);
					G[(t)*M*N*O + (m)*N*O + (n)*O + o] = computeFirstG(a, F, G, tm1, tm2, tm3, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl, a_0);
				}

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

void fillInitialCondition(REAL* a, REAL* F, REAL *G, size_t l, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl, int p, int q, int L, REAL* a_0){
	#pragma omp parallel for schedule(static) num_threads(10)
	for(size_t m=0; m<M; m++){
		for(size_t n=0; n<N; n++){
			for(size_t o=0; o<O; o++){
				a[(l)*M*N*O + (m)*N*O + (n)*O + o] = a_0[m];
				F[(l)*M*N*O + (m)*N*O + (n)*O + o] = (REAL)q*(dtheta*(REAL)n);
				G[(l)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/L - dphi*o);
			}
		}
	}
	
} 


