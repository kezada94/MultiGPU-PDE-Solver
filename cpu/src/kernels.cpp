#include "defines.h"
#include "kernels.h"
#include "EquationAlfa.h"
#include "EquationF.h"
#include "EquationG.h"
#include <fstream>
#include <limits>

void fillGhostPoints(REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O){
 
	// left
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O+2; phi++){
		for(size_t theta=0; theta<N+2; theta++){
			a[E(t, phi, theta, 0)] = a[E(t, phi, theta, 2)];
			F[E(t, phi, theta, 0)] = F[E(t, phi, theta, 2)];
			G[E(t, phi, theta, 0)] = G[E(t, phi, theta, 2)];
		}
	}

	// right
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O+2; phi++){
		for(size_t theta=0; theta<N+2; theta++){
			a[E(t, phi, theta, M+1)] = a[E(t, phi, theta, M-1)];
			F[E(t, phi, theta, M+1)] = F[E(t, phi, theta, M-1)];
			G[E(t, phi, theta, M+1)] = G[E(t, phi, theta, M-1)];
		}
	}

	// top
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		for(size_t phi=0; phi<N+2; phi++){
			a[E(t, phi, 0, r)] = a[E(t, phi, 2, r)];
			F[E(t, phi, 0, r)] = F[E(t, phi, 2, r)];
			G[E(t, phi, 0, r)] = G[E(t, phi, 2, r)];
		}
	}

	// bottom
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		for(size_t phi=0; phi<N+2; phi++){
			a[E(t, phi, N+1, r)] = a[E(t, phi, N-1, r)];
			F[E(t, phi, N+1, r)] = F[E(t, phi, N-1, r)];
			G[E(t, phi, N+1, r)] = G[E(t, phi, N-1, r)];
		}
	}

	// front
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		for(size_t theta=0; theta<N+2; theta++){
			a[E(t, 0, theta, r)] = a[E(t, 2, theta, r)];
			F[E(t, 0, theta, r)] = F[E(t, 2, theta, r)];
			G[E(t, 0, theta, r)] = G[E(t, 2, theta, r)];
		}
	}

	// back
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		for(size_t theta=0; theta<N+2; theta++){
			a[E(t, O+1, theta, r)] = a[E(t, O-1, theta, r)];
			F[E(t, O+1, theta, r)] = F[E(t, O-1, theta, r)];
			G[E(t, O+1, theta, r)] = G[E(t, O-1, theta, r)];
		}
	}
	
	//corners
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O+2; phi++){
		a[E(t, phi, 0, 0)] = a[E(t, phi, 2, 2)];
		F[E(t, phi, 0, 0)] = F[E(t, phi, 2, 2)];
		G[E(t, phi, 0, 0)] = G[E(t, phi, 2, 2)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O+2; phi++){
		a[E(t, phi, 0, M+1)] = a[E(t, phi, 2, M-1)];
		F[E(t, phi, 0, M+1)] = F[E(t, phi, 2, M-1)];
		G[E(t, phi, 0, M+1)] = G[E(t, phi, 2, M-1)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O+2; phi++){
		a[E(t, phi, N+1, 0)] = a[E(t, phi, N-1, 2)];
		F[E(t, phi, N+1, 0)] = F[E(t, phi, N-1, 2)];
		G[E(t, phi, N+1, 0)] = G[E(t, phi, N-1, 2)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<O+2; phi++){
		a[E(t, phi, N+1, M+1)] = a[E(t, phi, N-1, M-1)];
		F[E(t, phi, N+1, M+1)] = F[E(t, phi, N-1, M-1)];
		G[E(t, phi, N+1, M+1)] = G[E(t, phi, N-1, M-1)];
	}

	// border at phi=0
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		a[E(t, 0, 0, r)] = a[E(t, 2, 2, r)];
		F[E(t, 0, 0, r)] = F[E(t, 2, 2, r)];
		G[E(t, 0, 0, r)] = G[E(t, 2, 2, r)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		a[E(t, 0, N+1, r)] = a[E(t, 2, N-1, r)];
		F[E(t, 0, N+1, r)] = F[E(t, 2, N-1, r)];
		G[E(t, 0, N+1, r)] = G[E(t, 2, N-1, r)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t theta=0; theta<N+2; theta++){
		a[E(t, 0, theta, 0)] = a[E(t, 2, theta, 2)];
		F[E(t, 0, theta, 0)] = F[E(t, 2, theta, 2)];
		G[E(t, 0, theta, 0)] = G[E(t, 2, theta, 2)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t theta=0; theta<N+2; theta++){
		a[E(t, 0, theta, M+1)] = a[E(t, 2, theta, M-1)];
		F[E(t, 0, theta, M+1)] = F[E(t, 2, theta, M-1)];
		G[E(t, 0, theta, M+1)] = G[E(t, 2, theta, M-1)];
	}

	a[E(t, 0, 0, 0)] = a[E(t, 2, 2, 2)];
	F[E(t, 0, 0, 0)] = F[E(t, 2, 2, 2)];
	G[E(t, 0, 0, 0)] = G[E(t, 2, 2, 2)];

	a[E(t, 0, 0, M+1)] = a[E(t, 2, 2, M-1)];
	F[E(t, 0, 0, M+1)] = F[E(t, 2, 2, M-1)];
	G[E(t, 0, 0, M+1)] = G[E(t, 2, 2, M-1)];

	a[E(t, 0, N+1, 0)] = a[E(t, 2, N-1, 2)];
	F[E(t, 0, N+1, 0)] = F[E(t, 2, N-1, 2)];
	G[E(t, 0, N+1, 0)] = G[E(t, 2, N-1, 2)];

	a[E(t, 0, N+1, M+1)] = a[E(t, 2, N-1, M-1)];
	F[E(t, 0, N+1, M+1)] = F[E(t, 2, N-1, M-1)];
	G[E(t, 0, N+1, M+1)] = G[E(t, 2, N-1, M-1)];

	//Border at phi = O-1
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		a[E(t, O+1, 0, r)] = a[E(t, O-1, 2, r)];
		F[E(t, O+1, 0, r)] = F[E(t, O-1, 2, r)];
		G[E(t, O+1, 0, r)] = G[E(t, O-1, 2, r)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t r=0; r<M+2; r++){
		a[E(t, O+1, N+1, r)] = a[E(t, O-1, N-1, r)];
		F[E(t, O+1, N+1, r)] = F[E(t, O-1, N-1, r)];
		G[E(t, O+1, N+1, r)] = G[E(t, O-1, N-1, r)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t theta=0; theta<N+2; theta++){
		a[E(t, O+1, theta, 0)] = a[E(t, O-1, theta, 2)];
		F[E(t, O+1, theta, 0)] = F[E(t, O-1, theta, 2)];
		G[E(t, O+1, theta, 0)] = G[E(t, O-1, theta, 2)];
	}
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t theta=0; theta<N+2; theta++){
		a[E(t, O+1, theta, M+1)] = a[E(t, O-1, theta, M-1)];
		F[E(t, O+1, theta, M+1)] = F[E(t, O-1, theta, M-1)];
		G[E(t, O+1, theta, M+1)] = G[E(t, O-1, theta, M-1)];
	}

	a[E(t, O+1, 0, 0)] = a[E(t, O-1, 2, 2)];
	F[E(t, O+1, 0, 0)] = F[E(t, O-1, 2, 2)];
	G[E(t, O+1, 0, 0)] = G[E(t, O-1, 2, 2)];

	a[E(t, O+1, 0, M+1)] = a[E(t, O-1, 2, M-1)];
	F[E(t, O+1, 0, M+1)] = F[E(t, O-1, 2, M-1)];
	G[E(t, O+1, 0, M+1)] = G[E(t, O-1, 2, M-1)];

	a[E(t, O+1, N+1, 0)] = a[E(t, O-1, N-1, 2)];
	F[E(t, O+1, N+1, 0)] = F[E(t, O-1, N-1, 2)];
	G[E(t, O+1, N+1, 0)] = G[E(t, O-1, N-1, 2)];

	a[E(t, O+1, N+1, M+1)] = a[E(t, O-1, N-1, M-1)];
	F[E(t, O+1, N+1, M+1)] = F[E(t, O-1, N-1, M-1)];
	G[E(t, O+1, N+1, M+1)] = G[E(t, O-1, N-1, M-1)];

}

void fillTemporalGhostVolume(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, size_t phi_offset, size_t globalWidth, REAL dt, REAL dphi, REAL dtheta, REAL dr, REAL p){
	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for(size_t phi=0; phi<M; phi++){
		for(size_t theta=0; theta<N; theta++){
			for(size_t r=0; r<O; r++){
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
		cout << phi << endl;
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
