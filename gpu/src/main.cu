#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
 
using namespace Eigen;

#include "Grid.h"
#include "Linspace.h"
#include "kernels.cuh"

#include "EquationAlfa.cuh"
#include "EquationF.cuh"
#include "EquationG.cuh"
using namespace std;

const REAL PI = 3.14159265358979323846f;
const size_t buffSize = 4;
const size_t nfunctions= 3;


//void writeCheckpoint(ofstream &out, auto a, auto F, auto G, size_t t, size_t M, size_t N, size_t O, size_t l);
REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb);
MatrixXcd getU(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_0(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_2(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_3(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);

void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lamb);

vector<REAL> genLinspace(REAL start, REAL end, size_t n){
	vector<REAL> vec = vector<REAL>(n);
	REAL delta = (end - start)/(n-1);
	for(size_t i=0; i<n; ++i){
		vec[i] = start + i*delta;
	}
	return vec;
}

int main(int argc, char *argv[]){

    if (argc != 8){
        printf("Error. Try executing with\n\t./laplace <dt> <M> <N> <O> <p> <q> <n>\n");
        exit(1);
    }

    const size_t L = 999999;
    const size_t M = atoi(argv[2]);
    const size_t N = atoi(argv[3]);
    const size_t O = atoi(argv[4]);
    REAL dt = atof(argv[1]);
    cout << dt << endl;
    int p = atoi(argv[5]);
    int q = atoi(argv[6]);
    int n = atoi(argv[7]);

    vector<REAL> ax_r = genLinspace(0, 2*PI, M); // for r
    vector<REAL> ax_theta = genLinspace(0, PI, N); // for r
    vector<REAL> ax_phi = genLinspace(0, 2*PI, O); // for r

    REAL dr = ax_r[1] - ax_r[0];
    REAL dtheta = ax_theta[1] - ax_theta[0];
    REAL dphi = ax_phi[1] - ax_phi[0];

	size_t nelements = buffSize*M*N*O;
	cout << "Number of elements: " << nelements << endl;

// its better to separate fdm grid with continuous axes
	cout << "Generating grids..."; fflush(stdout);
    REAL *a;
	cudaMalloc(&a, nelements*sizeof(REAL));
    REAL *F;
	cudaMalloc(&F, nelements*sizeof(REAL));
    REAL *G;
	cudaMalloc(&G, nelements*sizeof(REAL));
	cout << "done" << endl;
    cout << "Reading values for a(r)...";fflush(stdout);

    REAL *a_0;
    cudaMallocManaged(&a_0, M*sizeof(REAL));
    int i = 0;

	fstream file("../alfa(r)-"+to_string(n)+"-"+to_string(q)+"-1000.csv");
    if (file.is_open()){
        string line;
        while(getline(file, line)){
            a_0[i] = stod(line);
            i++;
        }
    } else {
        cout << "Could not open file." << endl;
        exit(-190);
    }
    cout << "done. " << i << " elements red" << endl;;

    //REAL lamb = 4.0/6.0*(1.0/5.45*129.0)*50.9;
    REAL lamb = 1.0; //4.0/6.0*(1.0/5.45*129.0)*50.9;
    //REAL lamb = 4.0/6.0*(1.0/E*186.0)*50.9;

    REAL l_1 = 1.f;
    REAL l_2 = 1.f;

    string filename = "result-"+to_string(M)+".dat";

	dim3 g, b;
	b = dim3(8, 8, 8);
	g = dim3((M+b.x-1)/(b.x), (N+b.y-1)/b.y, (O+b.z-1)/(b.z));
	cout << "Grid(" << g.x << ", " << g.y << ", " << g.z << ")" << endl;
	cout << "Block(" << b.x << ", " << b.y << ", " << b.z << ")" << endl;

	
	cout << "Filling state 0..."; fflush(stdout);
	fillInitialCondition<<<g, b>>>(a, F, G, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
	cout << " done." << endl;

    cout << "Filling state 1..."; fflush(stdout);
	fillInitialCondition<<<g, b>>>(a, F, G, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
	cout << " done." << endl;

	cout << "Filling state 2..."; fflush(stdout);
    computeFirsta<<<g, b>>>(a, F, G, 2, 2, 1, 0, -1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
	cout << "a done." << endl;

    computeFirstF<<<g, b>>>(a, F, G, 2, 2, 1, 0, -1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
	cout << "F done." << endl;

    computeFirstG<<<g, b>>>(a, F, G, 2, 2, 1, 0, -1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
	cout << "G done." << endl;

    writeTimeSnapshot(filename, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb);
    cout << "Written" << endl;
    getchar();


    for (size_t l=3; l<L; ++l){
		cout << "Starting iteration l=" << l << endl;
		size_t t = l%buffSize;
		size_t tm1 = (l-1)%buffSize;
		size_t tm2 = (l-2)%buffSize;
		size_t tm3 = (l-3)%buffSize;

		cout << t << " " << tm1 << " " << tm2 << " " << tm3 << " "  << endl;

        computeNexta<<<g, b>>>(a, F, G, l, t, tm1, tm2, tm3, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
        computeNextF<<<g, b>>>(a, F, G, l, t, tm1, tm2, tm3, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);
	cudaDeviceSynchronize();
        computeNextG<<<g, b>>>(a, F, G, l, t, tm1, tm2, tm3, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb, p, q, 1, a_0);        
	cudaDeviceSynchronize();

		cout << "Finished iteration l=" << l << endl;

		cout << "Save? [y/n]" << endl;
		char key = getchar();
		if (key == 'y'){
            cout << "Saving values..." << endl;
            writeTimeSnapshot(filename, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lamb);
            cout << "done." << endl;
            getchar();
		}
    }
	
    //Boundary filll
    return 0;
}

static MatrixXcd i2x2 = [] {
    MatrixXcd matrix(2,2);
    matrix << 1, 0, 0, 1;
    return matrix;
}();
static MatrixXcd t1 = [] {
    MatrixXcd matrix(2,2);
    matrix << 0, 1, 1., 0;
    return matrix;
}();
static MatrixXcd t2 = [] {
    MatrixXcd matrix(2,2);
    matrix << 0, -1.if, 1.if, 0;
    return matrix;
}();
static MatrixXcd t3 = [] {
    MatrixXcd matrix(2,2) ;
    matrix << 1, 0, 0, -1;
    return matrix;
}();
MatrixXcd getU(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O){
    REAL anow = a[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Fnow = F[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Gnow = G[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    MatrixXcd U = cos(anow)*i2x2 + sin(anow)*(t1*n1+t2*n2+t3*n3);
    return U;
}
MatrixXcd getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O){
    REAL anow = a[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Fnow = F[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Gnow = G[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    MatrixXcd Um1 = cos(anow)*i2x2 - sin(anow)*(t1*n1+t2*n2+t3*n3);
    return Um1;
}

MatrixXcd getF(MatrixXcd L1, MatrixXcd L2){
    return (L1*L2 - L2*L1);
}

REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda){
    MatrixXcd Um1 = getUm1(a, F, G, t, r, theta, phi, M, N, O);
    MatrixXcd L_0 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, tm1, r, theta, phi, M, N, O))/dt); 
    MatrixXcd L_1 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r-1, theta, phi, M, N, O))/dr); 
    MatrixXcd L_2 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta-1, phi, M, N, O))/dtheta);
    MatrixXcd L_3 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta, phi-1, M, N, O))/dphi); 
    //REAL K = 4970.25;
    REAL K = 2.0;
    complex<double> cons = -K/2.0f;
    REAL t00 = ((cons)*(L_0*L_0 /*- 1.0/2.0*-1.0*(-1.0*L_0*L_0 + 0*L_1*L_1 + 0*L_2*L_2 + 0*L_3*L_3)*/).trace()).real();/*
                        + lambda/4.0*(-1.0*getF(L_0, L_0)*getF(L_0, L_0)
                                    +l_1*getF(L_0, L_1)*getF(L_0, L_1)
                                    +l_1*getF(L_0, L_2)*getF(L_0, L_2)
                                    +l_2*getF(L_0, L_3)*getF(L_0, L_3)
                            	    -(-1.0/4.0*(-1.0*getF(L_0, L_1)*getF(L_0, L_1)
                                        	-1.0*getF(L_0, L_2)*getF(L_0, L_2)
                                        	-1.0*getF(L_0, L_3)*getF(L_0, L_3)
                                        	-1.0*getF(L_1, L_0)*getF(L_1, L_0)
                                        	-1.0*getF(L_2, L_0)*getF(L_2, L_0)
                                        	-1.0*getF(L_3, L_0)*getF(L_3, L_0)
                                        	+l_1*l_1*getF(L_1, L_2)*getF(L_1, L_2)
                                        	+l_1*l_1*getF(L_2, L_1)*getF(L_2, L_1)
                                        	+l_1*l_2*getF(L_1, L_3)*getF(L_1, L_3)
                                        	+l_1*l_2*getF(L_3, L_1)*getF(L_3, L_1)
                                        	+l_2*l_1*getF(L_2, L_3)*getF(L_2, L_3)
                                        	+l_2*l_1*getF(L_3, L_2)*getF(L_3, L_2))) )).trace()).real();*/
    //return G[(0)*M*N*O + (r)*N*O + (theta)*O + phi];//t00;
    return t00;


}
void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda){
    int count = 0;
    ofstream file;
    file.open(filename);
    //file.open(filename, std::ofstream::app);
    double mm = 1;
    for (size_t m=1; m<M; m=round(mm)){
    	cout << m << endl;
        double nn = 1;
            for (size_t n=1; n<N; n=round(nn)){
            double oo = 1;
            for (size_t o=1; o<O; o=round(oo)){
                if (file.is_open()){
					//cout << m << ", " << n << ", " << o << endl;
                    file <<std::fixed  << getT00(a, F, G, t, tm1, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda) << "\n";
                    file.flush();
                }
                else{
                    std::cerr << "didn't write" << std::endl;
                }
                oo += (double)(O-2)/9.0;
            }
            nn += (double)(N-2)/99.0;
        }
        mm += (double)(M-2)/99.0;
    }
    file.close();

}
/*
void writeCheckpoint(ofstream &out, auto a, auto F, auto G, size_t t, size_t M, size_t N, size_t O, size_t l){
    for (size_t m=0; m<M; ++m){
	    for (size_t n=0; n<N; ++n){
		    for (size_t o=0; o<O; ++o){
		    }
	    }
    }
}
A
*/
