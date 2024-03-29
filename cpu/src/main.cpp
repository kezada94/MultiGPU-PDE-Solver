#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <Eigen/Dense>
 
using namespace Eigen;

#include "Grid.h"
#include "Linspace.h"
#include "kernels.h"


using namespace std;

const REAL PI = 3.14159265358979323846f;
const size_t buffSize = 4;
const size_t nfunctions= 3;


//void writeCheckpoint(ofstream &out, auto a, auto F, auto G, size_t t, size_t M, size_t N, size_t O, size_t l);
REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual);
MatrixXcd getU(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_0(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_2(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
MatrixXcd getL_3(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);

void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual);

vector<REAL> genLinspace(REAL start, REAL end, size_t n){
	vector<REAL> vec = vector<REAL>(n);
	REAL delta = (end - start)/(n-1);
	for(size_t i=0; i<n; ++i){
		vec[i] = start + i*delta;
		cout << vec[i] << endl;
	}
	return vec;
}

int main(int argc, char *argv[]){

    if (argc != 10){
        printf("Error. Try executing with\n\t./laplace <dt> <M> <N> <O> <p> <q> <n> <niter> <boundary>\n");
        exit(1);
    }

    const size_t M = atoi(argv[2]);
    const size_t N = atoi(argv[3]);
    const size_t O = atoi(argv[4]);
    REAL dt = atof(argv[1]);
    cout << dt << endl;
    int p = atoi(argv[5]);
    int q = atoi(argv[6]);
    int n = atoi(argv[7]);
    bool boundary = atoi(argv[9]);
    size_t niter = atoi(argv[8]);


    REAL dr = 2*PI/999.0;
    REAL dtheta = PI/999.0;
    REAL dphi = 2*PI/999.0;

    // +2 for ghost points offset
	size_t nelements = buffSize*(M+2)*(N+2)*(O+2);
	cout << "Number of elements: " << nelements << endl;

// its better to separate fdm grid with continuous axes
	cout << "Generating grids..."; fflush(stdout);
    REAL *a = new REAL[nelements];
    REAL *F = new REAL[nelements];
    REAL *G = new REAL[nelements];
	cout << "done" << endl;

    cout << "Reading values for a(r)...";fflush(stdout);
    REAL *a_0 = new REAL[M];
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
    //REAL lambda = 4.0/6.0*(1.0/5.45*129.0)*50.9;
    REAL lambda = 1.0; //4.0/6.0*(1.0/5.45*129.0)*50.9;
    //REAL lambda = 4.0/6.0*(1.0/E*186.0)*50.9;

    REAL l_1 = 1.f;
    REAL l_2 = 1.f;

    string filename = "result-"+to_string(M)+".dat";
    string filename0 = "result-"+to_string(M)+"-A.dat";
    string filename1 = "result-"+to_string(M)+"-F.dat";
    string filename2 = "result-"+to_string(M)+"-G.dat";

	

    cout << "Filling state 0..."; fflush(stdout);
    fillInitialCondition(a, F, G, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    if (boundary == 0){
	    fillDirichletBoundary(a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints(a, F, G, 0, M, N, O);
    } 
    cout << " done." << endl;
    writeTimeSnapshot(filename0, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    writeTimeSnapshot(filename1, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    writeTimeSnapshot(filename2, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;

    cout << "Filling state 1..."; fflush(stdout);
    fillInitialCondition(a, F, G, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    //computeFirstIteration(a, F, G, 1, 1, 0, -1, -2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    if (boundary == 0){
	    fillDirichletBoundary(a, F, G, 1, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints(a, F, G, 1, M, N, O);
    } 
    cout << " done." << endl;

    writeTimeSnapshot(filename0, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    writeTimeSnapshot(filename1, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    writeTimeSnapshot(filename2, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    getchar();

    cout << "Filling state 2..."; fflush(stdout);
    computeSecondIteration(a, F, G, 2, 2, 1, 0, -1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    if (boundary == 0){
	    fillDirichletBoundary(a, F, G, 2, 2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints(a, F, G, 2, M, N, O);
    } 


    writeTimeSnapshot(filename0, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    writeTimeSnapshot(filename1, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    writeTimeSnapshot(filename2, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    getchar();


    for (size_t l=3; l<niter; ++l){
	cout << "Starting iteration l=" << l << endl;
	size_t t = l%buffSize;
	size_t tm1 = (l-1)%buffSize;
	size_t tm2 = (l-2)%buffSize;
	size_t tm3 = (l-3)%buffSize;

	cout << t << " " << tm1 << " " << tm2 << " " << tm3 << " "  << endl;

	computeNextIteration(a, F, G, l, t, tm1, tm2, tm3, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
	if (boundary == 0){
		fillDirichletBoundary(a, F, G, l, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
	} else if (boundary == 1){
	fillGhostPoints(a, F, G, t, M, N, O);
	} 


	cout << "Finished iteration l=" << l << endl;

	//cout << "Save? [y/n]" << endl;
	//char key = getchar();
	if (l%10==0){
	    cout << "Saving values..." << endl;
	    writeTimeSnapshot(filename0, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
	    writeTimeSnapshot(filename1, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
	    writeTimeSnapshot(filename2, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
	    cout << "done." << endl;
	    //getchar();
	}
}
	
    //Boundary filll
    return 0;
}

static MatrixXcd i2x2 = [] {
    MatrixXcd matrix(2,2);
    matrix << 1., 0, 0, 1.;
    return matrix;
}();
static MatrixXcd t1 = [] {
    MatrixXcd matrix(2,2);
    matrix << 0, 1., 1., 0;
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
    REAL anow = a[I(t, r, theta, phi)];
    REAL Fnow = F[I(t, r, theta, phi)];
    REAL Gnow = G[I(t, r, theta, phi)];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    MatrixXcd U = cos(anow)*i2x2 + sin(anow)*(t1*n1+t2*n2+t3*n3);
    return U;
}
MatrixXcd getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O){
    REAL anow = a[I(t, r, theta, phi)];
    REAL Fnow = F[I(t, r, theta, phi)];
    REAL Gnow = G[I(t, r, theta, phi)];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    MatrixXcd Um1 = cos(anow)*i2x2 - sin(anow)*(t1*n1+t2*n2+t3*n3);
    return Um1;
}

MatrixXcd getF(MatrixXcd L1, MatrixXcd L2){
    return (L1*L2 - L2*L1);
}

REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual){
    MatrixXcd Um1 = getUm1(a, F, G, t, r, theta, phi, M, N, O);
    MatrixXcd L_0 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, tm1, r, theta, phi, M, N, O))/dt); 
    MatrixXcd L_1 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r-1, theta, phi, M, N, O))/(dr)); 
    MatrixXcd L_2 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta-1, phi, M, N, O))/(dtheta));
    MatrixXcd L_3 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta, phi-1, M, N, O))/(dphi)); 
    //REAL K = 4970.25;
    REAL K = 2.0;
    complex<double> cons = -K/2.0f;
    REAL t00 = ((cons)*(L_0*L_0 - 1.0/2.0*-1.0*(L_0*L_0 + 0*L_1*L_1 +0*L_2*L_2 +0*L_3*L_3)//).trace()).real();
                        + 0*lambda/4.0*(-1.0*getF(L_0, L_0)*getF(L_0, L_0)
                                    +getF(L_0, L_1)*getF(L_0, L_1)
                                    +getF(L_0, L_2)*getF(L_0, L_2)
                                    +getF(L_0, L_3)*getF(L_0, L_3)
                            	    -(-1.0/4.0*(getF(L_0, L_1)*getF(L_0, L_1)
                                        	+getF(L_0, L_2)*getF(L_0, L_2)
                                        	+getF(L_0, L_3)*getF(L_0, L_3)
                                        	+getF(L_1, L_0)*getF(L_1, L_0)
                                        	+getF(L_2, L_0)*getF(L_2, L_0)
                                        	+getF(L_3, L_0)*getF(L_3, L_0)
                                        	+getF(L_1, L_2)*getF(L_1, L_2)
                                        	+getF(L_2, L_1)*getF(L_2, L_1)
                                        	+getF(L_1, L_3)*getF(L_1, L_3)
                                        	+getF(L_3, L_1)*getF(L_3, L_1)
                                        	+getF(L_2, L_3)*getF(L_2, L_3)
                                        	+getF(L_3, L_2)*getF(L_3, L_2))) )).trace()).real();
	if (cual == 0){
    	return a[I(t, r, theta, phi)];//t00;
	} else if (cual == 1){
		return F[I(t, r, theta, phi)];//t00;
	} else if (cual == 2){
    	return G[I(t, r, theta, phi)];//t00;
	}
    return t00;


}
void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual){
    int count = 0;
    ofstream file;
    //file.open(filename);
    file.open(filename, std::ofstream::app);
    double mm = 0;
    for (size_t m=0; m<M; m=round(mm)){
    	cout << m << endl;
        double nn = 0;
            for (size_t n=0; n<N; n=round(nn)){
            double oo = 0;
            for (size_t o=0; o<O; o=round(oo)){
                if (file.is_open()){
					cout << m << ", " << n << ", " << o << endl;
                    file <<std::fixed << std::setprecision(62) << getT00(a, F, G, t, tm1, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, cual) << "\n";
                    file.flush();
                }
                else{
                    std::cerr << "didn't write" << std::endl;
                }
                oo += (double)(O-1)/9.0;
            }
            nn += (double)(N-1)/99.0;
        }
        mm += (double)(M-1)/99.0;
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
