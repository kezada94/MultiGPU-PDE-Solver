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
void printMSEG(REAL* func, size_t l, size_t tp1, REAL dt, REAL dr, REAL dtheta, REAL dphi, size_t M, size_t N, size_t O, REAL p, REAL L){
	REAL  mse = 0;
   
    #pragma omp parallel for shared(mse) num_threads(48)
    for (size_t o=0; o<O; o++){
        double nn = 0;
        for (size_t n=0; n<N; n=round(nn)){
            double mm = 0;
            for (size_t m=0; m<M; m=round(mm)){
                REAL sum = 0;
                sum = p*(l*dt/L - o*dphi);
                #pragma omp critical
                mse += (func[I(tp1, o, n, m)] - sum);
                mm += (double)(M-1)/99.0;
            }
            nn += (double)(N-1)/99.0;
        }
    }
	mse /= (O*100*100);
    cout << "Mean Error G: " << std::setprecision(64) << mse << endl;
}

void printMSEF(REAL* func, size_t l, size_t tp1, REAL dt, REAL dr, REAL dtheta, REAL dphi, size_t M, size_t N, size_t O, REAL p, REAL L){
	REAL mse = 0;
   
    double oo = 0;
    for (size_t o=0; o<O; o=round(oo)){
        #pragma omp parallel for shared(mse) num_threads(48)
        for (size_t n=0; n<N; n++){
            double mm = 0;
            for (size_t m=0; m<M; m=round(mm)){
                REAL sum = 0;
                sum = 3.0*(dtheta*n);
                #pragma omp critical
                mse += (func[I(tp1, o, n, m)] - sum);
                mm += (double)(M-1)/99.0;
            }
        }
        oo += (double)(O-1)/99.0;
    }
	mse /= (N*100*100);
    cout << "Mean Error F: " << std::setprecision(64) << mse << endl;
}

void printMSEa(REAL* func, size_t l, size_t tp1, REAL dt, REAL dr, REAL dtheta, REAL dphi, size_t M, size_t N, size_t O, REAL p, REAL L, REAL* a_0){
	REAL mse = 0;
   
    double newo = 0;
    double inc = (O-1)/99.0;
    #pragma omp parallel for shared(mse, inc) num_threads(48)
    for (size_t o=0; o<100; o++){
        double nn = 0;
        for (size_t n=0; n<N; n=round(nn)){
            for (size_t m=0; m<M; m++){
                REAL sum = 0;
                sum = a_0[m];
                size_t oo = o*inc;
                #pragma omp critical
                mse += (func[I(tp1, oo, n, m)] - sum);
            }
            nn += (double)(N-1)/99.0;
        }
    }
	mse /= (M*100*100);
    cout << "Mean Error a: " << std::setprecision(64) << mse << endl;
}

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


    REAL dr = 2*PI/(double)(M-1);
    REAL dtheta = PI/(double)(N-1);
    REAL dphi = 2*PI/(double)(O-1);

    // +2 for ghost points offset
	size_t nelements = buffSize*(M+GHOST_SIZE)*(N+GHOST_SIZE)*(O+GHOST_SIZE);
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
	fstream file("../alfa(r)-"+to_string(n)+"-"+to_string(q)+"-"+to_string(M)+".csv");
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
    //if (boundary == 0){
	    fillDirichletBoundary(a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    //} else if (boundary == 1){
	    fillGhostPoints(a, F, G, 0, M, N, O);
    //} 
    cout << " done." << endl;
    printMSEa(a, 0, 0, dt, dr, dtheta, dphi, M, N, O, p, 1.0, a_0);
    //printMSEF(F, 0, 0, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
    //printMSEG(G, 0, 0, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
    writeTimeSnapshot(filename0, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    //writeTimeSnapshot(filename1, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    //writeTimeSnapshot(filename2, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
/*
    cout << "Filling state 1..."; fflush(stdout);
    fillInitialCondition(a, F, G, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    //computeFirstIteration(a, F, G, 1, 1, 0, -1, -2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    if (boundary == 0){
	    fillDirichletBoundary(a, F, G, 1, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints(a, F, G, 1, M, N, O);
    } 
    cout << " done." << endl;
    printMSEa(a, 1, 1, dt, dr, dtheta, dphi, M, N, O, p, 1.0, a_0);
    printMSEF(F, 1, 1, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
    printMSEG(G, 1, 1, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
    writeTimeSnapshot(filename, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 3);
    //writeTimeSnapshot(filename0, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    //writeTimeSnapshot(filename1, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    //writeTimeSnapshot(filename2, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    getchar();

    cout << "Filling state 2..."; fflush(stdout);
    fillInitialCondition(a, F, G, 2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    //computeSecondIteration(a, F, G, 2, 2, 1, 0, -1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    if (boundary == 0){
	    fillDirichletBoundary(a, F, G, 2, 2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints(a, F, G, 2, M, N, O);
    } 

    printMSEa(a, 2, 2, dt, dr, dtheta, dphi, M, N, O, p, 1.0, a_0);
    printMSEF(F, 2, 2, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
    printMSEG(G, 2, 2, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
    writeTimeSnapshot(filename, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 3);
    //writeTimeSnapshot(filename0, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    //writeTimeSnapshot(filename1, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    //writeTimeSnapshot(filename2, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    getchar();

*/
    for (size_t l=1; l<niter; ++l){
        cout << "Starting iteration l=" << l << endl;
        size_t t = l%buffSize;
        size_t tm1 = (l-1)%buffSize;
        size_t tm2 = (l-2)%buffSize;
        size_t tm3 = (l-3)%buffSize;

        cout << t << " " << tm1 << " " << tm2 << " " << tm3 << " "  << endl;

        computeNextIteration(a, F, G, l, t, tm1, tm2, tm3, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
        //if (boundary == 0){
            fillDirichletBoundary(a, F, G, l, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
        //} else if (boundary == 1){
            fillGhostPoints(a, F, G, t, M, N, O);
        //} 


        cout << "Finished iteration l=" << l << endl;

        //cout << "Save? [y/n]" << endl;
        //char key = getchar();
        if (l%10==0 || true){
            cout << "Saving values..." << endl;
            printMSEa(a, l, t, dt, dr, dtheta, dphi, M, N, O, p, 1.0, a_0);
            //printMSEF(F, l, t, dt, dr, dtheta, dphi, M, N, O, p, 1.0);
            //printMSEG(G, l, t, dt, dr, dtheta, dphi, M, N, O, p, 1.0);

            //writeTimeSnapshot(filename, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 3);
            writeTimeSnapshot(filename0, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
            //writeTimeSnapshot(filename1, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
            //writeTimeSnapshot(filename2, a, F, G, t, tm1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
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
    REAL anow = a[I(t, phi, theta, r)];
    REAL Fnow = F[I(t, phi, theta, r)];
    REAL Gnow = G[I(t, phi, theta, r)];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    MatrixXcd U = (double)cos(anow)*i2x2 + (double)sin(anow)*(t1*(double)n1+t2*(double)n2+t3*(double)n3);
    return U;
}
MatrixXcd getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O){
    REAL anow = a[I(t, phi, theta, r)];
    REAL Fnow = F[I(t, phi, theta, r)];
    REAL Gnow = G[I(t, phi, theta, r)];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    MatrixXcd Um1 = (double)cos(anow)*i2x2 - (double)sin(anow)*(t1*(double)n1+t2*(double)n2+t3*(double)n3);
    return Um1;
}

MatrixXcd getF(MatrixXcd L1, MatrixXcd L2){
    return (L1*L2 - L2*L1);
}

REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual){
    MatrixXcd Um1 = getUm1(a, F, G, t, r, theta, phi, M, N, O);
    MatrixXcd L_0 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, tm1, r, theta, phi, M, N, O))/(double)dt); 
    MatrixXcd L_1 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r-1, theta, phi, M, N, O))/(double)(dr)); 
    MatrixXcd L_2 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta-1, phi, M, N, O))/(double)(dtheta));
    MatrixXcd L_3 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta, phi-1, M, N, O))/(double)(dphi)); 
    //REAL K = 4970.25;
    double K = 2.0;
    complex<double> cons = -K/2.0f;
    REAL t00 = ((cons)*(L_0*L_0 - 1.0/2.0*-1.0*(L_0*L_0 + L_1*L_1 +L_2*L_2 +L_3*L_3)//).trace()).real();
                        +((double)lambda)/4.0*(getF(L_0, L_1)*getF(L_0, L_1)
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
    return t00;


}
void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual){
    int count = 0;
    ofstream file;
    //file.open(filename);
    file.open(filename, std::ofstream::app);
    double oo = 0;
    for (size_t phi=1; phi<O; phi=round(oo)){
        double nn = 0;
            for (size_t theta=1; theta<N; theta=round(nn)){
            double mm = 0;
            for (size_t r=1; r<O; r=round(mm)){
                if (file.is_open()){
                    REAL val;
                    if (cual == 0){
                            val = a[I(t, phi, theta, r)];//t00;
                    } else if (cual == 1){
                            val = F[I(t, phi, theta, r)];//t00;
                    } else if (cual == 2){
                            val = G[I(t, phi, theta, r)];//t00;
                    } else if (cual == 3){
                            val = getT00(a, F, G, t, tm1, r, theta, phi, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, cual);
                    }
                    file <<std::fixed << std::setprecision(10) << val << "\n";
                    file.flush();
                } else {
                    std::cerr << "didn't write" << std::endl;
                }
                mm += (double)(M-1)/49.0;
            }
            nn += (double)(N-1)/49.0;
        }
        oo += (double)(O-1)/29.0;
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
