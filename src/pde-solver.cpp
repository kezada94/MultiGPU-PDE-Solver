#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <Eigen/Dense>
 
using namespace Eigen;

#include "Equation.h"
#include "Grid.h"
#include "Linspace.h"


using namespace std;

const REAL PI = 3.14159265358979323846f;
const REAL E = 2.71828;
const size_t buffSize = 4;

//void writeCheckpoint(ofstream &out, auto a, auto F, auto G, size_t t, size_t M, size_t N, size_t O, size_t l);
REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl);
Matrix2d getU(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
Matrix2d getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
Matrix2d getL_0(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
Matrix2d getL_1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
Matrix2d getL_2(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);
Matrix2d getL_3(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O);

void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl);


int main(int argc, char *argv[]){

    if (argc != 5){
        printf("Error. Try executing with\n\t./laplace <dt> <M> <N> <O>\n");
        exit(1);
    }

    const size_t L = 999999;
    const size_t M = atoi(argv[2]);
    const size_t N = atoi(argv[3]);
    const size_t O = atoi(argv[4]);
    REAL dt = atof(argv[1]);
    cout << dt << endl;

    

	cout << "Generating lispaces..."; fflush(stdout);
    vector<linspace_definition> linspaces = vector<linspace_definition>();
    linspaces.push_back(linspace_definition(0, 2*PI, M)); // for r
    linspaces.push_back(linspace_definition(0, PI, N)); // for theta
    linspaces.push_back(linspace_definition(0, 2*PI, O)); // for phi
	cout << "done" << endl;

// its better to separate fdm grid with continuous axes
	cout << "Generating grids..."; fflush(stdout);
    Grid *a = new Grid(linspaces, buffSize);
    Grid *F = new Grid(linspaces, buffSize);
    Grid *G = new Grid(linspaces, buffSize);
	cout << "done" << endl;
/*
    vector<decimal> X = vector<decimal>(M);
    vector<decimal> Y = vector<decimal>(N);
    vector<decimal> Z = vector<decimal>(O);
    vector<decimal> T = vector<decimal>(L);

    decimal dt = (1.f - 0)/(L-1);
    decimal dr = (2*PI - 0)/(M-1);
    decimal dtheta = (PI - 0)/(N-1);
    decimal dphi = (2*PI - 0)/(O-1);

    for (size_t x=0; x<M; x++){
	    X[x] = 0 + dr*x;
	    cout << X[x] << endl;
    }
    for (size_t y=0; y<N; y++){
	    Y[y] = 0 + dtheta*y;
    }
    for (size_t z=0; z<O; z++){
	    Z[z] = 0 + dphi*z;
    }
    for (size_t t=0; t<L; t++){
	    T[t] = 0 + dt*t;
    }
*/
    int p = 1;
    int q = 1;
    REAL bigl = 4.0/6.0*(1.0/5.45*129.0)*50.9;
    //REAL bigl = 4.0/6.0*(1.0/E*186.0)*50.9;

    REAL l_1 = 1.f;
    REAL l_2 = 1.f;

    string filename = "result-"+to_string(M)+"-"+to_string(N)+"-"+to_string(O)+".dat";

	cout << "Filling first 3 states..."; fflush(stdout);
    for (size_t l=0; l<3; ++l){
        for (size_t m=0; m<M; ++m){
            for (size_t n=0; n<N; ++n){
                for (size_t o=0; o<O; ++o){
                    a->data[(l)*M*N*O + (m)*N*O + (n)*O + o] = a->axes[0][m];
                    F->data[(l)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                    G->data[(l)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*(REAL)l)/bigl - a->axes[2][o]);
                }
            }
        }
    }
	cout << " done." << endl;

    REAL dr = a->deltas[0];
    REAL dtheta = a->deltas[1];
    REAL dphi = a->deltas[2];
	   
    writeTimeSnapshot(filename, a->data, F->data, G->data, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
    cout << "Written" << endl;
    getchar();

    omp_set_num_threads(4);
    for (size_t l=3; l<L; ++l){
        cout << "Performing iteration "<< l << endl;
        size_t t = l%buffSize;

        #pragma omp parallel for
        for (size_t m=2; m<M-2; ++m){
            cout << "r= "<< m << endl;
            for (size_t n=2; n<N-2; ++n){
                for (size_t o=2; o<O-2; ++o){
                    a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = Equation::computeNexta(a->data, F->data, G->data, t-1, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                    F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = Equation::computeNextF(a->data, F->data, G->data, t-1, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                    G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = Equation::computeNextG(a->data, F->data, G->data, t-1, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                }
            }
        }

        for (size_t m=0; m<M; ++m){
            for (size_t n=0; n<N; ++n){
                for (size_t o=0; o<O; ++o){
                    if (m == 0 || m == M-1 || m == M-2 || m == 1 ){
                            a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = (a->axes[0][m]);
                            F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                            G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - a->axes[2][o]);
                    }
                    if (n == 0 || n == N-1 || n == N-2 || n == 1 ){
                            a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = a->axes[0][m];
                            F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                            G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - a->axes[2][o]);
                    }
                    if (o == 0 || o == O-1 || o == O-2 || o == 1 ){
                            a->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = a->axes[0][m];
                            F->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = q*(a->axes[1][n]);
                            G->data[(t)*M*N*O + (m)*N*O + (n)*O + o] = p*((dt*l)/bigl - a->axes[2][o]);
                    }
                }
            }
        }
        writeTimeSnapshot(filename, a->data, F->data, G->data, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl);
        getchar();
    }

	
    //Boundary filll
    return 0;
}

static Matrix2d i2x2 = [] {
    Matrix2d matrix;
    matrix << 1, 0, 0, 1;
    return matrix;
}();
static Matrix2d t1 = [] {
    Matrix2d matrix;
    matrix << 0, 1, 1, 0;
    return matrix;
}();
static Matrix2d t2 = [] {
    Matrix2d matrix;
    matrix << 0, -1, 1, 0;
    return matrix;
}();
static Matrix2d t3 = [] {
    Matrix2d matrix;
    matrix << 0, 1, -1, 0;
    return matrix;
}();
Matrix2d getU(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O){
    REAL anow = a[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Fnow = F[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Gnow = G[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    Matrix2d U = cos(anow)*i2x2 + sin(anow)*(t1*n1+t2*n2+t3*n3);
    return U;
}
Matrix2d getUm1(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O){
    REAL anow = a[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Fnow = F[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL Gnow = G[(t)*M*N*O + (r)*N*O + (theta)*O + phi];
    REAL n1 = sin(Fnow)*cos(Gnow);
    REAL n2 = sin(Fnow)*sin(Gnow);
    REAL n3 = cos(Fnow);
    Matrix2d Um1 = cos(anow)*i2x2 - sin(anow)*(t1*n1+t2*n2+t3*n3);
    return Um1;
}

REAL getT00(REAL* a, REAL* F, REAL *G, size_t t, size_t r, size_t theta, size_t phi, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl){
    Matrix2d Um1 = getUm1(a, F, G, t, r, theta, phi, M, N, O);
    Matrix2d L_0 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t-1, r, theta, phi, M, N, O))/dt); 
    Matrix2d L_1 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r-1, theta, phi, M, N, O))/dr); 
    Matrix2d L_2 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta-1, phi, M, N, O))/dtheta);
    Matrix2d L_3 = Um1*((getU(a, F, G, t, r, theta, phi, M, N, O) - getU(a, F, G, t, r, theta, phi-1, M, N, O))/dphi); 

    REAL K = 13.5;
    REAL t00 = -K/2.0f*(L_0*L_0 - 1.0/2.0*-1.0*(L_0*L_0 + L_1*L_1 + L_2*L_2 + L_3*L_3)
                        + bigl/4.0*((-1.0*(L_0*L_0 - L_0*L_0)*(L_0*L_0 - L_0*L_0)
                                    +l_1*(L_0*L_1 - L_1*L_0)*(L_0*L_1 - L_1*L_0)
                                    +l_1*(L_0*L_2 - L_2*L_0)*(L_0*L_2 - L_2*L_0)
                                    +l_2*(L_0*L_3 - L_3*L_0)*(L_0*L_3 - L_3*L_0)) )).trace();
                            //+ -1.0/4.0*((L_0*L_0 - L_0*L_0)*(L_0*L_0 - L_0*L_0)
                             //           +(L_1*L_1 - L_1*L_1)*(L_1*L_1 - L_1*L_1)
                              //          +(L_2*L_2 - L_2*L_2)*(L_2*L_2 - L_2*L_2)
                               //         +(L_3*L_3 - L_3*L_3)*(L_3*L_3 - L_3*L_3)) )).trace();
    return t00;


}
void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL bigl){
    int count = 0;
    ofstream file;
    file.open(filename);
    for (size_t m=1; m<M; m++){
        for (size_t n=1; n<N; n++){
            for (size_t o=1; o<O; o++){
                if (file.is_open()){
                    //file << G[(t)*M*N*O + (m)*N*O + (n)*O + o] << "\n";
                    file << getT00(a, F, G, t, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl) << "\n";
                    file.flush();
                }
                else{
                    std::cerr << "didn't write" << std::endl;
                }
                //fprintf(file, "%f\n", getT00(a, F, G, t, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, bigl));
            }
        }
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
