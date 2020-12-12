#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

#include "update_equations.h"
#include "Grid.h"
#include "Linspace.h"

using namespace std;
typedef float decimal;


const decimal PI = 3.14159265358979323846f;

//void writeCheckpoint(ofstream &out, auto a, auto F, auto G, size_t t, size_t M, size_t N, size_t O, size_t l);

int main(int argc, char *argv[]){

    if (argc != 6){
        printf("Error. Try executing with\n\t./laplace <L> <M> <N> <O> <time buffer>\n");
        exit(1);
    }

    const int L = 500;// atoi(argv[1]);
    const int M = 500;// atoi(argv[2]);
    const int N = 500;// atoi(argv[3]);
    const int O = 500;// atoi(argv[4]);

    int buffSize = atoi(argv[5]);
    if (buffSize < 3) buffSize = 3;

    vector<linspace_definition> linspaces = vector<linspace_definition>();
    linspaces.push_back(linspace_definition(0, 1, 500)); // for t
    linspaces.push_back(linspace_definition(0, 2*PI, 500)); // for r
    linspaces.push_back(linspace_definition(0, PI, 500)); // for theta
    linspaces.push_back(linspace_definition(0, 2*PI, 500)); // for phi

    Grid *a = new Grid(linspaces);
    Grid *F = new Grid(linspaces);
    Grid *G = new Grid(linspaces);
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
    REAL bigl = 1.f;

    REAL l_1 = 1.f;
    REAL l_2 = 1.f;

    ofstream outfile;
    string filename = "result-"+to_string(L)+"-"+to_string(M)+"-"+to_string(N)+"-"+to_string(O)+".dat";
    outfile.open(filename);

    for (size_t l=0; l<2; ++l){
        for (size_t m=0; m<M; ++m){
            for (size_t n=0; n<N; ++n){
                for (size_t o=0; o<O; ++o){
                    INDEX4D(a->data, l, m, n, o, M, N, O) = a->axes[1][m];
                    INDEX4D(F->data, l, m, n, o, M, N, O) = q*(a->axes[2][n]);
                    INDEX4D(G->data, l, m, n, o, M, N, O) = p*(a->axes[0][l]/bigl - a->axes[3][o]);
                }
            }
        }
    }

    REAL dt = a->deltas[0];
    REAL dr = a->deltas[1];
    REAL dtheta = a->deltas[2];
    REAL dphi = a->deltas[3];

    for (size_t l=2; l<L; ++l){
        for (size_t m=1; m<M; ++m){
            for (size_t n=1; n<N; ++n){
                for (size_t o=1; o<O; ++o){
                    INDEX4D(a->data, l, m, n, o, M, N, O) = Update::computeNexta(a->data, F->data, G->data, l, m, n, o, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                    INDEX4D(F->data, l, m, n, o, M, N, O) = Update::computeNextF(a->data, F->data, G->data, l, m, n, o, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                    INDEX4D(G->data, l, m, n, o, M, N, O) = Update::computeNextG(a->data, F->data, G->data, l, m, n, o, dt, dr, dtheta, dphi, l_1, l_2, bigl);
                }
            }
	}

        //Dirichlet
        for (size_t m=0; m<M; ++m){
        }

        for (size_t n=0; n<N; ++n){
        }

        for (size_t o=0; o<O; ++o){
            INDEX4D(a->data, l, m, n, o, M, N, O) = a->axes[1][m];
            INDEX4D(F->data, l, m, n, o, M, N, O) = q*(a->axes[2][n]);
            INDEX4D(G->data, l, m, n, o, M, N, O) = p*(a->axes[0][l]/bigl - a->axes[3][o]);
        }
	//writeCheckpoint(outfile, a, F, G, l, M, N, O, l);
    }

    //Boundary filll
    for (size_t l=0; l<2; ++l){
        for (size_t m=0; m<M; ++m){
            for (size_t n=0; n<N; ++n){
                for (size_t o=0; o<O; ++o){
                    INDEX4D(a->data, l, m, n, o, M, N, O) = a->axes[1][m];
                    INDEX4D(F->data, l, m, n, o, M, N, O) = q*(a->axes[2][n]);
                    INDEX4D(G->data, l, m, n, o, M, N, O) = p*(a->axes[0][l]/bigl - a->axes[3][o]);
                }
            }
        }
    }
    outfile.close();
    return 0;
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
*/
