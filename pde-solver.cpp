#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

typedef float decimal;

const decimal M_PI = 3.14159265358979323846f;

void writeCheckpoint(ofstream &out, decimal* Z);

inline decimal computeNexta(decimal* a, decimal* F, decimal* G, size_t t, size_t r, size_t theta, size_t phi);
inline decimal computeNextF(decimal* a, decimal* F, decimal* G, size_t t, size_t r, size_t theta, size_t phi);
inline decimal computeNextG(decimal* a, decimal* F, decimal* G, size_t t, size_t r, size_t theta, size_t phi);

int main(int argc, char *argv[]){   

    if (argc != 6){
        printf("Error. Try executing with\n\t./laplace <L> <M> <N> <O> <time buffer>\n");
        exit(1);
    }

    int L = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int O = atoi(argv[4]);
    int buffSize = atoi(argv[5]);
	if (buffSize < 3) buffSize = 3;

	vector<float> X = vector<float>(M);
	vector<float> Y = vector<float>(N);
	vector<float> Z = vector<float>(O);
	vector<float> T = vector<float>(L);

	for (size_t x=0; x<M; x++){
		X[x] = 0 + (2*M_PI - 0)/(M-1)*x;
		cout << X[x] << endl;
	}
	for (size_t y=0; y<N; y++){
		Y[y] = 0 + (M_PI - 0)/(N-1)*y;
	}
	for (size_t z=0; z<O; z++){
		Z[z] = 0 + (2*M_PI - 0)/(O-1)*z;
	}
	for (size_t t=0; t<L; t++){
		T[t] = 0 + (1.f - 0)/(L-1)*t;
	}

	int p = 1;
	int q = 1;
	float bigl = 1.f;

	float l_1 = 1.f;
	float l_2 = 1.f;

    ofstream outfile;
    string filename = "result-"+to_string(L)+"-"+to_string(M)+"-"+to_string(N)+"-"+to_string(O)+".dat";
    outfile.open(filename);

    decimal *a = (decimal*)malloc(sizeof(decimal)*M*N*O*buffSize);
    decimal *F = (decimal*)malloc(sizeof(decimal)*M*N*O*buffSize);
    decimal *G = (decimal*)malloc(sizeof(decimal)*M*N*O*buffSize);

	for (size_t l=0; l<2; ++l){
		for (size_t m=0; m<M; ++m){
			for (size_t n=0; n<N; ++n){
				for (size_t o=0; o<O; ++o){
					a[l*M*N*O + m*N*O + n*O + o] = X[m];
					F[l*M*N*O + m*N*O + n*O + o] = q*Y[n];
					G[l*M*N*O + m*N*O + n*O + o] = p*(T[l]/bigl - Z[o]);
				}
			}
		}
	}

    for (int l=0; l<L; l++){
    }
    outfile.close();
    return 0;
}


void writeCheckpoint(ofstream &out, decimal* Z){

}


inline decimal computeNexta(decimal* a, decimal* F, decimal* G, size_t t, size_t r, size_t theta, size_t phi){
	return ---a---;
}

inline decimal computeNextF(decimal* a, decimal* F, decimal* G, size_t t, size_t r, size_t theta, size_t phi){
	return ---F---;
}

inline decimal computeNextG(decimal* a, decimal* F, decimal* G, size_t t, size_t r, size_t theta, size_t phi){
	return ---G---;
}
