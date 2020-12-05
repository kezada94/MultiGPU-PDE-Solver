#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

typedef float decimal;

void writeCheckpoint(ofstream &out, decimal* Z);

int main(int argc, char *argv[]){   

    if (argc != 5){
        printf("Error. Try executing with\n\t./laplace <L> <M> <N> <O>");
        exit(1);
    }

    int L = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int O = atoi(argv[4]);
    
    ofstream outfile;
    string filename = "result-"+to_string(L)+"-"+to_string(M)+"-"+to_string(N)+"-"+to_string(O)+".dat";
    outfile.open(filename);

    decimal *h_Z = (decimal*)malloc(sizeof(decimal)*M*N*O);


    for (int l=0; l<L; l++){
        writeCheckpoint(outfile, h_Z);
    }
    outfile.close();
    return 0;
}


void writeCheckpoint(ofstream &out, decimal* Z){
}
