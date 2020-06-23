#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cuda.h>

#define BSX 32
#define BSY 32
#define BS1D 1024

using namespace std;

typedef float decimal;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void laplaceSolver(decimal* d_Z1, decimal* d_Z2, int M, int N);
void writeCheckpoint(ofstream &out, decimal* Z, int w, int h);

int main(int argc, char *argv[]){   

    if (argc != 6){
        printf("Error. Try executing with\n\t./laplace <N> <M> <WIDTH> <HEIGHT> <L>");
        exit(1);
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int L = atoi(argv[5]);
    float WIDTH = atof(argv[3]);
    float HEIGHT= atof(argv[4]);
    
    ofstream outfile;
    string filename = "result-"+to_string(WIDTH)+"-"+to_string(HEIGHT)+"-"+to_string(M)+"-"+to_string(N)+"-"+to_string(L)+".dat";
    outfile.open(filename);

    decimal *h_Z = (decimal*)malloc(sizeof(decimal)*M*N);
    decimal *d_Z1, *d_Z2;

    gpuErrChk(cudaMalloc(&d_Z1, sizeof(decimal)*M*N));
    gpuErrChk(cudaMalloc(&d_Z2, sizeof(decimal)*M*N));

    for (int i=0; i<N; i++){
        for (int j=0; j<M; j++){
            if (i==N-1)
                h_Z[i*M+j] = sin((WIDTH/(M-1)*j));
            else
                h_Z[i*M+j] = 0.0;
        }
    }
    gpuErrChk(cudaMemcpy(d_Z1, h_Z, sizeof(decimal)*M*N, cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_Z2, h_Z, sizeof(decimal)*M*N, cudaMemcpyHostToDevice));

    dim3 bs(BSX, BSY);
    dim3 gs((M+BSX-1)/BSX, (N+BSY-1)/BSY);
    for (int l=0; l<L; l++){
        laplaceSolver<<<gs, bs>>>(d_Z1, d_Z2, M, N);
        gpuErrChk(cudaDeviceSynchronize());
        gpuErrChk(cudaMemcpy(h_Z, d_Z2, sizeof(decimal)*M*N, cudaMemcpyDeviceToHost));

        writeCheckpoint(outfile, h_Z, M, N);

        swap(d_Z1, d_Z2);
    }
    outfile.close();
    return 0;
}


__global__ void laplaceSolver(decimal* d_Z1, decimal* d_Z2, int M, int N){

    int bx0 = blockIdx.x*BSX;
    int by0 = blockIdx.y*BSY;


    int x = bx0+threadIdx.x;
    int y = by0+threadIdx.y;


    int gcol = (y*M+x)%M;
    int grow = (y*M+x)/M;
    
    if (gcol == 0 || gcol == M-1 || grow == 0 || grow == N-1){
        d_Z2[y*M+x] = d_Z1[y*M+x];
    } else {
        d_Z2[y*M+x] = (d_Z1[x+(y+1)*M] + 
                       d_Z1[x+(y-1)*M] + 
                       d_Z1[y*M + x+1] + 
                       d_Z1[y*M + x-1])/4;
    }
    

}
__global__ void laplaceSolvershmem(decimal* d_Z1, decimal* d_Z2, int M, int N){

    __shared__ decimal shmem[(BSY+2)][(BSX+2)];
    __shared__ decimal shmem_res[(BSY)][(BSX)];

    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    int wid = tid/32;

    int c0 = blockIdx.y*M*BSY+blockIdx.x*BSX;
    shmem[threadIdx.y+1][threadIdx.x+1] = d_Z1[c0+(threadIdx.y+1)*M
        + threadIdx.x+1];

    if (threadIdx.x == 0){
        shmem[threadIdx.y+1][0] = d_Z1[c0+(threadIdx.y)*M + threadIdx.x];
    } else if (threadIdx.x == BSX-1){
        shmem[threadIdx.y+1][BSX+2-1] = d_Z1[c0+(threadIdx.y)*M + threadIdx.x];
    } else if (threadIdx.y == 0){
        shmem[0][threadIdx.x+1] = d_Z1[c0+(threadIdx.y)*M + threadIdx.x];
    } else if (threadIdx.y == BSY-1){
        shmem[BSY+2-1][threadIdx.x+1] = d_Z1[c0+(threadIdx.y)*M + threadIdx.x];
    }
    __syncthreads();

    shmem_res[threadIdx.y][threadIdx.x] = (shmem[threadIdx.y+1-1][threadIdx.x+1]
        + shmem[threadIdx.y+1+1][threadIdx.x+1]
        +shmem[threadIdx.y+1][threadIdx.x+1-1]
        +shmem[threadIdx.y+1][threadIdx.x+1+1])/4;

    __syncthreads();

    d_Z2[c0+(threadIdx.y+1)*M+ threadIdx.x+1] = shmem_res[threadIdx.y][threadIdx.x];
}



void writeCheckpoint(ofstream &out, decimal* Z, int w, int h){
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            if (j != w-1){
                out << Z[i*w+j] << ",";
            } else {
                out << Z[i*w+j] << "\n";
            }
        }
    }
}
