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

#include "kernels.cuh"
#include "EquationAlfa.cuh"
#include "EquationF.cuh"
#include "EquationG.cuh"

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
void printTime(int tid, float *times, int n, char* msg);
void printBW(int tid, float *times, int n, size_t *byteCount, char* msg);

int main(int argc, char *argv[]){

    if (argc != 11){
        printf("Error. Try executing with\n\t./laplace <dt> <M> <N> <O> <p> <q> <n> <nGPU> <niter> <boundary>\n");
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
    int nGPU = atoi(argv[8]);
    bool boundary = atoi(argv[10]);
    size_t niter = atoi(argv[9]);


    if (nGPU < 0){
        cout << "Se necesitan al menos 3 gpus de 40gb para un problema de 1000x1000x1000." << endl;
        exit(-1);
    }

    float elapsed=0;
    cudaEvent_t start, stop;

    cucheck(cudaEventCreate(&start));
    cucheck(cudaEventCreate(&stop));

    size_t slicesStartIndex[nGPU+1];
    for (int i=0; i<nGPU; i++){
    	slicesStartIndex[i] = round((float)O/nGPU*i); 
		cout << "Corte " << i << ":  "<< slicesStartIndex[i] << endl;
    }
    slicesStartIndex[nGPU] = O; 
    cout << "Corte " << nGPU << ":  "<< slicesStartIndex[nGPU] << endl;

    REAL dr = 2*PI/(double)(M-1);
    REAL dtheta = PI/(double)(N-1);
    REAL dphi = 2*PI/(double)(O-1);

    // +2 for ghost points offset
	size_t nelements = buffSize*(M+GHOST_SIZE)*(N+GHOST_SIZE)*(O+GHOST_SIZE);
	cout << "Number of elements: " << nelements << endl;

    // its better to separate fdm grid with continuous axes
    cout << "Generating grids... "; fflush(stdout);
    cucheck( cudaEventRecord(start, 0));

    REAL *a;// = new REAL[nelements];
    REAL *F;// = new REAL[nelements];
    REAL *G;// = new REAL[nelements];

    cudaError_t status = cudaMallocHost((void**)&a, nelements*sizeof(REAL));
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    status = cudaMallocHost((void**)&F, nelements*sizeof(REAL));
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");
    
    status = cudaMallocHost((void**)&G, nelements*sizeof(REAL));
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    cucheck(cudaEventRecord(stop, 0));
    cucheck(cudaEventSynchronize (stop) );
    cucheck(cudaEventElapsedTime(&elapsed, start, stop) );

    cucheck(cudaEventDestroy(start));
    cucheck(cudaEventDestroy(stop));

    printf("Done. Allocating all arrays: %.3f GiB took %.3f ms\n", 3*nelements*sizeof(REAL)/(double)(1000*1000*1000), elapsed);


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
        cout << "No se puede abrir el archivo: " << "../alfa(r)-"+to_string(n)+"-"+to_string(q)+"-"+to_string(M)+".csv" << endl;
        exit(-190);
    }
    cout << "done. " << i << " elements read" << endl;;
    REAL lambda = 1.0; //4.0/6.0*(1.0/5.45*129.0)*50.9;

    REAL l_1 = 1.f;
    REAL l_2 = 1.f;

    string filename = "result-"+to_string(M)+".dat";
    string filename0 = "result-"+to_string(M)+"-A.dat";
    string filename1 = "result-"+to_string(M)+"-F.dat";
    string filename2 = "result-"+to_string(M)+"-G.dat";


    omp_set_num_threads(nGPU);
    float *tiempos = new float[nGPU];
    size_t *bytes = new size_t[nGPU];
    REAL** slices_ptra = new REAL*[nGPU];
    REAL** slices_ptrF = new REAL*[nGPU];
    REAL** slices_ptrG = new REAL*[nGPU];
    size_t* slices_widths = new size_t[nGPU];

    size_t sharedMemorySizeb = (buffSize)*nfunctions*(BSIZEX+2)*(BSIZEY+2)*(BSIZEZ+2)*sizeof(REAL);
    int carveout = 50;
    cudaFuncSetAttribute(computeFirsta, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(computeFirstF, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(computeFirstG, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

    cudaFuncSetAttribute(computeSeconda, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(computeSecondF, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(computeSecondG, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

    cudaFuncSetAttribute(computeNexta, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(computeNextF, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(computeNextG, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

    #pragma omp parallel shared(tiempos, bytes, slices_ptra, slices_ptrF, slices_ptrG, slices_widths)
    {
		int tid = omp_get_thread_num();
        cucheck(cudaSetDevice(tid));

        if (nGPU != 1){
            if (tid==0){
                cudaDeviceEnablePeerAccess(tid+1, 0);
            } else if (tid == nGPU-1){
                cudaDeviceEnablePeerAccess(tid-1, 0);
            } else{
                cudaDeviceEnablePeerAccess(tid-1, 0);
                cudaDeviceEnablePeerAccess(tid+1, 0);
            }
        }
        cudaEvent_t inicio, fin;

        cucheck(cudaEventCreate(&inicio));
        cucheck(cudaEventCreate(&fin));

		size_t GPUWidth = slicesStartIndex[tid+1] - slicesStartIndex[tid];

        slices_widths[tid] = GPUWidth;

        #pragma omp critical
        {
		    cout << "GPU " << tid << " works " << GPUWidth << " elements. Starting at " << slicesStartIndex[tid] <<  endl;
        }
		dim3 g, b;
		b = dim3(BSIZEX, BSIZEY, BSIZEZ);
		g = dim3((M+b.x-1)/(b.x), (N+b.y-1)/b.y, (GPUWidth+b.z-1)/(b.z));
    
        REAL *da_0;
        cucheck(cudaMalloc(&da_0, M*sizeof(REAL)));
        cucheck(cudaMemcpy(da_0, a_0, M*sizeof(REAL), cudaMemcpyHostToDevice));

        size_t nelements_slice = (M+GHOST_SIZE)*(N+GHOST_SIZE)*(GPUWidth+GHOST_SIZE)*buffSize;
        REAL* a_slice, *F_slice, *G_slice;

        #pragma omp critical
        {
            printf("Allocating volume slices in GPU %i: %.3f GB\n", tid, nelements_slice*sizeof(REAL)/(double)(1000*1000*1000));
        }
        cucheck( cudaEventRecord(inicio, 0));
        cucheck(cudaMalloc(&a_slice, nelements_slice*sizeof(REAL)));
        cucheck(cudaMalloc(&F_slice, nelements_slice*sizeof(REAL)));
        cucheck(cudaMalloc(&G_slice, nelements_slice*sizeof(REAL)));

        cucheck( cudaEventRecord(fin, 0));
        cucheck( cudaEventSynchronize(fin));
        cucheck( cudaEventElapsedTime(&tiempos[tid], inicio, fin) );
       
        #pragma omp barrier
        printTime(tid, tiempos, nGPU, "GPU allocation time: ");
        #pragma omp barrier

        slices_ptra[tid] = a_slice;
        slices_ptrF[tid] = F_slice;
        slices_ptrG[tid] = G_slice;
        

        // ESTADO 0, 1

        for (int time=0; time<2; time++){
            #pragma omp critical
            {
                printf("GPU %i - Iteration %i: filling initial condition\n", tid, time);
            }

            cucheck( cudaEventRecord(inicio, 0));
            fillInitialCondition<<<g, b>>>(a_slice, F_slice, G_slice, time, M, N, GPUWidth, slicesStartIndex[tid], O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, da_0);
            cucheck(cudaDeviceSynchronize());
            checkError();

            cucheck( cudaEventRecord(fin, 0));
            cucheck( cudaEventSynchronize(fin));
            cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );

            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier


            // CONDICION INICIAL
            #pragma omp critical
            {
                printf("GPU %i - Iteration %i: filling boundary\n", tid, time);
            }

            cucheck( cudaEventRecord(inicio, 0));
            fillDirichletBoundary<<<g, b>>>(a_slice, F_slice, G_slice, time, time, M, N, GPUWidth, slicesStartIndex[tid], O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, da_0);
            cucheck(cudaDeviceSynchronize());
            checkError();

            cucheck( cudaEventRecord(fin, 0));
            cucheck( cudaEventSynchronize(fin));
            cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );

            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier

            // copy por nvlink
            if (tid < nGPU-1 ){
                // Copiar seccion 0 de tid+1 al halo inferior de gpu tid
                #pragma omp critical
                {
                    printf("GPU %i - Iteration %i: sharing halo with neighboor gpus\n", tid, time);
                }//           
                size_t to    = time*(slices_widths[tid  ]+2)*(M+2)*(N+2) + (M+2)*(N+2)*(slices_widths[tid]+1);
                size_t from  = time*(slices_widths[tid+1]+2)*(M+2)*(N+2) + (M+2)*(N+2)*(1);
                size_t bytesCount = (M+2)*(N+2)*sizeof(REAL);
                
                cucheck( cudaEventRecord(inicio, 0));
                cucheck( cudaMemcpyPeer(slices_ptra[tid] + to, tid, slices_ptra[tid+1] + from, tid+1, bytesCount));
                cucheck( cudaEventRecord(fin, 0));
                cucheck( cudaEventSynchronize(fin));
                cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );
                bytes[tid] = bytesCount;


                cucheck( cudaMemcpyPeer(slices_ptrF[tid] + to, tid, slices_ptrF[tid+1] + from, tid+1, bytesCount));
                cucheck( cudaMemcpyPeer(slices_ptrG[tid] + to, tid, slices_ptrG[tid+1] + from, tid+1, bytesCount));


                // Copiar seccion GPUWidth de tid al halo superior de tid+1
                to      = time*(slices_widths[tid+1]+2)*(M+2)*(N+2) + (0)*(M+2)*(N+2);
                from    = time*(slices_widths[tid  ]+2)*(M+2)*(N+2) + (slices_widths[tid])*(M+2)*(N+2);
                cucheck(cudaMemcpyPeer(slices_ptra[tid+1] + to, tid+1, slices_ptra[tid] + from, tid, bytesCount));
                cucheck(cudaMemcpyPeer(slices_ptrF[tid+1] + to, tid+1, slices_ptrF[tid] + from, tid, bytesCount));
                cucheck(cudaMemcpyPeer(slices_ptrG[tid+1] + to, tid+1, slices_ptrG[tid] + from, tid, bytesCount));
            }
            #pragma omp barrier

            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier
            printBW(tid, tiempos, nGPU, bytes, "BW: ");
            #pragma omp barrier

            // Se copia el bloque de a+2 * theta+2 * phi (alfa y theta con halo, pero phi sin halo) esto debido a que los halos de phi se pueden sobreescribir entre tortas.
            #pragma omp critical
            {
                printf("GPU %i - Iteration %i: copying data back to host.\n", tid, time);
            }
            cucheck( cudaEventRecord(inicio, 0));
            cucheck(cudaMemcpy(a + I(time, slicesStartIndex[tid], -1, -1), a_slice + (GPUWidth+2)*time*(M+2)*(N+2) + (1)*(M+2)*(N+2), (GPUWidth)*(M+2)*(N+2)*sizeof(REAL), cudaMemcpyDeviceToHost));
            cucheck(cudaMemcpy(F + I(time, slicesStartIndex[tid], -1, -1), F_slice + (GPUWidth+2)*time*(M+2)*(N+2) + (1)*(M+2)*(N+2), (GPUWidth)*(M+2)*(N+2)*sizeof(REAL), cudaMemcpyDeviceToHost));
            cucheck(cudaMemcpy(G + I(time, slicesStartIndex[tid], -1, -1), G_slice + (GPUWidth+2)*time*(M+2)*(N+2) + (1)*(M+2)*(N+2), (GPUWidth)*(M+2)*(N+2)*sizeof(REAL), cudaMemcpyDeviceToHost));    
            cucheck(cudaDeviceSynchronize());
            checkError();

            cucheck( cudaEventRecord(fin, 0));
            cucheck( cudaEventSynchronize(fin));
            cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );
            bytes[tid] = (GPUWidth)*(M+2)*(N+2)*sizeof(REAL)*3;

            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier
            printBW(tid, tiempos, nGPU, bytes, "BW: ");
            #pragma omp barrier

            cucheck(cudaDeviceSynchronize());
            checkError();


            #pragma omp barrier
            if (tid == 0){
                writeTimeSnapshot(filename0, a, F, G, time, time-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
                //writeTimeSnapshot(filename1, a, F, G, time, time-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
                //writeTimeSnapshot(filename2, a, F, G, time, time-1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
                cout << "Written" << endl;
            }
            #pragma omp barrier
        }


		#pragma omp barrier
        for (size_t l=2; l<niter; ++l){
            size_t tp1 = l%buffSize;
            size_t t = (l-1)%buffSize;
            size_t tm1 = (l-2)%buffSize;
            size_t tm2 = (l-3)%buffSize;


            #pragma omp critical
            {
                printf("GPU %i - Iteration %i: computing state\n", tid, l);
            }
            cucheck( cudaEventRecord(inicio, 0));
            if (l == 2) {
                computeSecondIteration(a_slice, F_slice, G_slice, l, tp1, t, tm1, tm2, M, N, GPUWidth, slicesStartIndex[tid], O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, da_0, b, g, sharedMemorySizeb);
            } else {
                computeNextIteration(a_slice, F_slice, G_slice, l, tp1, t, tm1, tm2, M, N, GPUWidth, slicesStartIndex[tid], O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, da_0, b, g, sharedMemorySizeb);
            }
            cucheck( cudaEventRecord(fin, 0));
            cucheck( cudaEventSynchronize(fin));
            cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );

            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier


            #pragma omp critical
            {
                printf("GPU %i - Iteration %i: filling boundary\n", tid, l);
            }

            cucheck( cudaEventRecord(inicio, 0));
            fillDirichletBoundary<<<g, b>>>(a_slice, F_slice, G_slice, l, tp1, M, N, GPUWidth, slicesStartIndex[tid], O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, da_0);
            cucheck(cudaDeviceSynchronize());
            checkError();
            cucheck( cudaEventRecord(fin, 0));
            cucheck( cudaEventSynchronize(fin));
            cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );

            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier

            // copy por nvlink
            if (tid < nGPU-1 ){
                // Copiar seccion 0 de tid+1 al halo inferior de gpu tid
                int time = tp1;
                // Copiar seccion 0 de tid+1 al halo inferior de gpu tid
                #pragma omp critical
                {
                    printf("GPU %i - Iteration %i: sharing halo with neighboor gpus\n", tid, l);
                }//           
                size_t to    = time*(slices_widths[tid  ]+2)*(M+2)*(N+2) + (M+2)*(N+2)*(slices_widths[tid]+1);
                size_t from  = time*(slices_widths[tid+1]+2)*(M+2)*(N+2) + (M+2)*(N+2)*(1);
                size_t bytesCount = (M+2)*(N+2)*sizeof(REAL);
                
                cucheck( cudaEventRecord(inicio, 0));
                cucheck( cudaMemcpyPeer(slices_ptra[tid] + to, tid, slices_ptra[tid+1] + from, tid+1, bytesCount));
                cucheck( cudaEventRecord(fin, 0));
                cucheck( cudaEventSynchronize(fin));
                cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );
                bytes[tid] = bytesCount;


                cucheck( cudaMemcpyPeer(slices_ptrF[tid] + to, tid, slices_ptrF[tid+1] + from, tid+1, bytesCount));
                cucheck( cudaMemcpyPeer(slices_ptrG[tid] + to, tid, slices_ptrG[tid+1] + from, tid+1, bytesCount));


                // Copiar seccion GPUWidth de tid al halo superior de tid+1
                to      = time*(slices_widths[tid+1]+2)*(M+2)*(N+2) + (0)*(M+2)*(N+2);
                from    = time*(slices_widths[tid  ]+2)*(M+2)*(N+2) + (slices_widths[tid])*(M+2)*(N+2);
                cucheck(cudaMemcpyPeer(slices_ptra[tid+1] + to, tid+1, slices_ptra[tid] + from, tid, bytesCount));
                cucheck(cudaMemcpyPeer(slices_ptrF[tid+1] + to, tid+1, slices_ptrF[tid] + from, tid, bytesCount));
                cucheck(cudaMemcpyPeer(slices_ptrG[tid+1] + to, tid+1, slices_ptrG[tid] + from, tid, bytesCount));
            }
            #pragma omp barrier


            #pragma omp barrier
            printTime(tid, tiempos, nGPU, "Took: ");
            #pragma omp barrier
            printBW(tid, tiempos, nGPU, bytes, "BW: ");
            #pragma omp barrier

            if (l%10 == 0 || true){
                int time = tp1;
                // Se copia el bloque de a+2 * theta+2 * phi (alfa y theta con halo, pero phi sin halo) esto debido a que los halos de phi se pueden sobreescribir entre tortas.
                #pragma omp critical
                {
                    printf("GPU %i - Iteration %i: copying data back to host.\n", tid, l);
                }

                cucheck( cudaEventRecord(inicio, 0));
                cucheck(cudaMemcpy(a + I(time, slicesStartIndex[tid], -1, -1), a_slice + (GPUWidth+2)*time*(M+2)*(N+2) + (1)*(M+2)*(N+2), (GPUWidth)*(M+2)*(N+2)*sizeof(REAL), cudaMemcpyDeviceToHost));
                cucheck(cudaMemcpy(F + I(time, slicesStartIndex[tid], -1, -1), F_slice + (GPUWidth+2)*time*(M+2)*(N+2) + (1)*(M+2)*(N+2), (GPUWidth)*(M+2)*(N+2)*sizeof(REAL), cudaMemcpyDeviceToHost));
                cucheck(cudaMemcpy(G + I(time, slicesStartIndex[tid], -1, -1), G_slice + (GPUWidth+2)*time*(M+2)*(N+2) + (1)*(M+2)*(N+2), (GPUWidth)*(M+2)*(N+2)*sizeof(REAL), cudaMemcpyDeviceToHost));    
                cucheck(cudaDeviceSynchronize());
                checkError();
                cucheck( cudaEventRecord(fin, 0));
                cucheck( cudaEventSynchronize(fin));
                cucheck(cudaEventElapsedTime(&tiempos[tid], inicio, fin) );
                bytes[tid] = (GPUWidth)*(M+2)*(N+2)*sizeof(REAL)*3;

                #pragma omp barrier
                printTime(tid, tiempos, nGPU, "Done.\nTook: ");
                #pragma omp barrier
                printBW(tid, tiempos, nGPU, bytes, "BW: ");

                cucheck(cudaDeviceSynchronize());
                checkError();

		        #pragma omp barrier
                if (tid ==0){
                    cout << "Saving values..." << endl;
                    writeTimeSnapshot(filename0, a, F, G, tp1, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
                    //writeTimeSnapshot(filename1, a, F, G, tp1, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
                    //writeTimeSnapshot(filename2, a, F, G, tp1, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
                    cout << "done." << endl;
                }
		        #pragma omp barrier
            }
		    #pragma omp barrier
        }
    }
    
    /*if (boundary == 0){
	    fillDirichletBoundary<<<g, b>>>(a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints<<<g, b>>>(a, F, G, 0, M, N, O);
    } */
    cucheck(cudaDeviceSynchronize());
    checkError();
    cout << " done." << endl;
    //writeTimeSnapshot(filename0, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    //writeTimeSnapshot(filename1, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    //writeTimeSnapshot(filename2, a, F, G, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    exit(4);


/*


    cout << "Filling state 1..."; fflush(stdout);
    fillInitialCondition<<<g, b>>>(a, F, G, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    cucheck(cudaDeviceSynchronize());
    checkError();
    //computeFirstIteration(a, F, G, 1, 1, 0, -1, -2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    if (boundary == 0){
	    fillDirichletBoundary<<<g, b>>>(a, F, G, 1, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints<<<g, b>>>(a, F, G, 1, M, N, O);
    } 
    cucheck(cudaDeviceSynchronize());
    checkError();
    cout << " done." << endl;

    //writeTimeSnapshot(filename0, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    //writeTimeSnapshot(filename1, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    //writeTimeSnapshot(filename2, a, F, G, 1, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    getchar();

    cout << "Filling state 2..."; fflush(stdout);
    computeSecondIteration(a, F, G, 2, 2, 1, 0, 0, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    cucheck(cudaDeviceSynchronize());
    checkError();
    if (boundary == 0){
	    fillDirichletBoundary<<<g, b>>>(a, F, G, 2, 2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    } else if (boundary == 1){
	    fillGhostPoints<<<g, b>>>(a, F, G, 2, M, N, O);
    } 
    cucheck(cudaDeviceSynchronize());
    checkError();


    writeTimeSnapshot(filename0, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
    //writeTimeSnapshot(filename1, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
    //writeTimeSnapshot(filename2, a, F, G, 2, 1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
    cout << "Written" << endl;
    getchar();


    for (size_t l=3; l<niter; ++l){
	cout << "Starting iteration l=" << l << endl;
	size_t tp1 = l%buffSize;
	size_t t = (l-1)%buffSize;
	size_t tm1 = (l-2)%buffSize;
	size_t tm2 = (l-3)%buffSize;

	cout << tp1 << " " << t << " " << tm1 << " " << tm2 << " "  << endl;

	computeNextIteration(a, F, G, l, tp1, t, tm1, tm2, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
    	cucheck(cudaDeviceSynchronize());
    	checkError();
	if (boundary == 0){
		fillDirichletBoundary<<<g, b>>>(a, F, G, l, tp1, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, p, q, 1, a_0);
	} else if (boundary == 1){
	    fillGhostPoints<<<g, b>>>(a, F, G, tp1, M, N, O);
	} 
    	cucheck(cudaDeviceSynchronize());
    	checkError();


	cout << "Finished iteration l=" << l << endl;

	//cout << "Save? [y/n]" << endl;
	//char key = getchar();
	if (l%10==0){
	    cout << "Saving values..." << endl;
	    writeTimeSnapshot(filename0, a, F, G, tp1, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 0);
	    //writeTimeSnapshot(filename1, a, F, G, tp1, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 1);
	    //writeTimeSnapshot(filename2, a, F, G, tp1, t, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, 2);
	    cout << "done." << endl;
	    //getchar();
	}
}
	
    //Boundary filll*/
    return 0;
}

void printTime(int tid, float *times, int n, char* msg){
    if (tid == 0){
        printf("%s ", msg);
        for(int i=0; i<n; i++){
            printf("(%.2f ms) ", times[i]);   
        }
        printf("\n");
    }
}

void printBW(int tid, float *times, int n, size_t *byteCount, char* msg){
    if (tid == 0){
        printf("%s ", msg);
        for(int i=0; i<n; i++){
            printf("(%.3f GiB/s) ", byteCount[i]*1000./(double)times[i]/(double)(1024*1024*1024));   
        }
        printf("\n");
    }

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
	if (cual == 0){
            return a[I(t, phi, theta, r)];//t00;
	} else if (cual == 1){
	    	return F[I(t, phi, theta, r)];//t00;
	} else if (cual == 2){
    	    return G[I(t, phi, theta, r)];//t00;
	}
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
    return t00;


}
void writeTimeSnapshot(string filename, REAL* a, REAL* F, REAL *G, size_t t, size_t tm1, size_t M, size_t N, size_t O, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL lambda, int cual){
    int count = 0;
    ofstream file;
    //file.open(filename);
    file.open(filename, std::ofstream::app);
	if (!file.is_open()){
		std::cerr << "didn't write" << std::endl;
	}
    double oo = 0;
    for (size_t o=0; o<O; o=round(oo)){
    	cout << o << endl;
        double nn = 0;
            for (size_t n=0; n<N; n=round(nn)){
            double mm = 0;
            for (size_t m=0; m<M; m=round(mm)){
				file <<std::fixed << std::setprecision(32) << getT00(a, F, G, t, tm1, m, n, o, M, N, O, dt, dr, dtheta, dphi, l_1, l_2, lambda, cual) << "\n";
				file.flush();
                mm += (double)(M-1)/99.0;
            }
            nn += (double)(N-1)/99.0;
        }
        oo += (double)(O-1)/2.0;
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
