#pragma once

#define INDEX4D(arr, d0, d1, d2, d3, M, N, O) arr[d0*M*N*O + d1*N*O + d2*O + d3] 
#define INDEX3D(arr, d1, d2, d3, N, O){ arr[d1*N*O + d2*O + d3] }
#define INDEX2D(arr, d2, d3, O){ arr[d2*O + d3] }
#define INDEX1D(arr, d3){ arr[d3] }
typedef float REAL;
