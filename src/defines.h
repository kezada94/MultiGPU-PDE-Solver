#pragma once

#include <boost/multi_array.hpp>

//                   t   r   the phi M  N  O 
#define INDEX4D(arr, d0, d1, d2, d3, M, N, O) arr[d0*M*N*O + d1*N*O + d2*O + d3] 
#define INDEX3D(arr, d1, d2, d3, N, O){ arr[d1*N*O + d2*O + d3] }
#define INDEX2D(arr, d2, d3, O){ arr[d2*O + d3] }
#define INDEX1D(arr, d3){ arr[d3] }

typedef float REAL;

typedef boost::multi_array<REAL, 4> array4D; 
typedef array4D::index index4D;

typedef boost::multi_array<REAL, 3> array3D; 
typedef array3D::index index3D;

typedef boost::multi_array<REAL, 2> array2D; 
typedef array2D::index index2D;

typedef boost::multi_array<REAL, 1> array1D; 
typedef array1D::index index1D;
