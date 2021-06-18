#pragma once
#include <iostream>

/** CUDA check macro */
#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	exit(22);\
	}\
	}

#define cucheck_dev(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	assert(0);																												\
	}\
	}
#define checkError() \
   {\
   cudaError_t err = cudaGetLastError();        \
   if ( err != cudaSuccess )\
   {\
      printf("CUDA Error: %s\n", cudaGetErrorString(err));\
      exit(-1);\
   }\
   }

#define GHOST_SIZE 2
#define I(t, phi, theta, r) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi+1)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (theta+1)*(M+GHOST_SIZE) + r+1
#define E(t, phi, theta, r) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (theta)*(M+GHOST_SIZE) + r
#define CI(func, t, phi, theta, r) (func)*4*(blockDim.z+GHOST_SIZE)*(blockDim.y+GHOST_SIZE)*(blockDim.x+GHOST_SIZE) + (t)*(blockDim.z+GHOST_SIZE)*(blockDim.y+GHOST_SIZE)*(blockDim.x+GHOST_SIZE) + (phi+1)*(blockDim.y+GHOST_SIZE)*(blockDim.x+GHOST_SIZE) + (theta+1)*(blockDim.x+GHOST_SIZE) + r+1


#define E1 0.000
#define E2 0.000
#define E3 0.000
#define E4 0.000
#define E5 0.000
#define E6 0.000

#define PI_1 E1*sin(dr*r)*sin(dtheta*theta)*sin(dphi*global_phi)
#define PI_2 E2*sin(dr*r)*sin(dtheta*theta)*sin(dphi*global_phi)
#define PI_3 E3*sin(dr*r)*sin(dtheta*theta)*sin(dphi*global_phi)

typedef double REAL;
