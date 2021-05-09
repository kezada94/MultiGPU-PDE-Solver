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
#define I(t, phi, theta) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi+1)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (theta+1)*(M+GHOST_SIZE)
#define I(t, phi) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi+1)*(N+GHOST_SIZE)*(M+GHOST_SIZE) 
#define I(t) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE)

#define E(t, phi, theta, r) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (theta)*(M+GHOST_SIZE) + r
#define E(t, phi, theta) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (theta)*(M+GHOST_SIZE)
#define E(t, phi) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi)*(N+GHOST_SIZE)*(M+GHOST_SIZE)
#define E(t) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE)

#define E1 0.000
#define E2 0.000
#define E3 0.000
#define E4 0.000
#define E5 0.000
#define E6 0.000

#define PI_1 E1*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_2 E2*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_3 E3*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)

typedef double REAL;
