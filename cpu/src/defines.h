#pragma once
#include <iostream>

#define GHOST_SIZE 2
#define I(t, r, theta, phi) (t)*(M+GHOST_SIZE-1)*(N+GHOST_SIZE-1)*(O+GHOST_SIZE-1) + (r)*(N+GHOST_SIZE-1)*(O+GHOST_SIZE-1) + (theta)*(O+GHOST_SIZE-1) + phi
#define E(t, r, theta, phi) (t)*(M)*(N)*(O) + (r)*(N)*(O) + (theta)*(O) + phi

#define E1 0.001
#define E2 0.001
#define E3 0.001
#define E4 0.001
#define E5 0.001
#define E6 0.001

#define PI_1 0//E1*sin(dr*m)*sin(dtheta*n)*sin(dphi*o)
#define PI_2 0//E2*sin(dr*m)*sin(dtheta*n)*sin(dphi*o)
#define PI_3 0//E3*sin(dr*m)*sin(dtheta*n)*sin(dphi*o)
typedef double REAL;
