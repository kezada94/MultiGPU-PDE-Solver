#pragma once
extern "C" {
    #include "quadmath.h"
}
#include <iostream>
#include <boost/multiprecision/float128.hpp>

using namespace boost::multiprecision;

#define GHOST_SIZE 2
#define I(t, phi, theta, r) (t)*(O+GHOST_SIZE)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (phi+1)*(N+GHOST_SIZE)*(M+GHOST_SIZE) + (theta+1)*(M+GHOST_SIZE) + r+1
#define E(t, phi, theta, r) (t)*(O)*(N)*(M) + (phi)*(N)*(M) + (theta)*(M) + r

#define E1 0.00
#define E2 0.00
#define E3 0.00
#define E4 0.00
#define E5 0.00
#define E6 0.00

#define PI_1 E1*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_2 E2*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_3 E3*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_4 E3*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_5 E3*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)
#define PI_6 E3*sin(dr*r)*sin(dtheta*theta)*sin(dphi*phi)

//typedef float128 REAL;
//typedef cpp_dec_float_100 REAL;
typedef long double REAL;
//typedef double REAL;
