#pragma once

#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "defines.h"
using namespace std;

class Equation{
public:
	static REAL computeNexta(array4D &a, array4D &F, array4D &G, size_t t, size_t r, size_t theta, size_t phi, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L);

	static REAL computeNextF(array4D &a, array4D &F, array4D &G, size_t t, size_t r, size_t theta, size_t phi, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L);

	static REAL computeNextG(array4D &a, array4D &F, array4D &G, size_t t, size_t r, size_t theta, size_t phi, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L);
};

