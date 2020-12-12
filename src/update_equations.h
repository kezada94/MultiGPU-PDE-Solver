#pragma once

#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "defines.h"
using namespace std;

class Update{
public:
	static inline REAL computeNexta(REAL* a, REAL* F, REAL* G, size_t t, size_t r, size_t theta, size_t phi, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L){return 0.f;}

	static inline REAL computeNextF(REAL* a, REAL* F, REAL* G, size_t t, size_t r, size_t theta, size_t phi, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L){return 0.f;}

	static inline REAL computeNextG(REAL* a, REAL* F, REAL* G, size_t t, size_t r, size_t theta, size_t phi, REAL dt, REAL dr, REAL dtheta, REAL dphi, REAL l_1, REAL l_2, REAL L){return 0.f;}
};

