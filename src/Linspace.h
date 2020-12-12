#pragma once
#include <iostream>
#include "defines.h"

typedef struct linspace_definition{
	linspace_definition(REAL start, REAL end, size_t n){
		start = start;
		end = end;
		n = n;
	}
	REAL start;
	REAL end;
	size_t n;
} linspace_definition;

class Linspace{
public:
	Linspace(REAL start, REAL end, size_t size);
	~Linspace();

	REAL* d;
};
