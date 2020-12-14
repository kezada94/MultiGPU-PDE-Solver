#pragma once
#include <iostream>
#include "defines.h"

typedef struct linspace_definition{
	linspace_definition(REAL s, REAL e, size_t nn){
		start = s;
		end = e;
		n = nn;
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
