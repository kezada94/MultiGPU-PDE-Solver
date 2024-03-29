#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "defines.h"
#include "Linspace.h"

using namespace std;

class Grid{
public:

	uint8_t dimensions;
	vector<size_t> gridDimensions;
	vector<vector<REAL>> axes;
	vector<REAL> deltas;
	// parameters : Map with <dimension index, starting value, end value, number
	// of datapoints>
	Grid(vector<linspace_definition> &parameters, REAL buffSize);
	~Grid();
    void stepCircularBuffer();

	//Direct acces or too slow
	REAL* data;
	//REAL (*data)[_M][_N][_O];
private:

	vector<REAL> genLinspace(REAL start, REAL end, size_t n);
	template<typename T>
	void printVector(T vec, string inter);
};
