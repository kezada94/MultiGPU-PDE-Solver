#include "Grid.h"

Grid::Grid(vector<linspace_definition> &parameters, REAL buffSize){
	this->dimensions = parameters.size();
	this->gridDimensions = vector<size_t>(dimensions);
	this->axes = vector<vector<REAL>>(dimensions);
	this->deltas = vector<REAL>(dimensions);

	size_t linearSize = 1;
	for(uint8_t i=0; i<parameters.size(); ++i){
		linspace_definition d = parameters[i];
		linearSize *= d.n;
		cout << d.n << endl;
		//TODO has to be over 2 - end after start, non negative
		this->gridDimensions[i] = d.n;
		this->deltas[i] = (d.end - d.start)/(d.n-1);
		this->axes[i] = genLinspace(d.start, this->deltas[i], d.n);
	}
	cout << "Generated grid of: " <<buffSize << "x";
	printVector(this->gridDimensions, "x");
	cout << " = " << linearSize << " elements." << endl;
	
    data = vector<REAL*>(buffSize);
    for (uint8_t i=0; i<buffSize; i++){
        data[i] = new REAL[linearSize];
        for (size_t j=0; j<linearSize; j++){
            data[i][j] = 0.0;
        }
    }

}

Grid::~Grid(){

}

void Grid::stepCircularBuffer(){
    size_t buffSize = this->data.size();
    vector<REAL*> temp = vector<REAL*>(buffSize);

    temp[0] = this->data[buffSize-1];
    for (uint8_t i=1; i < buffSize; i++){
        temp[i%buffSize] = this->data[(i-1)%buffSize];
    }
    this->data = temp;
}
//maybe define as 4Dgrid

vector<REAL> Grid::genLinspace(REAL start, REAL delta, size_t n){
	vector<REAL> vec = vector<REAL>(n);
	for(size_t i=0; i<n; ++i){
		vec[i] = start + i*delta;
	}
	return vec;
}

template<typename T>
void Grid::printVector(T vec, string inter){
	for (int i=0; i<vec.size(); i++){
		if (i==vec.size()-1){
			cout << vec[i];
			break;
		}
		cout << vec[i] << inter;
	}
}
