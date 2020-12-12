#pragma once

class FdmSolver{

public:
	virtual void step() = 0;
	virtual void updateInnerPoints() = 0;
	virtual void updateNeumannBoundary() = 0;
	virtual void updateDirichletBoundary() = 0;
	virtual void () = 0;
};
