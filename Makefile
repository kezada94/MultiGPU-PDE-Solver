CXXFLAGS=-O3 -std=c++11
CUDAFLAGS=-arch=sm_75
LIBS= -lglfw -lGL -lGLEW
LIBDIRS=
INCDIRS=
all: 
	nvcc -o laplace *.cu $(LIBDIRS) $(INCDIRS) $(LIBS) $(CUDAFLAGS) $(CXXFLAGS)
