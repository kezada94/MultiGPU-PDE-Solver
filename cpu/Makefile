TARGET=pde
SRC_DIR=src
OBJ_DIR=obj
VERBOSE=no
GCC_FLAGS=-O3 -std=c++17 -fopenmp -pthread -lgmp -lmpfr
LIBS=-fopenmp -pthread -lgmp -lmpfr -lquadmath
DEFINES=
INCLUDES=-I../vendor/eigen-3.3.9 
CPP_FILES=$(wildcard $(SRC_DIR)/*cpp)

OBJ_FILES=$(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

GCC_OBJS=$(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))

all: $(TARGET)

$(TARGET) : $(GCC_OBJS) $(NVCC_OBJS)
	g++ $(LIBS) $(INCLUDES) -o $@ $?

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	g++ $(GCC_FLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm $(TARGET) $(OBJ_DIR)/*
