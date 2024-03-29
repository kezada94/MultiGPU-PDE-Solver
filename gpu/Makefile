TARGET=pde
SRC_DIR=src
OBJ_DIR=obj
VERBOSE=no
GCC_FLAGS=-O3 -std=c++17 -fopenmp
NVCC_FLAGS=-Xcompiler -fopenmp --disable-warnings -gencode=arch=compute_80,code=sm_80
LIBS= 
BSIZEX=34
BSIZEY=4
BSIZEZ=4
DEFINES=-DBSIZEX=${BSIZEX} -DBSIZEY=${BSIZEY} -DBSIZEZ=${BSIZEZ}
INCLUDES=-I../vendor/eigen-3.3.9
CPP_FILES=$(wildcard $(SRC_DIR)/*cpp)
CU_FILES=$(wildcard $(SRC_DIR)/*cu)

OBJ_FILES=$(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.o)))

GCC_OBJS=$(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
NVCC_OBJS=$(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

all: $(TARGET)

$(TARGET) : $(GCC_OBJS) $(NVCC_OBJS)
	nvcc $(DEFINES) $(NVCC_FLAGS) $(LIBS) $(INCLUDES) -o $@ $?

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	g++ $(GCC_FLAGS) $(INCLUDES) -c -o $@ $<

$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu
	nvcc $(DEFINES) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm $(TARGET) $(OBJ_DIR)/*
