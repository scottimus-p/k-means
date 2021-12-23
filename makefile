CC = g++
SRCS = ./src/*.cpp
CU_SRCS = ./src/*.cu
INC = ./src/
OPTS = -Xcompiler -mavx -O3
ARCH = -arch=sm_75
EXEC = ./bin/kmeans

CUDAC = nvcc -ccbin $(CC)

all: clean kmeans

kmeans:
	$(CUDAC) $(OPTS) -rdc=true $(SRCS) ./src/kmeans_cuda.cu ./src/kmeans_kernels.cu -I$(INC) $(ARCH) -o ./bin/kmeans

test:
	$(CUDAC) -rdc=true ./tests/unittest.cu ./src/kmeans_cuda.cu ./src/kmeans_kernels.cu ./src/helpers.cpp ./src/rng.cpp -I$(INC) -o ./bin/test

debug:
	$(CUDAC) -Xcompiler -O0 -g -rdc=true $(SRCS) ./src/kmeans_cuda.cu ./src/kmeans_kernels.cu -I$(INC) $(ARCH) -o ./bin/kmeans

clean:
	rm -f ./bin/*
