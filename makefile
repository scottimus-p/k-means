CC = g++
SRCS = ./src/*.cpp
CU_SRCS = ./src/*.cu
INC = ./src/
OPTS = -Xcompiler -O3
ARCH = -arch=sm_75
EXEC = ./bin/kmeans

CUDAC = nvcc -ccbin $(CC)

all: clean sequential

sequential:
	$(CC) -I$(INC)  $(SRCS) -D SEQUENTIAL -std=c++17 -O3 -o ./bin/kmeans_sequential

cuda:
	$(CUDAC) $(OPTS) -rdc=true $(SRCS) ./src/kmeans_cuda.cu ./src/kmeans_kernel.cu -I$(INC) $(ARCH) -o ./bin/kmeans_cuda

test:
	$(CUDAC) -rdc=true ./tests/unittest.cu ./src/kmeans_cuda.cu ./src/kmeans_kernel.cu ./src/helpers.cpp ./src/rng.cpp -I$(INC) -o ./bin/test

debug:
	$(CUDAC) -Xcompiler -O0 -g -rdc=true $(SRCS) ./src/kmeans_cuda.cu ./src/kmeans_kernel.cu -I$(INC) $(ARCH) -o ./bin/kmeans_cuda

clean:
	rm -f ./bin/*
