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

clean:
	rm -f ./bin/*
