# K-means Clustering

***k-means*** is an implementation of the [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) in C++ and CUDA.

## License
The code for this project is licensed under the MIT License. Refer to [LICENSE](https://github.com/scottimus-p/k-means/blob/main/LICENSE) for more information.

## Usage
The algorithm may be run in either a single-threaded, sequential CPU-only implementation or in a parallel GPU implementation. By default, the sequential implementation is run, but using the `--cuda` flag will run the parallel implementation (see below for supported GPUs).

### Flags
`-k` specifies the number of clusters to group the data into
`-d` tells the algorithm how many dimensions each data point has
`-i` is the file path for the input data
`-m` specifies the maximum number of iterations to run
`-s` a seed for the random number generator for initializing the random initial locations of the clusters
`-t` specifies a stopping threshold (see algorithm description below for the purpose of this)
`-c` an optional flag to output which cluster each data point belongs to (default output is to only display the location for each of the final clusters)
`--cuda` an optional flag to run the algorithm on a GPU (default is to run it on the CPU)

### Supported GPUs
The GPU codes are written in CUDA. As such, only Nvidia GPUs are supported. The included [makefile](https://github.com/scottimus-p/k-means/blob/main/makefile) compiles the code with CUDA compute capability 7.5. See https://developer.nvidia.com/cuda-gpus to determine supported GPU models. Modifying the `-arch` flag in the makefile to an older compute capability may allow some older GPUs to be supported as well.

### Input Data
The input data is specified using the `-i` flag to provide the file path of an input file. The input file should have the number of data points on the first line and then list the data points on all subsequent lines. Each data point should start with that data points id (this can be any number) with spaces separating the value for each dimension. For example, an input file with 5 data points with each data point having 3 dimensions would look like the following:
```
5
0 23.24 43.34 546.89
1 54.98 54.3897 348.9
2 34.59 89.0435 435.789
3 43.85 45.389 934.02
4 43.579 54.38 789.453
```
### Output
By default, the program will output the runtime, the number of iterations the algorithm went through and the location of each of the final clusters. Using the `-c` flag will output a list identifying which cluster each data point has been assigned to rather than the clusters themselves.

## Description of Algorithm
The algorithm is initiated by selecting `k` random locations for the clusters. It is then run through each of the following steps:
1. For each data point, calculate the Euclidean distance between that data point and each of the `k` clusters.
2. Assign each data point to the closest cluster
3. After assigning all data points to a cluster, update the location of each cluster to be the average location of all data points assigned to that cluster.

Each of these three steps will be performed repeatedly until either one of the two following conditions are satisfied:
1. The maximum number of iterations, specified by the `-m` flag, is reached, or
2. The clusters move less than the distance specified by the `-t` flag from their location after the previous iteration


