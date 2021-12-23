#ifndef __KMEANS_CUDA_H__
#define __KMEANS_CUDA_H__

#include <chrono>
#include <utility>
#include <tuple>

#include <argparse.h>
#include <double2d.h>

#include <cuda_runtime.h>

using std::tuple;

std::pair<int, std::chrono::milliseconds> run_kmeans_cuda(double2d &centroids, double2d &data, int *labels, const options_t &opts, int n_vals);

void calculateDistanceForAllDataPoints(double *distancesDevice, double *dataDevice, double *centroidsDevice, int n_vals, int dim, int k);
void find_nearest_centroids(int *nearestCentroidsDevice, double *distances, int n_vals, int k);

void average_labeled_centroids(double *centroids, double *dataDevice, int *labelsDevice, int n_vals, int dim, int k);

double * calculateSums(double *dataDevice, int *labelsDevice, int n_vals, int dim, int k);
int *calculateCounts(int *labels, int n_vals, int k);

void setDevice();
int getSPcores(cudaDeviceProp devProp);
tuple<int, int, int> getKernelLaunchParams(int n_vals, int dim);
unsigned int getNextPowerOfTwo(unsigned int value);

#endif