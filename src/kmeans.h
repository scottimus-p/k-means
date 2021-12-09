#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <double2d.h>
#include <argparse.h>

#include <chrono>

std::pair<int, std::chrono::milliseconds> run_kmeans_sequential(double2d &centroids, double2d &data, int *labels, const options_t &opts, int n_vals);

void find_nearest_centroids(int *labels, double2d &data, double2d &centroids, int n_vals, int dim, int numCentroids);
void cleanup(double **data, double **centroids, int *labels, int n_vals);

void average_labeled_centroids(double2d &centroids, double2d &data, int *labels, int n_vals, int dim, int k);

#endif