#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <cmath>

#include <argparse.h>
#include <double2d.h>

int readfile(double2d &data, const options_t &opts);
void readline(double *data, std::ifstream &str);

void initializeRandomCentroids(double2d &centroids, double2d &data, int n_vals, int dim, int k, unsigned int seed);

bool has_converged(double2d &prev_centroids, double2d &curr_centroids, int numCentroids, int dim, double threshold);

void printdata(double2d &data);

#if __CUDACC__
__host__ __device__
#endif
double calcSquareDistance(double *a, double *b, int dataDimension);

inline double calcDistance(double *a, double *b, int dataDimension)
{
    return sqrt(calcSquareDistance(a, b, dataDimension));
}

#endif