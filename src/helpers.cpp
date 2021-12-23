#include <fstream>
#include <sstream>
#include <cstdlib>
#include <immintrin.h> 

#include "helpers.h"


int readfile(double2d &data, const options_t &opts)
{
    // Open file
    std::ifstream in;
    in.open(opts.in_file);

    // Get num vals
    int n_vals = 0;

    in >> n_vals;
    
    // finish reading the rest of the first line
    std::string dummy;
    std::getline(in, dummy);

    data = double2d(n_vals, opts.dims);

    for (int i = 0; i < n_vals; i++)
    {
        readline(&(data(i, 0)), in);
    }

    in.close();

    return n_vals;
}


void readline(double *data, std::ifstream &str)
{
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string value;

    int i = 0;
    bool first = true;
    while (std::getline(lineStream, value, ' '))
    {
        // ignore the first value
        if (!first)
        {
            data[i] = std::stod(value);
            i++;
        }
        first = false;
    }
}


void initializeRandomCentroids(double2d &centroids, double2d &data, int n_vals, int dim, int k, unsigned int seed)
{
    srand(seed);

    for (int i = 0; i < k; i++)
    {
        int idx = rand() % n_vals;

        for (int j = 0; j < dim; j++)
        {
            centroids(i, j) = data(idx, j);
        }
    }
}


bool hasConverged(double2d &prev_centroids, double2d &curr_centroids, int numCentroids, int dim, double threshold)
{
    for (int i = 0; i < numCentroids; i++)
    {
        if (calcDistance(&prev_centroids(i, 0), &curr_centroids(i, 0), dim) > threshold)
        {
            return false;
        }
    
    }

    return true;
}


void printData(double2d &data)
{
    for (int i = 0; i < data.dim1; i++)
    {
        for (int j = 0; j < data.dim2; j++)
        {
            printf(" %f", data(i, j));
        }

        printf("\n");
    }
}


#if __CUDACC__
__host__ __device__
#endif
double calcSquareDistance(double *a, double *b, int dataDimension)
{
    auto stop = dataDimension - dataDimension % 4;

    __m256d avx_a, avx_b, avx_d, avx_d2, avx_sum;

    avx_sum = _mm256_setzero_pd();

    for (int i = 0; i < stop; i += 4)
    {
        avx_a = _mm256_loadu_pd(a + i);
        avx_b = _mm256_loadu_pd(b + i);

        avx_d = _mm256_sub_pd(avx_a, avx_b);

        avx_d2 = _mm256_mul_pd(avx_d, avx_d);

        avx_sum = _mm256_add_pd(avx_sum, avx_d2);
    }

    alignas(32) double result[] = {0, 0, 0, 0};

    _mm256_store_pd(result, avx_sum);

    switch (dataDimension % 4)
    {
    case 3:
        result[2] += (a[dataDimension - 3] - b[dataDimension - 3]) * (a[dataDimension - 3] - b[dataDimension - 3]);
        [[fallthrough]];
    case 2:
        result[1] += (a[dataDimension - 2] - b[dataDimension - 2]) * (a[dataDimension - 2] - b[dataDimension - 2]);
        [[fallthrough]];
    case 1:
        result[0] += (a[dataDimension - 1] - b[dataDimension - 1]) * (a[dataDimension - 1] - b[dataDimension - 1]);
    }

    return result[0] + result[1] + result[2] + result[3];
}