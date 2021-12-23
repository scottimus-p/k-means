#include <fstream>
#include <sstream>
#include <cstdlib>

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


bool has_converged(double2d &prev_centroids, double2d &curr_centroids, int numCentroids, int dim, double threshold)
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


void printdata(double2d &data)
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
    double distance = 0;

    for (int i = 0; i < dataDimension; i++)
    {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return distance;
}