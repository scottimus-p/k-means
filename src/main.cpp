#include <argparse.h>
#include <double2d.h>
#include <chrono>
#include <utility>

#include "kmeans.h"
#include "kmeans_cuda.h"

#include "helpers.h"

int main(int argc, char **argv)
{
    struct options_t opts;
    get_opts(argc, argv, &opts);

    double2d data;
    double2d centroids;
    int *labels = nullptr;

    int n_vals = readfile(data, opts);

    labels = new int[n_vals];

    centroids = double2d(opts.num_cluster, opts.dims);
    
    std::pair<int, std::chrono::milliseconds> runInfo;

    auto start = std::chrono::high_resolution_clock::now();
    
    if (opts.use_cuda)
    {
        runInfo = run_kmeans_cuda(centroids, data, labels, opts, n_vals);
    }
    else
    {
        runInfo = run_kmeans_sequential(centroids, data, labels, opts, n_vals);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int iterations = runInfo.first;
    double ms_per_iter = runInfo.second.count() / double(iterations);

    printf("total runtime: %ld\niterations: %d\nmillisec per iteration: %lf\n", diff.count(), iterations, ms_per_iter);

    if (!opts.c)
    {
        printf("clusters:\n");
        for (int i = 0; i < opts.num_cluster; i++)
        {
            printf(" %d", i);
            for (int j = 0; j < opts.dims; j++)
            {
                printf(" %f", centroids(i, j));
            }
            printf("\n");
        }
    }
    else
    {
        for (int i = 0; i < n_vals; i++)
        {
            printf(" %d: %d\n", i, labels[i]);
        }
    }

    delete [] labels;
    labels = nullptr;
    
    return 0;
}