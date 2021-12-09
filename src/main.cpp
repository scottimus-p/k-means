#include <argparse.h>
#include <double2d.h>
#include <chrono>
#include <utility>

#include "kmeans.h"

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
    
    std::pair<int, std::chrono::milliseconds> runInfo = run_kmeans_sequential(centroids, data, labels, opts, n_vals);

    int iterations = runInfo.first;
    double ms_per_iter = runInfo.second.count() / double(iterations);

    printf("%d,%lf\n", iterations, ms_per_iter);

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

    return 0;
}