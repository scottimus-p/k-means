#include <iostream>
#include <string.h>
#include <string>
#include <limits>
#include <argparse.h>

#include "kmeans.h"
#include "helpers.h"


std::pair<int, std::chrono::milliseconds> run_kmeans_sequential(double2d &centroids, double2d &data, int *labels, const options_t &opts, int n_vals)
{
    int dim = opts.dims;
    int k = opts.num_cluster;

    int iterations = 0;
    bool done = false;
    double2d old_centroids(centroids.dim1, centroids.dim2);

    initializeRandomCentroids(centroids, data, n_vals, dim, k, opts.seed);

    auto start = std::chrono::high_resolution_clock::now();

    if (iterations >= opts.max_num_iter)
    {
        done = true;
    }

    while (!done)
    {
        memcpy(old_centroids.data, centroids.data, sizeof(double) * centroids.dim1 * centroids.dim2);

        find_nearest_centroids(labels, data, centroids, n_vals, dim, k);

        average_labeled_centroids(centroids, data, labels, n_vals, dim, k);

        iterations++;

        if (iterations >= opts.max_num_iter || hasConverged(old_centroids, centroids, k, dim, opts.threshold))
        {
            done = true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    return { iterations, diff };
}


void find_nearest_centroids(int *labels, double2d &data, double2d &centroids, int n_vals, int dim, int numCentroids)
{
    for (int i = 0; i < n_vals; i++)
    {
        double closest_distance = std::numeric_limits<double>::max();
        int closest_centroid = -1;

        for (int j = 0; j < numCentroids; j++)
        {
            // finding the nearest squared distance is equivalent to find the nearest distance
            // so save time by not taking the sqrt
            double distance = calcSquareDistance(&data(i, 0), &centroids(j, 0), dim);

            if (distance < closest_distance)
            {
                closest_distance = distance;
                closest_centroid = j;
            }
        }

        labels[i] = closest_centroid;
    }
}


void average_labeled_centroids(double2d &centroids, double2d &data, int *labels, int n_vals, int dim, int k)
{
    int *centroidCount = new int[k];
    memset(centroidCount, 0, sizeof(int) * k);

    centroids.clear();

    for (int i = 0; i < n_vals; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            centroids(labels[i], j) += data(i, j);
        }

        centroidCount[labels[i]] += 1;
    }

    for (int i = 0; i < k; i++)
    {
        if (centroidCount[i] > 0)
        {
            for (int j = 0; j < dim; j++)
            {
                centroids(i, j) /= centroidCount[i];
            }
        }
    }

    delete [] centroidCount;
}