#include "kmeans_cuda.h"

#include <tuple>
//#include <cuda_runtime.h>
#include <vector>

#include "helpers.h"
#include "kmeans_kernels.h"

#define THREADS_PER_BLOCK 96

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


std::pair<int, std::chrono::milliseconds> run_kmeans_cuda(double2d &centroids, double2d &data, int *labels, const options_t &opts, int n_vals)
{
    int dim = opts.dims;
    int k = opts.num_cluster;

    int iterations = 0;
    bool done = false;
    double2d old_centroids(centroids.dim1, centroids.dim2);

    initializeRandomCentroids(centroids, data, n_vals, dim, k, opts.seed);

    setDevice();

    auto start = std::chrono::high_resolution_clock::now();

    // 1) initialize our device memory
    double *dataDevice, *centroidsDevice, *distancesDevice;
    int *nearestCentroidsDevice;

    cudaMalloc((void **) &dataDevice, sizeof(double) * n_vals * dim);
    cudaMalloc((void **) &centroidsDevice, sizeof(double) * k * dim);
    cudaMalloc((void **) &distancesDevice, sizeof(double) * n_vals * k);// * dim);
    cudaMalloc((void **) &nearestCentroidsDevice, sizeof(int) * n_vals);

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(dataDevice, data.data, sizeof(double) * n_vals * dim, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    // 2) enter the main loop
    if (iterations >= opts.max_num_iter)
    {
        done = true;
    }

    while (!done)
    {
        // 2.a) on the host, move the "current" position of the centroids, to the "previous" position of the centroids
        memcpy(old_centroids.data, centroids.data, sizeof(double) * k * dim);

        cudaMemcpy(centroidsDevice, centroids.data, sizeof(double) * k * dim, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        // 2.b) calculate distance between each data point and all of the centroids
        calculateDistanceForAllDataPoints(distancesDevice, dataDevice, centroidsDevice, n_vals, dim, k);

        // 2.c) for each data point, find the nearest centroid to classify that data point
        find_nearest_centroids(nearestCentroidsDevice, distancesDevice, n_vals, k);

        // 2.d) using the data point classifications, find the updated position of the centroids
        average_labeled_centroids(centroids.data, dataDevice, nearestCentroidsDevice, n_vals, dim, k);

        iterations++;

        if (iterations >= opts.max_num_iter || hasConverged(old_centroids, centroids, k, dim, opts.threshold))
        {
            done = true;
        }
    }

    cudaMemcpy(labels, nearestCentroidsDevice, sizeof(int) * n_vals, cudaMemcpyDeviceToHost);

    cudaFree(dataDevice);
    cudaFree(centroidsDevice);
    cudaFree(distancesDevice);
    cudaFree(nearestCentroidsDevice);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    return { iterations, diff };
}


void calculateDistanceForAllDataPoints(double *distancesDevice, double *dataDevice, double *centroidsDevice, int n_vals, int dim, int k)
{
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (n_vals * k * dim) / threadsPerBlock + 1;

    calculateDistances<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(distancesDevice, dataDevice, centroidsDevice, n_vals, k, dim);
    cudaDeviceSynchronize();

    gpuErrchk( cudaPeekAtLastError() );
}


void find_nearest_centroids(int *nearestCentroidsDevice, double *distances, int n_vals, int k)
{
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (n_vals * k) / threadsPerBlock + 1;

    findCentroids<<<numBlocks, threadsPerBlock>>>(nearestCentroidsDevice, distances, n_vals, k);
    //findCentroidsNew<<<numBlocks, threadsPerBlock, sizeof(int) * threadsPerBlock>>>(nearestCentroidsDevice, distances, n_vals, k);
    cudaDeviceSynchronize();

    gpuErrchk( cudaPeekAtLastError() );
}


void average_labeled_centroids(double *centroids, double *dataDevice, int *labelsDevice, int n_vals, int dim, int k)
{
    double *sumsDevice = calculateSums(dataDevice, labelsDevice, n_vals, dim, k);
    int *countsDevice = calculateCounts(labelsDevice, n_vals, k);
    cudaDeviceSynchronize();

    double *sumsHost = new double[k * dim];
    int *countsHost = new int[k];

    cudaMemcpy(sumsHost, sumsDevice, sizeof(double) * k * dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(countsHost, countsDevice, sizeof(int) * k, cudaMemcpyDeviceToHost);

    cudaFree(sumsDevice);
    cudaFree(countsDevice);

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            if (countsHost[i] > 0)
            {
                centroids[i * dim + j] = sumsHost[i * dim + j] / countsHost[i];
            }
        }
    }
    
    cudaDeviceSynchronize();

    delete [] countsHost;
    delete [] sumsHost;

    sumsHost = nullptr;
    countsHost = nullptr;
}


double * calculateSums(double *dataDevice, int *labelsDevice, int n_vals, int dim, int k)
{
    int threadsPerBlock, numBlocks, shmemBytesPerBlock;

    std::tie(numBlocks, threadsPerBlock, shmemBytesPerBlock) = getKernelLaunchParams(n_vals, dim);

    double *resultDevice;
    cudaMalloc((void **) &resultDevice, sizeof(double) * numBlocks * k * dim);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemset(resultDevice, 0, sizeof(double) * numBlocks * k * dim);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    // perform the initial conditional sum and receive the results by block
    for (int i = 0; i < k; i++)
    {
        sum_by_dim_if<<<numBlocks, threadsPerBlock, shmemBytesPerBlock>>>(resultDevice + i * numBlocks * dim, dataDevice, labelsDevice, i, n_vals, dim);
    }
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    // if required, sum the conditional results by block until the final results can be determined within a single block
    while (numBlocks > 1)
    {
        int prevNumBlocks = numBlocks;

        std::tie(numBlocks, threadsPerBlock, shmemBytesPerBlock) = getKernelLaunchParams(prevNumBlocks, dim);

        for (int i = 0; i < k; i++)
        {
            sum_by_dim<<<numBlocks, threadsPerBlock, shmemBytesPerBlock>>>(resultDevice + i * numBlocks * dim, resultDevice + i * prevNumBlocks * dim, prevNumBlocks, dim);
            cudaDeviceSynchronize();
            gpuErrchk( cudaPeekAtLastError() );
        }
    }

    return resultDevice;
}


int *calculateCounts(int *labels, int n_vals, int k)
{
    int threadsPerBlock = 1024; // max number of threads per block
    int numBlocks = n_vals % threadsPerBlock == 0 ? (n_vals / threadsPerBlock) : ((n_vals) / threadsPerBlock + 1);
    int shmem_per_block = sizeof(int) * threadsPerBlock;

    int *resultDevice;
    cudaMalloc((void **) &resultDevice, sizeof(int) * numBlocks * k);
    gpuErrchk( cudaPeekAtLastError() );

    // perform the initial conditional sum and receive the results by block
    for (int i = 0; i < k; i++)
    {
        count_if<<<numBlocks, threadsPerBlock, shmem_per_block>>>(resultDevice + i * numBlocks, labels, n_vals, i);
    }
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    int *otherResultDevice;
    cudaMalloc((void **) &otherResultDevice, sizeof(int) * numBlocks * k);
    cudaMemset(otherResultDevice, 0, sizeof(int) * numBlocks * k);
    gpuErrchk( cudaPeekAtLastError() );

    bool storeInOther = true;
    // if required, sum the conditional results by block until the final results can be determined within a single block
    bool keepReducing = numBlocks > 1;
    while (keepReducing)
    {
        int prevNumBlocks = numBlocks;
        numBlocks = (prevNumBlocks * k) % threadsPerBlock == 0 ? ((prevNumBlocks * k) / threadsPerBlock) : ((prevNumBlocks * k) / threadsPerBlock + 1);
        shmem_per_block = sizeof(int) * threadsPerBlock;
        
        if (storeInOther)
        {
            sum_by_block<<<numBlocks, threadsPerBlock, shmem_per_block>>>(otherResultDevice, resultDevice, prevNumBlocks * k, prevNumBlocks);
        }
        else
        {
            sum_by_block<<<numBlocks, threadsPerBlock, shmem_per_block>>>(resultDevice, otherResultDevice, prevNumBlocks * k, prevNumBlocks);
        }
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        storeInOther = !storeInOther;
        
        keepReducing = prevNumBlocks > threadsPerBlock;
    }

    if (!storeInOther)
        cudaMemcpy(resultDevice, otherResultDevice, sizeof(int) * numBlocks * k, cudaMemcpyDeviceToDevice);

    cudaFree(otherResultDevice);
    
    return resultDevice;
}


void setDevice()
{
    int count;
    cudaGetDeviceCount(&count);

    int maxCores = 0;
    int whichDevice = 0;

    for (int i = 0; i < count; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        int cores = getSPcores(props);
        if (cores > maxCores)
        {
            maxCores = cores;
            whichDevice = i;
        }
    }

    cudaSetDevice(whichDevice);
}


int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;

    switch (devProp.major)
    {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;

    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;

    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;

    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;

    case 8: // Ampere
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else printf("Unknown device type\n");
        break;

    default:
        printf("Unknown device type\n"); 
        break;
    }

    return cores;
}


tuple<int, int, int> getKernelLaunchParams(int n_vals, int dim)
{
    int maxShMemBytesPerBlock;

    cudaDeviceGetAttribute(&maxShMemBytesPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    int n_per_block = 0;

    auto calcShMemPerBlock = [dim](int n) -> int {
        return dim * sizeof(double) * n;
    };

    const int MAX_THREAD_BLOCK_SIZE = 1024;
    do
    {
        n_per_block++;
    } while (calcShMemPerBlock(n_per_block) < maxShMemBytesPerBlock && n_vals > n_per_block && (n_per_block + 1) * dim < MAX_THREAD_BLOCK_SIZE);

    int num_blocks = n_vals % n_per_block == 0 ? (n_vals * dim / (n_per_block * dim)) : (n_vals * dim / (n_per_block * dim) + 1);
    int threads_per_block = getNextPowerOfTwo(n_per_block * dim);

    int dataPointsPerBlock = threads_per_block % dim == 0 ? (threads_per_block / dim) : (threads_per_block / dim + 1);

    int bytes_per_block = calcShMemPerBlock(dataPointsPerBlock);

    return {num_blocks, threads_per_block, bytes_per_block};
}


unsigned int getNextPowerOfTwo(unsigned int value)
{
    float l = log2f(value);
    unsigned int u = l;

    return 1 << u == value ? 1 << u : 1 << (u + 1);
}