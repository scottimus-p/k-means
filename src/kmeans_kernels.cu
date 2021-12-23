#include "kmeans_kernels.h"

#include "kmeans.h"
#include <argparse.h>
#include <double2d.h>


__global__
void calculateDistances(double *distances, double *data, double *centroids, int n_vals, int k, int dim)
{
    extern __shared__ double sharedMemDistance[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int dataIdx = (tid / dim) / k;
    int centroidIdx = (tid / dim) % k;
    int dimIdx = tid % dim;

    // start by calculating the distance on each dimension
    if (tid < n_vals * k * dim)
    {
        double distance = data[dataIdx * dim + dimIdx] - centroids[centroidIdx * dim + dimIdx];
        
        sharedMemDistance[threadIdx.x] = distance * distance;
    }

    __syncthreads();

    int startStride = 1 << (sizeof(int) * 8 - __clz(dim));

    if (dimIdx >= (dim / 2) * 2)
    {
        sharedMemDistance[threadIdx.x - startStride] += sharedMemDistance[threadIdx.x];
    }

    __syncthreads();

    // next, add up the distance by dimension to get the square of the distance

    for (int i = startStride; i > 0; i /= 2)
    {
        if (dimIdx < i && (dimIdx + i) < dim)
        {
            sharedMemDistance[threadIdx.x] += sharedMemDistance[threadIdx.x + i];
        }
        __syncthreads();
    }

    // store the result
    if (dimIdx == 0 && dataIdx * k + centroidIdx < n_vals * k)
    {
        distances[dataIdx * k + centroidIdx] = sqrt(sharedMemDistance[threadIdx.x]);
    }
}


__global__ void findCentroids(int *nearestCentroidsDevice, double *distances, int n_vals, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_vals)
    {
        double currMinDistance = __DBL_MAX__;
        int currMinCentroid = 0;

        for (int i = 0; i < k; i++)
        {
            if (distances[tid * k + i] < currMinDistance)
            {
                currMinDistance = distances[tid * k + i];
                currMinCentroid = i;
            }
        }

        nearestCentroidsDevice[tid] = currMinCentroid;
    }
}


__global__ void findCentroidsNew(int *nearestCentroidsDevice, double *distances, int n_vals, int k)
{
    extern __shared__ int s[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataIdx = tid / k;
    int centroidIdx = tid % k;

    if (tid >= (n_vals * k)) return;

    int startStride = k / 2;

    s[threadIdx.x] = centroidIdx;

    for (int i = startStride; i > 0; i /= 2)
    {
        if (!(centroidIdx < i && distances[tid] < distances[tid + i]))
        //{
            //s[threadIdx.x] = s[threadIdx.x];
        //}
        //else
        {
            s[threadIdx.x] = s[threadIdx.x + i];
            distances[tid] = distances[tid + i];
        }

        __syncthreads();
    }
    
    if (centroidIdx == 0)
    {
        nearestCentroidsDevice[dataIdx] = s[threadIdx.x];
    }
}


__global__ void countDataPointsTransform(int *result, int *labels, int whichCentroid, int n_vals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_vals)
    {
        result[idx] = labels[idx] == whichCentroid ? 1 : 0;
    }
}


__global__ void count_if(int *result, int *data, int n_vals, int val)
{
    extern __shared__ int sidata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_vals && data[i] == val)
    {
        sidata[tid] = 1;
    }
    else
    {
        sidata[tid] = 0;
    }

    __syncthreads();
    
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sidata[tid] += sidata[tid + s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if (tid == 0)
    {
        result[blockIdx.x] = sidata[0];
    }
}


__global__ void sum_by_dim_if(double *result, double *dataRange, int *labels, int criteria, int n_vals, int dim)
{
    extern __shared__ double sddata[];

    unsigned int dataIdx = blockDim.x / dim * blockIdx.x + threadIdx.x / dim;
    unsigned int elementIdx = dataIdx * dim + threadIdx.x % dim;

    sddata[threadIdx.x] = elementIdx < (n_vals * dim) && labels[dataIdx] == criteria ? dataRange[elementIdx] : 0;

    __syncthreads();

    // do the reduction in shared memory
    int n_per_block = 1 << int(log2f(min(blockDim.x / dim, n_vals))) + 1;

    for (unsigned int s = n_per_block / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x / dim < s && (threadIdx.x + s * dim < dim * int(blockDim.x / dim)))
            sddata[threadIdx.x] += sddata[threadIdx.x + s * dim];

        __syncthreads();
    }

    if (threadIdx.x < dim)
    {
        result[blockIdx.x * dim + threadIdx.x] = sddata[threadIdx.x];
    }
}


__global__ void sum_by_dim(double *result, double *dataRange, int n_vals, int dim)
{
    extern __shared__ double sdata[];

    unsigned int dataIdx = blockDim.x / dim * blockIdx.x + threadIdx.x / dim;
    unsigned int elementIdx = dataIdx * dim + threadIdx.x % dim;

    sdata[threadIdx.x] = elementIdx < (n_vals * dim) ? dataRange[elementIdx] : 0;

    __syncthreads();

    // do the reduction in shared memory
    int n_per_block = 1 << int(log2f(min(blockDim.x / dim, n_vals))) + 1;

    for (unsigned int s = n_per_block / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x / dim < s && (threadIdx.x + s * dim < dim * int(blockDim.x / dim)))
            sdata[threadIdx.x] += sdata[threadIdx.x + s * dim];

        __syncthreads();
    }

    if (threadIdx.x < dim)
    {
        result[blockIdx.x * dim + threadIdx.x] = sdata[threadIdx.x];
    }
}


__global__ void sum_by_block(int *result, int *dataRange, int n_vals, int blockSize)
{
    extern __shared__ int sidata[];

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int whichBlock = tid % blockSize;
    unsigned int whichGroup = tid / blockSize;

    sidata[threadIdx.x] = tid < n_vals ? dataRange[tid] : 0;

    __syncthreads();

    if (whichBlock >= (blockSize / 2) * 2)
    {
        sidata[threadIdx.x - blockSize / 2] += sidata[threadIdx.x];
    }

    __syncthreads();

    // do the reduction in shared memory
    for (int s = blockSize / 2; s > 0; s /= 2)
    {
        if (whichBlock < s && tid < n_vals)
            sidata[threadIdx.x] += sidata[threadIdx.x + s];

        __syncthreads();
    }

    if (whichBlock == 0)
    {
        result[whichGroup] = sidata[threadIdx.x];
    }
}