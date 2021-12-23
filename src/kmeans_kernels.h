#ifndef __KMEANS_KERNEL_H__
#define __KMEANS_KERNEL_H__

__global__ void calculateDistances(double *distances, double *data, double *centroids, int n_vals, int k, int dim);
__global__ void findCentroids(int *nearestCentroidsDevice, double *distances, int n_vals, int k);
__global__ void countDataPointsTransform(int *result, int *labels, int whichCentroid, int n_vals);
__global__ void count_if(int *result, int *data, int n_vals, int val);
__global__ void sum_by_dim_if(double *result, double *dataRange, int *criteriaRange, int criteria, int n_vals, int dim);
__global__ void sum_by_dim(double *result, double *dataRange, int n_vals, int dim);
__global__ void sum_by_block(int *result, int *dataRange, int n_vals, int blockSize);


template <class T>
__global__ void sumNew(T *dataInput, T *dataOutput)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_data[];
    T *sddata = reinterpret_cast<T *>(shmem_data);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // load one element from global memory to shared memory in each thread
    sddata[threadIdx.x] = dataInput[tid];
    __syncthreads();

    // do the reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        sddata[threadIdx.x] += sddata[threadIdx.x + s];
    }

    if (threadIdx.x == 0)
    {
        dataOutput[blockIdx.x] = sddata[0];
    }
}


__global__ void findCentroidsNew(int *nearestCentroidsDevice, double *distances, int n_vals, int k);

/*
template< typename T >
__global__ void sum(T *result, T *data, int n_vals)
{
    extern __shared__ T sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_vals)
    {
        sdata[tid] = data[i];
    }
    else
    {
        sdata[tid] = 0;
    }

    __syncthreads();
    
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if (tid == 0)
    {
        result[blockIdx.x] = sdata[0];
    }
}
*/
#endif