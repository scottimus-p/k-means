
#ifndef __DOUBLE_2_D__
#define __DOUBLE_2_D__

#include <stdlib.h>
#include <string.h>

struct double2d
{
    int dim1;
    int dim2;

    double *data;

    #if __CUDACC__
    __host__ __device__
    #endif
    double &operator()(int idx1, int idx2)
    {
        return data[idx1 * dim2 + idx2];
    }

    #if __CUDACC__
    __host__ __device__
    #endif
    double operator()(int idx1, int idx2) const
    {
        return data[idx1 * dim2 + idx2];
    }

    #if __CUDACC__
    __host__ __device__
    #endif
    double2d() :
        dim1(0),
        dim2(0)
    {
        data = nullptr;
    }

    #if __CUDACC__
    __host__ __device__
    #endif
    double2d(int dimA, int dimB) :
        dim1(dimA),
        dim2(dimB)
    {
        data = new double[dim1 * dim2];
        memset(data, 0, dim1 * dim2 * sizeof(double));
    }

    #if __CUDACC__
    __host__ __device__
    #endif
    const double2d &operator=(const double2d &other)
    {
        if (this != &other)
        { // protect against invalid self-assignment
            // 1: allocate new memory and copy the elements
            dim1 = other.dim1;
            dim2 = other.dim2;

            double* new_data = new double[dim1 * dim2];

            memcpy(new_data, other.data, dim1 * dim2 * sizeof(double));

            // 2: deallocate old memory
            delete [] data;

            // 3: assign the new memory to the object
            data = new_data;
        }

        return *this;
    }
    
    #if __CUDACC__
    __host__ __device__
    #endif
    double2d(const double2d &copyFrom) :
        dim1(copyFrom.dim1),
        dim2(copyFrom.dim2)
    {
        data = new double[dim1 * dim2];
        memcpy(data, copyFrom.data, dim1 * dim2 * sizeof(double));
    }
    
    #if __CUDACC__
    __host__ __device__
    #endif
    ~double2d()
    {
        if (data != nullptr)
        {
            delete [] data;
        }
    }

    #if __CUDACC__
    __host__ __device__
    #endif
    void clear()
    {
        memset(data, 0, dim1 * dim2 * sizeof(double));
    }
};

#endif