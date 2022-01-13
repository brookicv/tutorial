#include <cuda.h>
#include <cstdio>

__global__ void kernel_dim()
{
    printf("Thread x = %d\n",threadIdx.x);
    printf("Thread y = %d\n",threadIdx.y);

    printf("Block x = %d\n",blockIdx.x);
    printf("Block y = %d\n",blockIdx.y);

    printf("Block dim x = %d\n",blockDim.x);
    printf("Block dim y = %d\n",blockDim.y);

    printf("Grid dim x = %d\n",gridDim.x);
    printf("Grid dim y = %d\n",gridDim.y);
}



int main()
{
    dim3 block(2,2);
    dim3 grid(3,3);

    kernel_dim<<<grid,block>>>();
    return 0;
}

