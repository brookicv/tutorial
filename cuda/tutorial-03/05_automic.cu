#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREAD 10000
#define SIZE 10 
#define BLOCK_WIDTH 100

__global__ void gpu_increment_without_atomic(int *d_a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    tid = tid % SIZE;
    d_a[tid] += 1; // 不同的现场读取相同的值，然后做+1操作，将结果写回显存
}

__global__ void gpu_increment_atomic(int *d_a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    tid = tid % SIZE;
    atomicAdd(d_a + tid,1);
}
int main()
{
    printf("%d total threads in %d blocks writing into %d array elements \n",NUM_THREAD,NUM_THREAD / BLOCK_WIDTH,SIZE);

    int h_a[SIZE];
    const int ARRAY_BYTES = SIZE *  sizeof(int);

    int *d_a;
    cudaMalloc((void**)&d_a,ARRAY_BYTES);
    cudaMemset((void**)d_a,0,ARRAY_BYTES);

    gpu_increment_atomic<< <NUM_THREAD / BLOCK_WIDTH,BLOCK_WIDTH>> >(d_a);

    cudaMemcpy(h_a,d_a,ARRAY_BYTES,cudaMemcpyDeviceToHost);
    for(int i = 0; i < SIZE; i ++){
        printf("index : %d ----> %d times\n",i,h_a[i]);
    }

    cudaFree(d_a);

    return 0;
}