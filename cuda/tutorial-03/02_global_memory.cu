#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

__global__ void global_memory_kernel(int *d_a)
{
    d_a[threadIdx.x] = threadIdx.x;
}

int main()
{
    int h_a[N];
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMemcpy((void*)d_a,(void*)h_a,sizeof(int) * N,cudaMemcpyHostToDevice);

    global_memory_kernel<<<1,N>>>(d_a);
    cudaMemcpy(h_a,d_a,sizeof(int) * N,cudaMemcpyDeviceToHost);

    for(int i = 0; i < N ; i ++) {
        printf("At Index :%d --> %d\n",i,h_a[i]);
    }

    return 0;
}
