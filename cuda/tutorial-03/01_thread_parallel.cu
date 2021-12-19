#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 50

__global__ void gpuAdd(int *d_a,int *d_b,int *d_c)
{
    // thread的唯一编号
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("=>>Thread id:%d\n",tid);
    while(tid < N)
    {
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += blockDim.x * gridDim.x;
        printf("====>>thread id:%d\n",tid);
    }
}

int main()
{
    int *d_a,*d_b,*d_c;
    cudaMalloc((void **)&d_a,N * sizeof(int));
    cudaMalloc((void **)&d_b,N * sizeof(int));
    cudaMalloc((void **)&d_c,N * sizeof(int));

    int h_a[N],h_b[N],h_c[N];
    for(int i = 0; i < N; i ++) {
        h_a[i] = 2 * i * i;
        h_b[i] = i;
    }

    // copy data from host to device memory
    cudaMemcpy(d_a,h_a,N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N * sizeof(int),cudaMemcpyHostToDevice);

    // kernel call
    gpuAdd<<<2,2>>>(d_a,d_b,d_c); 

    cudaMemcpy(h_c,d_c,N * sizeof(int),cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    for(int i = 0 ;i < N; i ++) {
        printf("%d + %d = %d\n",h_a[i],h_b[i],h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}