#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5
__global__ void gpuAdd(int *d_a,int *d_b,int *d_c)
{
    int tid = blockIdx.x; // Block index of current kernel
    if(tid < N){
        d_c[tid] = d_a[tid] + d_b[tid];
    }
    printf("Block idx:%d\n",tid);
}

int main()
{  
    int *d_a,*d_b,*d_c;
    // allocate the memory
    cudaMalloc((void**)&d_a,N * sizeof(int));
    cudaMalloc((void**)&d_b,N * sizeof(int));
    cudaMalloc((void**)&d_c,N * sizeof(int));

    int h_a[N],h_b[N],h_c[N];
    for(int i = 0; i < N ; i ++) {
        h_a[i] = 2 * i * i;
        h_b[i] = i;
    }

    // copy data from host to device memory
    cudaMemcpy(d_a,h_a,N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N * sizeof(int),cudaMemcpyHostToDevice);

    // kernel call
    gpuAdd<<<N,1>>>(d_a,d_b,d_c);

    cudaMemcpy(h_c,d_c,N * sizeof(int),cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i ++) {
        printf("The sum of %d element is %d + %d = %d\n",i,h_a[i],h_b[i],h_c[i]);
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}