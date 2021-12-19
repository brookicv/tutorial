#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

__global__ void gpuSquare(float *d_in,float *d_out)
{
    int tid = threadIdx.x;
    float temp = d_in[tid];
    d_out[tid] = temp * temp;
    printf("Thread idx:%d\n",tid);
}

int main()
{
    float *d_in,*d_out;
    cudaMalloc((void **)&d_in,N * sizeof(float));
    cudaMalloc((void **)&d_out,N * sizeof(float));

    float h_in[N];
    float h_out[N];
    for(int i = 0; i < N; i ++) {
        h_in[i] = i;
    }

    // copy data from host to device memory
    cudaMemcpy(d_in,h_in, N * sizeof(float),cudaMemcpyHostToDevice);

    // kernel call
    gpuSquare<<<1,N>>>(d_in,d_out);

    // copy result
    cudaMemcpy(h_out,d_out, N * sizeof(float),cudaMemcpyDeviceToHost);

    for(int i = 0 ;i < N ; i ++) {
        printf("The quare of %f is %f\n",h_in[i],h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}