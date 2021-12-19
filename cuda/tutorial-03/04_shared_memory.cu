#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpu_shared_memory(float *d_a)
{
    int index = threadIdx.x;
    __shared__ float sh_arr[10];
    sh_arr[index] = d_a[index];

    __syncthreads();
    float sum = 0.f;
    for(int i = 0; i <= index; i ++){
        sum += sh_arr[i];
    }

    float average = sum / (index + 1.0f);
    d_a[index] = average;
}

int main()
{
    float *d_a;
    cudaMalloc(&d_a, 10 * sizeof(float));

    float h_a[10];
    for(int i = 0; i < 10; i ++) {
        h_a[i] = i;
    }

    cudaMemcpy(d_a,h_a,sizeof(float) * 10,cudaMemcpyHostToDevice);
    
    gpu_shared_memory<<<1,10>>>(d_a);
    cudaMemcpy(h_a,d_a,10 * sizeof(float),cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10 ; i ++) {
        printf("The running average after %d element is %f \n",i,h_a[i]);
    }

    return 0;
}