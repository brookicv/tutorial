#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpuAddValue(int d_a,int d_b,int *d_c)
{
    *d_c = d_a + d_b;
}

int main()
{
    int h_c; // Host variable to store result
    int *d_c; // Device pointer

    // Allocate memory for device pointer
    cudaMalloc((void**)&d_c,sizeof(int));

    // kernel call
    gpuAddValue<<<1,1>>>(1,4,d_c);

    // Copy result from device memory to host memory
    cudaMemcpy(&h_c,d_c,sizeof(int),cudaMemcpyDeviceToHost);

    std::cout << "1 + 4 =" << h_c << std::endl;

    cudaFree(d_c);

    return 0;
}