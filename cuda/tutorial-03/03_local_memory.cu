#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpu_local_memory(int d_in)
{
    int t_local;
    t_local = d_in * threadIdx.x; // 寄存器中
    printf("Value of Local Variable in current thread is:%d\n",t_local);
    printf("Local variable address=%p\n",&t_local);
}

int main()
{
    gpu_local_memory<<<1,5>>>(5);
    cudaDeviceSynchronize();

    return 0;
}