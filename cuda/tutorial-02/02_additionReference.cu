#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpuAdd(int *d_a,int *d_b,int *d_c)
{
    *d_c = *d_a + * d_b;
}

int main()
{
    int h_a = 1;
    int h_b = 4;

    // 定义指向device的指针
    int *d_a;
    int *d_b;
    int *d_c;

    // 分配空间
    cudaMalloc((void**)&d_a,sizeof(int));
    cudaMalloc((void**)&d_b,sizeof(int));
    cudaMalloc((void**)&d_c,sizeof(int));

    // 从host复制值到device
    cudaMemcpy(d_a,&h_a,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&h_b,sizeof(int),cudaMemcpyHostToDevice);

    // Kernel call
    gpuAdd<<<1,1>>>(d_a,d_b,d_c);

    // 从device复制结果到host
    int h_c = 0;
    cudaMemcpy(&h_c,d_c,sizeof(int),cudaMemcpyDeviceToHost);

    std::cout << h_a << "+" << h_b << "=" << h_c << std::endl;

    // free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}