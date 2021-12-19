#include <iostream>

__global__ void firstKernel()
{
    //std::cout << "Hello! thread in block:" << blockIdx.x << std::endl;
    printf("Hello! thread in block:%d\n",blockIdx.x);
}

int main()
{
    firstKernel<<<16,1>>>();

    // 同步
    cudaDeviceSynchronize();

    std::cout << "All thread are finished" << std::endl;
    return 0;
}