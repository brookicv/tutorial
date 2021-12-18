#include <iostream>

__global__ void first_kernel()
{
}

int main()
{
    first_kernel<<<1,1>>>();
    std::cout << "Hello,CUDA" << std::endl;
    return 0;
}