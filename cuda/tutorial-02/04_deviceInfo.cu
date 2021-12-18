#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if(device_count == 0){
        std::cout << "There are no available device that support CUDA" << std::endl;
    } else {
        std::cout << "Detected " << device_count << " CUDA Capable devices" << std::endl;
    }

   
    for(int i = 0; i < device_count; i ++) {
        cudaDeviceProp device_property;
        cudaGetDeviceProperties(&device_property,i);
        std::cout << "Device:" << i << ":" << device_property.name << std::endl;

        int driver_version,runtime_version;
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);

        printf("CUDA driver Version:%d.%d\n",driver_version / 1000,(driver_version%100) / 10);
        printf("CUDA runtime Version:%d.%d\n",runtime_version / 1000,(runtime_version%100) / 10);

        printf("Total amount of global memory:%.0f MBytes (%llu bytes)\n",(float)device_property.totalGlobalMem/1048576.0f,(unsigned long long)device_property.totalGlobalMem);
        printf(" (%2d) Multiprocessors,",device_property.multiProcessorCount);
        printf(" GPU Max clock rate: %0.f MHz (%0.2f GHz)\n",device_property.clockRate * 1e-3f,device_property.clockRate * 1e-6f);
        
        printf("Maximum number of threads per multiprocessor:%d\n",device_property.maxThreadsPerMultiProcessor);
        printf("Maixmum number of threads per block:%d\n",device_property.maxThreadsPerBlock);
        printf("Max dimension size of a thread block (x,y,z):(%d,%d,%d)\n",device_property.maxThreadsDim[0],device_property.maxThreadsDim[1],device_property.maxThreadsDim[2]);
        printf("Max dimension size of a grid size (x,y,z):(%d,%d,%d)\n",device_property.maxGridSize[0],device_property.maxGridSize[1],device_property.maxGridSize[2]);
    }

    
    
}