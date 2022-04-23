#include <cuda_runtime.h>
#include <cstdio>


void printVector(int *data,size_t n)
{
    for(size_t i = 0; i < n; i ++){
        printf("%d ",data[i]);
    }
    printf("\n");
}

__device__ void printCUDAVector(int *data,size_t n)
{
    if(threadIdx.x == 0){
        for(size_t i = 0; i < n; i ++){
            printf("%d ",data[i]);
        }
        printf("\n");
    }

}


void ExclusiveScanSequtial(const int *input,int *output,size_t n)
{
	output[0] = 0;
	for(size_t i = 1; i < n; i ++){
		output[i] = output[i - 1] + input[i - 1];
	}
}

void InclusiveScan(const int *input,int *output,size_t n)
{
    int sum = 0;
    for(size_t i = 0; i < n; i ++){
        sum += input[i];
        output[i] = sum;
    }
}

bool check(const int *cpu_out,const int *gpu_out,size_t n)
{
    for(size_t i = 0; i < n; i ++){
        if(cpu_out[i] != gpu_out[i]){
            printf("===>>> Error,idx:%ld,cpu:%d,gpu:%d\n",i,cpu_out[i],gpu_out[i]);
            return false;
        }
    }

    return true;
}

typedef unsigned int uint32;

__global__ void warp_scan(const int *dev_in,int *dev_out)
{
    uint32 lane_id = threadIdx.x & 0x1f;
    int value = dev_in[lane_id];

    #pragma unroll
    for(int i = 1; i <= 32; i *= 2){
        int n = __shfl_up_sync(0xffffffff,value,i,32);
        if(lane_id >= i){
            value += n;
        }
    }
    
    dev_out[lane_id] = value;
}

__global__ void warp_reduce(const int *dev_in, int *dev_out)
{
    uint32 lane_id = threadIdx.x & 0x1f;
    int value = dev_in[lane_id];

    #pragma unroll
    for(int i = 16; i >= 1; i /= 2){
        value += __shfl_xor_sync(0xffffffff,value,i,32);
    }
}

int main()
{
    const size_t size = 32;
    
    int *host_data = new int[size];
    for(size_t i = 0; i < size; i ++){
        host_data[i] = i;
    }
    int *host_out = new int[size];

    int *exclusive_out = new int [size];
    ExclusiveScanSequtial(host_data,exclusive_out,size);

    int *inclusive_out = new int[size];
    InclusiveScan(host_data,inclusive_out,size);

    int *dev_data;
    int *dev_out;

    cudaMalloc<int>(&dev_data,size * sizeof(int));
    cudaMalloc<int>(&dev_out,size * sizeof(int));

    cudaMemcpy(dev_data,host_data,size * sizeof(int),cudaMemcpyKind::cudaMemcpyHostToDevice);
    warp_scan<<<1,32>>>(dev_data,dev_out);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out,dev_out,size * sizeof(int),cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printVector(host_out,size);
    printVector(inclusive_out,size);
    printVector(exclusive_out,size);
}