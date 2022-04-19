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

template<unsigned int block_size>
__global__ void BlellochScan(const int *d_in,int *d_out,size_t size)
{
    uint32 tid = threadIdx.x;
    uint32 x1 = tid;
    uint32 x2 = tid + block_size;

    extern __shared__ int shm[];
    shm[x1] = d_in[x1];
    shm[x2] = d_in[x2];
    __syncthreads();

    uint32 offset = 1;
    
    for(size_t d = block_size; d > 0; d /= 2){
        if(tid < d){
            uint32 x1 = 2 * offset * (tid + 1) - 1;
            uint32 x2 = x1 - offset;

            shm[x1] += shm[x2];
        }
        offset *= 2;
        __syncthreads();
    }

    if(tid == 0){
        shm[2 * block_size - 1] = 0;
    }
    
    for(size_t d = 1; d < block_size * 2; d *= 2){
        offset >>= 1;
        __syncthreads();

        if(tid < d){
            uint32 x1 = 2 * offset * (tid + 1) - 1;
            uint32 x2 = x1 - offset;

            int tmp = shm[x1];
            shm[x1] = shm[x2];
            shm[x2] += tmp;
        }
    }

    __syncthreads();
    
    d_out[x1] = shm[x1];
    d_out[x2] = shm[x2];
}

typedef unsigned int uint32_t;

__device__ void warp_scan(int *data,int *warp_sum)
{
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid & 31;

    volatile int *vdata = data;

    uint32_t stride = 1;
    for(size_t d = 16; d > 0; d /= 2){
        if(lane < d){
            uint32_t idx = 2 * stride * (lane + 1) - 1;
            vdata[idx] += vdata[idx - stride];
        }
        stride *= 2;
        __syncwarp();
    }
    //printCUDAVector(data,32);
    if(lane == 31){
        if(warp_sum != nullptr){
            *warp_sum = vdata[lane];
        }
        
        vdata[lane] = 0;
    }

    for(size_t d = 1; d < 32; d *= 2){
        stride >>= 1;
        __syncwarp();
        if(lane < d){
            uint32_t idx = 2 * stride * (lane + 1) - 1;
            int tmp = vdata[idx - stride];
            vdata[idx - stride] = vdata[idx];
            vdata[idx] += tmp;
        }
    }
}

__global__ void block_scan(const int *dev_in,int *dev_out,size_t n)
{
    extern __shared__ int shm[];
    uint32_t tid = threadIdx.x;

    extern __shared__ int warp_sum[32];

    shm[tid] = dev_in[tid];
    __syncthreads();

    uint32_t lane = tid & 31;
    uint32_t warp_id = tid >> 5;

    warp_scan(shm,warp_sum + warp_id);
    __syncthreads();

    printCUDAVector(warp_sum,32);

    if(warp_id == 0){
        warp_scan(warp_sum ,nullptr);
    }
    printCUDAVector(warp_sum,32);
    printCUDAVector(shm,64);
    __syncthreads();
    //if(warp_id > 0)
        shm[tid] += warp_sum[warp_id];
    __syncthreads();

    dev_out[tid] = shm[tid];
}

int main()
{
    const size_t size = 64;
    
    int *host_data = new int[size];
    for(size_t i = 0; i < size; i ++){
        host_data[i] = i;
    }
    int *host_out = new int[size];

    int *exclusive_out = new int [size];
    ExclusiveScanSequtial(host_data,exclusive_out,size);

    int *dev_data;
    int *dev_out;

    cudaMalloc<int>(&dev_data,size * sizeof(int));
    cudaMalloc<int>(&dev_out,size * sizeof(int));

    cudaMemcpy(dev_data,host_data,size * sizeof(int),cudaMemcpyKind::cudaMemcpyHostToDevice);

    block_scan<<<1,64,size * sizeof(int)>>>(dev_data,dev_out,size);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out,dev_out,size * sizeof(int),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if(check(exclusive_out,host_out,size)){
        printf("Blelloch Scan 1 success.\n");
    }
    printVector(host_out,size);
    printVector(exclusive_out,size);
}