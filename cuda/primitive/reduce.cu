#include <cuda_runtime.h>
#include <cstdio>

// baseline, Interleaved Addressing
__global__ void Reduce0(const int *dev_in,int *dev_out,size_t n)
{
    extern __shared__ int shm[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if(idx >= n) return ;

    shm[tid] = idx < n ? dev_in[idx]:0; // Load global data to shared memory
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if(tid % (2 * s) == 0){ // warp分化
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }
    // Save result to global memory
    if(tid == 0){
        dev_out[blockIdx.x] = shm[0];
    }
}

// 解决warp 分化的问题，但是存在bank conflict
__global__ void Reduce1(const int *dev_in,int *dev_out,size_t n)
{
    extern __shared__ int shm[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shm[tid] = idx < n ? dev_in[idx] : 0 ;
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        unsigned int index = 2 * tid * s;
        if(index < blockDim.x){
            shm[index] += shm[index + s]; // bank conflict
        }
        __syncthreads();
    }

    if(tid == 0){
        dev_out[blockIdx.x] = shm[0];
    }
}

// 解决bank conflict，但是存在线程闲置的情况
// 第一次循环只有一半的线程激活
__global__ void Reduce2(const int *dev_in,int *dev_out,size_t n)
{
    extern __shared__ int shm[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shm[tid] = idx < n ? dev_in[idx] : 0;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s /= 2){
        if(tid < s){
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        dev_out[blockIdx.x] = shm[0];
    }
}

// 处理2个数据块
__global__ void Reduce3(const int *dev_in,int *dev_out,size_t n)
{
    extern __shared__ int shm[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 2; // 处理两个block的数据

    shm[tid] = dev_in[tid] + dev_in[tid + blockDim.x];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s /= 2){
        if(tid < s){
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        dev_out[blockIdx.x] = shm[0];
    }
}

__global__ void Reduce4(const int *dev_in,int *dev_out,size_t n)
{
    extern __shared__ int shm[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    unsigned int girde_size = blockDim.x * 2 * gridDim.x;

    shm[tid] = 0;
    while(idx < n){
        shm[tid] += dev_in[idx] + dev_in[idx + blockDim.x];
        idx += girde_size;
    }
    __syncthreads();
}


__global__ void RedeuceTwoPassKernel(const int *dev_input,int *part_sum,size_t n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx == 0){
        printf("gridDim.x = %d,blockDim.x = %d,all = %d\n",gridDim.x,blockDim.x,gridDim.x * blockDim.x);
    }

    int sum = 0;
    for(int i = idx; i < n; i += gridDim.x * blockDim.x){
        sum += dev_input[i];
    }

    extern __shared__ int shm[];
    shm[tid] = sum;
    __syncthreads();

    for(unsigned int d = blockDim.x / 2;d > 0; d /= 2){
        if(tid < d){
            shm[tid] += shm[tid + d];
        }
        __syncthreads();
    }

    if(tid == 0){
        part_sum[blockIdx.x] = shm[0];
    }
}


void ReduceTwoPass(const int *dev_input,int *dev_part,int *sum,size_t n)
{
    const int block_size = 1024;
    const int block_num = 1024;

    size_t shm_size = block_size * sizeof(int);
    RedeuceTwoPassKernel<<<block_num,block_size,shm_size>>>(dev_input,dev_part,n);
    RedeuceTwoPassKernel<<<1,block_size,shm_size>>>(dev_part,sum,block_num);
}


int main()
{
    const size_t size = 2 << 22;
    printf("size = %ld.\n",size);

    int *data = new int[size];
    int sum = 0;
    for(size_t i = 0; i < size; i ++){
        data[i] = 1;
        sum += data[i];
    }

    int block_size = 1024;
    int block_num = (size + block_size - 1) / block_size;

    int *dev_data;
    int *dev_part;
    int *dev_out;

    cudaMalloc<int>(&dev_part,block_num * sizeof(int));
    cudaMalloc<int>(&dev_data,size * sizeof(int));
    cudaMalloc<int>(&dev_out,sizeof(int));

    int *host_out = new int[block_num];

    cudaMemcpy(dev_data,data,size * sizeof(int),cudaMemcpyKind::cudaMemcpyHostToDevice);


    printf("block_num:%d,block_size:%d\n",block_num,block_size);
    Reduce0<<<block_num,block_size,block_size * sizeof(int)>>>(dev_data,dev_part,block_size);
    //Reduce0<<<1,block_size,block_size * sizeof(int)>>>(dev_part,dev_out,block_size);

    //ReduceTwoPass(dev_data,dev_part,dev_out,size);
    cudaDeviceSynchronize();



    int out = 0;
    cudaMemcpy(host_out,dev_part,block_num * sizeof(int),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    for(int i = 0; i < block_num; i ++){
        out += host_out[i];
    }

    if(out == sum){
        printf("Reduce0 Success.\n");
    }else {
        printf("sum:%d,out:%d\n",sum,out);
    }
}