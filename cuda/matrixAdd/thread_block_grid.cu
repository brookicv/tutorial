#include <cuda_runtime.h>
#include <iostream>

using namespace std;

/*
https://anuradha-15.medium.com/cuda-thread-indexing-fb9910cba084
*/

__global__ void vector_add_block1(int *d_a,int *d_b,int *d_c,int length)
{
    int tid = threadIdx.x;
    if(tid < length){
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

__global__ void vector_add_block2(int *d_a,int *d_b,int *d_c,int length)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    printf("\t threadIdx.x = %d,threadIdx.y = %d,tid = %d\n",threadIdx.x,threadIdx.y,tid);
    if(tid < length) {
        d_c[tid] = d_a[tid] + d_b[tid];  
    }
}

__global__ void vector_add_grid1_block1(int *d_a,int *d_b,int *d_c,int length)
{
    int tid = blockIdx.x * gridDim.x + threadIdx.x;
    printf("\t threadIdx.x = %d,blockIdx.x = %d,tid = %d\n",threadIdx.x,blockIdx.x,tid);

    if(tid < length) {
        d_c[tid] = d_a[tid] + d_b[tid];  
    }
}

__global__ void vector_add_grid2_block2(int *d_a,int *d_b,int *d_c,int length)
{
    // 先计算块所在的索引 * 块的大小 + thread在block中的索引
    // 这种映射，同一个block中的线程，处理连续地址的数据
    // 一维数据
    int tid = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // 这种映射，同一个block中的线程处理的数据不连续
    // 适用于grid为1 x 1 x 1
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = idx + idy * 4; 

    printf("\t block = (%d,%d),thread = (%d,%d),tid = %d,index = %d,idx = %d,idy = %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,tid,index,idx,idy);

    if(index < length){
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}


int main()
{
    const int length = 16;
    int h_a[length],h_b[length],h_c[length];
    for(int i = 0; i < length; i ++){
        h_a[i] = h_b[i] = i;
    }

    int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,sizeof(int) * length);
    cudaMalloc(&d_b,sizeof(int) * length);
    cudaMalloc(&d_c,sizeof(int) * length);

    cudaMemcpy(d_a,h_a,length * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,length * sizeof(int),cudaMemcpyHostToDevice);

    dim3 block,grid;
    grid = dim3(1,1,1);
    block = dim3(4,4,1);
    cout << "====>>>>> block 2 dim" << endl;
    vector_add_block2<<<grid,block>>>(d_a,d_b,d_c,length);

    cudaMemcpy(h_c,d_c,sizeof(int) * length,cudaMemcpyDeviceToHost);

    for(int i = 0; i < length; i ++){
        cout << h_c[i] << " ";
    }
    cout << endl;


    grid = dim3(4,1,1);
    block = dim3(4,1,1);
    cout << "=====>>>>> 4block,4 thread/block" << endl;
    vector_add_grid1_block1<<<grid,block>>>(d_a,d_b,d_c,length);

    cudaMemcpy(h_c,d_c,sizeof(int) * length,cudaMemcpyDeviceToHost);
    for(int i = 0; i < length; i ++){
        cout << h_c[i] << " ";
    }
    cout << endl;

    grid = dim3(2,2);
    block = dim3(2,2);
    cout << "======>>>> 2 x 2 grid,2 x 2 block" << endl;
    vector_add_grid2_block2<<<grid,block>>>(d_a,d_b,d_c,length);

    cudaMemcpy(h_c,d_c,sizeof(int) * length,cudaMemcpyDeviceToHost);
    for(int i = 0; i < length; i ++){
        cout << h_c[i] << " ";
    }
    cout << endl;

    return 0;

}