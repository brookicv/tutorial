#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024
#define threadsPerBlock 512

__global__ void gpu_dot(float *d_a,float *d_b,float *d_c)
{
    __shared__ float partial_sum[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = threadIdx.x;

    float sum = 0;
    while(tid < N){
        sum += d_a[tid] * d_b[tid];
        tid += blockDim.x * gridDim.x; // 以block为单位进行计算，如果block进行一次计算无法完成计算，再进行第二轮
    }

    partial_sum[index] = sum; // 同一个block内的线程可以进行通信，将同一个block的线程计算结果保存下来

    __syncthreads();

    // 将同一个block的所有线程的值计算结果累
    // 记住：cuda的每一段代码都会在多个线程中并行
    int i = blockDim.x / 2;
    while(i != 0){
        if(index < i){
            partial_sum[index] += partial_sum[index + i];
        }
        __syncthreads();
        i /= 2;
    }

    if(index == 0){
        d_c[blockIdx.x] = partial_sum[0];
    }
}