#include "cudastart.h"
#include <cuda.h>
#include <chrono>


void sumMatrixOnCpu(float *matrixA,float *matrixB,float *matrixC,unsigned int nx,unsigned int ny)
{
    for(int i = 0; i < ny; i ++){

        // 指针移到第i行
        float *a = matrixA + i * nx;
        float *b = matrixB + i * nx;
        float *c = matrixC + i * nx;
        for(int j = 0; j < nx; j ++){
            c[j] = a[j] + b[j];
        }
    }
}

__global__ void sumMatrixOnGpu(float *matrixA,float *matrixB,float *matrixC,unsigned int nx,unsigned int ny)
{
    /*
        每个线程处理一个矩阵元素的加法
        线程分布在多个block中，利用下面公式计算出，当前线程要计算的矩阵元素的坐标
    */
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // 这种计算方式，处理的数据，按照block的排列依次分布block(0,0),处理前面blockDim.x * blockDim.y的数据，接着block(1,0)
    unsigned int index = (blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y) + threadIdx.x +threadIdx.y * blockDim.x;

    // 根据当前线程的坐标，计算出线程计算的矩阵元素的坐标
    // 这种thread坐标的映射方式，将thread分布在整个grid，就没有block了。这样，同一个block中处理的数据是不是连续，根据block划分不同而不同。
    // 在block中不同行的thread处理的数据是不连续的
    unsigned int idx = ix + iy * nx; 

    // 防止访问越界，要加限定条件
    // 优化点，访问存储，每计算一个元素都要写回Global Memory
    // 利用shared memory 缓存计算结果，分批次写回Global Memory 
    // 但是，shared memory没有大空间
    // 使用寄存器缓存，限制每个block的线程个数
    if(ix < nx && iy < ny){
        matrixC[idx] = matrixA[idx] + matrixB[idx];
    }

    // if(idx < 64){
    //     printf("block = (%d,%d),thread = (%d,%d),idx = %d,index = %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,idx,index);
    // }

    /*
        进行一次加法运算，需要从Global Memory中取两词数据，
        并且需要将结果写回Global Memory
        合并访问，由于一个warp中从Global Memory中连续取32个数，
        会进行合并访问主存，也就是一次计算，一个warp只需要访问
        2次Global Memory
        但是向Global Memory的写回操作，却没有办法合并。
    */

}

__global__ void matrixAdd(float *matrixA,float *matrixB,float *matrixC,unsigned int nx,unsigned int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int index  = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) * 2 + threadId;
    unsigned int blockSize = blockDim.x * blockDim.y;

    if(ix < nx && iy < ny){
         matrixC[index] = matrixA[index] + matrixB[index];
         matrixC[index + blockSize] = matrixA[index + blockSize] + matrixB[index + blockSize];
         matrixC[index + 2* blockSize] = matrixA[index + 2* blockSize] + matrixB[index + 2* blockSize];
         matrixC[index + 3 * blockSize] = matrixA[index + 3* blockSize] + matrixB[index + 3* blockSize];
    }

    // if(index < 64){
    //     printf("block = (%d,%d),thread = (%d,%d),index = %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,index);
    // }
    

}

__global__ void matrixAddSharedMemory(float *matrixA,float *matrixB,float *matrixC,unsigned int nx,unsigned int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int index  = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + threadId;

    // __shared__ float bufferA[512];
    // bufferA[threadId] = matrixA[index];

    if(ix < nx && iy < ny){
         matrixA[index] += matrixB[index];
    }
    // __syncthreads();
    // matrixC[index] = bufferA[threadId];
}

int main()
{
    printf("hello cuda\n");
    initDevice(0);

    // 输入二维矩阵，4096*4096，float
    const unsigned int nx = 1 << 13;
    const unsigned int ny = 1 << 13;
    const unsigned int nBytes = nx * ny * sizeof(float);

    float *h_a = new float[nBytes];
    float *h_b = new float[nBytes];
    float *h_c = new float[nBytes];
    float *h_reslut = new float[nBytes];

    initialData(h_a,nx*ny);
    initialData(h_b,nx * ny);

    float *d_a,*d_b,*d_c;
    CHECK(cudaMalloc((void **)&d_a,nBytes));
    CHECK(cudaMalloc((void **)&d_b,nBytes));
    CHECK(cudaMalloc((void **)&d_c,nBytes));

    CHECK(cudaMemcpy(d_a,h_a,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,h_b,nBytes,cudaMemcpyHostToDevice));

    // 线程块
    // 每一个线程计算一个位置的和
    // 32,一个wrap的线程个数，线程是以wrap为最小单元调度的
    dim3 block(32,16);
    dim3 grid((nx -1 ) / block.x + 1,(ny -1) / block.y + 1);

    printf("grid.x = %d,grid.y = %d\n",grid.x,grid.y);
    printf("All threads:%d\n",grid.x * grid.y * block.x * block.y);

    // 创建cuda event
    cudaEvent_t event_start,stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&stop);
    // 测试cpu时间
    auto start = std::chrono::steady_clock::now();

    // kernel call
    cudaEventRecord(event_start);
    
    sumMatrixOnGpu<<<grid,block>>>(d_a,d_b,d_c,nx,ny);

    //grid = dim3(grid.x / 2 ,grid.y / 2);
    //matrixAddSharedMemory<<<grid,block>>>(d_a,d_b,d_c,nx,ny);

    cudaEventRecord(stop);

    cudaEventSynchronize(event_start); // 等待时间结束
    cudaEventSynchronize(stop);

    float eventSpan;
    cudaEventElapsedTime(&eventSpan,event_start,stop);

    CHECK(cudaDeviceSynchronize());

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    auto ms_duration = duration.count() / 1000000.0f;
    printf("Gpu execution time:%lf ms\n",ms_duration);

    printf("Gpu envent time:%lf ms\n",eventSpan);

    // cpu
    cudaMemcpy(h_reslut,d_c,nBytes,cudaMemcpyDeviceToHost);

    start = std::chrono::steady_clock::now();

    sumMatrixOnCpu((float*)h_a,(float*)h_b,(float*)h_c,nx,ny);
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    auto ms  = duration.count() / 1000000.0f;
    printf("Cpu execution time:%lf ms \n",ms);

    checkResult((float*)h_c,(float*)h_reslut,nx * ny);

    // free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_reslut;

    return 0;
}