#include "cudastart.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

void sumMatrixOnCpu(float *matrixA,float *matrixB,float *matrixC,int nx,int ny)
{
    float *a = matrixA;
    float *b = matrixB;
    float *c = matrixC;

    for(int j = 0; j < ny; j ++){
        for(int i = 0; i < nx; i ++){
            c[i] = a[i] + b[i];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

// 核函数，每一个线程计算矩阵中的一个元素
__global__ void sumMatroxOnGpu(float *matrixA,float *MatrixB,float *matrixC,int nx,int ny)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    int idx = ix + iy * nx; // 线程当前计算的元素所在的位置

    if(ix < nx && iy < ny){
        matrixC[idx] = MatrixB[idx] + matrixA[idx];
    }
}

int main(int argc,char **argv)
{

    printf("hello cuda\n");
    initDevice(0);

    // 输入二维矩阵，4096*4096，float
    const int nx = 1 << 12;
    const int ny = 1 << 11;
    const int nBytes = nx * ny * sizeof(float);

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
    dim3 block(16,16);
    dim3 grid((nx -1 ) / block.x + 1,(ny -1) / block.y + 1);

    printf("grid.x = %d,grid.y = %d\n",grid.x,grid.y);
    printf("All threads:%d\n",grid.x * grid.y * block.x * block.y);

    // 测试cpu时间
    auto start = std::chrono::steady_clock::now();

    // kernel call
    sumMatroxOnGpu<<<grid,block>>>(d_a,d_b,d_c,nx,ny);

    CHECK(cudaDeviceSynchronize());

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    printf("Gpu execution time:%ld ns\n",duration.count());

    // cpu
    cudaMemcpy(h_reslut,d_c,nBytes,cudaMemcpyDeviceToHost);

    start = std::chrono::steady_clock::now();

    sumMatrixOnCpu((float*)h_a,(float*)h_b,(float*)h_c,nx,ny);
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    printf("Cpu execution time:%ld ns \n",duration.count());

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