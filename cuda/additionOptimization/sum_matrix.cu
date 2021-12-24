#include "cudastart.h"
#include <cuda_runtime.h>

void sumMatrixOnCpu(float *matrixA,float *matrixB,float *matrixC,int nx,int ny)
{
    auto a = matrixA;
    auto b = matrixB;
    auto c = matrixC;

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

    int idx = ix + iy * ny;

    if(ix < nx && iy < ny){
        matrixC[idx] = MatrixB[idx] + matrixC[idx];
    }
}

int main(int argc,char **argv)
{
    initDevice(0);

    // 输入二维矩阵，4096*4096，float
    const int nx = 1 << 12;
    const int ny = 1 << 12;
    int nBytes = nx * ny * sizeof(float);

    float h_a[nx][ny];
    float h_b[nx][ny];
    float h_c[nx][ny];

    initialData((float*)h_a,nx*ny);
    initialData((float*)h_b,nx * ny);

    float *d_a,*d_b,*d_c;
    CHECK(cudaMalloc((void **)&d_a,nBytes));
    CHECK(cudaMalloc((void **)&d_b,nBytes));
    CHECK(cudaMalloc((void **)&d_c,nBytes));

    CHECK(cudaMemcpy(d_a,h_a,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,h_b,nBytes,cudaMemcpyHostToDevice));

    // 线程块
    dim3 block(32,32);
    dim3 grid((nx -1 ) / block.x + 1,(ny -1) / block.y + 1);
}