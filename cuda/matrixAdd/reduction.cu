#include <cuda.h>
#include "cudastart.h"

// 将向量的原始累加,求和
// 相邻的元素求和，分成不同的block，计算完成后，再将block的结果相加

__global__ void reduceNeighbored(int *g_idata,int *g_odata,int nx)
{
    int tid = threadIdx.x;

    // 当前block求和数据的起始地址，grid 为一维
    int *data = g_idata + blockDim.x * blockIdx.x; 

    // 每次将相邻的原始相加,每次的步长加倍
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        // 每次迭代，只需要一半的线程计算
        if(tid % (2 * stride) == 0){
            data[tid] += data[tid + stride]; // 将相邻的元素相加
        }
        __syncthreads();
    }

    // 将结果写回,只需要一个线程执行即可
    if(tid == 0){
        g_odata[blockIdx.x] = data[0];
    }
    
    /*
    问题1. 每次迭代只使用一半的线程，而且warp内线程会执行不同的分支
    问题2. 访存不连续，导致不能合并访问
    */
}

// 解决warp线程分化
__global__ void reduceNeighboredLess(int *g_idata,int *g_odata,int nx)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // 对应要处理数据的idx
    
    // 避免数据越界
    if(idx >= nx) return; 

    int *idata = g_idata + blockIdx.x * blockDim.x;

    for(int stride = 1; stride < blockDim.x; stride *=2 ){
        // 将线程id映射为要处理数据的index，尽量保证连续的idx的线程执行相同的分支
        int index = tid * 2 * stride;
        if(index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        __syncthreads();
    }

    // 将结果写回,只需要一个线程执行即可
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }

    /*
    将线程的idx进行映射，使同一个warp线程执行相同的分支
    */
}

// 合并访问Global memory
__global__ void reduceInterleaved(int *g_idata,int *g_odata,int n)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= n) return;

    int *data = g_idata + blockDim.x * blockIdx.x;

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }

        __syncthreads();
    }

    // 将结果写回,只需要一个线程执行即可
    if(tid == 0){
        g_odata[blockIdx.x] = data[0];
    }
}


int main(int argc,char** argv)
{
	initDevice(0);
	
	//initialization
    
	int size = 1 << 24;
	printf("	with array size %d  ", size);

    int blocksize = 512;
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int * tmp = (int*)malloc(bytes);

	//initialize the array
	initialData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double timeStart, timeElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	//cpu reduction 对照组
	int cpu_sum = 0;
	timeStart = cpuSecond();
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	timeElaps = 1000*(cpuSecond() - timeStart);

	printf("cpu sum:%d \n", cpu_sum);
	printf("cpu reduction elapsed %lf ms cpu_sum: %d\n", timeElaps, cpu_sum);


	//kernel1 reduceNeighbored
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceNeighbored <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceNeighbored elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);

    //kernel 2 reduceNeighboredless
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceNeighboredLess <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceNeighboredless elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);


    //kernel 3 reduceInterleaved
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceInterleaved <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceInterleaved elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);

    return 0;
}