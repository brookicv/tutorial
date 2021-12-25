#include <cuda.h>
#include<cuda_runtime.h>
#include <cstdio>
#include "cudastart.h"

__global__ void reduceNeighbored(int *g_idata,int *g_odata,int n)
{
    int tid = threadIdx.x;
    if(tid >= n) return;

    // 当前block需要计算的数据块的起始地址
    int *idata = g_idata + blockIdx.x * blockDim.x;

    for(int stride = 1; stride < blockDim.x; stride *= 2){
        if((tid % ( 2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
    }

    __syncthreads();

    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

int recursiveReduce(int *data,int size)
{
    if(size == 1) return data[0];

    const int stride = size / 2;
    if(size % 2 == 1){
        for(int i = 0; i < stride; i ++){
            data[i] += data[i + stride];
        }
        data[0] += data[size-1];
    }else {
        for(int i = 0; i < stride; i ++){
            data[i] += data[i + stride];
        }
    }

    return recursiveReduce(data,stride);
}

// 重新组织线程，避免warp分化
__global__ void reduceNeighboredLess(int *g_idata,int *g_odata,int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx > n) return ;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    for(int stride = 1; stride < blockDim.x; stride *=2){

        // 将tid转换为数组的索引
        int index = 2 * stride * tid;
        if(index < blockDim.x){
            idata[index] += idata[index + stride];
        }

        __syncthreads();
    }

    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

// 重新组织内存访问
__global__ void reduceInterleaved(int *g_idata,int *g_odata,int n)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > n)
        return;

    int *data = g_idata + blockIdx.x * blockDim.x;

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }

        __syncthreads();
    }

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

	//execution configuration
	int blocksize = 512;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);   //从命令行输入设置block大小
	}
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


	//kernel reduceNeighbored

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

	// free host memory
	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	//reset device
	cudaDeviceReset();

	//check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success!\n");
	}
	return 0;
}
