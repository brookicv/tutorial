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


/*

如果每个线程处理两个数据块，那么我们需要的线程块总量会变为原来的一半，
这样处理后线程块减少了，与我们之前要使用尽量多线程块的理论不符。
但实际我们通过这种方式，让一个线程中有更多的独立内存加载/存储操作，
这样可以更好地隐藏内存延时，更好地使用设备内存读取吞吐量的指标，以产生更好的性能。

为了隐藏延时，我们需要合理地增加一个线程块中需要处理的数据量，以便线程束调度器进行调度。

*/

// 先在每个block里手动计算，两个块的和
__global__ void reduceUnroll2(int *g_idata,int *g_odata,int n)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;

    if(tid >= n) return ;

    // 每个block里计算两个块的和
    int *data = g_idata + blockDim.x * blockIdx.x * 2;
    if(idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }

    __syncthreads();

    // 再将一个块的数据计算和
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }

    // 将结果写回,只需要一个线程执行即可
    if(tid == 0){
        g_odata[blockIdx.x] = data[0];
    }
}

// 一个block计算4个数据块
__global__ void reduceUnroll4(int *g_idata,int *g_odata,int n)
{
    int tid = threadIdx.x;
    int idx  = threadIdx.x + blockIdx.x * blockDim.x * 4;
    if(idx >= n) return;

    // 手动展开计算4个块的和
    int *data = g_idata + blockIdx.x * blockDim.x * 4;
    if(idx + blockDim.x * 3 < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
    }
    __syncthreads();

    // 当前block内求数据块的和
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

// 一个block计算8个数据块
__global__ void reduceUnroll8(int *g_idata,int *g_odata,int n)
{
    int tid = threadIdx.x;
    int idx  = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if(idx >= n) return;

    // 手动展开计算4个块的和
    int *data = g_idata + blockIdx.x * blockDim.x * 8;
    if(idx + blockDim.x * 7 < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();

    // 当前block内求数据块的和
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

// 将最后一个warp展开
__global__ void reduceUnrollWarp8(int *g_idata,int *g_odata,int n)
{
        int tid = threadIdx.x;
    int idx  = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if(idx >= n) return;

    // 手动展开计算4个块的和
    int *data = g_idata + blockIdx.x * blockDim.x * 8;
    if(idx + blockDim.x * 7 < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();

    // 当前block内求数据块的和
    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    // 最后一轮迭代展开
    if(tid < 32){
        volatile int *vmem = data;
        vmem[tid] += vmem[tid + 32]; 
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0){
        g_odata[blockIdx.x] = data[0];
    }
} 

// 将循环完全展开
// 将最后一个warp展开
__global__ void reduceCompleteUnrollWarp8(int *g_idata,int *g_odata,int n)
{
        int tid = threadIdx.x;
    int idx  = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if(idx >= n) return;

    // 手动展开计算4个块的和
    int *data = g_idata + blockIdx.x * blockDim.x * 8;
    if(idx + blockDim.x * 7 < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();

    // 将循环完全展开
    if(blockDim.x>=1024 && tid <512)
		data[tid]+=data[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		data[tid]+=data[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		data[tid]+=data[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		data[tid]+=data[tid+64];
	__syncthreads();

    // 上面的写法是考虑到不同的block配置，如果默认是1024的话
    // 随着计算，激活的线程越来越少了
    if(tid < 512){
        data[tid] += data[tid + 512];
    } 
    __syncthreads();  
    if (tid < 256){
        data[tid] += data[tid + 256];
    } 
    __syncthreads();
    if (tid < 128) {
        data[tid] += data[tid + 128];
    }
    __syncthreads();
    if (tid < 64){
        data[tid] += data[tid + 64];
    }
    __syncthreads();


    // 最后一轮迭代展开
    if(tid < 32){
        volatile int *vmem = data;
        vmem[tid] += vmem[tid + 32]; 
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
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

    int blocksize = 1024;
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

    //kernel4 reduceUnroll2
    CHECK(cudaMemcpy(idata_dev,idata_host,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceUnroll2<<<grid.x / 2,block>>>(idata_dev,odata_dev,size);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(odata_host,odata_dev,grid.x / 2 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
	for (int i = 0; i < grid.x / 2; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll2 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x / 2, block.x);

    // kernel5 reduceUnroll4
    CHECK(cudaMemcpy(idata_dev,idata_host,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceUnroll4<<<grid.x / 4,block>>>(idata_dev,odata_dev,size);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(odata_host,odata_dev,grid.x / 4 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
	for (int i = 0; i < grid.x / 4; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll4 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x / 4, block.x);

    
    // kernel5 reduceUnroll8
    CHECK(cudaMemcpy(idata_dev,idata_host,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceUnroll8<<<grid.x / 8,block>>>(idata_dev,odata_dev,size);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(odata_host,odata_dev,grid.x / 8 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll8 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x / 8, block.x);


        // kernel6 reduceUnrollWarp8
    CHECK(cudaMemcpy(idata_dev,idata_host,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceUnrollWarp8<<<grid.x / 8,block>>>(idata_dev,odata_dev,size);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(odata_host,odata_dev,grid.x / 8 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll8 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x / 8, block.x);


    // kernel7 reduceUnrollWarp8
    CHECK(cudaMemcpy(idata_dev,idata_host,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceCompleteUnrollWarp8<<<grid.x / 8,block>>>(idata_dev,odata_dev,size);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(odata_host,odata_dev,grid.x / 8 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll8 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x / 8, block.x); 

    return 0;
}