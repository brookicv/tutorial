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
    for(size_t i = 0; i < n; i ++){
        printf("%d ",data[i]);
    }
    printf("\n");
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

// 存在bank conflict
__global__ void BlellochScan1(const int *d_in,int *d_out,size_t n)
{
    u_int32_t tid = threadIdx.x;
    if(tid >= n) return ;

    extern __shared__ int shm[];
    shm[tid] = d_in[tid];
    printf("%d\n",shm[tid]);
    __syncthreads();

    // reduce sweep
    for(size_t stride = 1; stride < n; stride *= 2){
        u_int32_t idx = 2 * stride * (tid + 1) - 1;
        if(idx < n){
            shm[idx] += shm[idx - stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        shm[n - 1] = 0;
        printf("%d\n",shm[n-1]);
    }
    __syncthreads();

    // up sweep
    for(size_t stride = n / 2; stride > 0; stride /= 2){
        u_int32_t idx = 2 * stride * (tid + 1) - 1;
        if(idx < n){
            int tmp = shm[idx - stride];
            shm[idx - stride] = shm[idx];
            shm[idx] += tmp;
        }
        __syncthreads();
    }

    d_out[tid] = shm[tid];
}

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__ void BlellochScan2(const int *d_in,int *d_out,size_t n)
{
    u_int32_t tid = threadIdx.x;
    if(tid >= n) return ;

    extern __shared__ int shm[];
    u_int32_t bank_offset = CONFLICT_FREE_OFFSET(tid);
    shm[tid + bank_offset] = d_in[tid];
    __syncthreads();


    // reduce sweep
    for(size_t stride = 1; stride < n; stride *= 2){
        u_int32_t idx = 2 * stride * (tid + 1) - 1;
        
        if(idx < n){
            u_int32_t x1 = idx;
            u_int32_t x2 = idx - stride;

            u_int32_t x1_offset = CONFLICT_FREE_OFFSET(x1);
            u_int32_t x2_offset = CONFLICT_FREE_OFFSET(x2);

            shm[x1 + x1_offset] += shm[x2 + x2_offset];
        }
        __syncthreads();
    }

    if(tid == 0){
        shm[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }
    __syncthreads();

    // up sweep
    for(size_t stride = n / 2; stride > 0; stride /= 2){
        u_int32_t idx = 2 * stride * (tid + 1) - 1;
        
        if(idx < n){
            u_int32_t x1 = idx;
            u_int32_t x2 = idx - stride;

            u_int32_t x1_offset = CONFLICT_FREE_OFFSET(x1);
            u_int32_t x2_offset = CONFLICT_FREE_OFFSET(x2);

            int tmp = shm[x2 + x2_offset];
            shm[x2 + x2_offset] = shm[x1 + x1_offset] ;
            shm[x1_offset + x1] += tmp;
        }
        __syncthreads();
    }
    d_out[tid] = shm[tid + bank_offset];
}

__global__ void BlellochScan3(const int *d_in,int *d_out,size_t n)
{
    u_int32_t tid = threadIdx.x;
    if(tid >= n) return ;

    extern __shared__ int shm[];
    u_int32_t x1 = tid;
    u_int32_t x2 = tid + n / 2;

    shm[x1] = d_in[x1];
    shm[x2] = d_in[x2];
    __syncthreads();

    int offset = 1;
    for(size_t d = n / 2;d > 0; d /= 2){
        if(tid < d){
            u_int32_t x1 = 2 * offset * (tid + 1) - 1;
            u_int32_t x2 = 2 * offset * (tid + 1) - 1 - offset;

            shm[x1] += shm[x2];
        }
        offset *= 2;
        __syncthreads();
    }

    if(tid == 0){
        shm[n - 1] = 0;
    }

    for(size_t d = 1; d < n; d *= 2){
        offset >>= 1;
        __syncthreads();

        if(tid < d){
            u_int32_t x1 = 2 * offset * (tid + 1) - 1;
            u_int32_t x2 = 2 * offset * (tid + 1) - 1 - offset;

            int tmp = shm[x2];
            shm[x2] = shm[x1];
            shm[x1] += tmp;
        }
    }
    __syncthreads();

    d_out[x1] = shm[x1];
    d_out[x2] = shm[x2];
}

__global__ void BlellochScan4(const int *d_in,int *d_out,size_t n)
{
    u_int32_t tid = threadIdx.x;
    if(tid >= n) return ;

    extern __shared__ int shm[];
    u_int32_t x1 = tid;
    u_int32_t x2 = tid + n / 2;

    u_int32_t offset_x1 = CONFLICT_FREE_OFFSET(x1);
    u_int32_t offset_x2 = CONFLICT_FREE_OFFSET(x2);

    shm[x1 + offset_x1] = d_in[x1];
    shm[x2 + offset_x2] = d_in[x2];
    __syncthreads();

    int offset = 1;
    for(size_t d = n / 2;d > 0; d /= 2){
        if(tid < d){
            u_int32_t x1 = 2 * offset * (tid + 1) - 1;
            u_int32_t x2 = x1 - offset;
            x1 += CONFLICT_FREE_OFFSET(x1);
            x2 += CONFLICT_FREE_OFFSET(x2);

            shm[x1] += shm[x2];
        }
        offset *= 2;
        __syncthreads();
    }

    if(tid == 0){
        shm[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for(size_t d = 1; d < n; d *= 2){
        offset >>= 1;
        __syncthreads();

        if(tid < d){
            u_int32_t x1 = 2 * offset * (tid + 1) - 1;
            u_int32_t x2 = x1 - offset;

            x1 += CONFLICT_FREE_OFFSET(x1);
            x2 += CONFLICT_FREE_OFFSET(x2);

            int tmp = shm[x2];
            shm[x2] = shm[x1];
            shm[x1] += tmp;
        }
    }
    __syncthreads();

    d_out[x1] = shm[x1 + offset_x1];
    d_out[x2] = shm[x2 + offset_x2];
}

__global__ void TestKernel(int *d_in){

    unsigned int tid = threadIdx.x;

    printf("%d\n",d_in[tid]);
    d_in[tid] += 1;

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

    TestKernel<<<1,size>>>(dev_data);

    //BlellochScan1<<<1,size,size * sizeof(int)>>>(dev_data,dev_out,size);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out,dev_out,size * sizeof(int),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if(check(exclusive_out,host_out,size)){
        printf("Blelloch Scan 1 success.\n");
    }

    // memset(host_out,0,size * sizeof(int));
    // cudaMemset(dev_out,0,size * sizeof(int));
    // BlellochScan2<<<1,size,2 * size * sizeof(int)>>>(dev_data,dev_out,size);
    // cudaDeviceSynchronize();

    // cudaMemcpy(host_out,dev_out,size * sizeof(int),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    // if(check(exclusive_out,host_out,size)){
    //     printf("Blelloch Scan 2 success.\n");
    // }

    // memset(host_out,0,size * sizeof(int));
    // cudaMemset(dev_out,0,size * sizeof(int));
    // BlellochScan3<<<1,size / 2,size * sizeof(int)>>>(dev_data,dev_out,size);
    // cudaDeviceSynchronize();

    // cudaMemcpy(host_out,dev_out,size * sizeof(int),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    // if(check(exclusive_out,host_out,size)){
    //     printf("Blelloch Scan 3 success.\n");
    // }

    if(size <= 64){
        printVector(host_data,size);
        printVector(host_out,size);
        printVector(exclusive_out,size);
    }


    return 0;
    
}