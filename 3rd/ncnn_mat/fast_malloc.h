
#ifndef NCNN_FAST_MALLOC_H
#define NCNN_FAST_MALLOC_H

namespace ncnn {

    #if __AVX__
    // the alignment of all the allocated buffers
    #define NCNN_MALLOC_ALIGN 32
    #else
    // the alignment of all the allocated buffers
    #define NCNN_MALLOC_ALIGN 16
    #endif

    // we have some optimized kernels that may overread buffer a bit in loop
    // it is common to interleave next-loop data load with arithmetic instructions
    // allocating more bytes keeps us safe from SEGV_ACCERR failure
    #define NCNN_MALLOC_OVERREAD 64

    // Aligns a pointer to the specified number of bytes
    // ptr Aligned pointer
    // n Alignment size that must be a power of two
    template<typename _Tp>
    static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
    {
        return (_Tp*)(((size_t)ptr + n - 1) & -n);
    }

    // Aligns a buffer size to the specified number of bytes
    // The function returns the minimum number that is greater or equal to sz and is divisible by n
    // sz Buffer size to align
    // n Alignment size that must be a power of two
    static inline size_t alignSize(size_t sz, int n)
    {
        return (sz + n - 1) & -n;
    }

    static inline void* fastMalloc(size_t size)
    {
        unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
        if (!udata)
            return 0;
        unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
        adata[-1] = udata;
        return adata;
    }

    static inline void fastFree(void* ptr)
    {
        if (ptr)
        {
            unsigned char* udata = ((unsigned char**)ptr)[-1];
            free(udata);
        }
    }


    static inline int NCNN_XADD(int* addr, int delta)
    {
        int tmp = *addr;
        *addr += delta;
        return tmp;
    }

}

#endif