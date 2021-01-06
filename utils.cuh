#pragma once

#ifndef _DEBUG
#define cudaCheckError()
#else
#define cudaCheckError()
    {                                                                                     \
        auto e = cudaGetLastError();                                                      \
        if (e != cudaSuccess)                                                             \
        {                                                                                 \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
            exit(0);                                                                      \
        }                                                                                 \
    }
    #endif

#ifndef _DEBUG
#define cudaSafeCall(ans) ans
#else
#define cudaSafeCall(ans)                                                      \
    {                                                                          \
        gpuAssertDebug((ans), __FILE__, __LINE__);                             \
    }
inline void
gpuAssertDebug(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,
                "GPUassert: %s %s %d\n",
                cudaGetErrorString(code),
                file,
                line);
        if (abort)
            exit(code);
    }
}
#endif

// atomicAdd with double is not defined if CUDA Version is not greater than or
// equal to 600 So we use this macro to keep a fully compatible program
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN
        // != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// atomicMin is only supported for unsigned int & long long, ushort is not
// supported by atomicCAS (even though the doc says so)
#if !defined(__CUDA_ARCH__)
#else
__device__ float atomicMin(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i,
                        assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN
        // != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// atomicMax is only supported for unsigned int & long long, ushort is not
// supported by atomicCAS (even though the doc says so)
#if !defined(__CUDA_ARCH__)
#else
__device__ float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i,
                        assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN
        // != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif