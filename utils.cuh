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