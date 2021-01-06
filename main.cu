#include <iostream>
#include "cuda.h"

#include "reduce.cuh"


/*! \brief init arrays for a reduce_add and get the result for later comparison */
template <typename T, typename U>
static U init_data_sum(T** d_data, U** d_result, size_t size)
{
    T* h_data = new T[size];
    cudaSafeCall(cudaMalloc((void**)d_data, sizeof(T) * size));
    cudaSafeCall(cudaMalloc((void**)d_result, sizeof(U)));
    cudaSafeCall(cudaMemset(*d_result, 0, sizeof(U)));

    U result = static_cast<U>(0);
    for (unsigned int i = 0; i < size; ++i)
    {
        const uint tmp = i % 10;
        h_data[i] = static_cast<T>(tmp);
        result += static_cast<U>(h_data[i]);
    }

    cudaSafeCall(cudaMemcpy(*d_data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice));

    delete[] h_data;

    return result;
}

/*! \brief Check the results of the gpu reduce and free ressources */
template <typename T, typename U>
static void check_result(U* d_result, U h_expected, T* d_data)
{
    U h_result;
    cudaSafeCall(cudaMemcpy(&h_result, d_result, sizeof(U), cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_result));
    cudaSafeCall(cudaFree(d_data));


    if (h_expected != h_result)
        std::cout << "Failure. Expected: " << h_expected << " Got: " << h_result << std::endl;
    else
        std::cout << "Success." << std::endl;
}


int main()
{
    constexpr size_t size = 100000; // 100 000
    int* d_data;
    int* d_result;

    int h_expected = init_data_sum(&d_data, &d_result, size);

    // Run the reduce on default stream
    reduce_add(d_data, d_result, size);

    // Wait until the end of computation
    // Use cudaStreamSynchronize if the reduce has been ran on a stream
    // different than the default stream
    cudaDeviceSynchronize();

    check_result(d_result, h_expected, d_data);

    return 1;
}