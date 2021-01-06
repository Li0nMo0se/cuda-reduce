#include <iostream>
#include <sstream>
#include "cuda.h"

#include "reduce.cuh"
#include "timer.h"


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
static void check_result(U* const d_result, const U h_expected, T* const d_data)
{
    U h_result;
    cudaSafeCall(cudaMemcpy(&h_result, d_result, sizeof(U), cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_result));
    cudaSafeCall(cudaFree(d_data));


    if (h_expected != h_result)
        std::cout << "failure. Expected: " << h_expected << " Got: " << h_result << std::endl;
    else
        std::cout << "success." << std::endl;
}


template <typename T, typename U>
void make_test(const size_t size)
{
    // T is the input array type
    // U is the result type (could be T)
    T* d_data;
    U* d_result;

    U h_expected = init_data_sum(&d_data, &d_result, size);

    // Timer (better use nivdia nsight compute for a better timer)
    GpuTimer timer;
    timer.Start();
    // Run the reduce on default stream
    reduce_add(d_data, d_result, size);

    // Wait until the end of computation
    // Use cudaStreamSynchronize if the reduce has been ran on a stream
    // different than the default stream
    cudaDeviceSynchronize();
    timer.Stop();

    std::cout << "Reduce of " << size << " elements: ";
    // Error of precision may accurate if the size is too large
    check_result(d_result, h_expected, d_data);
    std::cout << "Time elapsed: " << timer.Elapsed() << " ms." << std::endl;
}

int main(int argc, char* argv[])
{
    // parse argv
    if (argc != 2)
    {
        std::cout << "Usage: ./" << argv[0] << " array_size" << std::endl;
        return EXIT_FAILURE;
    }
    std::stringstream size_ss(argv[1]);
    size_t size;
    size_ss >> size;

    make_test<float, double>(size);

    return 1;
}