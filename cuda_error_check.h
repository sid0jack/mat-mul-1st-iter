// cuda_error_check.h
#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_LAST_CUDA_ERROR() checkCudaError(__FILE__, __LINE__)

inline void checkCudaError(const char *file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in " << file << " at line " << line << std::endl;
        exit(-1);
    }
}

#endif // CUDA_ERROR_CHECK_H
