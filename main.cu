#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gemm1.cu"
// Assume the gemm_v00 and launch_gemm_kernel_v00 templates are defined above this code

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)

// Example main function
int main() {
    size_t m = 1024; // Number of rows in matrices A and C
    size_t n = 1024; // Number of columns in matrices B and C
    size_t k = 1024; // Number of columns in matrix A and rows in matrix B

    float alpha = 1.0f;
    float beta = 0.0f;

    float *A, *B, *C;

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMallocManaged(&A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, m * n * sizeof(float)));

    // Initialize matrices A and B with some values
    for (size_t i = 0; i < m * k; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < k * n; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Launch the kernel
    launch_gemm_kernel_v00<float>(m, n, k, &alpha, A, k, B, n, &beta, C, n, 0);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Optionally verify the result
    // This part is skipped for brevity

    // Free GPU memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return 0;
}