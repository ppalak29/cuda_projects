#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32
#define MATRIX_SIZE 4096

/*
Matrix sizes:
MxK * KxN = MxN
*/

__global__ 
void sgemm_coalesced(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        
        C[x * N + y] = (alpha * tmp) + (beta * C[x * N + y]);
    }
}

void initialize_matrix(float* matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < n * n; i++) {
        matrix[i] = dis(gen);
    }
}

int main() {
    const int n = MATRIX_SIZE;
    size_t size = n * n * sizeof(float);  // Changed: int -> size_t

    // Host memory allocation
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    initialize_matrix(A, n);
    initialize_matrix(B, n);

    // Device memory allocation
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1)/BLOCK_SIZE, (n + BLOCK_SIZE - 1)/BLOCK_SIZE);

    sgemm_coalesced<<<gridSize, blockSize>>>(n, n, n, 2.0, d_A, d_B, 2.0, d_C);
    
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result to verify it worked
    printf("C[0] = %f\n", C[0]);
    printf("Kernel completed successfully!\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    
    return 0;
}