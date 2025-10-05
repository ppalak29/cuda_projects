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
void sgemm_shared_memory(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const u_int cRow = blockIdx.x;
    const u_int cCol = blockIdx.y;

    __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];

    // coalescing from kernel 2
    const u_int threadRow = threadIdx.x / BLOCK_SIZE;
    const u_int threadCol = threadIdx.x % BLOCK_SIZE;

    A += cRow * K * BLOCK_SIZE; // col 0, row cRow
    B += cCol * BLOCK_SIZE; //col = cCol, row = 0
    C += (cRow * N * BLOCK_SIZE) + (cCol * BLOCK_SIZE); //row = cRow, col = cCol

    float tmp = 0.0;
    for (int tile = 0; tile < K; tile += BLOCK_SIZE) { //loading tiles
        if (cRow * BLOCK_SIZE + threadRow < M && tile + threadCol < K) {
            A_shared[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
        }
        else {
            A_shared[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        }

        if (tile + threadRow < K && cCol * BLOCK_SIZE + threadCol < N) {
            B_shared[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];
        }
        else {
             B_shared[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        }

        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; i ++) {
            tmp += A_shared[threadRow * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + threadCol];
        }
        
        __syncthreads();

        A += BLOCK_SIZE;
        B += (N * BLOCK_SIZE);
    }

    C[threadRow * N + threadCol] = (alpha * tmp) + (beta * C[threadRow * N + threadCol]);
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
    size_t size = n * n * sizeof(float);

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

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE); // coalescing pattern requires 1d blocks
    dim3 gridSize((n + BLOCK_SIZE - 1)/BLOCK_SIZE, (n + BLOCK_SIZE - 1)/BLOCK_SIZE);

    sgemm_shared_memory<<<gridSize, blockSize>>>(n, n, n, 2.0, d_A, d_B, 2.0, d_C);
    
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