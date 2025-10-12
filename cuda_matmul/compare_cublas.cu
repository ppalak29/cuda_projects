#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define MATRIX_SIZE 4096

// My kernel 6 code
__global__ void sgemm_vectorized(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
    // As is now transposed (BK x BM instead of BM x BK)
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint strideA = numThreadsBlocktile / (BK / 4);
    
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint strideB = numThreadsBlocktile / (BN / 4);
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A with float4 and transpose
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
            
            // Transpose: store as column-major
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }
        
        // Load B with float4
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
        }
        
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load from transposed As (now contiguous)
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += 
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        
        __syncthreads();
    }
    
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // Vectorized write to C
            float4 tmp;
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 0];
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 1];
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 2];
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 3];
            
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
        }
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

void run_your_kernel(int N, float* d_A, float* d_B, float* d_C) {
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 blockDim((BM * BN) / (TM * TN));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        sgemm_vectorized<<<gridDim, blockDim>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    cudaDeviceSynchronize();
    
    // Timed runs (average of 10)
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        sgemm_vectorized<<<gridDim, blockDim>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 10.0f; // Average
    
    long long flops = 2LL * N * N * N;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;
    
    printf("Your Kernel 6:\n");
    printf("  Duration: %.3f ms\n", milliseconds);
    printf("  Performance: %.2f GFLOPs\n", gflops);
    printf("  %% of A100 peak: %.2f%%\n", (gflops / 19500.0) * 100);
}

void run_cublas(int N, float* d_A, float* d_B, float* d_C) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N,
                    d_A, N,
                    &beta,
                    d_C, N);
    }
    cudaDeviceSynchronize();
    
    // Timed runs (average of 10)
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N,
                    d_A, N,
                    &beta,
                    d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 10.0f; // Average
    
    long long flops = 2LL * N * N * N;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;
    
    printf("\ncuBLAS:\n");
    printf("  Duration: %.3f ms\n", milliseconds);
    printf("  Performance: %.2f GFLOPs\n", gflops);
    printf("  %% of A100 peak: %.2f%%\n", (gflops / 19500.0) * 100);
    
    cublasDestroy(handle);
}

int main() {
    const int N = MATRIX_SIZE;
    size_t size = N * N * sizeof(float);
    
    printf("=== Matrix Multiply Comparison ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Hardware: A100 (19.5 TFLOPs FP32 peak)\n\n");
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C1 = (float*)malloc(size);
    float *h_C2 = (float*)malloc(size);
    
    initialize_matrix(h_A, N);
    initialize_matrix(h_B, N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Run your kernel
    run_your_kernel(N, d_A, d_B, d_C);
    cudaMemcpy(h_C1, d_C, size, cudaMemcpyDeviceToHost);
    
    // Run cuBLAS
    run_cublas(N, d_A, d_B, d_C);
    cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify correctness (compare first few elements)
    printf("\n=== Correctness Check ===\n");
    bool correct = true;
    float max_diff = 0.0f;
    for (int i = 0; i < 100; i++) {
        float diff = fabs(h_C1[i] - h_C2[i]);
        max_diff = fmax(max_diff, diff);
        if (diff > 1e-3) {
            correct = false;
        }
    }
    printf("Max difference: %.6f\n", max_diff);
    printf("Results match: %s\n", correct ? "YES" : "NO");
    
    // Calculate performance gap
    printf("\n=== Performance Gap Analysis ===\n");
    // Re-run to get clean numbers for comparison
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 blockDim((BM * BN) / (TM * TN));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sgemm_vectorized<<<gridDim, blockDim>>>(N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_yours = 0;
    cudaEventElapsedTime(&time_yours, start, stop);
    float gflops_yours = (2.0 * N * N * N / (time_yours / 1000.0)) / 1e9;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_cublas = 0;
    cudaEventElapsedTime(&time_cublas, start, stop);
    float gflops_cublas = (2.0 * N * N * N / (time_cublas / 1000.0)) / 1e9;
    
    printf("Your Kernel: %.2f GFLOPs\n", gflops_yours);
    printf("cuBLAS:      %.2f GFLOPs\n", gflops_cublas);
    printf("Gap:         %.2f GFLOPs (%.1f%% slower)\n", 
           gflops_cublas - gflops_yours,
           ((gflops_cublas - gflops_yours) / gflops_cublas) * 100);
    printf("Your kernel is %.2fx slower than cuBLAS\n", gflops_cublas / gflops_yours);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    cublasDestroy(handle);
    
    return 0;
}