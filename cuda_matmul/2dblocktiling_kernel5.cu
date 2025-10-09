#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define MATRIX_SIZE 4096

__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) 
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] = 
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = 
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
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
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = 
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
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

int main() {
    const int n = MATRIX_SIZE;
    size_t size = n * n * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    initialize_matrix(h_A, n);
    initialize_matrix(h_B, n);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    
    dim3 gridDim((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 blockDim((BM * BN) / (TM * TN));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    sgemm_2d_blocktiling<<<gridDim, blockDim>>>(n, n, n, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    sgemm_2d_blocktiling<<<gridDim, blockDim>>>(n, n, n, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    long long flops = 2LL * n * n * n;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;
    
    printf("\nKernel 5 Performance: \n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Duration: %.2f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPs\n", gflops);
    printf("Percentage of peak (19.5 TF): %.2f%%\n", (gflops / 19500.0) * 100);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("C[0] = %f\n", h_C[0]);
    printf("Kernel completed successfully!\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}