#include <cuda_runtime.h>
#include <random>
#include <iostream>

#define BM 64      // Block rows
#define BN 64      // Block cols  
#define BK 8       // Tile depth, each thread loads 8 from B
#define TM 8       // Results per thread
#define MATRIX_SIZE 4096

/*
Matrix sizes:
MxK * KxN = MxN
*/

__global__ 
void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // faster performance with y as row, x as col
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Shared memory for tiles
    __shared__ float As[BM * BK];  // 64×8
    __shared__ float Bs[BK * BN];  // 8×64

    // Thread indexing for computation
    const int threadCol = threadIdx.x % BN;  
    const int threadRow = threadIdx.x / BN;  

    // Thread indexing for loading 
    const uint innerColA = threadIdx.x % BK;  // 0-7
    const uint innerRowA = threadIdx.x / BK;  // 0-63
    const uint innerColB = threadIdx.x % BN;  // 0-63
    const uint innerRowB = threadIdx.x / BN;  // 0-7

    // Advance pointers to this block's region
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Thread-local accumulator (in registers so fast!)
    float threadResults[TM] = {0.0};

    // Outer loop: iterate over K dimension in chunks of BK
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load tile into shared memory
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        
        __syncthreads();

        // Advance pointers for next iteration
        A += BK;
        B += BK * N;

        // Compute using the tile in shared memory
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load B once, reuse it TM times
            float tmpB = Bs[dotIdx * BN + threadCol];
            
            // Each thread computes TM outputs
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += 
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        
        __syncthreads();
    }

    // Write results to global memory
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = 
            alpha * threadResults[resIdx] + 
            beta * C[(threadRow * TM + resIdx) * N + threadCol];
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

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    initialize_matrix(A, n);
    initialize_matrix(B, n);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Each block has BM*BK threads for loading (64*8 = 512 threads)
    dim3 blockSize(BM * BK);  // 512 threads
    dim3 gridSize((n + BN - 1) / BN, (n + BM - 1) / BM);

    sgemm_1d_blocktiling<<<gridSize, blockSize>>>(n, n, n, 1.0, d_A, d_B, 0.0, d_C);
    
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

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