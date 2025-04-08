#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // 每个 Block 处理 16×16 的矩阵块

// 🔹 CUDA GEMM Kernel（基于 Block Tiling）
__global__ void gemm_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // 🔹 共享内存
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // 🔹 线程索引
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 🔹 遍历 K 维度的 Block
    for (int bk = 0; bk < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++bk) {
        // 读取 A 和 B 子矩阵到共享内存
        if (row < M && bk * BLOCK_SIZE + tx < K)
            Asub[ty][tx] = A[row * K + bk * BLOCK_SIZE + tx];
        else
            Asub[ty][tx] = 0.0f;

        if (col < N && bk * BLOCK_SIZE + ty < K)
            Bsub[ty][tx] = B[(bk * BLOCK_SIZE + ty) * N + col];
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        // 计算当前 16×16 子块的矩阵乘法
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads();
    }

    // 🔹 计算结果写回 C
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// 🔹 Host 代码：分配内存、调用 Kernel
int main() {
    int M = 512, N = 512, K = 512;  // 512×512 矩阵乘法

    // 🔹 1️⃣ 分配 Host 内存
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // 🔹 2️⃣ 初始化矩阵数据
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    // 🔹 3️⃣ 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 🔹 4️⃣ 复制数据到 GPU
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 🔹 5️⃣ 启动 Kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // 🔹 6️⃣ 复制结果回 CPU
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 🔹 7️⃣ 打印部分结果
    std::cout << "C[0,0] = " << h_C[0] << std::endl;

    // 🔹 8️⃣ 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

