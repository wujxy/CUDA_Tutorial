#include <cuda_runtime.h>
#include <stdio.h>

__global__ void addVectors(int n, float *a, float *b, float *c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的全局索引
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int N = 1 << 20;        // 1M 元素
    float *a, *b, *c;       // 主机端指针
    float *d_a, *d_b, *d_c; // 设备端指针

    // 分配主机和设备内存
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&c, N * sizeof(float));

    // 初始化数据
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // 启动内核：<<<grid, block>>>
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(N, a, b, c);
    cudaDeviceSynchronize(); // 等待 GPU 完成

    // 检查结果（简单验证前几个）
    printf("c[0] = %f, c[1] = %f\n", c[0], c[1]); // 应该输出 0, 3

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}