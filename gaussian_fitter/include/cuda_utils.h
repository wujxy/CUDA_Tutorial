#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>

/**
 * CUDA错误检查宏
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

/**
 * 简约的数组规约求和（使用thrust）
 */
float arraySum(const float* d_array, int size);

/**
 * 打印GaussianParams
 */
void printParams(const char* label, const struct GaussianParams& params);

/**
 * 比较两个参数的差异
 */
float paramsDiff(const struct GaussianParams& p1, const struct GaussianParams& p2);

#endif // CUDA_UTILS_H
