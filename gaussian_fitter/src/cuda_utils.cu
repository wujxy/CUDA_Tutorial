#include "../include/cuda_utils.h"
#include "../include/model.h"
#include <cstdio>

/**
 * 打印GaussianParams
 */
void printParams(const char* label, const GaussianParams& params) {
    printf("%s: A=%.2f, x0=%.3f, y0=%.3f, sigma_x=%.3f, sigma_y=%.3f, rho=%.3f\n",
           label, params.A, params.x0, params.y0, params.sigma_x, params.sigma_y, params.rho);
}

/**
 * 比较两个参数的差异（返回最大相对差异）
 */
float paramsDiff(const GaussianParams& p1, const GaussianParams& p2) {
    float diff_A = fabsf(p1.A - p2.A) / (fabsf(p1.A) + 1e-6f);
    float diff_x0 = fabsf(p1.x0 - p2.x0) / (fabsf(p1.x0) + 1e-6f);
    float diff_y0 = fabsf(p1.y0 - p2.y0) / (fabsf(p1.y0) + 1e-6f);
    float diff_sx = fabsf(p1.sigma_x - p2.sigma_x) / (fabsf(p1.sigma_x) + 1e-6f);
    float diff_sy = fabsf(p1.sigma_y - p2.sigma_y) / (fabsf(p1.sigma_y) + 1e-6f);
    float diff_rho = fabsf(p1.rho - p2.rho) / (fabsf(p1.rho) + 1e-6f);

    return fmaxf(fmaxf(fmaxf(diff_A, diff_x0), fmaxf(diff_y0, diff_sx)),
                 fmaxf(diff_sy, diff_rho));
}

// 简单的串行求和（替代thrust，避免额外依赖）
float arraySum(const float* d_array, int size) {
    float* h_array = new float[size];
    cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += h_array[i];
    }

    delete[] h_array;
    return sum;
}
