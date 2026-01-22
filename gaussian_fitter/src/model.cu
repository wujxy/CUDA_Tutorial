#include "../include/model_kernels.cuh"
#include <math.h>
#include <cstdio>

// 数值稳定性常数
constexpr float MIN_EXPECTED = 1e-10f;  // 避免log(0)

/**
 * 设备端函数：计算2D高斯PDF值（归一化版本）
 * 对于撒点方式生成的直方图，PDF需要正确归一化
 * 使得积分等于总样本数A
 */
__device__ float gaussian2d(float x, float y, GaussianParams params) {
    // 计算相对位置
    float dx = x - params.x0;
    float dy = y - params.y0;

    // 2x2协方差矩阵的逆矩阵相关计算
    float sigma_x_safe = fmaxf(params.sigma_x, 1e-6f);
    float sigma_y_safe = fmaxf(params.sigma_y, 1e-6f);
    float rho_safe = fmaxf(-0.999f, fminf(params.rho, 0.999f));

    float inv_1_minus_rho2 = 1.0f / (1.0f - rho_safe * rho_safe);

    float dx_sigma = dx / sigma_x_safe;
    float dy_sigma = dy / sigma_y_safe;

    float Q = inv_1_minus_rho2 * (
        dx_sigma * dx_sigma +
        dy_sigma * dy_sigma -
        2.0f * rho_safe * dx_sigma * dy_sigma
    );

    // 限制指数参数避免溢出
    Q = fminf(Q, 50.0f);

    // 对于归一化的2D高斯，积分等于A时：
    // PDF(x,y) = A * f(x,y)
    // 其中 f(x,y) 是归一化的PDF，积分等于1
    // 完整的2D高斯归一化因子是: 2π * σₓ * σᵧ * sqrt(1-ρ²)

    float norm_factor = 2.0f * 3.14159265359f * sigma_x_safe * sigma_y_safe * sqrtf(1.0f - rho_safe * rho_safe);

    return params.A * expf(-0.5f * Q) / norm_factor;
}

/**
 * CUDA Kernel: 计算期望值和泊松似然（使用bin中心点）
 */
__global__ void poissonLikelihoodKernel(
    float* __restrict__ expected,
    float* __restrict__ likelihood,
    const int* __restrict__ observed,
    GaussianParams params,
    int nbins,
    float x_min, float x_max, int nx,
    float y_min, float y_max, int ny
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbins) return;

    // 计算当前bin的索引
    int ix = idx % nx;
    int iy = idx / nx;

    // 计算bin中心坐标
    float dx = (x_max - x_min) / nx;
    float dy = (y_max - y_min) / ny;
    float x = x_min + (ix + 0.5f) * dx;
    float y = y_min + (iy + 0.5f) * dy;

    // 计算期望值（使用bin中心点的PDF值）
    float lambda = gaussian2d(x, y, params);

    // 数值稳定性：确保lambda > 0
    lambda = fmaxf(lambda, MIN_EXPECTED);

    expected[idx] = lambda;

    // 泊松似然（每个bin的贡献）
    // L_i = λ_i - n_i * log(λ_i)
    int n = observed[idx];
    likelihood[idx] = lambda - n * logf(lambda);
}

/**
 * CUDA Kernel: 数值微分计算梯度（使用bin中心点和相对步长）
 */
__global__ void numericalGradientKernel(
    float* __restrict__ gradient,
    const int* __restrict__ observed,
    GaussianParams params,
    float epsilon_rel,
    int nbins,
    float x_min, float x_max, int nx,
    float y_min, float y_max, int ny
) {
    // 每个线程块处理一个参数的梯度
    int param_idx = blockIdx.x;  // 0-5 对应6个参数

    if (param_idx >= 6) return;

    // 计算每个参数的绝对步长（相对步长 * 参数值或默认值）
    float epsilon = epsilon_rel;

    switch (param_idx) {
        case 0: epsilon = fmaxf(epsilon_rel * params.A, 1.0f); break;
        case 1: epsilon = epsilon_rel * (fabsf(params.x0) + 0.1f); break;
        case 2: epsilon = epsilon_rel * (fabsf(params.y0) + 0.1f); break;
        case 3: epsilon = epsilon_rel * params.sigma_x; break;
        case 4: epsilon = epsilon_rel * params.sigma_y; break;
        case 5: epsilon = epsilon_rel * 0.01f; break;
        default: epsilon = epsilon_rel; break;
    }

    // 扰动后的参数
    GaussianParams p_plus = params;
    switch (param_idx) {
        case 0: p_plus.A += epsilon; break;
        case 1: p_plus.x0 += epsilon; break;
        case 2: p_plus.y0 += epsilon; break;
        case 3: p_plus.sigma_x += epsilon; break;
        case 4: p_plus.sigma_y += epsilon; break;
        case 5: p_plus.rho = fminf(0.99f, fmaxf(-0.99f, params.rho + epsilon)); break;
    }

    // 计算bin尺寸
    float dx_bin = (x_max - x_min) / nx;
    float dy_bin = (y_max - y_min) / ny;

    // 线程块内规约求和
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int idx = tid;

    float sum = 0.0f;
    while (idx < nbins) {
        // 计算当前bin的索引
        int ix = idx % nx;
        int iy = idx / nx;

        // 计算bin中心坐标
        float x = x_min + (ix + 0.5f) * dx_bin;
        float y = y_min + (iy + 0.5f) * dy_bin;

        // 使用bin中心点计算扰动前后的期望值
        float lambda_plus = gaussian2d(x, y, p_plus);
        lambda_plus = fmaxf(lambda_plus, MIN_EXPECTED);

        float lambda = gaussian2d(x, y, params);
        lambda = fmaxf(lambda, MIN_EXPECTED);

        int n = observed[idx];

        // (L(θ+ε) - L(θ)) / ε
        float l_plus = lambda_plus - n * logf(lambda_plus);
        float l = lambda - n * logf(lambda);

        sum += (l_plus - l) / epsilon;
        idx += blockDim.x;
    }

    // 线程块规约
    s_data[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 第一个线程写入结果
    if (tid == 0) {
        gradient[param_idx] = s_data[0];
    }
}
