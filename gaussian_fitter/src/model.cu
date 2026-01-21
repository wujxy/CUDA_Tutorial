#include "../include/model_kernels.cuh"
#include <math.h>
#include <cstdio>

// 数值稳定性常数
constexpr float MIN_EXPECTED = 1e-10f;  // 避免log(0)
constexpr float MAX_EXP_ARG = 50.0f;    // 避免exp溢出

/**
 * 设备端函数：计算2D高斯函数值
 */
__device__ float gaussian2d(float x, float y, GaussianParams params) {
    // 计算相对位置
    float dx = x - params.x0;
    float dy = y - params.y0;

    // 2x2协方差矩阵的逆矩阵相关计算
    // Q = (1/(1-ρ²)) * [dx²/σₓ² + dy²/σᵧ² - 2ρ*dx*dy/(σₓσᵧ)]

    float sigma_x_safe = fmaxf(params.sigma_x, 1e-6f);  // 避免除零
    float sigma_y_safe = fmaxf(params.sigma_y, 1e-6f);
    float rho_safe = fmaxf(-0.999f, fminf(params.rho, 0.999f));  // 限制|ρ|<1

    float inv_1_minus_rho2 = 1.0f / (1.0f - rho_safe * rho_safe);

    float dx_sigma = dx / sigma_x_safe;
    float dy_sigma = dy / sigma_y_safe;

    float Q = inv_1_minus_rho2 * (
        dx_sigma * dx_sigma +
        dy_sigma * dy_sigma -
        2.0f * rho_safe * dx_sigma * dy_sigma
    );

    // 限制指数参数避免溢出
    Q = fminf(Q, MAX_EXP_ARG);

    return params.A * expf(-0.5f * Q);
}

/**
 * CUDA Kernel: 计算期望值和泊松似然
 */
__global__ void poissonLikelihoodKernel(
    float* __restrict__ expected,
    float* __restrict__ likelihood,
    const int* __restrict__ observed,
    const float* __restrict__ x_coords,
    const float* __restrict__ y_coords,
    GaussianParams params,
    int nbins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbins) return;

    // 计算当前bin的期望值
    float lambda = gaussian2d(x_coords[idx], y_coords[idx], params);

    // 数值稳定性：确保lambda > 0
    lambda = fmaxf(lambda, MIN_EXPECTED);

    expected[idx] = lambda;

    // 泊松似然（每个bin的贡献）
    // L_i = λ_i - n_i * log(λ_i)
    int n = observed[idx];
    likelihood[idx] = lambda - n * logf(lambda);
}

/**
 * 辅助Kernel：计算单个参数扰动后的似然值
 * 用于数值微分
 */
__global__ void singleParamLikelihoodKernel(
    float* __restrict__ likelihood,
    const int* __restrict__ observed,
    const float* __restrict__ x_coords,
    const float* __restrict__ y_coords,
    GaussianParams params,
    int nbins,
    int param_idx,  // 0=A, 1=x0, 2=y0, 3=sigma_x, 4=sigma_y, 5=rho
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbins) return;

    // 扰动指定参数
    GaussianParams p = params;
    switch (param_idx) {
        case 0: p.A += epsilon; break;
        case 1: p.x0 += epsilon; break;
        case 2: p.y0 += epsilon; break;
        case 3: p.sigma_x += epsilon; break;
        case 4: p.sigma_y += epsilon; break;
        case 5: p.rho += epsilon; break;
    }

    float lambda = gaussian2d(x_coords[idx], y_coords[idx], p);
    lambda = fmaxf(lambda, MIN_EXPECTED);

    int n = observed[idx];
    likelihood[idx] = lambda - n * logf(lambda);
}

/**
 * CUDA Kernel: 数值微分计算梯度（使用相对步长）
 */
__global__ void numericalGradientKernel(
    float* __restrict__ gradient,
    const int* __restrict__ observed,
    const float* __restrict__ x_coords,
    const float* __restrict__ y_coords,
    GaussianParams params,
    float epsilon_rel,  // 相对步长
    int nbins
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
        case 5: epsilon = epsilon_rel * 0.01f; break;  // rho 小步长
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

    // 线程块内规约求和
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int idx = tid;

    float sum = 0.0f;
    while (idx < nbins) {
        // 计算扰动后的似然贡献
        float lambda_plus = gaussian2d(x_coords[idx], y_coords[idx], p_plus);
        lambda_plus = fmaxf(lambda_plus, MIN_EXPECTED);

        float lambda = gaussian2d(x_coords[idx], y_coords[idx], params);
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

/**
 * 创建坐标数组（CPU端实现）
 */
void createCoordinateArrays(
    float* x_coords,
    float* y_coords,
    const Histogram2D& hist
) {
    float dx = (hist.x_max - hist.x_min) / hist.nx;
    float dy = (hist.y_max - hist.y_min) / hist.ny;

    for (int iy = 0; iy < hist.ny; iy++) {
        for (int ix = 0; ix < hist.nx; ix++) {
            int idx = iy * hist.nx + ix;
            // bin中心坐标
            x_coords[idx] = hist.x_min + (ix + 0.5f) * dx;
            y_coords[idx] = hist.y_min + (iy + 0.5f) * dy;
        }
    }
}
