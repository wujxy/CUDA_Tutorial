#ifndef MODEL_KERNELS_CUH
#define MODEL_KERNELS_CUH

#include "model.h"

/**
 * CUDA设备端函数：计算2D高斯函数值
 * λ(x,y) = A * exp(-Q/2)
 *
 * Q = (1/(1-ρ²)) * [
 *     (x-x₀)²/σₓ² +
 *     (y-y₀)²/σᵧ² -
 *     2ρ(x-x₀)(y-y₀)/(σₓσᵧ)
 * ]
 */
__device__ float gaussian2d(float x, float y, GaussianParams params);

/**
 * CUDA Kernel: 计算所有bin的期望值和泊松似然（使用bin中心点）
 *
 * 泊松似然函数（负对数似然，需要最小化）:
 * L(θ) = Σᵢ [λᵢ(θ) - nᵢ * log(λᵢ(θ))]
 *
 * 参数:
 *   expected    - 输出: 每个bin的期望值 λᵢ [nbins]
 *   likelihood  - 输出: 每个bin的似然贡献 [nbins]
 *   observed    - 输入: 观测值 nᵢ [nbins]
 *   params      - 输入: 模型参数
 *   nbins       - 输入: 总bin数
 *   x_min, x_max, nx - x方向范围和bin数
 *   y_min, y_max, ny - y方向范围和bin数
 */
__global__ void poissonLikelihoodKernel(
    float* __restrict__ expected,
    float* __restrict__ likelihood,
    const int* __restrict__ observed,
    GaussianParams params,
    int nbins,
    float x_min, float x_max, int nx,
    float y_min, float y_max, int ny
);

/**
 * CUDA Kernel: 通过数值微分计算梯度（使用bin中心点和相对步长）
 *
 * 对每个参数进行扰动，计算似然变化，得到梯度近似:
 * ∇L ≈ [L(θ+εeᵢ) - L(θ)] / ε
 *
 * 参数:
 *   gradient    - 输出: 6个参数的梯度
 *   observed    - 输入: 观测值
 *   params      - 输入: 当前模型参数
 *   epsilon_rel - 输入: 数值微分的相对扰动步长
 *   nbins       - 输入: 总bin数
 *   x_min, x_max, nx - x方向范围和bin数
 *   y_min, y_max, ny - y方向范围和bin数
 */
__global__ void numericalGradientKernel(
    float* __restrict__ gradient,
    const int* __restrict__ observed,
    GaussianParams params,
    float epsilon_rel,
    int nbins,
    float x_min, float x_max, int nx,
    float y_min, float y_max, int ny
);

#endif // MODEL_KERNELS_CUH
