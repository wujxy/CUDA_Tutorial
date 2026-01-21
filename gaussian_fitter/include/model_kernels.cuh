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
 * CUDA Kernel: 计算所有bin的期望值和泊松似然
 *
 * 泊松似然函数（负对数似然，需要最小化）:
 * L(θ) = Σᵢ [λᵢ(θ) - nᵢ * log(λᵢ(θ))]
 *
 * 参数:
 *   expected    - 输出: 每个bin的期望值 λᵢ [nbins]
 *   likelihood  - 输出: 每个bin的似然贡献 [nbins]
 *   observed    - 输入: 观测值 nᵢ [nbins]
 *   x_coords    - 输入: 每个bin的x坐标 [nbins]
 *   y_coords    - 输入: 每个bin的y坐标 [nbins]
 *   params      - 输入: 模型参数
 *   nbins       - 输入: 总bin数
 */
__global__ void poissonLikelihoodKernel(
    float* __restrict__ expected,
    float* __restrict__ likelihood,
    const int* __restrict__ observed,
    const float* __restrict__ x_coords,
    const float* __restrict__ y_coords,
    GaussianParams params,
    int nbins
);

/**
 * CUDA Kernel: 通过数值微分计算梯度（使用相对步长）
 *
 * 对每个参数进行扰动，计算似然变化，得到梯度近似:
 * ∇L ≈ [L(θ+εeᵢ) - L(θ)] / ε
 *
 * 参数:
 *   gradient    - 输出: 6个参数的梯度
 *   observed    - 输入: 观测值
 *   x_coords    - 输入: x坐标
 *   y_coords    - 输入: y坐标
 *   params      - 输入: 当前模型参数
 *   epsilon_rel - 输入: 数值微分的相对扰动步长
 *   nbins       - 输入: 总bin数
 */
__global__ void numericalGradientKernel(
    float* __restrict__ gradient,
    const int* __restrict__ observed,
    const float* __restrict__ x_coords,
    const float* __restrict__ y_coords,
    GaussianParams params,
    float epsilon_rel,
    int nbins
);

#endif // MODEL_KERNELS_CUH
