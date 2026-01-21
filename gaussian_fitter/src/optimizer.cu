#include "../include/optimizer.h"
#include "../include/cuda_utils.h"
#include "../include/model_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// 用于数组求和的辅助kernel
__global__ void sumKernel(float* __restrict__ output, const float* __restrict__ input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_data[];

    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
    }
    s_data[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = s_data[0];
    }
}

/**
 * 计算总似然值
 */
float GradientDescentOptimizer::computeLikelihood(const GaussianParams& params) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (nbins + threadsPerBlock - 1) / threadsPerBlock;

    // 第一步：计算每个bin的似然
    poissonLikelihoodKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_expected, d_likelihood, d_observed, d_x_coords, d_y_coords, params, nbins
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 第二步：规约求和
    int num_blocks = blocksPerGrid;
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, num_blocks * sizeof(float)));

    sumKernel<<<num_blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_temp, d_likelihood, nbins
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 如果需要多级规约（当bin数很大时）
    while (num_blocks > 1) {
        int prev_blocks = num_blocks;
        num_blocks = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;

        sumKernel<<<num_blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_expected, d_temp, prev_blocks
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 交换指针
        float* swap = d_temp;
        d_temp = d_expected;
        d_expected = swap;
    }

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_temp));

    return result;
}

/**
 * 计算梯度（数值微分）
 */
void GradientDescentOptimizer::computeGradient(const GaussianParams& params, float* gradient) {
    const int threadsPerBlock = 256;

    // 使用numericalGradientKernel计算所有6个参数的梯度
    numericalGradientKernel<<<6, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_gradient, d_observed, d_x_coords, d_y_coords, params, config.gradient_epsilon, nbins
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝回主机
    CUDA_CHECK(cudaMemcpy(gradient, d_gradient, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    // 对每个参数进行自适应归一化
    // 使用参数值的倒数作为归一化因子
    float scale[6];
    scale[0] = 1.0f / (fabsf(params.A) + 100.0f);         // A
    scale[1] = 1.0f / (fabsf(params.x0) + 0.5f);         // x0
    scale[2] = 1.0f / (fabsf(params.y0) + 0.5f);         // y0
    scale[3] = 1.0f / (fabsf(params.sigma_x) + 0.5f);    // sigma_x
    scale[4] = 1.0f / (fabsf(params.sigma_y) + 0.5f);    // sigma_y
    scale[5] = 1.0f;                                      // rho (already small)

    for (int i = 0; i < 6; i++) {
        gradient[i] *= scale[i];
    }
}

/**
 * 应用参数约束
 */
void GradientDescentOptimizer::applyConstraints(GaussianParams& params) {
    // sigma 必须为正
    params.sigma_x = fmaxf(params.sigma_x, 0.1f);
    params.sigma_y = fmaxf(params.sigma_y, 0.1f);

    // rho 必须在 (-1, 1) 之间
    params.rho = fmaxf(-0.99f, fminf(params.rho, 0.99f));

    // 幅值通常为正
    params.A = fmaxf(params.A, 0.0f);
}

/**
 * 参数更新
 */
void GradientDescentOptimizer::updateParams(GaussianParams& params, const float* gradient) {
    // 直接使用梯度（已经被归一化）
    params.A        -= config.learning_rate * gradient[0] * 10.0f;     // A 需要较大更新
    params.x0       -= config.learning_rate * gradient[1];
    params.y0       -= config.learning_rate * gradient[2];
    params.sigma_x  -= config.learning_rate * gradient[3];
    params.sigma_y  -= config.learning_rate * gradient[4];
    params.rho      -= config.learning_rate * gradient[5] * 0.01f;      // rho 需要更小更新

    applyConstraints(params);
}

/**
 * 构造函数
 */
GradientDescentOptimizer::GradientDescentOptimizer(const OptimizerConfig& cfg)
    : config(cfg), d_observed(nullptr), d_x_coords(nullptr), d_y_coords(nullptr),
      nbins(0), d_expected(nullptr), d_likelihood(nullptr), d_gradient(nullptr)
{
}

/**
 * 析构函数
 */
GradientDescentOptimizer::~GradientDescentOptimizer() {
    if (d_expected)   cudaFree(d_expected);
    if (d_likelihood) cudaFree(d_likelihood);
    if (d_gradient)   cudaFree(d_gradient);
}

/**
 * 设置数据
 */
void GradientDescentOptimizer::setData(const Histogram2D& hist) {
    nbins = hist.nx * hist.ny;

    // 分配临时内存
    CUDA_CHECK(cudaMallocManaged(&d_expected, nbins * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_likelihood, nbins * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_gradient, 6 * sizeof(float)));

    // 坐标数组（统一内存）
    float* x_coords;
    float* y_coords;
    CUDA_CHECK(cudaMallocManaged(&x_coords, nbins * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&y_coords, nbins * sizeof(float)));

    createCoordinateArrays(x_coords, y_coords, hist);

    d_x_coords = x_coords;
    d_y_coords = y_coords;
    d_observed = hist.data;
}

/**
 * 执行优化
 */
OptimizationResult GradientDescentOptimizer::optimize(const GaussianParams& initial_params) {
    OptimizationResult result;
    result.params = initial_params;
    result.iterations = 0;
    result.converged = false;

    float prev_likelihood = computeLikelihood(initial_params);
    result.final_likelihood = prev_likelihood;

    if (config.verbose) {
        printf("\n=== Gradient Descent Optimization ===\n");
        printf("Initial likelihood: %.6f\n", prev_likelihood);
        printParams("Initial", result.params);
    }

    for (int iter = 0; iter < config.max_iterations; iter++) {
        // 计算梯度
        float gradient[6];
        computeGradient(result.params, gradient);

        // 保存旧参数用于收敛判断
        GaussianParams old_params = result.params;

        // 更新参数
        updateParams(result.params, gradient);

        // 计算新似然
        float new_likelihood = computeLikelihood(result.params);
        result.final_likelihood = new_likelihood;
        result.iterations = iter + 1;

        // 打印进度
        if (config.verbose && (iter % 10 == 0 || iter == config.max_iterations - 1)) {
            printf("Iter %4d: L = %.6f", iter + 1, new_likelihood);
            printf(" | grad: [%.3e, %.3e, %.3e, %.3e, %.3e, %.3e]\n",
                   gradient[0], gradient[1], gradient[2],
                   gradient[3], gradient[4], gradient[5]);
        }

        // 检查收敛
        float param_change = paramsDiff(old_params, result.params);
        if (param_change < config.tolerance) {
            result.converged = true;
            if (config.verbose) {
                printf("\nConverged after %d iterations!\n", iter + 1);
            }
            break;
        }

        // 检查似然是否增加（应该下降）
        if (new_likelihood > prev_likelihood + 1.0f) {
            if (config.verbose) {
                printf("Warning: Likelihood increased! Consider reducing learning rate.\n");
            }
        }
        prev_likelihood = new_likelihood;
    }

    if (config.verbose) {
        printf("\nFinal likelihood: %.6f\n", result.final_likelihood);
        printParams("Final", result.params);
        printf("Converged: %s\n", result.converged ? "Yes" : "No");
        printf("=====================================\n\n");
    }

    return result;
}
