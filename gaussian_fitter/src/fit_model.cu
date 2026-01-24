#include "../include/fit_model.h"
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
float SimpleOptimizer::computeLikelihood(const GaussianParams& params) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (nbins + threadsPerBlock - 1) / threadsPerBlock;

    // 计算每个bin的似然
    poissonLikelihoodKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_expected, d_likelihood, d_observed, params, nbins,
        x_min, x_max, nx, y_min, y_max, ny
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 规约求和
    int num_blocks = blocksPerGrid;
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, num_blocks * sizeof(float)));

    sumKernel<<<num_blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_temp, d_likelihood, nbins
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    while (num_blocks > 1) {
        int prev_blocks = num_blocks;
        num_blocks = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;

        sumKernel<<<num_blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_expected, d_temp, prev_blocks
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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
void SimpleOptimizer::computeGradient(const GaussianParams& params, float* gradient) {
    const int threadsPerBlock = 256;

    // 使用CUDA计算所有6个参数的梯度
    numericalGradientKernel<<<6, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_gradient, d_observed, params, config.gradient_epsilon, nbins,
        x_min, x_max, nx, y_min, y_max, ny
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝回主机
    CUDA_CHECK(cudaMemcpy(gradient, d_gradient, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

/**
 * 应用参数约束
 */
void SimpleOptimizer::applyConstraints(GaussianParams& params) {
    params.sigma_x = fmaxf(params.sigma_x, 0.1f);
    params.sigma_y = fmaxf(params.sigma_y, 0.1f);
    params.rho = fmaxf(-0.99f, fminf(params.rho, 0.99f));
    params.A = fmaxf(params.A, 0.0f);
}

/**
 * 构造函数
 */
SimpleOptimizer::SimpleOptimizer(const OptimizerConfig& cfg)
    : config(cfg), d_observed(nullptr), nbins(0),
      x_min(0), x_max(0), y_min(0), y_max(0), nx(0), ny(0),
      d_expected(nullptr), d_likelihood(nullptr), d_gradient(nullptr)
{
}

/**
 * 析构函数
 */
SimpleOptimizer::~SimpleOptimizer() {
    if (d_expected)   cudaFree(d_expected);
    if (d_likelihood) cudaFree(d_likelihood);
    if (d_gradient)   cudaFree(d_gradient);
}

/**
 * 设置数据
 */
void SimpleOptimizer::setData(const Histogram2D& hist) {
    nbins = hist.nx * hist.ny;

    x_min = hist.x_min;
    x_max = hist.x_max;
    y_min = hist.y_min;
    y_max = hist.y_max;
    nx = hist.nx;
    ny = hist.ny;

    CUDA_CHECK(cudaMallocManaged(&d_expected, nbins * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_likelihood, nbins * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_gradient, 6 * sizeof(float)));

    d_observed = hist.data;
}

/**
 * 执行优化
 */
OptimizationResult SimpleOptimizer::optimize(const GaussianParams& initial_params) {
    OptimizationResult result;
    result.params = initial_params;
    result.iterations = 0;
    result.converged = false;

    float current_likelihood = computeLikelihood(initial_params);
    result.final_likelihood = current_likelihood;
    result.likelihood_history.push_back(current_likelihood);

    int skipped_count = 0;  // 连续跳过的次数

    if (config.verbose) {
        printf("\n=== Gradient Descent Optimization ===\n");
        printf("Initial likelihood: %.6f\n", current_likelihood);
        printParams("Initial", result.params);
    }

    for (int iter = 0; iter < config.max_iterations; iter++) {
        // 记录迭代开始时间
        auto iter_start = std::chrono::high_resolution_clock::now();

        // 计算梯度
        float gradient[6];
        computeGradient(result.params, gradient);

        // 保存旧参数
        GaussianParams old_params = result.params;
        float old_likelihood = current_likelihood;

        // 更新参数（使用较小的固定步长）
        result.params.A        -= config.learning_rate * gradient[0];
        result.params.x0       -= config.learning_rate * gradient[1] * 0.01f;
        result.params.y0       -= config.learning_rate * gradient[2] * 0.01f;
        result.params.sigma_x  -= config.learning_rate * gradient[3] * 0.001f;
        result.params.sigma_y  -= config.learning_rate * gradient[4] * 0.001f;
        result.params.rho      -= config.learning_rate * gradient[5] * 0.01f;

        applyConstraints(result.params);

        // 计算新似然
        float new_likelihood = computeLikelihood(result.params);
        result.iterations = iter + 1;

        // 记录迭代时间（只在指定间隔保存）
        auto iter_end = std::chrono::high_resolution_clock::now();
        float iter_time_ms = std::chrono::duration<float, std::milli>(iter_end - iter_start).count();

        // 只在指定间隔保存时间（第一次总是保存，之后每timing_save_interval次保存一次）
        if (iter == 0 || (iter + 1) % config.timing_save_interval == 0) {
            result.iteration_times.push_back(iter_time_ms);
        }

        // 检查是否跳过这次更新
        float likelihood_change = new_likelihood - old_likelihood;
        float relative_change = fabsf(likelihood_change) / (fabsf(old_likelihood) + 1e-10f);
        bool is_warmup = (iter < 100);  // 前100次迭代作为预热期

        // 计算参数变化幅度（使用相对变化）
        float param_change = fabsf(result.params.A - old_params.A) / (fabsf(old_params.A) + 1e-10f) +
                            fabsf(result.params.x0 - old_params.x0) +
                            fabsf(result.params.y0 - old_params.y0) +
                            fabsf(result.params.sigma_x - old_params.sigma_x) +
                            fabsf(result.params.sigma_y - old_params.sigma_y) +
                            fabsf(result.params.rho - old_params.rho);

        if (new_likelihood > old_likelihood && relative_change > 1e-8f && !is_warmup && param_change > 1e-8f) {
            // 似然显著增加且参数有实际变化，恢复旧参数
            result.params = old_params;
            current_likelihood = old_likelihood;
            result.likelihood_history.push_back(current_likelihood);
            skipped_count++;

            if (config.verbose && iter % 100 == 0) {
                printf("Iter %4d: Skipped (likelihood increased: %.6e -> %.6e)\n",
                       iter + 1, old_likelihood, new_likelihood);
            }
        } else {
            // 接受新参数
            current_likelihood = new_likelihood;
            result.final_likelihood = new_likelihood;
            result.likelihood_history.push_back(new_likelihood);
            skipped_count = 0;

            if (config.verbose && iter % 100 == 0) {
                printf("Iter %4d: L = %.6f, LR = %.6e, Time = %.2f ms\n",
                       iter + 1, new_likelihood, config.learning_rate, iter_time_ms);
            }

            // 检查收敛（使用相对变化）
            if (!is_warmup && iter >= 100) {
                if (param_change < config.tolerance) {
                    result.converged = true;
                    if (config.verbose) {
                        printf("\nConverged after %d iterations!\n", iter + 1);
                    }
                    break;
                }
            }
        }

        // 如果连续跳过太多，说明可能已经收敛
        if (skipped_count > 500) {
            result.converged = true;
            if (config.verbose) {
                printf("\nConverged (skipped too many iterations)!\n");
            }
            break;
        }
    }

    if (config.verbose) {
        printf("\nFinal likelihood: %.6f\n", result.final_likelihood);
        printParams("Final", result.params);
        printf("Converged: %s\n", result.converged ? "Yes" : "No");

        // 计算平均迭代时间
        if (!result.iteration_times.empty()) {
            float total_time = 0.0f;
            for (float t : result.iteration_times) {
                total_time += t;
            }
            printf("Average iteration time: %.2f ms\n", total_time / result.iteration_times.size());
            printf("Total optimization time: %.2f ms\n", total_time);
        }
        printf("=====================================\n\n");
    }

    return result;
}
