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

/**
 * 计算总似然值
 */
float AdamOptimizer::computeLikelihood(const GaussianParams& params) {
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
void AdamOptimizer::computeGradient(const GaussianParams& params, float* gradient) {
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
void AdamOptimizer::applyConstraints(GaussianParams& params) {
    params.sigma_x = fmaxf(params.sigma_x, 0.1f);
    params.sigma_y = fmaxf(params.sigma_y, 0.1f);
    params.rho = fmaxf(-0.99f, fminf(params.rho, 0.99f));
    params.A = fmaxf(params.A, 0.0f);
}

/**
 * 初始化 Adam 优化器的动量和二阶矩
 */
void AdamOptimizer::initializeAdamParameters() {
    // 为6个参数初始化一阶矩（动量）和二阶矩
    for (int i = 0; i < 6; i++) {
        m[i] = 0.0f;
        v[i] = 0.0f;
    }
}

/**
 * 构造函数
 */
AdamOptimizer::AdamOptimizer(const OptimizerConfig& cfg)
    : config(cfg), d_observed(nullptr), nbins(0),
      x_min(0), x_max(0), y_min(0), y_max(0), nx(0), ny(0),
      d_expected(nullptr), d_likelihood(nullptr), d_gradient(nullptr),
      m(nullptr), v(nullptr)
{
    // 设置 Adam 默认参数（如果未在配置中指定）
    if (config.beta1 < 0.0f) config.beta1 = 0.9f;
    if (config.beta2 < 0.0f) config.beta2 = 0.999f;
    if (config.epsilon < 0.0f) config.epsilon = 1e-8f;
}

/**
 * 析构函数
 */
AdamOptimizer::~AdamOptimizer() {
    if (d_expected)   cudaFree(d_expected);
    if (d_likelihood) cudaFree(d_likelihood);
    if (d_gradient)   cudaFree(d_gradient);
    if (m)            delete[] m;
    if (v)            delete[] v;
}

/**
 * 设置数据
 */
void AdamOptimizer::setData(const Histogram2D& hist) {
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

    // 初始化 Adam 动量和二阶矩
    m = new float[6];
    v = new float[6];
    initializeAdamParameters();
}

/**
 * 执行优化
 */
OptimizationResult AdamOptimizer::optimize(const GaussianParams& initial_params) {
    OptimizationResult result;
    result.params = initial_params;
    result.iterations = 0;
    result.converged = false;

    float current_likelihood = computeLikelihood(initial_params);
    result.final_likelihood = current_likelihood;
    result.likelihood_history.push_back(current_likelihood);

    int skipped_count = 0;  // 连续跳过的次数

    if (config.verbose) {
        printf("\n=== Adam Optimization ===\n");
        printf("Initial likelihood: %.6f\n", current_likelihood);
        printf("Beta1: %.4f, Beta2: %.4f, Epsilon: %.2e\n", config.beta1, config.beta2, config.epsilon);
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

        // Adam 更新规则
        // 注意：Adam 有自适应学习率机制，直接使用原始梯度，不需要额外缩放
        // （Simple GD 需要缩放是因为不同参数的梯度尺度差异很大）
        const float* g = gradient;  // 直接使用原始梯度

        // 更新一阶矩和二阶矩
        float beta1 = config.beta1;
        float beta2 = config.beta2;

        for (int i = 0; i < 6; i++) {
            // m_t = β1 * m_{t-1} + (1-β1) * g_t
            m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];

            // v_t = β2 * v_{t-1} + (1-β2) * g_t²
            v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        }

        // 偏差修正（bias correction）
        float beta1_t = powf(beta1, iter + 1);
        float beta2_t = powf(beta2, iter + 1);
        float m_hat[6], v_hat[6];

        for (int i = 0; i < 6; i++) {
            m_hat[i] = m[i] / (1.0f - beta1_t);
            v_hat[i] = v[i] / (1.0f - beta2_t);
        }

        // 更新参数
        // θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
        result.params.A        -= config.learning_rate * m_hat[0] / (sqrtf(v_hat[0]) + config.epsilon);
        result.params.x0       -= config.learning_rate * m_hat[1] / (sqrtf(v_hat[1]) + config.epsilon);
        result.params.y0       -= config.learning_rate * m_hat[2] / (sqrtf(v_hat[2]) + config.epsilon);
        result.params.sigma_x  -= config.learning_rate * m_hat[3] / (sqrtf(v_hat[3]) + config.epsilon);
        result.params.sigma_y  -= config.learning_rate * m_hat[4] / (sqrtf(v_hat[4]) + config.epsilon);
        result.params.rho      -= config.learning_rate * m_hat[5] / (sqrtf(v_hat[5]) + config.epsilon);

        applyConstraints(result.params);

        // 计算新似然
        float new_likelihood = computeLikelihood(result.params);
        result.iterations = iter + 1;

        // 记录迭代时间
        auto iter_end = std::chrono::high_resolution_clock::now();
        float iter_time_ms = std::chrono::duration<float, std::milli>(iter_end - iter_start).count();

        // 只在指定间隔保存时间
        if (iter == 0 || (iter + 1) % config.timing_save_interval == 0) {
            result.iteration_times.push_back(iter_time_ms);
        }

        // 检查是否跳过这次更新
        float likelihood_change = new_likelihood - old_likelihood;
        float relative_change = fabsf(likelihood_change) / (fabsf(old_likelihood) + 1e-10f);
        bool is_warmup = (iter < 100);

        // 计算参数变化幅度
        float param_change = fabsf(result.params.A - old_params.A) / (fabsf(old_params.A) + 1e-10f) +
                            fabsf(result.params.x0 - old_params.x0) +
                            fabsf(result.params.y0 - old_params.y0) +
                            fabsf(result.params.sigma_x - old_params.sigma_x) +
                            fabsf(result.params.sigma_y - old_params.sigma_y) +
                            fabsf(result.params.rho - old_params.rho);

        if (new_likelihood > old_likelihood && relative_change > 1e-8f && !is_warmup && param_change > 1e-8f) {
            result.params = old_params;
            current_likelihood = old_likelihood;
            result.likelihood_history.push_back(current_likelihood);
            skipped_count++;

            if (config.verbose && iter % 100 == 0) {
                printf("Iter %4d: Skipped (likelihood increased: %.6e -> %.6e)\n",
                       iter + 1, old_likelihood, new_likelihood);
            }
        } else {
            current_likelihood = new_likelihood;
            result.final_likelihood = new_likelihood;
            result.likelihood_history.push_back(new_likelihood);
            skipped_count = 0;

            if (config.verbose && iter % 100 == 0) {
                printf("Iter %4d: L = %.6f, LR = %.6e, Time = %.2f ms\n",
                       iter + 1, new_likelihood, config.learning_rate, iter_time_ms);
            }

            // 检查收敛
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

// ============================================================================
// LikelihoodScanner Implementation
// ============================================================================

LikelihoodScanner::LikelihoodScanner()
    : d_observed(nullptr), nbins(0),
      x_min(0), x_max(0), y_min(0), y_max(0), nx(0), ny(0),
      d_expected(nullptr), d_likelihood(nullptr)
{
}

LikelihoodScanner::~LikelihoodScanner() {
    if (d_expected) cudaFree(d_expected);
    if (d_likelihood) cudaFree(d_likelihood);
}

void LikelihoodScanner::setData(const Histogram2D& hist) {
    this->d_observed = hist.data;
    this->nbins = hist.nx * hist.ny;
    this->nx = hist.nx;
    this->ny = hist.ny;
    this->x_min = hist.x_min;
    this->x_max = hist.x_max;
    this->y_min = hist.y_min;
    this->y_max = hist.y_max;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_expected, nbins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_likelihood, nbins * sizeof(float)));
}

float LikelihoodScanner::computeLikelihood(const GaussianParams& params) {
    // 计算期望值
    int threadsPerBlock = 256;
    int blocksPerGrid = (nbins + threadsPerBlock - 1) / threadsPerBlock;
    poissonLikelihoodKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_expected, d_likelihood, d_observed,
        params, nbins,
        x_min, x_max, nx,
        y_min, y_max, ny
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 规约求和 (与 SimpleOptimizer 相同的方法)
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

LikelihoodScan2DResult LikelihoodScanner::scan2D(
    const GaussianParams& fixed_params,
    float x0_center, float y0_center,
    float x0_range, float y0_range,
    int nx, int ny
) {
    LikelihoodScan2DResult result;
    result.nx = nx;
    result.ny = ny;
    result.likelihood.resize(nx * ny);
    result.x0_values.resize(nx);
    result.y0_values.resize(ny);

    // 生成扫描点
    for (int i = 0; i < nx; i++) {
        result.x0_values[i] = x0_center - x0_range + 2.0f * x0_range * i / (nx - 1);
    }
    for (int j = 0; j < ny; j++) {
        result.y0_values[j] = y0_center - y0_range + 2.0f * y0_range * j / (ny - 1);
    }

    result.min_likelihood = 1e30f;
    int min_idx = 0;

    // 扫描所有 (x0, y0) 组合
    printf("\n=== 2D Likelihood Scan ===\n");
    printf("Scanning %dx%d = %d points...\n", nx, ny, nx * ny);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            GaussianParams params = fixed_params;
            params.x0 = result.x0_values[i];
            params.y0 = result.y0_values[j];

            float likelihood = computeLikelihood(params);
            result.likelihood[j * nx + i] = likelihood;

            if (likelihood < result.min_likelihood) {
                result.min_likelihood = likelihood;
                result.x0_at_min = params.x0;
                result.y0_at_min = params.y0;
                min_idx = j * nx + i;
            }
        }
        printf("Progress: %d/%d rows completed\n", j + 1, ny);
    }

    printf("Minimum likelihood: %.6f at (x0=%.4f, y0=%.4f)\n",
           result.min_likelihood, result.x0_at_min, result.y0_at_min);

    // 使用 Δχ² = 1 确定 1σ 置信区域
    float threshold = result.min_likelihood + 1.0f;

    // 找到最小值在网格中的位置
    int min_idx_i = 0, min_idx_j = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (result.likelihood[j * nx + i] < result.likelihood[min_idx_j * nx + min_idx_i]) {
                min_idx_i = i;
                min_idx_j = j;
            }
        }
    }

    // 在 y0 ≈ y0_at_min 这一行找 x0 的误差
    // 向左找（likelihood 从小于阈值到大于阈值）
    result.x0_error_minus = x0_range;  // 默认值：扫描范围
    for (int i = min_idx_i - 1; i >= 0; i--) {
        int idx = min_idx_j * nx + i;
        if (result.likelihood[idx] >= threshold) {
            // 在 i 和 i+1 之间插值
            float x1 = result.x0_values[i];
            float x2 = result.x0_values[i + 1];
            float y1 = result.likelihood[idx];
            float y2 = result.likelihood[idx + 1];

            float t = (threshold - y1) / (y2 - y1);
            float x_intersect = x1 + t * (x2 - x1);
            result.x0_error_minus = result.x0_at_min - x_intersect;
            break;
        }
    }

    // 向右找
    result.x0_error_plus = x0_range;  // 默认值：扫描范围
    for (int i = min_idx_i + 1; i < nx; i++) {
        int idx = min_idx_j * nx + i;
        if (result.likelihood[idx] >= threshold) {
            // 在 i-1 和 i 之间插值
            float x1 = result.x0_values[i - 1];
            float x2 = result.x0_values[i];
            float y1 = result.likelihood[idx - 1];
            float y2 = result.likelihood[idx];

            float t = (threshold - y1) / (y2 - y1);
            float x_intersect = x1 + t * (x2 - x1);
            result.x0_error_plus = x_intersect - result.x0_at_min;
            break;
        }
    }

    // 在 x0 ≈ x0_at_min 这一列找 y0 的误差
    // 向下找
    result.y0_error_minus = y0_range;  // 默认值：扫描范围
    for (int j = min_idx_j - 1; j >= 0; j--) {
        int idx = j * nx + min_idx_i;
        if (result.likelihood[idx] >= threshold) {
            // 在 j 和 j+1 之间插值
            float y1 = result.y0_values[j];
            float y2 = result.y0_values[j + 1];
            float l1 = result.likelihood[idx];
            float l2 = result.likelihood[idx + nx];

            float t = (threshold - l1) / (l2 - l1);
            float y_intersect = y1 + t * (y2 - y1);
            result.y0_error_minus = result.y0_at_min - y_intersect;
            break;
        }
    }

    // 向上找
    result.y0_error_plus = y0_range;  // 默认值：扫描范围
    for (int j = min_idx_j + 1; j < ny; j++) {
        int idx = j * nx + min_idx_i;
        if (result.likelihood[idx] >= threshold) {
            // 在 j-1 和 j 之间插值
            float y1 = result.y0_values[j - 1];
            float y2 = result.y0_values[j];
            float l1 = result.likelihood[idx - nx];
            float l2 = result.likelihood[idx];

            float t = (threshold - l1) / (l2 - l1);
            float y_intersect = y1 + t * (y2 - y1);
            result.y0_error_plus = y_intersect - result.y0_at_min;
            break;
        }
    }

    printf("Error estimates (Δχ² = 1):\n");
    printf("  x0: %.4f + %.4f - %.4f\n", result.x0_at_min, result.x0_error_plus, result.x0_error_minus);
    printf("  y0: %.4f + %.4f - %.4f\n", result.y0_at_min, result.y0_error_plus, result.y0_error_minus);
    printf("=====================================\n\n");

    return result;
}
