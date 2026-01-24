#ifndef FIT_MODEL_H
#define FIT_MODEL_H

#include "model.h"
#include <vector>
#include <chrono>

/**
 * 优化器配置
 */
struct OptimizerConfig {
    float learning_rate;     // 学习率
    int max_iterations;      // 最大迭代次数
    float tolerance;         // 收敛容差
    float gradient_epsilon;  // 数值微分的扰动步长
    bool verbose;            // 是否打印详细信息
};

/**
 * 优化结果
 */
struct OptimizationResult {
    GaussianParams params;   // 最优参数
    float final_likelihood;  // 最终似然值
    int iterations;          // 实际迭代次数
    bool converged;          // 是否收敛

    // 性能监测数据
    std::vector<float> iteration_times;  // 每次迭代的时间（毫秒）
    std::vector<float> likelihood_history;  // 每次迭代的似然值
};

/**
 * 简单的梯度下降优化器
 * 使用CUDA计算梯度，CPU端更新参数
 */
class SimpleOptimizer {
private:
    OptimizerConfig config;

    // 直方图数据
    const int* d_observed;
    int nbins;

    // 直方图边界信息
    float x_min, x_max;
    float y_min, y_max;
    int nx, ny;

    // 临时内存（设备端）
    float* d_expected;
    float* d_likelihood;
    float* d_gradient;

    float computeLikelihood(const GaussianParams& params);
    void computeGradient(const GaussianParams& params, float* gradient);
    void applyConstraints(GaussianParams& params);

public:
    SimpleOptimizer(const OptimizerConfig& cfg);
    ~SimpleOptimizer();

    void setData(const Histogram2D& hist);
    OptimizationResult optimize(const GaussianParams& initial_params);
};

#endif // FIT_MODEL_H
