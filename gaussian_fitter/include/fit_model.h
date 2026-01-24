#ifndef FIT_MODEL_H
#define FIT_MODEL_H

#include "model.h"
#include <vector>
#include <chrono>

/**
 * 优化器类型枚举
 */
enum class OptimizerType {
    SIMPLE = 0,   // 简单梯度下降
    ADAM = 1       // Adam优化器
};

/**
 * 优化器配置
 */
struct OptimizerConfig {
    OptimizerType optimizer_type;  // 优化器类型
    float learning_rate;           // 学习率
    int max_iterations;            // 最大迭代次数
    float tolerance;               // 收敛容差
    float gradient_epsilon;        // 数值微分的扰动步长
    bool verbose;                  // 是否打印详细信息
    int timing_save_interval;     // 保存迭代时间的间隔（默认1，即每次都保存）

    // Adam优化器特有参数
    float beta1;                   // Adam: 一阶矩估计的指数衰减率
    float beta2;                   // Adam: 二阶矩估计的指数衰减率
    float epsilon;                  // Adam: 避免除零的小常数
};

/**
 * 优化结果
 */
struct OptimizationResult {
    GaussianParams params;        // 最优参数
    float final_likelihood;         // 最终似然值
    int iterations;                // 实际迭代次数
    bool converged;                // 是否收敛

    // 性能监测数据
    std::vector<float> iteration_times;  // 每次迭代的时间（毫秒）
    std::vector<float> likelihood_history;  // 每次迭代的似然值
};

/**
 * 优化器基类
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void setData(const Histogram2D& hist) = 0;
    virtual OptimizationResult optimize(const GaussianParams& initial_params) = 0;
};

/**
 * 简单的梯度下降优化器
 * 使用CUDA计算梯度，CPU端更新参数
 */
class SimpleOptimizer : public Optimizer {
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

    void setData(const Histogram2D& hist) override;
    OptimizationResult optimize(const GaussianParams& initial_params) override;
};

/**
 * Adam优化器
 * 自适应矩估计优化器
 */
class AdamOptimizer : public Optimizer {
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

    // Adam优化器状态
    float* m;  // 一阶矩估计（动量）
    float* v;  // 二阶矩估计

    float computeLikelihood(const GaussianParams& params);
    void computeGradient(const GaussianParams& params, float* gradient);
    void applyConstraints(GaussianParams& params);
    void initializeAdamParameters();

public:
    AdamOptimizer(const OptimizerConfig& cfg);
    ~AdamOptimizer();

    void setData(const Histogram2D& hist) override;
    OptimizationResult optimize(const GaussianParams& initial_params) override;
};

#endif // FIT_MODEL_H
