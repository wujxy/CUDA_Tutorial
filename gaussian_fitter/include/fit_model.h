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

/**
 * 2D似然扫描结果
 */
struct LikelihoodScan2DResult {
    std::vector<float> x0_values;      // x0扫描点
    std::vector<float> y0_values;      // y0扫描点
    std::vector<float> likelihood;     // 似然值网格 (nx × ny)
    int nx;                            // x0方向扫描点数
    int ny;                            // y0方向扫描点数
    float min_likelihood;              // 最小似然值
    float x0_at_min;                   // 最小似然处的x0
    float y0_at_min;                   // 最小似然处的y0
    float x0_error_plus;               // x0正误差 (1σ)
    float x0_error_minus;              // x0负误差 (1σ)
    float y0_error_plus;               // y0正误差 (1σ)
    float y0_error_minus;              // y0负误差 (1σ)
};

/**
 * 2D似然函数扫描器
 * 用于绘制似然函数等值线和参数置信区域
 */
class LikelihoodScanner {
private:
    const int* d_observed;
    int nbins;
    float x_min, x_max, y_min, y_max;
    int nx, ny;
    float* d_expected;
    float* d_likelihood;

    float computeLikelihood(const GaussianParams& params);

public:
    LikelihoodScanner();
    ~LikelihoodScanner();

    void setData(const Histogram2D& hist);

    /**
     * 在(x0, y0)二维参数空间中扫描似然函数
     * @param fixed_params 固定的参数 (A, sigma_x, sigma_y, rho)
     * @param x0_center x0扫描中心
     * @param y0_center y0扫描中心
     * @param x0_range x0扫描范围 (±)
     * @param y0_range y0扫描范围 (±)
     * @param nx x0方向扫描点数
     * @param ny y0方向扫描点数
     */
    LikelihoodScan2DResult scan2D(
        const GaussianParams& fixed_params,
        float x0_center, float y0_center,
        float x0_range, float y0_range,
        int nx, int ny
    );
};

#endif // FIT_MODEL_H
