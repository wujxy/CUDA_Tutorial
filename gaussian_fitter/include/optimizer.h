#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "model.h"

/**
 * 梯度下降优化器配置
 */
struct OptimizerConfig {
    float learning_rate;     // 学习率（步长）
    int max_iterations;      // 最大迭代次数
    float tolerance;         // 收敛容差（参数变化小于此值时停止）
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
};

/**
 * 梯度下降优化器类
 *
 * 使用数值微分计算梯度，然后执行梯度下降更新:
 * θ_{new} = θ_{old} - lr * ∇L(θ)
 */
class GradientDescentOptimizer {
private:
    OptimizerConfig config;

    // 直方图数据（设备端指针）
    const int* d_observed;
    const float* d_x_coords;
    const float* d_y_coords;
    int nbins;

    // 临时内存（设备端）
    float* d_expected;
    float* d_likelihood;
    float* d_gradient;

    /**
     * 计算当前参数的总似然值
     * 调用CUDA kernel并返回规约后的结果
     */
    float computeLikelihood(const GaussianParams& params);

    /**
     * 通过数值微分计算梯度
     * 对每个参数扰动epsilon，计算似然变化
     */
    void computeGradient(const GaussianParams& params, float* gradient);

    /**
     * 应用参数约束
     * 确保sigma_x, sigma_y > 0 且 |rho| < 1
     */
    void applyConstraints(GaussianParams& params);

    /**
     * 参数更新步（梯度下降）
     */
    void updateParams(GaussianParams& params, const float* gradient);

public:
    /**
     * 构造函数
     */
    GradientDescentOptimizer(const OptimizerConfig& cfg);

    /**
     * 析构函数（释放设备内存）
     */
    ~GradientDescentOptimizer();

    /**
     * 设置数据（将数据拷贝到设备）
     */
    void setData(const Histogram2D& hist);

    /**
     * 执行优化
     *
     * 参数:
     *   initial_params - 初始参数猜测
     *
     * 返回:
     *   OptimizationResult - 优化结果
     */
    OptimizationResult optimize(const GaussianParams& initial_params);
};

#endif // OPTIMIZER_H
