#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

#include "include/model.h"
#include "include/optimizer.h"
#include "include/cuda_utils.h"

using namespace std;

/**
 * 生成泊松随机数
 */
int poissonRandom(float lambda, mt19937& gen) {
    if (lambda < 10.0f) {
        // Knuth算法（适合小lambda）
        float L = expf(-lambda);
        int k = 0;
        float p = 1.0f;
        do {
            k++;
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            p *= dist(gen);
        } while (p > L);
        return k - 1;
    } else {
        // 正态近似（适合大lambda）
        normal_distribution<float> norm(0.0f, 1.0f);
        return static_cast<int>(lambda + sqrtf(lambda) * norm(gen));
    }
}

/**
 * Step 1: 生成2D高斯分布测试数据
 * 使用已知参数生成理想高斯分布，然后添加泊松噪声
 */
Histogram2D generateTestData(const GaussianParams& true_params, int nx, int ny) {
    Histogram2D hist;
    hist.nx = nx;
    hist.ny = ny;
    hist.x_min = -5.0f;
    hist.x_max = 5.0f;
    hist.y_min = -5.0f;
    hist.y_max = 5.0f;

    int nbins = nx * ny;
    CUDA_CHECK(cudaMallocManaged(&hist.data, nbins * sizeof(int)));

    // 生成坐标
    float* x_coords;
    float* y_coords;
    CUDA_CHECK(cudaMallocManaged(&x_coords, nbins * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&y_coords, nbins * sizeof(float)));
    createCoordinateArrays(x_coords, y_coords, hist);

    // 随机数生成器
    random_device rd;
    mt19937 gen(rd());

    // 计算理想高斯值并添加泊松噪声
    // 使用与模型相同的2D高斯函数（包含rho项）
    float sum_counts = 0.0f;
    for (int i = 0; i < nbins; i++) {
        // 使用完整的2D高斯公式（与CUDA kernel一致）
        float dx = x_coords[i] - true_params.x0;
        float dy = y_coords[i] - true_params.y0;

        float sigma_x = true_params.sigma_x;
        float sigma_y = true_params.sigma_y;
        float rho = true_params.rho;

        float inv_1_minus_rho2 = 1.0f / (1.0f - rho * rho);
        float dx_sigma = dx / sigma_x;
        float dy_sigma = dy / sigma_y;

        float Q = inv_1_minus_rho2 * (
            dx_sigma * dx_sigma +
            dy_sigma * dy_sigma -
            2.0f * rho * dx_sigma * dy_sigma
        );

        float lambda = true_params.A * expf(-0.5f * Q);
        lambda = fmaxf(lambda, 0.0f);  // 确保非负

        // 添加泊松噪声
        hist.data[i] = poissonRandom(lambda, gen);
        sum_counts += hist.data[i];
    }

    cout << "Generated test data with " << nx << "x" << ny << " bins" << endl;
    cout << "Total counts: " << static_cast<int>(sum_counts) << endl;

    cudaFree(x_coords);
    cudaFree(y_coords);

    return hist;
}

/**
 * Step 2: 创建泊松似然拟合器
 * （已在optimizer.cu中实现）
 */

/**
 * Step 3: 执行拟合程序（bin-to-bin在CUDA上计算）
 */
OptimizationResult runFitting(const Histogram2D& hist, const GaussianParams& initial_guess) {
    // 配置优化器 - 调整后的参数
    OptimizerConfig config;
    config.learning_rate = 0.01f;      // 增加学习率（因为梯度已被归一化）
    config.max_iterations = 500;       // 更多迭代
    config.tolerance = 1e-6f;          // 严格收敛容差
    config.gradient_epsilon = 1e-3f;   // 相对步长
    config.verbose = true;             // 打印详细输出

    // 创建优化器
    GradientDescentOptimizer optimizer(config);
    optimizer.setData(hist);

    // 执行优化
    return optimizer.optimize(initial_guess);
}

/**
 * 比较拟合参数与真实参数
 */
void printComparison(const GaussianParams& true_params, const GaussianParams& fit_params) {
    cout << "\n=== Parameter Comparison ===" << endl;
    cout << left << setw(12) << "Parameter"
         << setw(15) << "True Value"
         << setw(15) << "Fitted Value"
         << setw(15) << "Error (%)" << endl;
    cout << string(57, '-') << endl;

    auto printRow = [](const char* name, float true_val, float fit_val) {
        float error = (fit_val - true_val) / true_val * 100.0f;
        cout << left << setw(12) << name
             << setw(15) << true_val
             << setw(15) << fit_val
             << setw(14) << fixed << setprecision(2) << error << "%" << endl;
    };

    printRow("A", true_params.A, fit_params.A);
    printRow("x0", true_params.x0, fit_params.x0);
    printRow("y0", true_params.y0, fit_params.y0);
    printRow("sigma_x", true_params.sigma_x, fit_params.sigma_x);
    printRow("sigma_y", true_params.sigma_y, fit_params.sigma_y);
    printRow("rho", true_params.rho, fit_params.rho);

    cout << string(57, '-') << endl;
}

/**
 * 主函数
 */
int main() {
    cout << "========================================" << endl;
    cout << "  CUDA Poisson Likelihood Gaussian Fit" << endl;
    cout << "========================================" << endl;

    // 定义真实参数 - 简化版本（rho=0）
    GaussianParams true_params;
    true_params.A = 1000.0f;
    true_params.x0 = 0.0f;
    true_params.y0 = 0.0f;
    true_params.sigma_x = 1.0f;
    true_params.sigma_y = 1.5f;
    true_params.rho = 0.0f;  // 不相关

    cout << "\nTrue parameters:" << endl;
    printParams("True", true_params);

    // Step 1: 生成测试数据
    constexpr int NX = 64;
    constexpr int NY = 64;
    Histogram2D hist = generateTestData(true_params, NX, NY);

    // 初始猜测 - 接近真实值
    GaussianParams initial_guess;
    initial_guess.A = 900.0f;
    initial_guess.x0 = 0.1f;
    initial_guess.y0 = -0.1f;
    initial_guess.sigma_x = 0.9f;
    initial_guess.sigma_y = 1.4f;
    initial_guess.rho = 0.0f;  // 固定为0

    cout << "\nInitial guess:" << endl;
    printParams("Initial", initial_guess);

    // Step 2 & 3: 运行拟合（CUDA计算期望值和梯度）
    OptimizationResult result = runFitting(hist, initial_guess);

    // 打印结果比较
    printComparison(true_params, result.params);

    // 清理
    cudaFree(hist.data);

    return 0;
}
