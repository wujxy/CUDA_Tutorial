#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>

#include "include/model.h"
#include "include/fit_model.h"
#include "include/config.h"
#include "include/cuda_utils.h"
#include "include/visualization.h"

using namespace std;

/**
 * 使用撒点方式生成2D高斯直方图
 * 生成多组(x, y)样本点，然后填充到直方图bins中
 */
Histogram2D generateGaussianHistogram(const GaussianParams& params, int nx, int ny,
                                   float x_min, float x_max, float y_min, float y_max,
                                   int num_samples) {
    Histogram2D hist;
    hist.nx = nx;
    hist.ny = ny;
    hist.x_min = x_min;
    hist.x_max = x_max;
    hist.y_min = y_min;
    hist.y_max = y_max;

    int nbins = nx * ny;
    CUDA_CHECK(cudaMallocManaged(&hist.data, nbins * sizeof(int)));

    // 初始化直方图为0
    for (int i = 0; i < nbins; i++) {
        hist.data[i] = 0;
    }

    // 使用C++标准库生成高斯分布样本
    random_device rd;
    mt19937 gen(rd());

    // 生成2D高斯样本（Cholesky分解）
    float sigma_x = params.sigma_x;
    float sigma_y = params.sigma_y;
    float rho = params.rho;

    float l11 = sigma_x;
    float l21 = rho * sigma_y;
    float l22 = sigma_y * sqrtf(1.0f - rho * rho);

    normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_samples; i++) {
        float z1 = dist(gen);
        float z2 = dist(gen);

        float x = params.x0 + l11 * z1;
        float y = params.y0 + l21 * z1 + l22 * z2;

        // 计算所属的bin
        if (x >= hist.x_min && x < hist.x_max &&
            y >= hist.y_min && y < hist.y_max) {

            float dx = (hist.x_max - hist.x_min) / nx;
            float dy = (hist.y_max - hist.y_min) / ny;

            int ix = static_cast<int>((x - hist.x_min) / dx);
            int iy = static_cast<int>((y - hist.y_min) / dy);

            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                int idx = iy * nx + ix;
                hist.data[idx]++;
            }
        }
    }

    // 计算总counts
    int total_counts = 0;
    for (int i = 0; i < nbins; i++) {
        total_counts += hist.data[i];
    }

    cout << "Generated histogram using " << num_samples << " samples" << endl;
    cout << "Histogram size: " << nx << "x" << ny << " bins" << endl;
    cout << "Total counts: " << total_counts << endl;

    return hist;
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
 * 打印使用帮助
 */
void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [config_file]" << endl;
    cout << endl;
    cout << "Arguments:" << endl;
    cout << "  config_file  Path to configuration file (default: config_default.conf)" << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  " << program_name << "                    # Use default config" << endl;
    cout << "  " << program_name << " my_config.conf     # Use custom config" << endl;
}

/**
 * 主函数
 */
int main(int argc, char* argv[]) {
    cout << "========================================" << endl;
    cout << "  CUDA 2D Gaussian Fit" << endl;
    cout << "========================================" << endl;

    // 解析命令行参数
    string config_file = "config_default.conf";
    if (argc > 1) {
        config_file = argv[1];
    }

    // 显示配置文件路径
    cout << "\nConfig file: " << config_file << endl;

    // 加载配置
    Config config;
    if (!config.loadFromFile(config_file)) {
        cerr << "Failed to load config file, using default values." << endl;
    }

    // 打印配置信息
    config.print();

    // 使用配置中的参数

    // Step 1: 使用撒点方式生成测试数据
    Histogram2D hist = generateGaussianHistogram(
        config.true_params,
        config.nx, config.ny,
        config.x_min, config.x_max,
        config.y_min, config.y_max,
        config.num_samples
    );

    cout << "\nInitial guess:" << endl;
    printParams("Initial", config.initial_guess);

    // Step 2: 配置优化器
    OptimizerConfig opt_config;
    opt_config.learning_rate = config.learning_rate;
    opt_config.max_iterations = config.max_iterations;
    opt_config.tolerance = config.tolerance;
    opt_config.gradient_epsilon = config.gradient_epsilon;
    opt_config.verbose = config.verbose;

    // Step 3: 创建优化器并执行拟合
    SimpleOptimizer optimizer(opt_config);
    optimizer.setData(hist);

    cout << "\nStarting optimization..." << endl;
    OptimizationResult result = optimizer.optimize(config.initial_guess);

    // Step 4: 打印结果比较
    printComparison(config.true_params, result.params);

    // Step 5: 生成可视化输出
    if (config.save_plots) {
        cout << "\nGenerating visualization outputs..." << endl;
        Visualizer viz(config.output_dir);
        viz.generateAllOutputs(hist, result, config.true_params, config);
    }

    // 清理
    cudaFree(hist.data);

    return 0;
}
