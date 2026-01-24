#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "model.h"
#include "fit_model.h"
#include "config.h"
#include <string>
#include <vector>

/**
 * 1维投影直方图数据
 */
struct Projection1D {
    std::vector<float> bin_centers;  // bin中心坐标
    std::vector<int> counts;          // 每个bin的计数
    float min_val;
    float max_val;
    int nbins;
};

/**
 * 可视化类 - 负责生成和保存拟合结果图
 */
class Visualizer {
private:
    std::string output_dir;

    /**
     * 创建输出目录
     */
    bool createOutputDirectory();

    /**
     * 2D高斯直方图投影到1D
     */
    Projection1D projectToX(const Histogram2D& hist);
    Projection1D projectToY(const Histogram2D& hist);

    /**
     * 计算1D高斯分布曲线
     * 对于2D高斯的边缘分布：
     * - X边缘: N(x; x0, sigma_x)
     * - Y边缘: N(y; y0, sigma_y)
     */
    std::vector<float> compute1DGaussianCurve(
        const Projection1D& proj,
        float mean,
        float sigma,
        float total_counts
    );

    /**
     * 保存配置信息到文件
     */
    bool saveConfig(const std::string& output_path, const Config& config);

    /**
     * 保存优化结果到文件
     */
    bool saveResults(const std::string& output_path,
                     const OptimizationResult& result,
                     const GaussianParams& true_params);

    /**
     * 保存性能数据（迭代时间）到CSV
     */
    bool saveTimingData(const std::string& output_path,
                        const std::vector<float>& iteration_times,
                        int save_interval);

public:
    Visualizer(const std::string& output_dir);

    /**
     * 生成并保存所有可视化结果
     */
    bool generateAllOutputs(
        const Histogram2D& hist,
        const OptimizationResult& result,
        const GaussianParams& true_params,
        const Config& config
    );

    /**
     * 绘制1维直方图和拟合曲线
     * - X方向投影
     * - Y方向投影
     */
    bool plot1DHistogramWithFit(
        const Histogram2D& hist,
        const GaussianParams& fit_params,
        const GaussianParams& true_params
    );

    /**
     * 绘制2D直方图热图
     */
    bool plot2DHistogram(const Histogram2D& hist);

    /**
     * 绘制迭代时间曲线
     */
    bool plotIterationTimes(
        const std::vector<float>& iteration_times,
        int save_interval
    );

    /**
     * 绘制似然值变化曲线
     */
    bool plotLikelihoodHistory(
        const std::vector<float>& likelihood_history
    );
};

#endif // VISUALIZATION_H
