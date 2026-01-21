#ifndef MODEL_H
#define MODEL_H

/**
 * 2D高斯模型参数
 * 完整的2D高斯分布有6个自由参数
 */
struct GaussianParams {
    float A;        // 幅值 (amplitude) - 总强度
    float x0;       // x中心位置
    float y0;       // y中心位置
    float sigma_x;  // x方向标准差（宽度）
    float sigma_y;  // y方向标准差（宽度）
    float rho;      // 相关系数 (correlation, [-1, 1])
};

/**
 * 2D直方图数据
 */
struct Histogram2D {
    int* data;      // 观测计数 [nx * ny]
    int nx;         // x方向bin数量
    int ny;         // y方向bin数量
    float x_min;    // x范围最小值
    float x_max;    // x范围最大值
    float y_min;    // y范围最小值
    float y_max;    // y范围最大值
};

/**
 * 辅助函数：创建坐标数组（CPU端调用）
 */
void createCoordinateArrays(
    float* x_coords,
    float* y_coords,
    const Histogram2D& hist
);

#endif // MODEL_H
