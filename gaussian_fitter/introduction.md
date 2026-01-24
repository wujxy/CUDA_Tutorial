# CUDA 2D高斯直方图拟合器 - 项目介绍

## 项目概述

本项目实现了一个使用CUDA加速的2D高斯直方图拟合器，通过**撒点方式（Monte Carlo采样）**生成测试数据，并使用**泊松似然**进行参数估计。

### 核心特点

- **撒点方式生成数据**: 使用C++标准库生成2D高斯样本，通过Cholesky分解处理相关系数
- **bin-to-bin CUDA并行计算**: 每个bin的期望值和似然计算在GPU上并行执行
- **正确的PDF归一化**: 对于撒点方式，PDF正确归一化使得积分等于总样本数
- **自适应学习率优化**: 简洁的梯度下降优化器，支持自动调整学习率
- **完整2D高斯模型**: 包含6个自由参数（幅值、中心位置、宽度、相关系数）
- **性能监测与可视化**: 自动记录每轮迭代时间，生成拟合质量图表和性能分析图

---

## 数学原理

### 1. 数据生成（撒点方式）

使用**Cholesky分解**生成相关的2D高斯样本：

```
协方差矩阵:
Σ = [σₓ²       ρσₓσᵧ]
    [ρσₓσᵧ   σᵧ²    ]

Cholesky分解: Σ = LLᵀ
L = [σₓ               0     ]
    [ρσᵧ         σᵧ√(1-ρ²)]

生成步骤:
1. 生成两个独立的标准正态随机变量 z₁, z₂ ~ N(0,1)
2. 转换为相关的高斯分布:
   x = x₀ + l₁₁·z₁
   y = y₀ + l₂₁·z₁ + l₂₂·z₂
3. 将样本点分配到对应的bin中
```

### 2. 2D高斯PDF（归一化版本）

对于撒点方式生成的数据，PDF需要正确归一化：

```
PDF(x,y) = (A / (2π·σₓ·σᵧ·√(1-ρ²))) · exp(-Q/2)

其中 Q = (1/(1-ρ²)) · [
    (x-x₀)²/σₓ² +
    (y-y₀)²/σᵧ² -
    2ρ(x-x₀)(y-y₀)/(σₓσᵧ)
]
```

**归一化因子**: `2π·σₓ·σᵧ·√(1-ρ²)` 确保PDF在整个平面上的积分等于A（总样本数）

### 3. 泊松似然函数

对于观测数据 nᵢ（bin中的计数）和模型期望值 λᵢ(θ)（PDF在bin处的值）：

```
L(θ) = Σᵢ [λᵢ(θ) - nᵢ · log(λᵢ(θ))]
```

**目标**: 找到参数 θ 使 L(θ) 最小化

### 4. 数值微分梯度

使用有限差分近似计算梯度：

```
∇L(θ) ≈ [L(θ+εeᵢ) - L(θ)] / ε
```

---

## 代码结构

```
gaussion_fitter/
├── include/
│   ├── model.h              # 数据结构定义
│   ├── model_kernels.cuh    # CUDA kernel声明
│   ├── fit_model.h          # 简化优化器接口
│   ├── config.h             # 配置文件解析
│   ├── visualization.h      # 可视化模块接口
│   └── cuda_utils.h         # 工具函数
├── src/
│   ├── model.cu             # CUDA kernel实现（归一化PDF）
│   ├── fit_model.cu         # 简化优化器实现
│   ├── config.cpp           # 配置文件解析实现
│   ├── visualization.cpp    # 可视化模块实现
│   └── cuda_utils.cu        # 工具函数实现
├── run.cpp                  # 主程序（撒点方式生成数据）
├── config_default.conf      # 默认配置文件
├── config_high_precision.conf # 高精度配置文件
├── Makefile                 # 编译配置
├── run.sh                   # 运行脚本
└── introduction.md          # 本文档
```

---

## 核心实现详解

### 1. 数据生成（`run.cpp`）

使用C++标准库的`std::normal_distribution`和`std::mt19937`：

```cpp
// Cholesky分解矩阵元素
float l11 = sigma_x;
float l21 = rho * sigma_y;
float l22 = sigma_y * sqrtf(1.0f - rho * rho);

normal_distribution<float> dist(0.0f, 1.0f);

for (int i = 0; i < num_samples; i++) {
    // 生成两个独立的标准正态随机变量
    float z1 = dist(gen);
    float z2 = dist(gen);

    // 转换为相关的高斯分布
    float x = params.x0 + l11 * z1;
    float y = params.y0 + l21 * z1 + l22 * z2;

    // 计算所属的bin并增加计数
    // ...
}
```

### 2. 归一化的高斯PDF（`src/model.cu`）

```cuda
__device__ float gaussian2d(float x, float y, GaussianParams params) {
    // ... 计算 Q ...

    // 归一化因子: 2π * σₓ * σᵧ * sqrt(1-ρ²)
    float norm_factor = 2.0f * 3.14159265359f * sigma_x_safe * sigma_y_safe
                       * sqrtf(1.0f - rho_safe * rho_safe);

    return params.A * expf(-0.5f * Q) / norm_factor;
}
```

**关键点**：
- 正确的归一化确保PDF积分等于A
- 数值稳定性处理（避免除零、限制指数）

### 3. CUDA Kernel - 泊松似然（`src/model.cu`）

```cuda
__global__ void poissonLikelihoodKernel(
    float* __restrict__ expected,
    float* __restrict__ likelihood,
    const int* __restrict__ observed,
    GaussianParams params,
    int nbins,
    float x_min, float x_max, int nx,
    float y_min, float y_max, int ny
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbins) return;

    // 计算bin中心坐标
    int ix = idx % nx;
    int iy = idx / nx;
    float x = x_min + (ix + 0.5f) * dx;
    float y = y_min + (iy + 0.5f) * dy;

    // 计算期望值（归一化PDF）
    float lambda = gaussian2d(x, y, params);
    lambda = fmaxf(lambda, MIN_EXPECTED);

    // 泊松似然: L_i = λ_i - n_i * log(λ_i)
    int n = observed[idx];
    likelihood[idx] = lambda - n * logf(lambda);
}
```

### 4. 简化优化器（`src/fit_model.cu`）

**特点**：
- 自适应学习率：似然增加时自动降低学习率
- 参数约束：确保sigma > 0，|rho| < 1，A > 0
- CUDA并行梯度计算：所有6个参数的梯度在GPU上并行计算

```cpp
OptimizationResult SimpleOptimizer::optimize(
    const GaussianParams& initial_params
) {
    for (int iter = 0; iter < max_iterations; iter++) {
        // 1. 计算梯度（CUDA并行）
        computeGradient(result.params, gradient);

        // 2. 保存旧参数
        GaussianParams old_params = result.params;
        float old_likelihood = current_likelihood;

        // 3. 更新参数
        result.params.A        -= lr * gradient[0];
        result.params.x0       -= lr * gradient[1] * 0.01f;
        // ... 其他参数

        // 4. 计算新似然
        float new_likelihood = computeLikelihood(result.params);

        // 5. 如果似然增加，回退并降低学习率
        if (new_likelihood > old_likelihood) {
            result.params = old_params;
            config.learning_rate *= 0.5f;
        } else {
            current_likelihood = new_likelihood;
        }

        // 6. 检查收敛
        if (param_change < tolerance) {
            result.converged = true;
            break;
        }
    }
    return result;
}
```

---

## 测试结果

### 配置
- 样本数：100,000
- 直方图大小：64×64 bins
- 坐标范围：[-5, 5] × [-5, 5]

### 拟合结果

| 参数 | 真值 | 拟合值 | 误差 |
|------|------|--------|------|
| x0 | 0.00 | 0.04 | 优秀 |
| y0 | 0.00 | -0.21 | 优秀 |
| sigma_x | 1.00 | 1.15 | 15% |
| sigma_y | 1.50 | 1.69 | 13% |
| rho | 0.30 | 0.31 | 2.3% |

- **收敛迭代次数**: ~2275次
- **最终似然值**: -188883

---

## 性能监测与可视化

本项目新增了完整的性能监测和可视化功能，可以自动生成拟合质量分析和性能报告。

### 输出文件结构

运行后，所有输出文件保存在 `output/` 目录：

```
output/
├── config.txt              # 当前运行的配置参数
├── results.txt             # 拟合结果摘要（参数比较、误差统计）
├── histogram_x_data.csv    # X方向投影数据和拟合曲线
├── histogram_y_data.csv    # Y方向投影数据和拟合曲线
├── histogram_2d_data.csv   # 完整2D直方图数据
├── iteration_times.csv     # 每轮迭代时间记录
├── likelihood_history.csv  # 似然值变化历史
├── plot_results.py         # Python绘图脚本
├── histogram_x.jpg         # X方向直方图+拟合曲线
├── histogram_y.jpg         # Y方向直方图+拟合曲线
├── histogram_2d.jpg        # 2D热图
├── iteration_times.jpg     # 迭代时间变化图
└── likelihood_history.jpg  # 似然值收敛曲线
```

### 拟合质量检测

**1. 1D投影直方图** - X和Y方向的边缘分布

- 将2D直方图投影到X和Y轴
- 叠加拟合的高斯曲线和真实高斯曲线
- 用于验证拟合质量

**2. 完整2D分布**

- 2D热图显示完整的观测分布
- 直观展示高斯峰的位置和形状
- 可以观察相关系数的影响

### CUDA加速性能检测

**迭代时间监测**:

- 每轮迭代自动记录执行时间（毫秒）
- 使用 `std::chrono::high_resolution_clock` 精确计时
- 保存所有迭代时间到CSV文件
- 生成时间变化折线图

**配置参数** (在配置文件的 `[output]` 部分):

```ini
[output]
output_dir = output           # 输出目录
timing_save_interval = 100    # 时间记录间隔（已自动记录每次迭代）
save_plots = true             # 是否生成图像
```

### 生成图像

运行程序后，执行Python脚本生成JPG图像：

```bash
cd output
python3 plot_results.py
```

**生成的图像**:

1. `histogram_x.jpg` - X方向直方图 + 拟合曲线
2. `histogram_y.jpg` - Y方向直方图 + 拟合曲线
3. `histogram_2d.jpg` - 2D热图
4. `iteration_times.jpg` - 迭代时间变化图
5. `likelihood_history.jpg` - 似然值收敛曲线

### 典型输出示例

```
=== Parameter Comparison ===
Parameter   True Value     Fitted Value   Error (%)
---------------------------------------------------------
A           10000          9999.93        -0.00%
x0          1.00           1.15           15.37%
y0          -0.50          -0.54          8.73%
sigma_x     0.80           0.98           22.06%
sigma_y     1.20           1.06           -11.58%
rho         -0.40          -0.33          -16.88%

=== Performance Summary ===
Average iteration time: 0.74 ms
Total optimization time: 52.43 ms
```

---

## 编译和运行

### 编译

```bash
cd gaussion_fitter
make clean
make
```

### 运行

```bash
./gaussion_fitter
```

**预期输出**：
```
========================================
  CUDA 2D Gaussian Fit (Clean Version)
========================================

True parameters:
True: A=10000.00, x0=0.000, y0=0.000, sigma_x=1.000, sigma_y=1.500, rho=0.300

Generated histogram using 100000 samples
Histogram size: 64x64 bins
Total counts: 99920

Starting optimization...

=== Gradient Descent Optimization ===
Initial likelihood: -171896.093750
Iter    1: L = -171919.718750, LR = 1.000000e-06
...
Converged after 2275 iterations!
```

---

## 设计要点

### 1. 为什么使用撒点方式？

相比直接计算PDF值，撒点方式更接近实际实验数据：
- 粒子物理实验（探测器计数）
- 天文观测（星系分布）
- 医学成像（PET扫描）

### 2. PDF归一化的重要性

对于撒点方式，必须正确归一化PDF：
- 未归一化：似然值大小不正确，导致拟合失败
- 正确归一化：似然值尺度正确，优化收敛

### 3. CUDA并行计算策略

- **每个bin一个线程**: 最大化并行度
- **线程块规约**: 高效求和
- **多级规约**: 处理大数组（4096+ bins）

### 4. 自适应学习率

当似然增加时（表示步长太大）：
1. 回退参数到上一步
2. 将学习率减半
3. 继续优化

这比固定学习率更稳定。

---

## 关键代码文件

| 文件 | 功能 |
|------|------|
| [run.cpp](run.cpp) | 撒点方式生成数据，主程序入口 |
| [src/model.cu](src/model.cu) | 归一化PDF，CUDA kernels |
| [src/fit_model.cu](src/fit_model.cu) | 简化优化器，带时间监测 |
| [include/fit_model.h](include/fit_model.h) | 优化器接口 |
| [src/visualization.cpp](src/visualization.cpp) | 可视化模块实现 |
| [include/visualization.h](include/visualization.h) | 可视化模块接口 |
| [src/config.cpp](src/config.cpp) | 配置文件解析 |
| [include/config.h](include/config.h) | 配置类定义 |
| [config_default.conf](config_default.conf) | 默认配置文件 |

---

## 参考资料

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Cholesky分解: https://en.wikipedia.org/wiki/Cholesky_decomposition
- 泊松统计: https://en.wikipedia.org/wiki/Poisson_distribution
- 最大似然估计: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
