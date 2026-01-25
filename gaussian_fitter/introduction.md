# CUDA 2D高斯直方图拟合器 - 项目介绍

## 项目概述

本项目实现了一个使用CUDA加速的2D高斯直方图拟合器，通过**撒点方式（Monte Carlo采样）**生成测试数据，并使用**泊松似然**进行参数估计。

### 核心特点

- **撒点方式生成数据**: 使用C++标准库生成2D高斯样本，通过Cholesky分解处理相关系数
- **bin-to-bin CUDA并行计算**: 每个bin的期望值和似然计算在GPU上并行执行
- **bin积分计算期望值**: 使用8×8数值采样点在每个bin内进行积分，而非简单的bin中心点计算
- **正确的PDF归一化**: 对于撒点方式，PDF正确归一化使得积分等于总样本数
- **多种优化算法**: 支持简单梯度下降(Simple GD)和Adam优化器
- **完整2D高斯模型**: 包含6个自由参数（幅值、中心位置、宽度、相关系数）
- **ctypes Python接口**: 提供Python前端，支持配置文件和结果可视化
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

### 3. Bin积分计算期望值

为提高拟合精度，使用**数值积分**而非简单的bin中心点计算：

```
对于每个bin (x_bin_start, x_bin_end) × (y_bin_start, y_bin_end):

λᵢ = ∫∫_bin PDF(x,y) dx dy

使用8×8网格采样点进行数值积分:
λᵢ ≈ (bin_area / 64) × ΣⱼΣₖ PDF(xⱼ, yₖ)

其中 xⱼ, yₖ 是bin内的采样点
```

### 4. 泊松似然函数

对于观测数据 nᵢ（bin中的计数）和模型期望值 λᵢ(θ)（bin积分）：

```
L(θ) = Σᵢ [λᵢ(θ) - nᵢ · log(λᵢ(θ))]
```

**目标**: 找到参数 θ 使 L(θ) 最小化

### 5. 数值微分梯度

使用有限差分近似计算梯度：

```
∇L(θ) ≈ [L(θ+εeᵢ) - L(θ)] / ε
```

### 6. 优化算法

#### 简单梯度下降 (Simple GD)

```
参数更新: θ_t = θ_{t-1} - α · ∇L(θ)
```

- 使用参数特定的缩放因子处理不同梯度尺度
- 自适应学习率：似然增加时自动降低学习率
- 参数约束：确保sigma > 0，|rho| < 1，A > 0

#### Adam优化器 (Adaptive Moment Estimation)

```
一阶矩估计: m_t = β₁·m_{t-1} + (1-β₁)·g_t
二阶矩估计: v_t = β₂·v_{t-1} + (1-β₂)·g_t²

偏差修正:
    m̂_t = m_t / (1-β₁^t)
    v̂_t = v_t / (1-β₂^t)

参数更新: θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)
```

- 自适应学习率：每个参数有独立的学习率调整
- 动量加速：利用一阶矩估计加速收敛
- 典型参数：β₁=0.9, β₂=0.999, ε=1e-8

---

## 代码结构

```
gaussian_fitter/
├── include/
│   ├── model.h              # 数据结构定义
│   ├── fit_model.h          # 优化器类定义(Simple/Adam基类)
│   ├── ctypes_interface.h   # ctypes C接口
│   ├── config.h             # 配置文件解析
│   └── cuda_utils.h         # 工具函数
├── src/
│   ├── model.cu             # CUDA kernel实现（含bin积分）
│   ├── fit_model.cu         # Simple/Adam优化器实现
│   ├── ctypes_interface.cpp # ctypes接口实现
│   ├── config.cpp           # 配置文件解析实现
│   └── cuda_utils.cu        # 工具函数实现
├── share/
│   └── run_ctypes.py        # Python前端（ctypes调用）
├── config/
│   └── config_default.conf  # 默认配置文件
├── run.sh                   # 运行脚本
├── Makefile                 # 编译配置
└── introduction.md          # 本文档
```

---

## 核心实现详解

### 1. Bin积分（`src/model.cu`）

```cuda
__device__ float integrateGaussianOverBin(
    float x_bin_start, float x_bin_end,
    float y_bin_start, float y_bin_end,
    GaussianParams params
) {
    constexpr int SUBSAMPLES = 8;  // 8x8 = 64个采样点
    float dx = (x_bin_end - x_bin_start) / SUBSAMPLES;
    float dy = (y_bin_end - y_bin_start) / SUBSAMPLES;
    float sum = 0.0f;

    // 二维采样：计算每个采样点的PDF值
    for (int i = 0; i < SUBSAMPLES; i++) {
        for (int j = 0; j < SUBSAMPLES; j++) {
            float x = x_bin_start + (i + 0.5f) * dx;
            float y = y_bin_start + (j + 0.5f) * dy;
            sum += gaussian2d(x, y, params);
        }
    }

    float bin_area = (x_bin_end - x_bin_start) * (y_bin_end - y_bin_start);
    return sum / (SUBSAMPLES * SUBSAMPLES) * bin_area;
}
```

### 2. CUDA Kernel - 泊松似然（`src/model.cu`）

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

    // 计算bin边界
    int ix = idx % nx;
    int iy = idx / nx;
    float x_bin_start = x_min + ix * dx;
    float x_bin_end = x_bin_start + dx;
    float y_bin_start = y_min + iy * dy;
    float y_bin_end = y_bin_start + dy;

    // 使用bin积分计算期望值
    float lambda = integrateGaussianOverBin(
        x_bin_start, x_bin_end,
        y_bin_start, y_bin_end,
        params
    );
    lambda = fmaxf(lambda, MIN_EXPECTED);

    // 泊松似然: L_i = λ_i - n_i * log(λ_i)
    int n = observed[idx];
    likelihood[idx] = lambda - n * logf(lambda);
}
```

### 3. 优化器架构（`include/fit_model.h`）

使用多态设计支持多种优化算法：

```cpp
// 优化器类型枚举
enum class OptimizerType {
    SIMPLE = 0,   // 简单梯度下降
    ADAM = 1       // Adam优化器
};

// 基类
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void setData(const Histogram2D& hist) = 0;
    virtual OptimizationResult optimize(const GaussianParams& initial_params) = 0;
};

// Simple GD实现
class SimpleOptimizer : public Optimizer { ... };

// Adam优化器实现
class AdamOptimizer : public Optimizer {
private:
    float* m;  // 一阶矩估计（动量）
    float* v;  // 二阶矩估计
    // ...
};
```

### 4. Adam优化器实现（`src/fit_model.cu`）

```cpp
// Adam参数更新（无额外缩放，利用自适应机制）
const float* g = gradient;  // 直接使用原始梯度

// 更新一阶矩和二阶矩
for (int i = 0; i < 6; i++) {
    m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
}

// 偏差修正
float beta1_t = powf(beta1, iter + 1);
float beta2_t = powf(beta2, iter + 1);
for (int i = 0; i < 6; i++) {
    m_hat[i] = m[i] / (1.0f - beta1_t);
    v_hat[i] = v[i] / (1.0f - beta2_t);
}

// 更新参数: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
result.params.A   -= lr * m_hat[0] / (sqrtf(v_hat[0]) + epsilon);
result.params.x0  -= lr * m_hat[1] / (sqrtf(v_hat[1]) + epsilon);
// ... 其他参数
```

**关键设计决策**:
- Adam不使用额外的参数缩放因子，因为其自适应机制能自动处理不同梯度尺度
- Simple GD使用缩放因子是因为固定学习率无法自动适应不同参数的梯度差异

---

## 测试结果对比

### 配置
- 样本数：100,000
- 直方图大小：128×128 bins
- 坐标范围：[-5, 5] × [-5, 5]

### 优化器性能对比

| 优化器 | 学习率 | 迭代次数 | 最大误差 | 收敛状态 |
|--------|--------|----------|----------|----------|
| Simple GD | 1e-5 | 3328 | y0: -1.43% | ✓ |
| Adam | 1e-2 | 655 | sigma_x: -0.23% | ✓ |

**Adam优势**:
- 收敛速度快约**5倍**
- 拟合精度更高（误差从1.43%降至0.23%）
- 可使用更大的学习率

### 典型拟合结果 (Adam)

| 参数 | 真值 | 拟合值 | 误差 |
|------|------|--------|------|
| A | 100000 | 99999.0 | -0.00% |
| x0 | 0.500 | 0.499 | -0.24% |
| y0 | 0.500 | 0.499 | -0.20% |
| sigma_x | 1.000 | 0.998 | -0.23% |
| sigma_y | 1.500 | 1.499 | -0.05% |
| rho | 0.300 | 0.301 | 0.29% |

---

## 配置文件

配置文件采用INI格式，支持多种优化器配置：

```ini
# ==================== 真实参数（用于生成测试数据） ====================
[true_params]
A = 100000
x0 = 0.5
y0 = 0.5
sigma_x = 1.0
sigma_y = 1.5
rho = 0.3

# ==================== 直方图配置 ====================
[histogram]
nx = 128           # x方向bin数量
ny = 128           # y方向bin数量
x_min = -5.0
x_max = 5.0
y_min = -5.0
y_max = 5.0
num_samples = 100000

# ==================== 初始猜测 ====================
[initial_guess]
A = 100000
x0 = 0.8           # 故意设置偏离真值
y0 = -0.1
sigma_x = 1.3
sigma_y = 1.3
rho = 0.25

# ==================== 优化器配置 ====================
[optimizer]
# 优化器类型: simple (简单梯度下降) 或 adam (自适应矩估计)
optimizer_type = adam

# 学习率（Adam通常可以使用更大的学习率）
learning_rate = 0.01
max_iterations = 100000
tolerance = 1.0e-6
gradient_epsilon = 1.0e-5

# Adam优化器特有参数（仅当optimizer_type=adam时使用）
beta1 = 0.9       # 一阶矩衰减率
beta2 = 0.999     # 二阶矩衰减率
epsilon = 1.0e-8  # 数值稳定性常数

# ==================== 输出配置 ====================
[output]
output_dir = output/adam
timing_save_interval = 100
save_plots = true
```

---

## 输出文件结构

运行后，所有输出文件保存在配置的输出目录：

```
output/
├── config.txt          # 当前运行的配置参数
├── results.txt         # 拟合结果摘要（参数比较、误差统计）
├── results.csv         # CSV格式的拟合结果（便于程序解析）
├── histogram_x.jpg     # X方向直方图+拟合曲线
├── histogram_y.jpg     # Y方向直方图+拟合曲线
├── histogram_2d.jpg    # 2D热图
└── iteration_times.jpg # 迭代时间变化图
```

---

## 编译和运行

### 编译

```bash
cd gaussian_fitter
make clean
make ctypes    # 编译ctypes共享库
```

### 运行

```bash
# 使用默认配置文件
./run.sh

# 使用自定义配置文件
python3 share/run_ctypes.py config/my_config.conf
```

### 预期输出

```
==================================================
  CUDA 2D Gaussian Fitter
  Python Frontend with ctypes
==================================================

Config file: config/config_default.conf

==================================================
  Configuration
==================================================

[True Parameters]
  A = 100000.0, x0 = 0.5, y0 = 0.5
  sigma_x = 1.0, sigma_y = 1.5, rho = 0.3

[Histogram]
  bins: 128x128, num_samples: 100000

[Optimizer]
  optimizer_type = Adam
  learning_rate = 0.01, max_iterations = 100000
==================================================

Loading CUDA library...

Generating histogram...
Histogram size: 128x128 bins
Total counts: 99840

Projecting histogram to 1D...

Starting optimization...
Optimizer: Adam

=== Adam Optimization ===
Initial likelihood: -196776.625000
Beta1: 0.9000, Beta2: 0.9990, Epsilon: 1.00e-08
Initial: A=100000.00, x0=0.800, y0=-0.100, sigma_x=1.300, sigma_y=1.300, rho=0.250
Iter    1: L = -198309.937500, LR = 1.000000e-02, Time = 1.56 ms
Iter  101: L = -222353.875000, LR = 1.000000e-02, Time = 1.48 ms
...
Converged after 655 iterations!

==================================================
  Parameter Comparison
==================================================
Parameter    True Value   Fitted Value Error (%)
--------------------------------------------------
A            100000.0000  99999.0312   -0.00
x0           0.5000       0.4988       -0.24
y0           0.5000       0.4990       -0.20
sigma_x      1.0000       0.9977       -0.23
sigma_y      1.5000       1.4992       -0.05
rho          0.3000       0.3009       0.29
--------------------------------------------------

Iterations: 655
Converged: True
Final Likelihood: -222395.3750
==================================================

All outputs saved to: output/adam/
```

---

## 设计要点

### 1. 为什么使用bin积分？

相比简单的bin中心点计算，bin积分有以下优势：
- **更精确的期望值**: 考虑了bin内PDF的变化
- **更好的拟合质量**: 特别是对于大bins或高梯度区域
- **物理上更正确**: 真实观测是bin内的积分计数

### 2. Adam vs Simple GD

| 特性 | Simple GD | Adam |
|------|-----------|------|
| 学习率调整 | 手动（全局固定） | 自动（每参数独立） |
| 动量 | 无 | 有（一阶矩） |
| 梯度尺度敏感 | 高（需要手动缩放） | 低（自适应） |
| 收敛速度 | 慢 | 快（~5倍） |
| 参数调节 | 需要careful tuning | 更鲁棒 |

### 3. CUDA并行计算策略

- **每个bin一个线程**: 最大化并行度
- **bin内积分**: 每个线程独立完成8×8采样
- **coalesced内存访问**: 连续访问observed数组

### 4. 为什么Adam不需要参数缩放？

```cpp
// Simple GD需要缩放（因为不同参数梯度差异很大）
result.params.sigma_x -= lr * gradient[3] * 0.001f;  // 手动缩放

// Adam不需要（自适应机制自动处理）
result.params.sigma_x -= lr * m_hat[3] / (sqrtf(v_hat[3]) + epsilon);
```

Adam的二阶矩估计 `v_t` 记录了梯度的平方，参数更新时除以 `√v̂_t`，相当于为每个参数自动调整了学习率，额外缩放反而会干扰这种自适应机制。

---

## 关键代码文件

| 文件 | 功能 |
|------|------|
| [share/run_ctypes.py](share/run_ctypes.py) | Python前端，ctypes接口调用 |
| [src/ctypes_interface.cpp](src/ctypes_interface.cpp) | ctypes C接口实现 |
| [include/ctypes_interface.h](include/ctypes_interface.h) | ctypes C接口声明 |
| [src/model.cu](src/model.cu) | CUDA kernels（bin积分） |
| [src/fit_model.cu](src/fit_model.cu) | Simple/Adam优化器实现 |
| [include/fit_model.h](include/fit_model.h) | 优化器类定义 |
| [config/config_default.conf](config/config_default.conf) | 默认配置文件 |

---

## 参考资料

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Adam优化器论文: Kingma & Ba (2015) - "Adam: A Method for Stochastic Optimization"
- Cholesky分解: https://en.wikipedia.org/wiki/Cholesky_decomposition
- 泊松统计: https://en.wikipedia.org/wiki/Poisson_distribution
- 最大似然估计: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
