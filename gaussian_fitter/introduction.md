# CUDA Poisson Likelihood Gaussian Fitter - 实现介绍

## 项目概述

本项目实现了一个使用CUDA加速的泊松似然2D高斯拟合器，用于拟合直方图数据到2D高斯分布模型。核心特点是：

- **bin-to-bin CUDA计算**: 每个bin的期望值计算在GPU上并行执行
- **自定义最小化器**: 梯度下降 + 数值微分
- **完整2D高斯模型**: 包含6个自由参数（幅值、中心位置、宽度、相关系数）

---

## 数学原理

### 1. 2D高斯模型

完整的2D高斯分布有6个自由参数：

```cpp
struct GaussianParams {
    float A;        // 幅值 (总强度)
    float x0;       // x中心位置
    float y0;       // y中心位置
    float sigma_x;  // x方向标准差
    float sigma_y;  // y方向标准差
    float rho;      // 相关系数 [-1, 1]
};
```

**高斯函数公式**：

```
λ(x,y) = A · exp(-Q/2)

其中 Q = (1/(1-ρ²)) · [
    (x-x₀)²/σₓ² +
    (y-y₀)²/σᵧ² -
    2ρ(x-x₀)(y-y₀)/(σₓσᵧ)
]
```

### 2. 泊松似然函数

对于观测数据 nᵢ 和模型期望值 λᵢ(θ)，负对数泊松似然为：

```
L(θ) = Σᵢ [λᵢ(θ) - nᵢ · log(λᵢ(θ))]
```

**目标**: 找到参数 θ 使 L(θ) 最小化

### 3. 数值微分梯度

使用有限差分近似计算梯度：

```
∇L(θ) ≈ [L(θ+εeᵢ) - L(θ)] / ε
```

为处理不同尺度的参数，使用**相对步长**：

| 参数 | 扰动步长 |
|------|----------|
| A | max(ε·A, 1.0) |
| x0, y0 | ε·(|参数| + 0.1) |
| sigma_x, sigma_y | ε·参数值 |
| rho | ε·0.01 |

---

## 代码结构

```
gaussion_fitter/
├── include/
│   ├── model.h              # 数据结构定义
│   ├── model_kernels.cuh    # CUDA kernel声明
│   ├── optimizer.h          # 优化器接口
│   └── cuda_utils.h         # 工具函数
├── src/
│   ├── model.cu             # CUDA kernel实现
│   ├── optimizer.cu         # 优化器实现
│   └── cuda_utils.cu        # 工具函数实现
├── run.cpp                  # 主程序
├── Makefile                 # 编译配置
└── introduction.md          # 本文档
```

---

## 核心实现详解

### 1. 数据结构 (`include/model.h`)

```cpp
// 2D直方图数据结构
struct Histogram2D {
    int* data;      // 观测计数 [nx * ny]
    int nx, ny;     // bin数量
    float x_min, x_max, y_min, y_max;  // 坐标范围
};
```

### 2. CUDA Kernel 实现 (`src/model.cu`)

#### 2.1 设备端高斯函数

```cuda
__device__ float gaussian2d(float x, float y, GaussianParams params) {
    // 计算相对位置
    float dx = x - params.x0;
    float dy = y - params.y0;

    // 数值稳定性处理
    float sigma_x_safe = fmaxf(params.sigma_x, 1e-6f);
    float sigma_y_safe = fmaxf(params.sigma_y, 1e-6f);
    float rho_safe = fmaxf(-0.999f, fminf(params.rho, 0.999f));

    // 计算 Q 值并返回高斯值
    float Q = /* ... */;
    return params.A * expf(-0.5f * Q);
}
```

**关键点**：
- 使用 `fmaxf` 避免除零
- 限制 rho 在 (-1, 1) 范围内
- 限制指数参数防止溢出

#### 2.2 泊松似然 Kernel

```cuda
__global__ void poissonLikelihoodKernel(
    float* __restrict__ expected,    // 输出: 期望值
    float* __restrict__ likelihood,  // 输出: 似然贡献
    const int* __restrict__ observed, // 输入: 观测值
    const float* __restrict__ x_coords,
    const float* __restrict__ y_coords,
    GaussianParams params,
    int nbins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbins) return;

    // 计算期望值
    float lambda = gaussian2d(x_coords[idx], y_coords[idx], params);
    lambda = fmaxf(lambda, MIN_EXPECTED);  // 避免log(0)

    expected[idx] = lambda;

    // 泊松似然: L_i = λ_i - n_i * log(λ_i)
    int n = observed[idx];
    likelihood[idx] = lambda - n * logf(lambda);
}
```

**执行配置**：
```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (nbins + threadsPerBlock - 1) / threadsPerBlock;
poissonLikelihoodKernel<<<blocksPerGrid, threadsPerBlock>>>(/* args */);
```

#### 2.3 数值微分梯度 Kernel

```cuda
__global__ void numericalGradientKernel(
    float* __restrict__ gradient,
    /* ... 其他参数 ... */,
    GaussianParams params,
    float epsilon_rel,  // 相对步长
    int nbins
) {
    // 每个线程块处理一个参数的梯度 (0-5)
    int param_idx = blockIdx.x;

    // 计算该参数的绝对扰动步长
    float epsilon;
    switch (param_idx) {
        case 0: epsilon = fmaxf(epsilon_rel * params.A, 1.0f); break;
        case 1: epsilon = epsilon_rel * (fabsf(params.x0) + 0.1f); break;
        // ... 其他参数
    }

    // 线程块内规约求和
    extern __shared__ float s_data[];
    // ... 计算梯度并规约 ...
}
```

**规约模式**：
```
线程块: [t0, t1, t2, t3, t4, t5, t6, t7]
步1:    [0+1, 2+3, 4+5, 6+7, ..., ..., ..., ...]
步2:    [0+1+2+3, 4+5+6+7, ..., ...]
步3:    [sum, ...]
```

### 3. 梯度下降优化器 (`src/optimizer.cu`)

#### 3.1 类结构

```cpp
class GradientDescentOptimizer {
private:
    OptimizerConfig config;
    const int* d_observed;      // 设备端数据指针
    const float* d_x_coords;
    const float* d_y_coords;
    float* d_expected;          // 临时内存
    float* d_likelihood;
    float* d_gradient;

    float computeLikelihood(const GaussianParams& params);
    void computeGradient(const GaussianParams& params, float* gradient);
    void applyConstraints(GaussianParams& params);
    void updateParams(GaussianParams& params, const float* gradient);

public:
    void setData(const Histogram2D& hist);
    OptimizationResult optimize(const GaussianParams& initial_params);
};
```

#### 3.2 似然计算流程

```cpp
float GradientDescentOptimizer::computeLikelihood(const GaussianParams& params) {
    // Step 1: 并行计算每个bin的似然
    poissonLikelihoodKernel<<<blocks, threads>>>(/* args */);
    cudaDeviceSynchronize();

    // Step 2: 规约求和（多级规约处理大数组）
    sumKernel<<<blocks, threads, shared_mem>>>(/* args */);

    return total_likelihood;
}
```

#### 3.3 梯度计算与归一化

```cpp
void GradientDescentOptimizer::computeGradient(
    const GaussianParams& params, float* gradient
) {
    // Step 1: CUDA计算原始梯度
    numericalGradientKernel<<<6, 256, shared_mem>>>(/* args */);

    // Step 2: 参数自适应归一化
    float scale[6];
    scale[0] = 1.0f / (fabsf(params.A) + 100.0f);      // A
    scale[1] = 1.0f / (fabsf(params.x0) + 0.5f);       // x0
    scale[2] = 1.0f / (fabsf(params.y0) + 0.5f);       // y0
    scale[3] = 1.0f / (fabsf(params.sigma_x) + 0.5f);  // sigma_x
    scale[4] = 1.0f / (fabsf(params.sigma_y) + 0.5f);  // sigma_y
    scale[5] = 1.0f;                                     // rho

    for (int i = 0; i < 6; i++) {
        gradient[i] *= scale[i];
    }
}
```

**为什么需要归一化**？
不同参数的梯度尺度差异巨大：
- A 的梯度 ~ 10⁻¹
- sigma_x 的梯度 ~ 10⁵

归一化后梯度尺度相近，便于统一学习率。

#### 3.4 参数更新

```cpp
void GradientDescentOptimizer::updateParams(
    GaussianParams& params, const float* gradient
) {
    // 梯度下降: θ_new = θ_old - lr * ∇L
    params.A        -= config.learning_rate * gradient[0] * 10.0f;
    params.x0       -= config.learning_rate * gradient[1];
    params.y0       -= config.learning_rate * gradient[2];
    params.sigma_x  -= config.learning_rate * gradient[3];
    params.sigma_y  -= config.learning_rate * gradient[4];
    params.rho      -= config.learning_rate * gradient[5] * 0.01f;

    applyConstraints(params);  // 确保参数在有效范围
}
```

#### 3.5 优化循环

```cpp
OptimizationResult GradientDescentOptimizer::optimize(
    const GaussianParams& initial_params
) {
    GaussianParams params = initial_params;

    for (int iter = 0; iter < max_iterations; iter++) {
        // 1. 计算梯度
        float gradient[6];
        computeGradient(params, gradient);

        // 2. 保存旧参数（用于收敛判断）
        GaussianParams old_params = params;

        // 3. 更新参数
        updateParams(params, gradient);

        // 4. 计算新似然
        float new_likelihood = computeLikelihood(params);

        // 5. 检查收敛
        if (paramsDiff(old_params, params) < tolerance) {
            result.converged = true;
            break;
        }
    }

    return result;
}
```

### 4. 主程序流程 (`run.cpp`)

```cpp
int main() {
    // Step 1: 生成测试数据
    GaussianParams true_params = {1000, 0, 0, 1.0, 1.5, 0};
    Histogram2D hist = generateTestData(true_params, 64, 64);

    // Step 2: 设置初始猜测
    GaussianParams initial_guess = {900, 0.1, -0.1, 0.9, 1.4, 0};

    // Step 3: 配置优化器
    OptimizerConfig config = {
        .learning_rate = 0.01f,
        .max_iterations = 500,
        .tolerance = 1e-6f,
        .gradient_epsilon = 1e-3f,
        .verbose = true
    };

    // Step 4: 运行拟合
    GradientDescentOptimizer optimizer(config);
    optimizer.setData(hist);
    OptimizationResult result = optimizer.optimize(initial_guess);

    // Step 5: 比较结果
    printComparison(true_params, result.params);
}
```

---

## CUDA 编程要点

### 1. 统一内存 (Unified Memory)

```cpp
// 使用 cudaMallocManaged 简化内存管理
cudaMallocManaged(&data, size * sizeof(float));

// CPU 和 GPU 都可以直接访问
data[i] = value;        // CPU 写入
kernel<<<grid, block>>>(data);  // GPU 读取
```

### 2. Kernel 执行配置

```cuda
kernel<<<grid, block, shared_mem>>>(args);

// grid: 网格维度（线程块数量）
// block: 线程块维度（每块线程数量）
// shared_mem: 动态共享内存大小（字节）
```

### 3. 线程索引计算

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 示例：4096 个元素，256 线程/块
// Block 0: 线程 0-255 → 索引 0-255
// Block 1: 线程 0-255 → 索引 256-511
// ...
```

### 4. 共享内存规约

```cuda
extern __shared__ float s_data[];

// 每个线程计算部分和
s_data[tid] = partial_sum;
__syncthreads();

// 树形规约
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
}
```

---

## 编译和运行

### 编译

```bash
cd gaussion_fitter
make clean
make
```

**Makefile 关键配置**：
```makefile
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11 -arch=sm_52

$(TARGET): $(CPP_SRC) $(CU_SRC)
    $(NVCC) $(INCLUDE) $(NVCC_FLAGS) $(CPP_SRC) $(CU_SRC) -o $(TARGET)
```

### 运行

```bash
./gaussion_fitter
```

**预期输出**：
```
========================================
  CUDA Poisson Likelihood Gaussian Fit
========================================

True parameters:
True: A=1000.00, x0=0.000, y0=0.000, sigma_x=1.000, sigma_y=1.500, rho=0.000

Generated test data with 64x64 bins
Total counts: 385130

Initial guess:
Initial: A=900.00, x0=0.100, y0=-0.100, sigma_x=0.900, sigma_y=1.400, rho=0.000

=== Gradient Descent Optimization ===
Initial likelihood: -1870115.125000
Iter    1: L = -1887660.875000 | grad: [...]
...
```

---

## 优化方向

### 当前实现的挑战

1. **梯度下降对泊松似然不够稳定**
   - 似然函数对参数变化非常敏感
   - 需要精细调整学习率和归一化

2. **可能的改进方向**

| 方法 | 优点 | 实现难度 |
|------|------|----------|
| Nelder-Mead 单纯形法 | 无需梯度，更稳定 | 中等 |
| L-BFGS | 收敛快，适合平滑函数 | 较高 |
| Adam 优化器 | 自适应学习率 | 中等 |
| 多起点优化 | 避免局部最优 | 简单 |

### CUDA 性能优化

1. **使用共享内存**缓存坐标数组
2. **Coalesced 内存访问**模式
3. **异步执行**重叠计算和数据传输

---

## 参考资料

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 泊松统计: https://en.wikipedia.org/wiki/Poisson_distribution
- 梯度下降: https://en.wikipedia.org/wiki/Gradient_descent
- 中文博客教程: https://face2ai.com/CUDA-F-1-0-并行计算与计算机架构/
