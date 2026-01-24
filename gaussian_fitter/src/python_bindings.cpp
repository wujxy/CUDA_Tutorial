#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "model.h"
#include "fit_model.h"
#include "config.h"

namespace py = pybind11;

/**
 * Python模块封装
 * 将核心拟合功能暴露给Python
 */

// 生成2D高斯直方图
Histogram2D generateGaussianHistogramPy(
    float A, float x0, float y0, float sigma_x, float sigma_y, float rho,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    int num_samples
) {
    GaussianParams params;
    params.A = A;
    params.x0 = x0;
    params.y0 = y0;
    params.sigma_x = sigma_x;
    params.sigma_y = sigma_y;
    params.rho = rho;

    return generateGaussianHistogram(params, nx, ny, x_min, x_max, y_min, y_max, num_samples);
}

// 执行拟合
OptimizationResult fitHistogramPy(
    py::array_t<int> histogram_data,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    float A_init, float x0_init, float y0_init,
    float sigma_x_init, float sigma_y_init, float rho_init,
    float learning_rate,
    int max_iterations,
    float tolerance,
    float gradient_epsilon,
    bool verbose
) {
    // 从numpy数组获取数据
    py::buffer_info buf = histogram_data.request();
    int* ptr = static_cast<int*>(buf.ptr);

    // 创建Histogram2D结构
    Histogram2D hist;
    hist.nx = nx;
    hist.ny = ny;
    hist.x_min = x_min;
    hist.x_max = x_max;
    hist.y_min = y_min;
    hist.y_max = y_max;

    // 分配内存并复制数据
    int nbins = nx * ny;
    CUDA_CHECK(cudaMallocManaged(&hist.data, nbins * sizeof(int)));
    for (int i = 0; i < nbins; i++) {
        hist.data[i] = ptr[i];
    }

    // 设置初始参数
    GaussianParams init_params;
    init_params.A = A_init;
    init_params.x0 = x0_init;
    init_params.y0 = y0_init;
    init_params.sigma_x = sigma_x_init;
    init_params.sigma_y = sigma_y_init;
    init_params.rho = rho_init;

    // 配置优化器
    OptimizerConfig opt_config;
    opt_config.learning_rate = learning_rate;
    opt_config.max_iterations = max_iterations;
    opt_config.tolerance = tolerance;
    opt_config.gradient_epsilon = gradient_epsilon;
    opt_config.verbose = verbose;

    // 创建优化器并拟合
    SimpleOptimizer optimizer(opt_config);
    optimizer.setData(hist);
    OptimizationResult result = optimizer.optimize(init_params);

    // 清理GPU内存（但保留result中的数据）
    cudaFree(hist.data);

    return result;
}

// 投影2D直方图到1D
py::dict projectHistogramPy(
    py::array_t<int> histogram_data,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max
) {
    py::buffer_info buf = histogram_data.request();
    const int* data = static_cast<const int*>(buf.ptr);

    // X投影
    std::vector<float> bin_centers_x(nx);
    std::vector<int> counts_x(nx, 0);
    float dx = (x_max - x_min) / nx;
    for (int i = 0; i < nx; i++) {
        bin_centers_x[i] = x_min + (i + 0.5f) * dx;
    }
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            counts_x[ix] += data[iy * nx + ix];
        }
    }

    // Y投影
    std::vector<float> bin_centers_y(ny);
    std::vector<int> counts_y(ny, 0);
    float dy = (y_max - y_min) / ny;
    for (int i = 0; i < ny; i++) {
        bin_centers_y[i] = y_min + (i + 0.5f) * dy;
    }
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            counts_y[iy] += data[iy * nx + ix];
        }
    }

    return py::dict(
        "x_bins"_a = bin_centers_x,
        "x_counts"_a = counts_x,
        "y_bins"_a = bin_centers_y,
        "y_counts"_a = counts_y
    );
}

// 计算1D高斯曲线
py::array_t<float> compute1DGaussianCurvePy(
    py::array_t<float> bin_centers,
    float mean,
    float sigma,
    float total_counts
) {
    py::buffer_info buf = bin_centers.request();
    const float* bins = static_cast<const float*>(buf.ptr);
    int nbins = buf.size;

    std::vector<float> curve(nbins);
    float norm_factor = 1.0f / (sigma * sqrtf(2.0f * 3.14159265359f));

    // 估算bin宽度
    float bin_width = (nbins > 1) ? (bins[1] - bins[0]) : 1.0f;

    for (int i = 0; i < nbins; i++) {
        float x = bins[i];
        float z = (x - mean) / sigma;
        float pdf = norm_factor * expf(-0.5f * z * z);
        curve[i] = pdf * bin_width * total_counts;
    }

    return py::array_t<float>(curve.size(), curve.data());
}

PYBIND11_MODULE(gaussian_fitter_core, m) {
    m.doc() = "CUDA-accelerated 2D Gaussian histogram fitting";

    // 数据结构
    py::class_<GaussianParams>(m, "GaussianParams")
        .def(py::init<>())
        .def_readwrite("A", &GaussianParams::A)
        .def_readwrite("x0", &GaussianParams::x0)
        .def_readwrite("y0", &GaussianParams::y0)
        .def_readwrite("sigma_x", &GaussianParams::sigma_x)
        .def_readwrite("sigma_y", &GaussianParams::sigma_y)
        .def_readwrite("rho", &GaussianParams::rho);

    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def(py::init<>())
        .def_readwrite("params", &OptimizationResult::params)
        .def_readwrite("final_likelihood", &OptimizationResult::final_likelihood)
        .def_readwrite("iterations", &OptimizationResult::iterations)
        .def_readwrite("converged", &OptimizationResult::converged)
        .def_readwrite("iteration_times", &OptimizationResult::iteration_times)
        .def_readwrite("likelihood_history", &OptimizationResult::likelihood_history);

    // 函数
    m.def("generate_histogram", &generateGaussianHistogramPy,
          "Generate 2D Gaussian histogram by scatter sampling",
          py::arg("A"), py::arg("x0"), py::arg("y0"),
          py::arg("sigma_x"), py::arg("sigma_y"), py::arg("rho"),
          py::arg("nx"), py::arg("ny"),
          py::arg("x_min"), py::arg("x_max"),
          py::arg("y_min"), py::arg("y_max"),
          py::arg("num_samples"));

    m.def("fit_histogram", &fitHistogramPy,
          "Fit 2D Gaussian to histogram data",
          py::arg("histogram_data"),
          py::arg("nx"), py::arg("ny"),
          py::arg("x_min"), py::arg("x_max"),
          py::arg("y_min"), py::arg("y_max"),
          py::arg("A_init"), py::arg("x0_init"), py::arg("y0_init"),
          py::arg("sigma_x_init"), py::arg("sigma_y_init"), py::arg("rho_init"),
          py::arg("learning_rate") = 1.0e-6f,
          py::arg("max_iterations") = 5000,
          py::arg("tolerance") = 1.0e-6f,
          py::arg("gradient_epsilon") = 1.0e-4f,
          py::arg("verbose") = true);

    m.def("project_histogram", &projectHistogramPy,
          "Project 2D histogram to 1D (X and Y margins)",
          py::arg("histogram_data"),
          py::arg("nx"), py::arg("ny"),
          py::arg("x_min"), py::arg("x_max"),
          py::arg("y_min"), py::arg("y_max"));

    m.def("compute_1d_curve", &compute1DGaussianCurvePy,
          "Compute 1D Gaussian curve for plotting",
          py::arg("bin_centers"),
          py::arg("mean"), py::arg("sigma"), py::arg("total_counts"));
}
