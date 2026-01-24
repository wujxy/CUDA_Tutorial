#include "../include/ctypes_interface.h"
#include "../include/model.h"
#include "../include/fit_model.h"
#include "../include/cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <random>
#include <algorithm>

/**
 * Generate 2D Gaussian histogram - C interface for ctypes
 * Implements scatter sampling using Cholesky decomposition
 */
extern "C" int* generate_histogram_ctypes(
    float A, float x0, float y0, float sigma_x, float sigma_y, float rho,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    int num_samples,
    int* nx_out, int* ny_out, int* total_counts_out
) {
    // Set output parameters
    *nx_out = nx;
    *ny_out = ny;

    // Allocate histogram data
    int nbins = nx * ny;
    int* hist_data;
    CUDA_CHECK(cudaMallocManaged(&hist_data, nbins * sizeof(int)));

    // Initialize histogram to 0
    for (int i = 0; i < nbins; i++) {
        hist_data[i] = 0;
    }

    // Cholesky decomposition for correlated Gaussian
    float l11 = sigma_x;
    float l21 = rho * sigma_y;
    float l22 = sigma_y * sqrtf(1.0f - rho * rho);

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate samples
    for (int i = 0; i < num_samples; i++) {
        float z1 = dist(gen);
        float z2 = dist(gen);

        float x = x0 + l11 * z1;
        float y = y0 + l21 * z1 + l22 * z2;

        // Bin the sample
        if (x >= x_min && x < x_max && y >= y_min && y < y_max) {
            float dx = (x_max - x_min) / nx;
            float dy = (y_max - y_min) / ny;

            int ix = static_cast<int>((x - x_min) / dx);
            int iy = static_cast<int>((y - y_min) / dy);

            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                int idx = iy * nx + ix;
                hist_data[idx]++;
            }
        }
    }

    // Calculate total counts
    int total_counts = 0;
    for (int i = 0; i < nbins; i++) {
        total_counts += hist_data[i];
    }
    *total_counts_out = total_counts;

    // Return the data pointer (caller must free using free_histogram_ctypes)
    return hist_data;
}

/**
 * Fit result structure layout for ctypes
 * [0]: final_likelihood
 * [1]: A_fit
 * [2]: x0_fit
 * [3]: y0_fit
 * [4]: sigma_x_fit
 * [5]: sigma_y_fit
 * [6]: rho_fit
 */
extern "C" float* fit_histogram_ctypes(
    const int* histogram_data,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    float A_init, float x0_init, float y0_init,
    float sigma_x_init, float sigma_y_init, float rho_init,
    float learning_rate,
    int max_iterations,
    float tolerance,
    float gradient_epsilon,
    int verbose,
    int* iterations_out, int* converged_out,
    float** iteration_times_out, int* num_times_out
) {
    // Create Histogram2D from input data
    Histogram2D hist;
    hist.nx = nx;
    hist.ny = ny;
    hist.x_min = x_min;
    hist.x_max = x_max;
    hist.y_min = y_min;
    hist.y_max = y_max;

    int nbins = nx * ny;
    CUDA_CHECK(cudaMallocManaged(&hist.data, nbins * sizeof(int)));
    for (int i = 0; i < nbins; i++) {
        hist.data[i] = histogram_data[i];
    }

    // Set initial parameters
    GaussianParams init_params;
    init_params.A = A_init;
    init_params.x0 = x0_init;
    init_params.y0 = y0_init;
    init_params.sigma_x = sigma_x_init;
    init_params.sigma_y = sigma_y_init;
    init_params.rho = rho_init;

    // Configure optimizer
    OptimizerConfig opt_config;
    opt_config.learning_rate = learning_rate;
    opt_config.max_iterations = max_iterations;
    opt_config.tolerance = tolerance;
    opt_config.gradient_epsilon = gradient_epsilon;
    opt_config.verbose = verbose != 0;

    // Run optimization
    SimpleOptimizer optimizer(opt_config);
    optimizer.setData(hist);
    OptimizationResult result = optimizer.optimize(init_params);

    // Set output parameters
    *iterations_out = result.iterations;
    *converged_out = result.converged ? 1 : 0;

    // Copy iteration times to managed memory
    int num_times = result.iteration_times.size();
    float* iteration_times_array = nullptr;
    if (num_times > 0) {
        CUDA_CHECK(cudaMallocManaged(&iteration_times_array, num_times * sizeof(float)));
        for (int i = 0; i < num_times; i++) {
            iteration_times_array[i] = result.iteration_times[i];
        }
    }
    *iteration_times_out = iteration_times_array;
    *num_times_out = num_times;

    // Allocate result array and fill with results
    float* result_array = new float[7];
    result_array[0] = result.final_likelihood;
    result_array[1] = result.params.A;
    result_array[2] = result.params.x0;
    result_array[3] = result.params.y0;
    result_array[4] = result.params.sigma_x;
    result_array[5] = result.params.sigma_y;
    result_array[6] = result.params.rho;

    // Clean up histogram data
    cudaFree(hist.data);

    return result_array;
}

/**
 * Free histogram data allocated by generate_histogram_ctypes
 */
extern "C" void free_histogram_ctypes(int* data) {
    cudaFree(data);
}

/**
 * Free fit result allocated by fit_histogram_ctypes
 */
extern "C" void free_fit_result_ctypes(float* data) {
    delete[] data;
}

/**
 * Free iteration times array allocated by fit_histogram_ctypes
 */
extern "C" void free_iteration_times_ctypes(float* data) {
    cudaFree(data);
}
