#ifndef CTYPES_INTERFACE_H
#define CTYPES_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible interface for ctypes

/**
 * Generate 2D Gaussian histogram
 * Returns a pointer to allocated histogram data (caller must free)
 * Output parameters: nx_out, ny_out, total_counts_out
 */
int* generate_histogram_ctypes(
    float A, float x0, float y0, float sigma_x, float sigma_y, float rho,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    int num_samples,
    int* nx_out, int* ny_out, int* total_counts_out
);

/**
 * Fit histogram data
 * histogram_data: array of histogram counts (length = nx * ny)
 * Returns a pointer to result structure (caller must free)
 *
 * Optimizer types:
 *   - 0: Simple gradient descent
 *   - 1: Adam optimizer
 *
 * Output parameters:
 *   - iterations_out: number of iterations performed
 *   - converged_out: 1 if converged, 0 otherwise
 *   - iteration_times_out: array of iteration times in milliseconds (caller must free)
 *   - num_times_out: number of iteration times in array
 */
float* fit_histogram_ctypes(
    const int* histogram_data,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    float A_init, float x0_init, float y0_init,
    float sigma_x_init, float sigma_y_init, float rho_init,
    int optimizer_type,              // 0=Simple, 1=Adam
    float learning_rate,
    int max_iterations,
    float tolerance,
    float gradient_epsilon,
    int verbose,
    int timing_save_interval,
    // Adam optimizer specific parameters (can be set to negative to use defaults)
    float beta1,                      // Adam: first moment decay rate (default 0.9)
    float beta2,                      // Adam: second moment decay rate (default 0.999)
    float epsilon,                    // Adam: numerical stability constant (default 1e-8)
    int* iterations_out, int* converged_out,
    float** iteration_times_out, int* num_times_out
);

/**
 * Free histogram data
 */
void free_histogram_ctypes(int* data);

/**
 * Free fit result
 */
void free_fit_result_ctypes(float* data);

/**
 * Free iteration times array
 */
void free_iteration_times_ctypes(float* data);

/**
 * 2D Likelihood scan for (x0, y0) parameters
 * Scans the likelihood function over a 2D grid of (x0, y0) values
 * while keeping other parameters fixed at their fitted values.
 *
 * @param histogram_data: observed histogram data
 * @param nx, ny: histogram dimensions
 * @param x_min, x_max, y_min, y_max: coordinate ranges
 * @param A_fit, sigma_x_fit, sigma_y_fit, rho_fit: fixed fitted parameters
 * @param x0_center, y0_center: center of scan region (usually fitted values)
 * @param x0_range, y0_range: half-width of scan region
 * @param nx_scan, ny_scan: number of scan points in each direction
 *
 * Returns: pointer to result array containing:
 *   [0]: min_likelihood
 *   [1]: x0_at_min
 *   [2]: y0_at_min
 *   [3]: x0_error_plus
 *   [4]: x0_error_minus
 *   [5]: y0_error_plus
 *   [6]: y0_error_minus
 *   [7-7+nx_scan]: x0_values array
 *   [7+nx_scan-7+nx_scan+ny_scan]: y0_values array
 *   [7+nx_scan+ny_scan-end]: likelihood grid (nx_scan * ny_scan)
 *
 * Output parameters:
 *   - nx_out: number of x0 scan points
 *   - ny_out: number of y0 scan points
 *   - likelihood_size_out: size of likelihood grid
 */
float* likelihood_scan_2d_ctypes(
    const int* histogram_data,
    int nx, int ny,
    float x_min, float x_max,
    float y_min, float y_max,
    float A_fit, float sigma_x_fit, float sigma_y_fit, float rho_fit,
    float x0_center, float y0_center,
    float x0_range, float y0_range,
    int nx_scan, int ny_scan,
    int* nx_out, int* ny_out, int* likelihood_size_out
);

/**
 * Free likelihood scan result
 */
void free_likelihood_scan_ctypes(float* data);

#ifdef __cplusplus
}
#endif

#endif // CTYPES_INTERFACE_H
