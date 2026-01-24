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
    float learning_rate,
    int max_iterations,
    float tolerance,
    float gradient_epsilon,
    int verbose,
    int timing_save_interval,
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

#ifdef __cplusplus
}
#endif

#endif // CTYPES_INTERFACE_H
