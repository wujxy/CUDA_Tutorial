#!/usr/bin/env python3
"""
CUDA 2D Gaussian Fitter - Python Frontend (using ctypes)

This script uses ctypes to call the CUDA-accelerated fitting code.
No Python headers required - only needs a compiled shared library.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ctypes
import ctypes.util
from pathlib import Path


class GaussianFitterLib:
    """Wrapper for the CUDA Gaussian fitter library using ctypes"""

    def __init__(self, lib_path=None):
        if lib_path is None:
            # Try to find the library
            lib_path = self._find_library()

        self.lib = ctypes.CDLL(lib_path)

        # Configure function signatures
        self._configure_functions()

    def _find_library(self):
        """Find the shared library"""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(script_dir, 'build')

        # Possible library paths - prioritize build directory
        lib_paths = [
            os.path.join(build_dir, 'libgaussian_fitter.so'),
            os.path.join(script_dir, 'libgaussian_fitter.so'),
            './build/libgaussian_fitter.so',
            './libgaussian_fitter.so',
        ]

        for path in lib_paths:
            if os.path.exists(path):
                return path

        # Try to build it
        print("Library not found. Attempting to build...")
        os.system("make ctypes")
        print("Build complete. Loading library...")

        for path in lib_paths:
            if os.path.exists(path):
                return path

        raise RuntimeError("Cannot find gaussian_fitter library. Run 'make ctypes' first.")

    def _configure_functions(self):
        """Configure ctypes function signatures"""

        # generate_histogram_ctypes
        self.lib.generate_histogram_ctypes.restype = ctypes.POINTER(ctypes.c_int)
        self.lib.generate_histogram_ctypes.argtypes = [
            ctypes.c_float,  # A
            ctypes.c_float,  # x0
            ctypes.c_float,  # y0
            ctypes.c_float,  # sigma_x
            ctypes.c_float,  # sigma_y
            ctypes.c_float,  # rho
            ctypes.c_int,    # nx
            ctypes.c_int,    # ny
            ctypes.c_float,  # x_min
            ctypes.c_float,  # x_max
            ctypes.c_float,  # y_min
            ctypes.c_float,  # y_max
            ctypes.c_int,    # num_samples
            ctypes.POINTER(ctypes.c_int),  # nx_out
            ctypes.POINTER(ctypes.c_int),  # ny_out
            ctypes.POINTER(ctypes.c_int),  # total_counts_out
        ]

        # fit_histogram_ctypes
        self.lib.fit_histogram_ctypes.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.fit_histogram_ctypes.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # histogram_data
            ctypes.c_int,    # nx
            ctypes.c_int,    # ny
            ctypes.c_float,  # x_min
            ctypes.c_float,  # x_max
            ctypes.c_float,  # y_min
            ctypes.c_float,  # y_max
            ctypes.c_float,  # A_init
            ctypes.c_float,  # x0_init
            ctypes.c_float,  # y0_init
            ctypes.c_float,  # sigma_x_init
            ctypes.c_float,  # sigma_y_init
            ctypes.c_float,  # rho_init
            ctypes.c_float,  # learning_rate
            ctypes.c_int,    # max_iterations
            ctypes.c_float,  # tolerance
            ctypes.c_float,  # gradient_epsilon
            ctypes.c_int,    # verbose
            ctypes.c_int,    # timing_save_interval
            ctypes.POINTER(ctypes.c_int),  # iterations_out
            ctypes.POINTER(ctypes.c_int),  # converged_out
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # iteration_times_out
            ctypes.POINTER(ctypes.c_int),  # num_times_out
        ]

        # free_histogram_ctypes
        self.lib.free_histogram_ctypes.argtypes = [ctypes.POINTER(ctypes.c_int)]

        # free_fit_result_ctypes
        self.lib.free_fit_result_ctypes.argtypes = [ctypes.POINTER(ctypes.c_float)]

        # free_iteration_times_ctypes
        self.lib.free_iteration_times_ctypes.argtypes = [ctypes.POINTER(ctypes.c_float)]

    def generate_histogram(self, A, x0, y0, sigma_x, sigma_y, rho,
                          nx, ny, x_min, x_max, y_min, y_max, num_samples):
        """Generate 2D Gaussian histogram"""
        nx_out = ctypes.c_int()
        ny_out = ctypes.c_int()
        total_out = ctypes.c_int()

        data_ptr = self.lib.generate_histogram_ctypes(
            ctypes.c_float(A), ctypes.c_float(x0), ctypes.c_float(y0),
            ctypes.c_float(sigma_x), ctypes.c_float(sigma_y), ctypes.c_float(rho),
            ctypes.c_int(nx), ctypes.c_int(ny),
            ctypes.c_float(x_min), ctypes.c_float(x_max),
            ctypes.c_float(y_min), ctypes.c_float(y_max),
            ctypes.c_int(num_samples),
            ctypes.byref(nx_out), ctypes.byref(ny_out), ctypes.byref(total_out)
        )

        # Copy data to numpy array
        nbins = nx * ny
        histogram = np.array([data_ptr[i] for i in range(nbins)], dtype=np.int32)

        # Free the GPU memory
        self.lib.free_histogram_ctypes(data_ptr)

        return {
            'data': histogram.reshape(ny, nx),
            'nx': nx,
            'ny': ny,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'total_counts': total_out.value
        }

    def fit_histogram(self, histogram_data, nx, ny, x_min, x_max, y_min, y_max,
                     A_init, x0_init, y0_init, sigma_x_init, sigma_y_init, rho_init,
                     learning_rate=1.0e-6, max_iterations=5000, tolerance=1.0e-6,
                     gradient_epsilon=1.0e-4, verbose=True, timing_save_interval=100):
        """Fit 2D Gaussian to histogram data"""

        # Ensure contiguous array
        hist_flat = np.ascontiguousarray(histogram_data.flatten(), dtype=np.int32)
        hist_ptr = hist_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        iterations_out = ctypes.c_int()
        converged_out = ctypes.c_int()
        iteration_times_out = ctypes.POINTER(ctypes.c_float)()
        num_times_out = ctypes.c_int()

        result_ptr = self.lib.fit_histogram_ctypes(
            hist_ptr,
            ctypes.c_int(nx), ctypes.c_int(ny),
            ctypes.c_float(x_min), ctypes.c_float(x_max),
            ctypes.c_float(y_min), ctypes.c_float(y_max),
            ctypes.c_float(A_init), ctypes.c_float(x0_init), ctypes.c_float(y0_init),
            ctypes.c_float(sigma_x_init), ctypes.c_float(sigma_y_init), ctypes.c_float(rho_init),
            ctypes.c_float(learning_rate),
            ctypes.c_int(max_iterations),
            ctypes.c_float(tolerance),
            ctypes.c_float(gradient_epsilon),
            ctypes.c_int(1 if verbose else 0),
            ctypes.c_int(timing_save_interval),
            ctypes.byref(iterations_out),
            ctypes.byref(converged_out),
            ctypes.byref(iteration_times_out),
            ctypes.byref(num_times_out)
        )

        # Copy result
        result = {
            'final_likelihood': result_ptr[0],
            'params': {
                'A': result_ptr[1],
                'x0': result_ptr[2],
                'y0': result_ptr[3],
                'sigma_x': result_ptr[4],
                'sigma_y': result_ptr[5],
                'rho': result_ptr[6],
            },
            'iterations': iterations_out.value,
            'converged': converged_out.value != 0,
            'iteration_times': []
        }

        # Copy iteration times
        num_times = num_times_out.value
        if num_times > 0 and iteration_times_out:
            # Cast to array to access elements
            array_type = ctypes.c_float * num_times
            times_array = ctypes.cast(iteration_times_out, ctypes.POINTER(array_type)).contents
            result['iteration_times'] = [times_array[i] for i in range(num_times)]
            self.lib.free_iteration_times_ctypes(iteration_times_out)

        # Free result
        self.lib.free_fit_result_ctypes(result_ptr)

        return result


class GaussianFitterConfig:
    """Configuration class for Gaussian fitting"""
    def __init__(self):
        # True parameters (for generating test data)
        self.true_A = 10000.0
        self.true_x0 = 0.0
        self.true_y0 = 0.0
        self.true_sigma_x = 1.0
        self.true_sigma_y = 1.5
        self.true_rho = 0.3

        # Histogram settings
        self.nx = 64
        self.ny = 64
        self.x_min = -5.0
        self.x_max = 5.0
        self.y_min = -5.0
        self.y_max = 5.0
        self.num_samples = 100000

        # Initial guess
        self.init_A = 10000.0
        self.init_x0 = 0.5
        self.init_y0 = -0.5
        self.init_sigma_x = 1.2
        self.init_sigma_y = 1.7
        self.init_rho = 0.2

        # Optimizer settings
        self.learning_rate = 1.0e-6
        self.max_iterations = 5000
        self.tolerance = 1.0e-6
        self.gradient_epsilon = 1.0e-4
        self.verbose = True

        # Output settings
        self.output_dir = "output"
        self.timing_save_interval = 100  # Save iteration times every n iterations
        self.save_plots = True


def load_config_from_file(filename):
    """Load configuration from file"""
    config = GaussianFitterConfig()

    if not os.path.exists(filename):
        return config

    # Simple INI parser
    current_section = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].lower()
            elif '=' in line and current_section:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()

                if current_section == 'true_params':
                    if key == 'a': config.true_A = float(value)
                    elif key == 'x0': config.true_x0 = float(value)
                    elif key == 'y0': config.true_y0 = float(value)
                    elif key == 'sigma_x': config.true_sigma_x = float(value)
                    elif key == 'sigma_y': config.true_sigma_y = float(value)
                    elif key == 'rho': config.true_rho = float(value)
                elif current_section == 'histogram':
                    if key == 'nx': config.nx = int(value)
                    elif key == 'ny': config.ny = int(value)
                    elif key == 'x_min': config.x_min = float(value)
                    elif key == 'x_max': config.x_max = float(value)
                    elif key == 'y_min': config.y_min = float(value)
                    elif key == 'y_max': config.y_max = float(value)
                    elif key == 'num_samples': config.num_samples = int(value)
                elif current_section == 'initial_guess':
                    if key == 'a': config.init_A = float(value)
                    elif key == 'x0': config.init_x0 = float(value)
                    elif key == 'y0': config.init_y0 = float(value)
                    elif key == 'sigma_x': config.init_sigma_x = float(value)
                    elif key == 'sigma_y': config.init_sigma_y = float(value)
                    elif key == 'rho': config.init_rho = float(value)
                elif current_section == 'optimizer':
                    if key == 'learning_rate': config.learning_rate = float(value)
                    elif key == 'max_iterations': config.max_iterations = int(value)
                    elif key == 'tolerance': config.tolerance = float(value)
                    elif key == 'gradient_epsilon': config.gradient_epsilon = float(value)
                    elif key == 'verbose': config.verbose = value.lower() in ('true', '1', 'yes')
                elif current_section == 'output':
                    if key == 'output_dir': config.output_dir = value
                    elif key == 'timing_save_interval': config.timing_save_interval = int(value)
                    elif key == 'save_plots': config.save_plots = value.lower() in ('true', '1', 'yes')

    return config


def compute_1d_projection(histogram_2d, x_min, x_max, y_min, y_max):
    """Project 2D histogram to 1D (X and Y)"""
    ny, nx = histogram_2d.shape

    # X projection (sum over Y)
    x_counts = np.sum(histogram_2d, axis=0)
    x_bins = np.linspace(x_min, x_max, nx, endpoint=False) + (x_max - x_min) / nx / 2

    # Y projection (sum over X)
    y_counts = np.sum(histogram_2d, axis=1)
    y_bins = np.linspace(y_min, y_max, ny, endpoint=False) + (y_max - y_min) / ny / 2

    return {
        'x_bins': x_bins,
        'x_counts': x_counts,
        'y_bins': y_bins,
        'y_counts': y_counts
    }


def compute_1d_gaussian_curve(bin_centers, mean, sigma, total_counts):
    """Compute 1D Gaussian curve for plotting"""
    norm_factor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    z = (bin_centers - mean) / sigma
    pdf = norm_factor * np.exp(-0.5 * z ** 2)
    curve = pdf * bin_width * total_counts

    return curve


def plot_results(histogram_2d, projections, result, true_params, config):
    """Generate and save all plots"""
    if not config.save_plots:
        return

    os.makedirs(config.output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Generating Plots")
    print("=" * 50)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10

    output_path = Path(config.output_dir)

    # 1. X Distribution
    print("Plotting X distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_bins = projections['x_bins']
    x_counts = projections['x_counts']
    x_width = x_bins[1] - x_bins[0] if len(x_bins) > 1 else 1.0

    ax.bar(x_bins, x_counts, width=x_width*0.8,
           alpha=0.6, label='Observed', color='steelblue', edgecolor='black')

    total_counts = np.sum(histogram_2d)
    fit_curve_x = compute_1d_gaussian_curve(x_bins, result['params']['x0'],
                                            result['params']['sigma_x'], total_counts)
    ax.plot(x_bins, fit_curve_x, 'r-', linewidth=2, label='Fitted', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Counts')
    ax.set_title('X Distribution: Histogram vs Fitted Gaussian')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'histogram_x.jpg', format='jpg', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'histogram_x.jpg'}")

    # 2. Y Distribution
    print("Plotting Y distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    y_bins = projections['y_bins']
    y_counts = projections['y_counts']
    y_width = y_bins[1] - y_bins[0] if len(y_bins) > 1 else 1.0

    ax.bar(y_bins, y_counts, width=y_width*0.8,
           alpha=0.6, label='Observed', color='steelblue', edgecolor='black')

    fit_curve_y = compute_1d_gaussian_curve(y_bins, result['params']['y0'],
                                            result['params']['sigma_y'], total_counts)
    ax.plot(y_bins, fit_curve_y, 'r-', linewidth=2, label='Fitted', alpha=0.8)

    ax.set_xlabel('Y')
    ax.set_ylabel('Counts')
    ax.set_title('Y Distribution: Histogram vs Fitted Gaussian')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'histogram_y.jpg', format='jpg', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'histogram_y.jpg'}")

    # 3. 2D Histogram
    print("Plotting 2D histogram...")
    fig, ax = plt.subplots(figsize=(8, 7))

    x_edges = np.linspace(config.x_min, config.x_max, config.nx + 1)
    y_edges = np.linspace(config.y_min, config.y_max, config.ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    im = ax.pcolormesh(x_centers, y_centers, histogram_2d,
                       shading='auto', cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Gaussian Histogram')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts')
    plt.tight_layout()
    plt.savefig(output_path / 'histogram_2d.jpg', format='jpg', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'histogram_2d.jpg'}")

    # 4. Iteration Time vs Iteration Number
    print("Plotting iteration times...")
    if 'iteration_times' in result and result['iteration_times']:
        fig, ax = plt.subplots(figsize=(10, 5))
        iter_times = result['iteration_times']

        # Calculate actual iteration numbers based on timing_save_interval
        # Iterations are saved at: 1, 1+interval, 1+2*interval, ...
        interval = config.timing_save_interval
        iterations = [1 + i * interval for i in range(len(iter_times))]

        ax.plot(iterations, iter_times, 'b-o', linewidth=1, markersize=3, alpha=0.7)
        ax.set_xlabel('Iteration Number')
        ax.set_ylabel('Iteration Time (ms)')
        ax.set_title(f'Iteration Time vs Iteration Number (saved every {interval} iterations)')
        ax.grid(True, alpha=0.3)

        # Add average line
        avg_time = np.mean(iter_times)
        ax.axhline(y=avg_time, color='r', linestyle='--',
                   label=f'Average: {avg_time:.2f} ms')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / 'iteration_times.jpg', format='jpg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path / 'iteration_times.jpg'}")

        # Print statistics
        print(f"\n  Iteration time statistics (saved every {interval} iterations):")
        print(f"    Average: {avg_time:.2f} ms")
        print(f"    Min: {min(iter_times):.2f} ms")
        print(f"    Max: {max(iter_times):.2f} ms")
        print(f"    Total recorded time: {sum(iter_times):.2f} ms ({sum(iter_times)/1000:.2f} s)")

    print("=" * 50)

    # Save configuration and results to text files
    save_config_and_results(output_path, result, true_params, config)


def save_config_and_results(output_path, result, true_params, config):
    """Save configuration and fitting results to text files"""

    # Save configuration
    print("\nSaving configuration and results...")
    config_file = output_path / 'config.txt'
    with open(config_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  Configuration\n")
        f.write("=" * 60 + "\n\n")

        f.write("[true_params]\n")
        f.write(f"  A = {config.true_A}\n")
        f.write(f"  x0 = {config.true_x0}\n")
        f.write(f"  y0 = {config.true_y0}\n")
        f.write(f"  sigma_x = {config.true_sigma_x}\n")
        f.write(f"  sigma_y = {config.true_sigma_y}\n")
        f.write(f"  rho = {config.true_rho}\n\n")

        f.write("[histogram]\n")
        f.write(f"  nx = {config.nx}\n")
        f.write(f"  ny = {config.ny}\n")
        f.write(f"  x_min = {config.x_min}\n")
        f.write(f"  x_max = {config.x_max}\n")
        f.write(f"  y_min = {config.y_min}\n")
        f.write(f"  y_max = {config.y_max}\n")
        f.write(f"  num_samples = {config.num_samples}\n\n")

        f.write("[initial_guess]\n")
        f.write(f"  A = {config.init_A}\n")
        f.write(f"  x0 = {config.init_x0}\n")
        f.write(f"  y0 = {config.init_y0}\n")
        f.write(f"  sigma_x = {config.init_sigma_x}\n")
        f.write(f"  sigma_y = {config.init_sigma_y}\n")
        f.write(f"  rho = {config.init_rho}\n\n")

        f.write("[optimizer]\n")
        f.write(f"  learning_rate = {config.learning_rate}\n")
        f.write(f"  max_iterations = {config.max_iterations}\n")
        f.write(f"  tolerance = {config.tolerance}\n")
        f.write(f"  gradient_epsilon = {config.gradient_epsilon}\n")
        f.write(f"  verbose = {config.verbose}\n\n")

        f.write("[output]\n")
        f.write(f"  output_dir = {config.output_dir}\n")
        f.write(f"  timing_save_interval = {config.timing_save_interval}\n")
        f.write(f"  save_plots = {config.save_plots}\n")

    print(f"  Saved: {config_file}")

    # Save fitting results
    results_file = output_path / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  Fitting Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Optimization Status:\n")
        f.write(f"  Iterations: {result['iterations']}\n")
        f.write(f"  Converged: {result['converged']}\n")
        f.write(f"  Final Likelihood: {result['final_likelihood']:.6f}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'Parameter':<12} {'True Value':<15} {'Fitted Value':<15} {'Error (%)':<12}\n")
        f.write("-" * 60 + "\n")

        params = [
            ('A', true_params['A'], result['params']['A']),
            ('x0', true_params['x0'], result['params']['x0']),
            ('y0', true_params['y0'], result['params']['y0']),
            ('sigma_x', true_params['sigma_x'], result['params']['sigma_x']),
            ('sigma_y', true_params['sigma_y'], result['params']['sigma_y']),
            ('rho', true_params['rho'], result['params']['rho']),
        ]

        for name, true_val, fit_val in params:
            error = (fit_val - true_val) / true_val * 100.0 if true_val != 0 else 0
            f.write(f"{name:<12} {true_val:<15.6f} {fit_val:<15.6f} {error:<12.2f}\n")

        f.write("-" * 60 + "\n")

        # Save iteration time statistics if available
        if 'iteration_times' in result and result['iteration_times']:
            f.write(f"\nIteration Time Statistics:\n")
            f.write(f"  Number of recorded iterations: {len(result['iteration_times'])}\n")
            f.write(f"  Average: {np.mean(result['iteration_times']):.2f} ms\n")
            f.write(f"  Min: {min(result['iteration_times']):.2f} ms\n")
            f.write(f"  Max: {max(result['iteration_times']):.2f} ms\n")
            f.write(f"  Total: {sum(result['iteration_times']):.2f} ms ({sum(result['iteration_times'])/1000:.2f} s)\n")

    print(f"  Saved: {results_file}")

    # Also save results as CSV for easy parsing
    csv_file = output_path / 'results.csv'
    with open(csv_file, 'w') as f:
        f.write("parameter,true_value,fitted_value,error_percent\n")
        for name, true_val, fit_val in params:
            error = (fit_val - true_val) / true_val * 100.0 if true_val != 0 else 0
            f.write(f"{name},{true_val:.6f},{fit_val:.6f},{error:.2f}\n")
        f.write(f"iterations,-,{result['iterations']},-\n")
        f.write(f"converged,-,{1 if result['converged'] else 0},-\n")
        f.write(f"final_likelihood,-,{result['final_likelihood']:.6f},-\n")

    print(f"  Saved: {csv_file}")


def print_results(true_params, fit_params, result):
    """Print fitting results"""
    print("\n" + "=" * 50)
    print("  Parameter Comparison")
    print("=" * 50)
    print(f"{'Parameter':<12} {'True Value':<12} {'Fitted Value':<12} {'Error (%)':<12}")
    print("-" * 50)

    params = [
        ('A', true_params['A'], fit_params['A']),
        ('x0', true_params['x0'], fit_params['x0']),
        ('y0', true_params['y0'], fit_params['y0']),
        ('sigma_x', true_params['sigma_x'], fit_params['sigma_x']),
        ('sigma_y', true_params['sigma_y'], fit_params['sigma_y']),
        ('rho', true_params['rho'], fit_params['rho']),
    ]

    for name, true_val, fit_val in params:
        error = (fit_val - true_val) / true_val * 100.0 if true_val != 0 else 0
        print(f"{name:<12} {true_val:<12.4f} {fit_val:<12.4f} {error:<12.2f}")

    print("-" * 50)
    print(f"\nIterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    print(f"Final Likelihood: {result['final_likelihood']:.4f}")
    print("=" * 50)


def main():
    """Main entry point"""
    print("=" * 50)
    print("  CUDA 2D Gaussian Fitter")
    print("  Python Frontend with ctypes")
    print("=" * 50)

    # Parse command line arguments
    config_file = "config_default.conf"
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python run_ctypes.py [config_file]")
            print("  config_file: Path to configuration file (default: config_default.conf)")
            sys.exit(0)
        config_file = sys.argv[1]

    print(f"\nConfig file: {config_file}")

    # Load configuration
    config = load_config_from_file(config_file)

    print("\n" + "=" * 50)
    print("  Configuration")
    print("=" * 50)
    print(f"\n[True Parameters]")
    print(f"  A = {config.true_A}, x0 = {config.true_x0}, y0 = {config.true_y0}")
    print(f"  sigma_x = {config.true_sigma_x}, sigma_y = {config.true_sigma_y}, rho = {config.true_rho}")
    print(f"\n[Histogram]")
    print(f"  bins: {config.nx}x{config.ny}, num_samples: {config.num_samples}")
    print(f"\n[Optimizer]")
    print(f"  learning_rate = {config.learning_rate}, max_iterations = {config.max_iterations}")
    print("=" * 50)

    # Load the library
    print("\nLoading CUDA library...")
    lib = GaussianFitterLib()

    # Step 1: Generate histogram using C++ code
    print("\nGenerating histogram...")
    hist_result = lib.generate_histogram(
        config.true_A, config.true_x0, config.true_y0,
        config.true_sigma_x, config.true_sigma_y, config.true_rho,
        config.nx, config.ny,
        config.x_min, config.x_max,
        config.y_min, config.y_max,
        config.num_samples
    )

    histogram_2d = hist_result['data']
    print(f"Histogram size: {config.nx}x{config.ny} bins")
    print(f"Total counts: {hist_result['total_counts']}")

    # Step 2: Project to 1D (for plotting)
    print("\nProjecting histogram to 1D...")
    projections = compute_1d_projection(histogram_2d, config.x_min, config.x_max,
                                       config.y_min, config.y_max)

    # Step 3: Fit the histogram
    print("\nStarting optimization...")
    result = lib.fit_histogram(
        histogram_2d, config.nx, config.ny,
        config.x_min, config.x_max, config.y_min, config.y_max,
        config.init_A, config.init_x0, config.init_y0,
        config.init_sigma_x, config.init_sigma_y, config.init_rho,
        config.learning_rate, config.max_iterations, config.tolerance,
        config.gradient_epsilon, config.verbose, config.timing_save_interval
    )

    # Step 4: Print results
    true_params = {
        'A': config.true_A, 'x0': config.true_x0, 'y0': config.true_y0,
        'sigma_x': config.true_sigma_x, 'sigma_y': config.true_sigma_y, 'rho': config.true_rho
    }
    print_results(true_params, result['params'], result)

    # Step 5: Generate plots
    plot_results(histogram_2d, projections, result, true_params, config)

    print(f"\nAll outputs saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
