#!/usr/bin/env python3
"""
CUDA 2D Gaussian Fitter - Python Frontend

This script serves as the frontend for the CUDA-accelerated 2D Gaussian fitter.
It uses pybind11 to call C++ code directly without intermediate files.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Try to import the core module
try:
    import gaussian_fitter_core as gfc
except ImportError:
    print("Error: Cannot import gaussian_fitter_core module.")
    print("Please build the project first: make pybind")
    sys.exit(1)


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
        self.save_plots = True


def load_config_from_file(filename):
    """Load configuration from file"""
    config = GaussianFitterConfig()

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

                # Parse based on section and key
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
                    elif key == 'save_plots': config.save_plots = value.lower() in ('true', '1', 'yes')

    return config


def print_config(config):
    """Print configuration"""
    print("=" * 50)
    print("  Configuration")
    print("=" * 50)
    print(f"\n[True Parameters]")
    print(f"  A = {config.true_A}")
    print(f"  x0 = {config.true_x0}")
    print(f"  y0 = {config.true_y0}")
    print(f"  sigma_x = {config.true_sigma_x}")
    print(f"  sigma_y = {config.true_sigma_y}")
    print(f"  rho = {config.true_rho}")

    print(f"\n[Histogram]")
    print(f"  bins: {config.nx} x {config.ny}")
    print(f"  x range: [{config.x_min}, {config.x_max}]")
    print(f"  y range: [{config.y_min}, {config.y_max}]")
    print(f"  num_samples: {config.num_samples}")

    print(f"\n[Initial Guess]")
    print(f"  A = {config.init_A}")
    print(f"  x0 = {config.init_x0}")
    print(f"  y0 = {config.init_y0}")
    print(f"  sigma_x = {config.init_sigma_x}")
    print(f"  sigma_y = {config.init_sigma_y}")
    print(f"  rho = {config.init_rho}")

    print(f"\n[Optimizer]")
    print(f"  learning_rate = {config.learning_rate}")
    print(f"  max_iterations = {config.max_iterations}")
    print(f"  tolerance = {config.tolerance}")
    print(f"  verbose = {config.verbose}")
    print("=" * 50)


def print_results(true_params, fit_params, result):
    """Print fitting results"""
    print("\n" + "=" * 50)
    print("  Parameter Comparison")
    print("=" * 50)
    print(f"{'Parameter':<12} {'True Value':<12} {'Fitted Value':<12} {'Error (%)':<12}")
    print("-" * 50)

    params = [
        ('A', true_params.A, fit_params.A),
        ('x0', true_params.x0, fit_params.x0),
        ('y0', true_params.y0, fit_params.y0),
        ('sigma_x', true_params.sigma_x, fit_params.sigma_x),
        ('sigma_y', true_params.sigma_y, fit_params.sigma_y),
        ('rho', true_params.rho, fit_params.rho),
    ]

    for name, true_val, fit_val in params:
        error = (fit_val - true_val) / true_val * 100.0 if true_val != 0 else 0
        print(f"{name:<12} {true_val:<12.4f} {fit_val:<12.4f} {error:<12.2f}")

    print("-" * 50)

    if result.iteration_times:
        avg_time = np.mean(result.iteration_times)
        total_time = np.sum(result.iteration_times)
        print(f"\nIterations: {result.iterations}")
    print(f"Converged: {result.converged}")
    print(f"Final Likelihood: {result.final_likelihood:.4f}")
    if result.iteration_times:
        print(f"Avg Iteration Time: {avg_time:.4f} ms")
        print(f"Total Time: {total_time:.4f} ms")
    print("=" * 50)


def plot_results(histogram_2d, projections, result, true_params, config):
    """Generate and save all plots"""

    if not config.save_plots:
        return

    # Create output directory
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

    # Compute fitted curve
    total_counts = np.sum(histogram_2d)
    fit_curve_x = gfc.compute_1d_curve(
        x_bins.astype(np.float32),
        result.params.x0, result.params.sigma_x, total_counts
    )
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

    fit_curve_y = gfc.compute_1d_curve(
        y_bins.astype(np.float32),
        result.params.y0, result.params.sigma_y, total_counts
    )
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

    im = ax.pcolormesh(x_centers, y_centers, histogram_2d.reshape(config.ny, config.nx),
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

    # 4. Iteration Times
    if result.iteration_times:
        print("Plotting iteration times...")
        fig, ax = plt.subplots(figsize=(10, 5))
        iterations = np.arange(1, len(result.iteration_times) + 1)
        ax.plot(iterations, result.iteration_times, 'b-', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Iteration Time vs Iteration Number')

        avg_time = np.mean(result.iteration_times)
        ax.axhline(y=avg_time, color='r', linestyle='--', linewidth=2,
                   label=f'Average: {avg_time:.2f} ms')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / 'iteration_times.jpg', format='jpg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path / 'iteration_times.jpg'}")

    # 5. Likelihood History
    if result.likelihood_history:
        print("Plotting likelihood history...")
        fig, ax = plt.subplots(figsize=(10, 5))
        iterations = np.arange(1, len(result.likelihood_history) + 1)
        ax.plot(iterations, result.likelihood_history, 'b-', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Likelihood')
        ax.set_title('Likelihood History')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / 'likelihood_history.jpg', format='jpg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path / 'likelihood_history.jpg'}")

    print("=" * 50)


def main():
    """Main entry point"""
    print("=" * 50)
    print("  CUDA 2D Gaussian Fitter")
    print("  Python Frontend with pybind11")
    print("=" * 50)

    # Parse command line arguments
    config_file = "config_default.conf"
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python run_py.py [config_file]")
            print("  config_file: Path to configuration file (default: config_default.conf)")
            sys.exit(0)
        config_file = sys.argv[1]

    print(f"\nConfig file: {config_file}")

    # Load configuration
    try:
        config = load_config_from_file(config_file)
    except FileNotFoundError:
        print(f"Warning: Config file '{config_file}' not found, using defaults.")
        config = GaussianFitterConfig()

    print_config(config)

    # Step 1: Generate histogram using C++ code
    print("\nGenerating histogram...")
    hist = gfc.generate_histogram(
        config.true_A, config.true_x0, config.true_y0,
        config.true_sigma_x, config.true_sigma_y, config.true_rho,
        config.nx, config.ny,
        config.x_min, config.x_max,
        config.y_min, config.y_max,
        config.num_samples
    )

    # Convert to numpy array
    histogram_2d = np.array(hist.data, dtype=np.int32).reshape(config.ny, config.nx)
    total_counts = np.sum(histogram_2d)
    print(f"Histogram size: {config.nx}x{config.ny} bins")
    print(f"Total counts: {total_counts}")

    # Step 2: Project to 1D (for plotting)
    print("\nProjecting histogram to 1D...")
    projections = gfc.project_histogram(
        np.ascontiguousarray(histogram_2d.flatten()),
        config.nx, config.ny,
        config.x_min, config.x_max,
        config.y_min, config.y_max
    )

    # Step 3: Fit the histogram
    print("\nStarting optimization...")
    result = gfc.fit_histogram(
        np.ascontiguousarray(histogram_2d.flatten()),
        config.nx, config.ny,
        config.x_min, config.x_max,
        config.y_min, config.y_max,
        config.init_A, config.init_x0, config.init_y0,
        config.init_sigma_x, config.init_sigma_y, config.init_rho,
        config.learning_rate,
        config.max_iterations,
        config.tolerance,
        config.gradient_epsilon,
        config.verbose
    )

    # Step 4: Print results
    true_params = gfc.GaussianParams()
    true_params.A = config.true_A
    true_params.x0 = config.true_x0
    true_params.y0 = config.true_y0
    true_params.sigma_x = config.true_sigma_x
    true_params.sigma_y = config.true_sigma_y
    true_params.rho = config.true_rho

    print_results(true_params, result.params, result)

    # Step 5: Generate plots
    plot_results(histogram_2d, projections, result, true_params, config)

    print(f"\nAll outputs saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
