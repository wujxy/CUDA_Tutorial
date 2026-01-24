#include "../include/config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <ctype.h>

Config::Config() {
    // 默认真实参数
    true_params.A = 10000.0f;
    true_params.x0 = 0.0f;
    true_params.y0 = 0.0f;
    true_params.sigma_x = 1.0f;
    true_params.sigma_y = 1.5f;
    true_params.rho = 0.3f;

    // 默认直方图配置
    nx = 64;
    ny = 64;
    x_min = -5.0f;
    x_max = 5.0f;
    y_min = -5.0f;
    y_max = 5.0f;
    num_samples = 100000;

    // 默认初始猜测
    initial_guess.A = 10000.0f;
    initial_guess.x0 = 0.5f;
    initial_guess.y0 = -0.5f;
    initial_guess.sigma_x = 1.2f;
    initial_guess.sigma_y = 1.7f;
    initial_guess.rho = 0.2f;

    // 默认优化器配置
    learning_rate = 1.0e-6f;
    max_iterations = 5000;
    tolerance = 1.0e-6f;
    gradient_epsilon = 1.0e-4f;
    verbose = true;

    // 默认输出配置
    output_dir = "output";
    timing_save_interval = 100;
    save_plots = true;
}

bool Config::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open config file: " << filename << std::endl;
        return false;
    }

    std::map<std::string, std::string> params;
    std::string current_section = "";

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (!parseLine(line, current_section, params)) {
            std::cerr << "Warning: Invalid line " << line_num << " in " << filename << std::endl;
        }
    }

    file.close();

    // 读取参数
    // [true_params]
    true_params.A = getFloat(params, "true_params", "A", true_params.A);
    true_params.x0 = getFloat(params, "true_params", "x0", true_params.x0);
    true_params.y0 = getFloat(params, "true_params", "y0", true_params.y0);
    true_params.sigma_x = getFloat(params, "true_params", "sigma_x", true_params.sigma_x);
    true_params.sigma_y = getFloat(params, "true_params", "sigma_y", true_params.sigma_y);
    true_params.rho = getFloat(params, "true_params", "rho", true_params.rho);

    // [histogram]
    nx = getInt(params, "histogram", "nx", nx);
    ny = getInt(params, "histogram", "ny", ny);
    x_min = getFloat(params, "histogram", "x_min", x_min);
    x_max = getFloat(params, "histogram", "x_max", x_max);
    y_min = getFloat(params, "histogram", "y_min", y_min);
    y_max = getFloat(params, "histogram", "y_max", y_max);
    num_samples = getInt(params, "histogram", "num_samples", num_samples);

    // [initial_guess]
    initial_guess.A = getFloat(params, "initial_guess", "A", initial_guess.A);
    initial_guess.x0 = getFloat(params, "initial_guess", "x0", initial_guess.x0);
    initial_guess.y0 = getFloat(params, "initial_guess", "y0", initial_guess.y0);
    initial_guess.sigma_x = getFloat(params, "initial_guess", "sigma_x", initial_guess.sigma_x);
    initial_guess.sigma_y = getFloat(params, "initial_guess", "sigma_y", initial_guess.sigma_y);
    initial_guess.rho = getFloat(params, "initial_guess", "rho", initial_guess.rho);

    // [optimizer]
    learning_rate = getFloat(params, "optimizer", "learning_rate", learning_rate);
    max_iterations = getInt(params, "optimizer", "max_iterations", max_iterations);
    tolerance = getFloat(params, "optimizer", "tolerance", tolerance);
    gradient_epsilon = getFloat(params, "optimizer", "gradient_epsilon", gradient_epsilon);
    verbose = getBool(params, "optimizer", "verbose", verbose);

    // [output]
    output_dir = params["output_output_dir"].empty() ? output_dir : params["output_output_dir"];
    timing_save_interval = getInt(params, "output", "timing_save_interval", timing_save_interval);
    save_plots = getBool(params, "output", "save_plots", save_plots);

    std::cout << "Loaded config from: " << filename << std::endl;
    return true;
}

bool Config::parseLine(const std::string& line, std::string& section, std::map<std::string, std::string>& params) {
    // 去除前后空白
    std::string trimmed = line;
    size_t start = trimmed.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        // 空行或纯空白行，跳过
        return true;
    }
    size_t end = trimmed.find_last_not_of(" \t\r\n");
    trimmed = trimmed.substr(start, end - start + 1);

    // 跳过注释行
    if (trimmed.empty() || trimmed[0] == '#') {
        return true;
    }

    // 检查section header: [section_name]
    if (trimmed[0] == '[' && trimmed[trimmed.length() - 1] == ']') {
        section = trimmed.substr(1, trimmed.length() - 2);
        // 转换为小写
        std::transform(section.begin(), section.end(), section.begin(), ::tolower);
        return true;
    }

    // 解析 key = value
    size_t eq_pos = trimmed.find('=');
    if (eq_pos != std::string::npos) {
        std::string key = trimmed.substr(0, eq_pos);
        std::string value = trimmed.substr(eq_pos + 1);

        // 去除key和value的空白
        start = key.find_first_not_of(" \t");
        end = key.find_last_not_of(" \t");
        if (start != std::string::npos) {
            key = key.substr(start, end - start + 1);
        }
        start = value.find_first_not_of(" \t");
        end = value.find_last_not_of(" \t");
        if (start != std::string::npos) {
            value = value.substr(start, end - start + 1);
        }

        // 转换key为小写
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);

        // 存储参数: section_key = value
        std::string full_key = section + "_" + key;
        params[full_key] = value;

        return true;
    }

    return false;
}

float Config::getFloat(const std::map<std::string, std::string>& params, const std::string& section, const std::string& key, float default_value) {
    std::string full_key = section + "_" + key;
    auto it = params.find(full_key);
    if (it != params.end()) {
        return std::stof(it->second);
    }
    return default_value;
}

int Config::getInt(const std::map<std::string, std::string>& params, const std::string& section, const std::string& key, int default_value) {
    std::string full_key = section + "_" + key;
    auto it = params.find(full_key);
    if (it != params.end()) {
        return std::stoi(it->second);
    }
    return default_value;
}

bool Config::getBool(const std::map<std::string, std::string>& params, const std::string& section, const std::string& key, bool default_value) {
    std::string full_key = section + "_" + key;
    auto it = params.find(full_key);
    if (it != params.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return (value == "true" || value == "1" || value == "yes");
    }
    return default_value;
}

void Config::print() const {
    std::cout << "========================================" << std::endl;
    std::cout << "  Configuration" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[True Parameters]" << std::endl;
    std::cout << "  A = " << true_params.A << std::endl;
    std::cout << "  x0 = " << true_params.x0 << std::endl;
    std::cout << "  y0 = " << true_params.y0 << std::endl;
    std::cout << "  sigma_x = " << true_params.sigma_x << std::endl;
    std::cout << "  sigma_y = " << true_params.sigma_y << std::endl;
    std::cout << "  rho = " << true_params.rho << std::endl;

    std::cout << "\n[Histogram]" << std::endl;
    std::cout << "  bins: " << nx << " x " << ny << std::endl;
    std::cout << "  x range: [" << x_min << ", " << x_max << "]" << std::endl;
    std::cout << "  y range: [" << y_min << ", " << y_max << "]" << std::endl;
    std::cout << "  num_samples: " << num_samples << std::endl;

    std::cout << "\n[Initial Guess]" << std::endl;
    std::cout << "  A = " << initial_guess.A << std::endl;
    std::cout << "  x0 = " << initial_guess.x0 << std::endl;
    std::cout << "  y0 = " << initial_guess.y0 << std::endl;
    std::cout << "  sigma_x = " << initial_guess.sigma_x << std::endl;
    std::cout << "  sigma_y = " << initial_guess.sigma_y << std::endl;
    std::cout << "  rho = " << initial_guess.rho << std::endl;

    std::cout << "\n[Optimizer]" << std::endl;
    std::cout << "  learning_rate = " << learning_rate << std::endl;
    std::cout << "  max_iterations = " << max_iterations << std::endl;
    std::cout << "  tolerance = " << tolerance << std::endl;
    std::cout << "  gradient_epsilon = " << gradient_epsilon << std::endl;
    std::cout << "  verbose = " << (verbose ? "true" : "false") << std::endl;

    std::cout << "\n[Output]" << std::endl;
    std::cout << "  output_dir = " << output_dir << std::endl;
    std::cout << "  timing_save_interval = " << timing_save_interval << std::endl;
    std::cout << "  save_plots = " << (save_plots ? "true" : "false") << std::endl;

    std::cout << "========================================" << std::endl;
}
