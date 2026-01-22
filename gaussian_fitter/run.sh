#!/bin/bash

# CUDA 2D高斯拟合器运行脚本

# 设置默认配置文件
CONFIG_FILE="config_high_precision.conf"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-c|--config CONFIG_FILE]"
            echo ""
            echo "Arguments:"
            echo "  -c, --config  Specify configuration file"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Use default config"
            echo "  $0 -c config_high_precision.conf   # Use custom config"
            echo "  $0 -c my_test.conf               # Use custom config"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "  CUDA 2D Gaussian Fitter"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# 编译（如果需要）
if [ ! -f "./gaussion_fitter" ]; then
    echo "Binary not found, compiling..."
    make clean
    make
    echo ""
fi

# 运行程序
./gaussion_fitter "$CONFIG_FILE"
