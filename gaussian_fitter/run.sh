#!/bin/bash

# CUDA 2D高斯拟合器运行脚本
# 使用Python作为前端，通过ctypes调用CUDA拟合库

# 设置Python路径
PYTHON_BIN="/home/yoru/py_venv/ml_env/bin/python"

# 设置默认配置文件
CONFIG_FILE="config_default.conf"

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
            echo "  $0                                      # Use default config"
            echo "  $0 -c config_high_precision.conf       # Use custom config"
            echo "  $0 -c my_test.conf                      # Use custom config"
            echo ""
            echo "Environment:"
            echo "  PYTHON_BIN: Path to Python interpreter (default: /home/yoru/py_venv/ml_env/bin/python)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

# 检查Python是否存在
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python not found at $PYTHON_BIN"
    echo "Please update PYTHON_BIN in run.sh or install Python"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "  CUDA 2D Gaussian Fitter"
echo "  Python Frontend (ctypes)"
echo "=========================================="
echo "Python: $PYTHON_BIN"
echo "Config file: $CONFIG_FILE"
echo ""

# 编译（如果需要）
if [ ! -f "./build/libgaussian_fitter.so" ]; then
    echo "Library not found, compiling..."
    make ctypes
    echo ""
fi

# 运行Python程序
"$PYTHON_BIN" share/run_ctypes.py "$CONFIG_FILE"
