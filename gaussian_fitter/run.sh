#!/bin/bash

# CUDA Poisson Likelihood Gaussian Fitter - 编译和运行脚本

echo "========================================="
echo "  CUDA Gaussian Fitter"
echo "========================================="
echo ""

# 清理之前的编译结果
echo "[1/3] Cleaning previous build..."
make clean

# 编译
echo ""
echo "[2/3] Compiling..."
make
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

# 运行
echo ""
echo "[3/3] Running..."
echo "========================================="
./gaussion_fitter

echo ""
echo "========================================="
echo "Done!"
