# CUDA Makefile for Gaussian Fitter

NVCC = nvcc
TARGET = gaussion_fitter
PYTHON_MODULE = gaussian_fitter_core
CTYPES_LIB_NAME = libgaussian_fitter.so
BUILD_DIR = build
CTYPES_LIB = $(BUILD_DIR)/$(CTYPES_LIB_NAME)

# 源文件
CPP_SRC = run.cpp
CU_SRC = src/model.cu src/fit_model.cu src/config.cpp src/cuda_utils.cu src/visualization.cpp

# pybind11 源文件
PYBIND_SRC = src/python_bindings.cpp

# ctypes 接口源文件
CTYPES_SRC = src/ctypes_interface.cpp

# 包含目录
INCLUDE = -Iinclude

# pybind11 设置
# 检测pybind11安装路径
PYTHON_VERSION := $(shell python3 --version 2>&1 | grep -oP '\d+\.\d+')
PYTHON_INCLUDE_DIR := /usr/include/python3.12

# 尝试常见的pybind11安装位置
PYBIND11_INCLUDE := $(shell python3 -c 'import pybind11; print(pybind11.get_include())' 2>/dev/null)
ifeq ($(PYBIND11_INCLUDE),)
    # 备用路径
    PYBIND11_INCLUDE := /usr/include/pybind11
endif
ifeq ($(wildcard $(PYBIND11_INCLUDE)),)
    PYBIND11_INCLUDE := /usr/local/include/pybind11
endif

# 编译选项
CFLAGS = -O3 -std=c++11
NVCC_FLAGS = -O3 -std=c++11 -arch=sm_52 -Xcompiler -fPIC
PYBIND_FLAGS = -O3 -std=c++11 -shared \
              -Xcompiler -fPIC \
              -I$(PYTHON_INCLUDE_DIR) \
              -I$(PYBIND11_INCLUDE) \
              $(INCLUDE)

# ctypes 编译选项（不需要Python头文件）
CTYPES_FLAGS = -O3 -std=c++11 -shared \
               -Xcompiler -fPIC \
               $(INCLUDE)

# 默认目标
all: $(TARGET)

# 链接规则（standalone executable）
$(TARGET): $(CPP_SRC) $(CU_SRC)
	$(NVCC) $(INCLUDE) $(NVCC_FLAGS) $(CPP_SRC) $(CU_SRC) -o $(TARGET)

# Python模块（使用pybind11）
pybind: $(PYTHON_MODULE).so

$(PYTHON_MODULE).so: $(PYBIND_SRC) $(CU_SRC)
	@echo "Python include: $(PYTHON_INCLUDE_DIR)"
	@echo "pybind11 include: $(PYBIND11_INCLUDE)"
	$(NVCC) $(PYBIND_FLAGS) $(NVCC_FLAGS) $^ -o $@.so

# ctypes 共享库（不需要Python头文件）
ctypes: $(CTYPES_LIB)

$(CTYPES_LIB): $(CTYPES_SRC) $(CU_SRC) | $(BUILD_DIR)
	$(NVCC) $(CTYPES_FLAGS) $(NVCC_FLAGS) $^ -o $@
	@echo "Built ctypes library: $@"

# 创建构建目录
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# 安装pybind11（如果没有）
install_pybind11:
	@echo "Installing pybind11..."
	python3 -m pip install pybind11

# 运行
run: $(TARGET)
	./$(TARGET)

# 运行Python版本（ctypes）
run_py: ctypes
	python3 run_ctypes.py $(ARGS)

# 清理
clean:
	rm -f $(TARGET) $(PYTHON_MODULE).so $(CTYPES_LIB)
	rm -rf $(BUILD_DIR)

.PHONY: all pybind ctypes install_pybind11 run run_py clean
