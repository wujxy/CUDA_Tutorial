# CUDA Makefile for Gaussian Fitter

NVCC = nvcc
TARGET = gaussion_fitter

# 源文件
CPP_SRC = run.cpp
CU_SRC = src/model.cu src/optimizer.cu src/cuda_utils.cu

# 包含目录
INCLUDE = -Iinclude

# 编译选项
CFLAGS = -O3 -std=c++11
NVCC_FLAGS = -O3 -std=c++11 -arch=sm_52

# 默认目标
all: $(TARGET)

# 链接规则
$(TARGET): $(CPP_SRC) $(CU_SRC)
	$(NVCC) $(INCLUDE) $(NVCC_FLAGS) $(CPP_SRC) $(CU_SRC) -o $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET)

.PHONY: all run clean
