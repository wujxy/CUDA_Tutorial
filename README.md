# CUDA_Tutorial
In order to learn cuda programing
based on blog : https://face2ai.com/CUDA-F-1-0-%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97%E4%B8%8E%E8%AE%A1%E7%AE%97%E6%9C%BA%E6%9E%B6%E6%9E%84/
reference : 《CUDA C编程权威指南》

# quick start: 

1. download nvcc : 
   sudo apt-get update
   sudo apt install nvidia-cuda-toolkit

2. download gcc : 
   sudo apt-get update
   sudo apt-get install build-essential

3. simple compile && run : nvcc test.cu -o test && ./test