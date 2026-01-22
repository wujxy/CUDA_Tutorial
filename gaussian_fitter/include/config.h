#ifndef CONFIG_H
#define CONFIG_H

#include "model.h"
#include <string>
#include <map>

/**
 * 配置参数类
 */
class Config {
public:
    // 真实参数
    GaussianParams true_params;

    // 直方图配置
    int nx, ny;
    float x_min, x_max;
    float y_min, y_max;
    int num_samples;

    // 初始猜测
    GaussianParams initial_guess;

    // 优化器配置
    float learning_rate;
    int max_iterations;
    float tolerance;
    float gradient_epsilon;
    bool verbose;

    /**
     * 默认构造函数 - 设置默认值
     */
    Config();

    /**
     * 从配置文件加载
     */
    bool loadFromFile(const std::string& filename);

    /**
     * 打印配置信息
     */
    void print() const;

private:
    /**
     * 解析配置行
     * 格式: key = value
     */
    bool parseLine(const std::string& line, std::string& section, std::map<std::string, std::string>& params);

    /**
     * 获取浮点参数
     */
    float getFloat(const std::map<std::string, std::string>& params, const std::string& section, const std::string& key, float default_value);

    /**
     * 获取整数参数
     */
    int getInt(const std::map<std::string, std::string>& params, const std::string& section, const std::string& key, int default_value);

    /**
     * 获取布尔参数
     */
    bool getBool(const std::map<std::string, std::string>& params, const std::string& section, const std::string& key, bool default_value);
};

#endif // CONFIG_H
