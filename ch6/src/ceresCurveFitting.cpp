#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>

struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // 计算残差
    template<typename T>
    bool operator() (
        const T *const abc,
        T *residual) const {
            residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
            return true;
        }
    const double _x, _y;
};

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    std::vector<double> x_data, y_data;      // 数据
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};

    // 构建最小二乘问题
    ceres::Problem problem;
    for (int i = 0; i < N; i++)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            nullptr,
            abc
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summery;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summery);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;

    std::cout << summery.BriefReport() << std::endl;
    std::cout << "estimated a,b,c = ";
    for (auto a:abc) std::cout << a << " ";
    std::cout << std::endl;
    
    return 0;
}

