#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << std::endl;
        return 1;
    }
    std::ifstream fin(argv[1]);
    if (!fin)
    {
        std::cout << "file " << argv[1] << " does not exist." << std::endl;
        return 1;
    }
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; 
    
    return 0;
}
