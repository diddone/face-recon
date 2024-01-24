#ifndef CERES_OPTIMIZER_H
#define CERES_OPTIMIZER_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "bfm_manager.h"
#include "utils.h"

//include ceres cost functions
#include "cost_functions.cpp"

class CeresOptimizer{
    public: 
    static ceres::Solver::Summary optimize(const std::shared_ptr<const BfmManager>& pBfmManager,
                                           const ImageUtilityThing& imageUtility,
                                           const Matrix3d& cameraMatrix,
                                           const std::vector<size_t>& vertexIds,
                                           const std::vector<Eigen::Vector2d>& landmarkUVs,
                                           const std::vector<double>& weights,
                                           double* pose,
                                           double* shapeCoefs,
                                           double* exprCoefs,
                                           double* texCoefs)
    {
        //define problem
        ceres::Problem problem;

        //add residual blocks for cost functions
        //now only for sparse
        for(size_t i = 0; i < vertexIds.size(); ++i){
            ceres::CostFunction* sparseCostFunction = SparseCostFunction::create(pBfmManager, cameraMatrix, vertexIds[i], 
                                                        landmarkUVs[i], weights[i]);
            problem.AddResidualBlock(sparseCostFunction, nullptr, pose, shapeCoefs, exprCoefs);

            //TODO: For dense optimization, add other residual functions for other cost functions
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        
        //enable displaying progress update 
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        return summary;                         
    }

};

#endif // CERES_OPTIMIZER_H