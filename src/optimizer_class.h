#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "utils.h"
#include <utility>
#include "cost_functions.cpp"

class Optimizer {

    shared_ptr<BfmManager> pBfmManager;
    const ImageUtilityThing& imageUtility;
    double sparseWeight;
    double priorWeight;

    // only sparse for now
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

public:
    Optimizer(shared_ptr<BfmManager> _pBfmManager, const ImageUtilityThing& _imageUtility, double sparseWeight=1.0, double priorWeight=1.0):
    imageUtility(_imageUtility), sparseWeight(sparseWeight), priorWeight(priorWeight) {
        pBfmManager = std::move(_pBfmManager);

        configureOptions();
    }

    void solveSparse() {
        preparePriorConstraints();
        prepareSparseConstraints();
        return ceres::Solve(options, &problem, &summary);
    }

    void printReport() {
        std::cout << summary.BriefReport() << std::endl;
    }

    void setPriorWeight(double newWeight) {
        priorWeight = newWeight;
    }

private:
    void configureOptions() {
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 4;
        options.num_threads = 8;
    }

    void preparePriorConstraints() {
        problem.AddResidualBlock(
            PriorShapeCostFunction::create(pBfmManager->m_nIdPcs, priorWeight), nullptr,
            pBfmManager->m_aShapeCoef
        );
        problem.AddResidualBlock(
            PriorExprCostFunction::create(pBfmManager->m_nExprPcs, priorWeight), nullptr,
            pBfmManager->m_aExprCoef
        );
        problem.AddResidualBlock(
            PriorTexCostFunction::create(pBfmManager->m_nIdPcs, priorWeight), nullptr,
            pBfmManager->m_aTexCoef
        );
    }
    void prepareSparseConstraints() {
        // use bfm and image thing to create and add residuals to the (sparse)problem
        Eigen::VectorXi imageLandmarks = imageUtility.getUVLandmarks();

        for (size_t iLandmark = 0; iLandmark < pBfmManager->m_mapLandmarkIndices.size(); ++iLandmark) {
            Eigen::Vector2i landmark(imageLandmarks[2 * iLandmark], imageLandmarks[2 * iLandmark + 1]);
            problem.AddResidualBlock(SparseCostFunction::create(
                    pBfmManager, imageUtility.camera_matrix, iLandmark, landmark, sparseWeight
            ), nullptr, pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef);
        }
    }
};