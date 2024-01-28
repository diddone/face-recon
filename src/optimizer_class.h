#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "utils.h"
#include <utility>
#include "cost_functions.cpp"

class Optimizer {

    shared_ptr<BfmManager> pBfmManager;
    const ImageUtilityThing& imageUtility;

    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

public:
    Optimizer(shared_ptr<BfmManager> _pBfmManager, const ImageUtilityThing& _imageUtility): imageUtility(_imageUtility) {
        pBfmManager = std::move(_pBfmManager);
        configureOptions();
    }

    void setNumIterations(const size_t maxNumIterations) {
        options.max_num_iterations = maxNumIterations;
    }
    void setNumThreads(const size_t numThreads) {
        options.num_threads = numThreads;
    }

    void resetConstraints() {
        problem = ceres::Problem();
    }

    void addPriorConstraints(double priorWeight) {
        size_t numberOfShapeParams = pBfmManager->m_nIdPcs;
        problem.AddResidualBlock(
                PriorShapeCostFunction::create(numberOfShapeParams, priorWeight / numberOfShapeParams), nullptr,
                pBfmManager->m_aShapeCoef
        );

        size_t numberOfExpressionParams = pBfmManager->m_nExprPcs;
        problem.AddResidualBlock(
                PriorExprCostFunction::create(numberOfExpressionParams, priorWeight / numberOfExpressionParams), nullptr,
                pBfmManager->m_aExprCoef
        );

        problem.AddResidualBlock(
                PriorTexCostFunction::create(numberOfShapeParams, priorWeight / numberOfShapeParams), nullptr,
                pBfmManager->m_aTexCoef
        );
    }

    void addSparseConstraints(double sparseWeight) {
        Eigen::VectorXi imageLandmarks = imageUtility.getUVLandmarks();
        size_t numberOfLandmarks = pBfmManager->m_mapLandmarkIndices.size();

        for (size_t iLandmark = 0; iLandmark < numberOfLandmarks; ++iLandmark) {
            Eigen::Vector2i landmark(imageLandmarks[2 * iLandmark], imageLandmarks[2 * iLandmark + 1]);
            problem.AddResidualBlock(SparseCostFunction::create(
                    pBfmManager, imageUtility.camera_matrix, iLandmark, landmark, sparseWeight / numberOfLandmarks
            ), nullptr, pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef);
        }
    }

    void addDepthConstraints(double depthWeight) {
        for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {
            problem.AddResidualBlock(DepthP2PCostFunction::create(
                pBfmManager, imageUtility, vertexInd, depthWeight / pBfmManager->m_nVertices
            ), nullptr, pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef);
        }
    }

    void addColorConstraints(double colorWeight) {
        // same as other two
        for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {
            problem.AddResidualBlock(ColorCostFunction::create(
                pBfmManager, imageUtility, vertexInd, colorWeight / pBfmManager->m_nVertices
            ), nullptr, pBfmManager->m_aTexCoef);
        }
    }

    void solve() {
        return ceres::Solve(options, &problem, &summary);
    }

    void printReport() {
        std::cout << summary.BriefReport() << std::endl;
    }

private:
    void configureOptions() {
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 10;
        options.num_threads = 8;
    }
};