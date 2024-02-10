#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "utils.h"
#include <utility>
#include "cost_functions.cpp"

#include <chrono>

class Optimizer {

    shared_ptr<BfmManager> pBfmManager;
    const shared_ptr<const ImageRGBOnly> pImageUtility;

    // ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

public:
    ceres::Problem problem;
    Optimizer(shared_ptr<BfmManager> _pBfmManager, const shared_ptr<const ImageRGBOnly> _imageUtility): pImageUtility(_imageUtility) {
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
        summary = ceres::Solver::Summary();
    }

    void addPriorConstraints(double shapePriorWeight, double exprPriorWeight, double texPriorWeight) {
        size_t numberOfShapeParams = pBfmManager->m_nIdPcs;
        problem.AddResidualBlock(
                PriorShapeCostFunction::create(numberOfShapeParams, pBfmManager->m_vecShapeEv, shapePriorWeight), nullptr,
                pBfmManager->m_aShapeCoef
        );

        size_t numberOfExpressionParams = pBfmManager->m_nExprPcs;
        problem.AddResidualBlock(
                PriorExprCostFunction::create(numberOfExpressionParams, exprPriorWeight), nullptr,
                pBfmManager->m_aExprCoef
        );

        problem.AddResidualBlock(
                PriorTexCostFunction::create(numberOfShapeParams, pBfmManager->m_vecTexEv, texPriorWeight), nullptr,
                pBfmManager->m_aTexCoef
        );
    }

    void addSparseConstraints(double sparseWeight) {
        Eigen::VectorXi imageLandmarks = pImageUtility->getUVLandmarks();
        size_t numberOfLandmarks = pBfmManager->m_mapLandmarkIndices.size();

        for (size_t iLandmark = 0; iLandmark < numberOfLandmarks; ++iLandmark) {
            Eigen::Vector2i landmark(imageLandmarks[2 * iLandmark], imageLandmarks[2 * iLandmark + 1]);
            problem.AddResidualBlock(SparseCostFunction::create(
                    pBfmManager, pImageUtility->getIntMat(), iLandmark, landmark, sparseWeight / numberOfLandmarks
            ), nullptr, pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef);
        }
    }

    void addDepthConstraints(double depthWeight) {
        std::shared_ptr<const ImageUtilityThing> pImageUtilityWithDepth = std::static_pointer_cast<const ImageUtilityThing>(pImageUtility);
        for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {
            problem.AddResidualBlock(DepthP2PCostFunction::create(
                pBfmManager, pImageUtilityWithDepth, vertexInd, depthWeight / pBfmManager->m_nVertices
            ), nullptr, pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef);
        }
    }

    void addDepthWithNormalsConstraints(double p2PointWeight, double p2PlaneWeight) {
      std::shared_ptr<const ImageUtilityThing> pImageUtilityWithDepth = std::static_pointer_cast<const ImageUtilityThing>(pImageUtility);
      for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {
          problem.AddResidualBlock(DepthP2PlaneCostFunction::create(
              pBfmManager, pImageUtilityWithDepth, vertexInd, p2PointWeight / pBfmManager->m_nVertices, p2PlaneWeight / pBfmManager->m_nVertices
          ), nullptr, pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef);
      }
    }

    void addColorConstraints(double colorWeight) {
        // same as other two
        for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {
            problem.AddResidualBlock(ColorCostFunction::create(
                pBfmManager, pImageUtility, vertexInd, colorWeight / pBfmManager->m_nVertices
            ), nullptr, pBfmManager->m_aTexCoef);
        }
    }

    void solve() {
      ceres::Solve(options, &problem, &summary);
      pBfmManager->updateFaceUsingParams();
    }

    void printReport() {
        std::cout << summary.BriefReport() << std::endl;
    }

    void solveWithDepthConstraints(
      size_t maxNumIterations, double sparseWeight, double p2PointWeight, double p2PlaneWeight,
        double shapePriorWeight = 1., double exprPriorWeight = 1.0, double texPriorWeight = 1.0
      ) {

      // we will do one step at time as we need to recompute normals
      options.max_num_iterations = 1;
      options.linear_solver_type = ceres::ITERATIVE_SCHUR;
      for (size_t iter = 0; iter < maxNumIterations; ++iter) {
          std::cout << "Starting iteration " << iter << std::endl;
          resetConstraints();

          pBfmManager->computeVertexNormals();
          // here we divide by scale to rescale sigmas
          addPriorConstraints(shapePriorWeight / pBfmManager->m_dScale, exprPriorWeight, texPriorWeight);
          addSparseConstraints(sparseWeight);
          addDepthWithNormalsConstraints(p2PointWeight, p2PlaneWeight);

          // here we also update params
          solve();
          printReport();
      }

      pBfmManager->updateFaceUsingParams();
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