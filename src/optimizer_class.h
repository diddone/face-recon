#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "utils.h"
#include <utility>

class Optimizer {

    shared_ptr<BfmManager> pBfmManager;
    shared_ptr<ImageUtilityThing> imageUtility;

    // only sparse for now
    ceres::Problem sparseProblem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

public:
    Optimizer(shared_ptr<BfmManager> _pBfmManager, shared_ptr<ImageUtilityThing> _imageUtility) {
        pBfmManager = std::move(_pBfmManager);
        imageUtility = std::move(_imageUtility);

        configureOptions();
        prepareConstraints();
    }

    void solveSparse() {
        return ceres::Solve(options, &sparseProblem, &summary);
    }

    void printReport() {
        std::cout << summary.BriefReport() << std::endl;
    }

private:
    void configureOptions() {
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 1;
        options.num_threads = 8;
    }

    void prepareConstraints() {
        // use bfm and image thing to create and add residuals to the (sparse)problem
    }
};