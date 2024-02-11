#include "utils.h"
#include "bfm_manager.h"

#include <fstream>
#include <iostream>
#include <array>
#include <memory>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core/mat.hpp>

#include <string>
#include <vector>

#include "procrustes_aligner.h"

#include "visualizer.h"

#include "optimizer_class.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

const std::string LOG_PATH("./log");
const std::string WEIGHTS_FILE_PATH("../Data/uninit.txt"); // Path to the weights file

int main(int argc, char *argv[])
{
    // logging
    boost::filesystem::path data_path("../Data");
    std::string sBfmH5Path = (data_path / "model2017-1_face12_nomouth.h5").string();
    std::string sLandmarkIdxPath = (data_path / "landmark_68.anl").string();

    // Check if the log directory exists, and create it if it doesn't
    boost::filesystem::path log_path(LOG_PATH);
    if (!boost::filesystem::exists(log_path)) {
        if (!boost::filesystem::create_directory(log_path)) {
            std::cerr << "Could not create log directory." << std::endl;
            return 1;
        }
    }

    // Initialize Google Logging here, after the check for the log directory
    bool isInitGlog = initGlog(argc, argv, LOG_PATH);
    if (!isInitGlog) {
        std::cout << "Glog problem\n";
        return 1;
    }

    //Test logging
    LOG(INFO) << "Logging initialized successfully.";

	std::shared_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));

  // intrinsics parameters
  std::string imageFile = (data_path/"RGB/small.png").string();
  std::string landmarkFile = (data_path / "RGB/landmrks_pipnet_small.txt").string();
  std::shared_ptr<ImageRGBOnly> pImageUtility(new ImageRGBOnly);
  pImageUtility->input(imageFile, landmarkFile);

  // pBfmManager->genAvgFace();

  // Perform optimization if weights were not loaded
  {
      // this is for better starting initialisation
      pBfmManager->m_dScale = 0.001;
      pBfmManager->genExtParams();
      Optimizer optimizer(pBfmManager, pImageUtility);
      optimizer.setNumThreads(1);
      optimizer.setNumIterations(50);
      optimizer.addPriorConstraints(50., 50., 0.);
      optimizer.addSparseConstraints(0.0006);
    //   optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aShapeCoef);
    //   optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aExprCoef);
      optimizer.solve();
      optimizer.printReport();
      optimizer.problem.SetParameterBlockVariable(pBfmManager->m_aShapeCoef);
      optimizer.problem.SetParameterBlockVariable(pBfmManager->m_aExprCoef);
      pBfmManager->genTransforms();
      std::cout << "Rotation\n" << pBfmManager->m_matR << std::endl;
      std::cout << "Translation\n" << pBfmManager->m_vecT << std::endl;
      std::cout << "scale \n" << pBfmManager->m_dScale << std::endl;
    }

    // {
    //   Optimizer optimizer(pBfmManager, pImageUtility);
    //   optimizer.setNumThreads(1);
    //   optimizer.setNumIterations(20);
    //   optimizer.addPriorConstraints(100., 100., 0.);
    //   optimizer.addSparseConstraints(0.0001);

    //   optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aExtParams.data());
    //   optimizer.solve();
    //   optimizer.printReport();
    // }

    // setCurrentTexAsImage(pBfmManager, pImageUtility);
    Visualizer visualizer(argc, *argv);
    visualizer.setupImage(imageFile);
    visualizer.setupFace(pBfmManager, pImageUtility);
    while (visualizer.shouldRenderFrame()) {
       visualizer.setupFrame();
       visualizer.renderImage();
       visualizer.renderFaceMesh();
       visualizer.finishFrame();
    }

    visualizer.closeOpenGL();
    google::ShutdownGoogleLogging();
	return 0;
}