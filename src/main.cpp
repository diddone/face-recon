#include "bfm_manager.h"
#include "utils.h"

#include <array>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>

#include <string>
#include <vector>

#include "procrustes_aligner.h"

#include "visualizer.h"

#include "optimizer_class.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

const std::string LOG_PATH("./log");

int main(int argc, char *argv[]) {
  // logging
  boost::filesystem::path data_path("../Data");
  std::string sBfmH5Path =
      (data_path / "model2017-1_face12_nomouth.h5").string();
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

  // Test logging
  LOG(INFO) << "Logging initialized successfully.";

  // intrinsics parameters
  std::string cameraInfoPath((data_path / "camera_info.yaml").string());
  std::shared_ptr<ImageUtilityThing> pImageUtility(
      new ImageUtilityThing(cameraInfoPath));
  std::shared_ptr<BfmManager> pBfmManager(
      new BfmManager(sBfmH5Path, sLandmarkIdxPath));
  // Define a flag to check if weights are loaded

  std::string imageFile = (data_path / "image.png").string();
  std::string cloudFile = (data_path / "cloud.pcd").string();
  std::string landmarkFile = (data_path / "image_landmarks_dlib.txt").string();
  const std::string WEIGHTS_FILE_PATH(
      "../Data/none.txt"); // Path to the weights file
  const std::string outWeightPath = "bright_color_light_weights.txt";
  double sparseWeight = 0.0004;

  bool weightsLoaded = false;
  // Load weights if they exist
  if (boost::filesystem::exists(WEIGHTS_FILE_PATH)) {
    try {
      pBfmManager->loadWeights(WEIGHTS_FILE_PATH);
      pBfmManager->updateFaceUsingParams(); // Apply the loaded parameters
      LOG(INFO) << "Weights loaded successfully from " << WEIGHTS_FILE_PATH;
      weightsLoaded = true;
    } catch (const std::exception &e) {
      LOG(ERROR) << "Error loading weights: " << e.what();
      return 1;
    }
  } else {
    LOG(INFO) << "No weights file found. Proceeding without loading weights.";
  }

  pImageUtility->input(imageFile, cloudFile, landmarkFile);
  VectorXd imageLandmarks = pImageUtility->getXYZLandmarks();

  ProcrustesAligner procruster;
  ExtrinsicTransform transform = procruster.estimatePose(
      pBfmManager->m_vecLandmarkCurrentBlendshape, imageLandmarks);
  pBfmManager->setRotTransScParams(transform.rotation, transform.translation,
                                   transform.scale);

  pBfmManager->computeVertexNormals();
  pImageUtility->computeNormals();
  {
    double p2PWeight = 700.;
    double p2PlaneCoef = 9.;
    double p2PlaneWeight = p2PlaneCoef * p2PWeight;
    std::cout << "Init depth cost functions" << std::endl;
    std::cout << computeDepthCostFunction(pBfmManager, pImageUtility);

    pImageUtility->computeNormals();
    Optimizer optimizer(pBfmManager, pImageUtility);
    optimizer.setNumThreads(8);
    optimizer.solveWithDepthConstraints(20, sparseWeight, p2PWeight,
                                        p2PlaneWeight, 150., 150., 0., 2);

    pBfmManager->writePly("adepth_res.ply");
    pImageUtility->writePly("adepth_target_cloud.ply");

    std::cout << "Final Depth cost functions" << std::endl;
    std::cout << computeDepthCostFunction(pBfmManager, pImageUtility);
  }

  // optimise for color
  {
    pBfmManager->computeVertexNormals();
    Optimizer optimizer(pBfmManager, pImageUtility);
    optimizer.setNumThreads(8);
    optimizer.setNumIterations(30);
    optimizer.addPriorConstraints(0., 0., 3.);
    optimizer.addColorWithLightConstraints(1.0);
    optimizer.solve();
    optimizer.printReport();
    pBfmManager->saveWeights(outWeightPath);
    std::cout << "SH" << std::endl;

    // set all SH coefficients as RGB and increasee intensity
    for (size_t k = 0; k < 3; ++k) {
      for (size_t t = 0; t < 9; ++t) {
        pBfmManager->m_aSHCoef[9 * k + t] =
            1.5 * pBfmManager->m_aSHCoef[9 * 2 + t];
      }
    }

    // setCurrentTexAsImageMinusLight(pBfmManager, pImageUtility);
    addLightToTexture(pBfmManager);
  }

  pBfmManager->saveWeights(outWeightPath);
  LOG(INFO) << "Weights saved to " << outWeightPath << "\n";

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