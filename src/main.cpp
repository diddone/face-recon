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
const std::string WEIGHTS_FILE_PATH("../Data/bfm_weights.txt"); // Path to the weights file

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

  // intrinsics parameters
  std::string cameraInfoPath((data_path / "camera_info.yaml").string());
	std::shared_ptr<ImageUtilityThing> pImageUtility(new ImageUtilityThing(cameraInfoPath));
	std::shared_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));

    // Define a flag to check if weights are loaded
    bool weightsLoaded = false;

    // Load weights if they exist
    if (boost::filesystem::exists(WEIGHTS_FILE_PATH)) {
        try {
            pBfmManager->loadWeights(WEIGHTS_FILE_PATH);
            pBfmManager->updateFaceUsingParams(); // Apply the loaded parameters
            LOG(INFO) << "Weights loaded successfully from " << WEIGHTS_FILE_PATH;
            weightsLoaded = true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error loading weights: " << e.what();
            return 1;
        }
    } else {
        LOG(INFO) << "No weights file found. Proceeding without loading weights.";
    }


    std::string imageFile = (data_path/"rgbd_face_dataset_training/013_11_image.png").string();
    std::string cloudFile = (data_path/"rgbd_face_dataset_training/013_11_cloud.pcd").string();
    std::string landmarkFile = (data_path / "samples/landmarkc_013_11_image.txt").string();
    pImageUtility->input(imageFile, cloudFile, landmarkFile);
    VectorXd imageLandmarks = pImageUtility->getXYZLandmarks();

    ProcrustesAligner procruster;
    ExtrinsicTransform transform = procruster.estimatePose(pBfmManager->m_vecLandmarkCurrentBlendshape, imageLandmarks);
    pBfmManager->setRotTransScParams(transform.rotation, transform.translation, transform.scale);
    // this functions inits extrinsics of the BfmManager

    // pBfmManager->genAvgFace();
    // pBfmManager->writePlyNew("avg_face_transformed_neg.ply");
    if (!weightsLoaded) {
        // Perform optimization if weights were not loaded
        Optimizer optimizer(pBfmManager, pImageUtility);
        optimizer.setNumThreads(4);
        optimizer.setNumIterations(30);
        optimizer.addPriorConstraints(1.0 / pBfmManager->m_dScale, 1., 1.);
        optimizer.addSparseConstraints(0.001);
        optimizer.solve();
        optimizer.printReport();

        // optimizer.resetConstraints();
        // optimizer.setNumIterations(10);
        // optimizer.addPriorConstraints(1.0 / pBfmManager->m_dScale, 1., 1.);
        // optimizer.addSparseConstraints(0.0001);
        // optimizer.addDepthConstraints(16.);
        // optimizer.solve();
        // optimizer.printReport();
    }

    pBfmManager->transformShapeExprBFM();
    pBfmManager->updateFaceUsingParams();
    pBfmManager->writePly("proc_face.ply", false);

    // Save the current weights to a file for further texture work
    // std::string weightsFilePath = "../Data/bfm_weights.txt";
    // pBfmManager->saveWeights(weightsFilePath);
    // LOG(INFO) << "Weights saved to " << weightsFilePath;

    // Important, dont forget to regenerate face (using coefs and extr)
    std::cout << "Ext params translation" << pBfmManager->m_aExtParams[3] << " " << pBfmManager->m_aExtParams[4] << " " << pBfmManager->m_aExtParams[5] << "\n";
    std::cout << "Old color\n";
    // pBfmManager->transformShapeExprBFM();
    // pBfmManager->updateFaceUsingParams();
    for (size_t t = 0; t < 3; ++t) {
        size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
        auto uvVec = pImageUtility->getUVLandmarks();
        std::cout << "Landmarks XYZ " << pImageUtility->UVtoXYZ(uvVec[2 * t], uvVec[2 * t + 1]) << std::endl;
        std::cout << "BFM " << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 1] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 2] << "\n";
    }


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