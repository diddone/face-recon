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

#include "ceres_optimizer.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

const std::string LOG_PATH("./log");

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
	ImageUtilityThing imageUtility(cameraInfoPath);
	std::shared_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));

    std::string imageFile = (data_path/"image.png").string();
    std::string cloudFile = (data_path/"cloud.pcd").string();
    std::string landmarkFile = (data_path / "image_landmarks_dlib.txt").string();
    imageUtility.input(imageFile, cloudFile, landmarkFile);
    VectorXd imageLandmarks = imageUtility.getXYZLandmarks();

    // imageUtility.getUVLandmarks()
    // Procruster: XYZ vector from BFM manager and XYZ vector from ImageUtility
    //
    ProcrustesAligner procruster;
    ExtrinsicTransform transform = procruster.estimatePose(pBfmManager->m_vecLandmarkCurrentBlendshape, imageLandmarks);

    pBfmManager->setRotTransScParams(transform.rotation, transform.translation, transform.scale);
    // pBfmManager->genAvgFace();
    //pBfmManager->writePly("avg_face_transformed.ply");
    //pBfmManager->writePlyNew("avg_face_transformed_neg.ply");

    Visualizer visualizer(argc, *argv);
    visualizer.setupImage(imageFile);
    visualizer.setupFace(pBfmManager);
    while (visualizer.shouldRenderFrame()) {
        visualizer.setupFrame();
        visualizer.renderImage();
        visualizer.renderFaceMesh();
        visualizer.finishFrame();
    }

    visualizer.closeOpenGL();

    //Sparse optimization
    //TODO: Initialize necessary parameters
    
    ceres::Solver::Summary summary = CeresOptimizer::optimize(pBfmManager, imageUtility, cameraMatrix, vertexIds, landmarkUVs, weights, pose, shapeCoefs, exprCoefs, texCoefs);
    std::cout << summary.FullReport() << std::endl;

	google::ShutdownGoogleLogging();
	return 0;
}


// int main() {
//     // init_glog;

//     // init bfm manager
//     // read image -> cv::Mat
//     // read detection -> std::vector<Eigen::Vector2i>
//     // run procrusters ->
// }