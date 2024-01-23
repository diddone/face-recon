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
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "visualizer.h"

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
    std::string cameraInfoPath((data_path / ("rgbd_face_dataset_training/camera_info.yaml")).string());
	ImageUtilityThing imageUtility(cameraInfoPath);
	std::unique_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));

    std::string imageFile = (data_path/"image.png").string();
    std::string cloudFile = (data_path/"cloud.pcd").string();
    std::string landmarkFile = (data_path / "image_landmarks_dlib.txt").string();
    imageUtility.input(imageFile, cloudFile, landmarkFile);
    VectorXd imageLandmarks = imageUtility.getXYZLandmarks();

    // imageUtility.getUVLandmarks()
    // Procruster: XYZ vector from BFM manager and XYZ vector from ImageUtility
    //
    ProcrustesAligner procruster;

    // std::cout<<"Landmark Blendshape \n";
    // for (int i = 0; i <5; ++i) {
    //     std::cout << pBfmManager->m_vecLandmarkCurrentBlendshape[3 * i] << " "
    //     << pBfmManager->m_vecLandmarkCurrentBlendshape[3 * i + 1] << " "
    //     << pBfmManager->m_vecLandmarkCurrentBlendshape[3 * i + 2] << std::endl;
    // }

    // std::cout<<"Blendshape \n";
    // for (int i = 0; i <5; ++i) {
    //     std::cout << pBfmManager->m_vecCurrentBlendshape[3 * i] << " "
    //     << pBfmManager->m_vecCurrentBlendshape[3 * i + 1] << " "
    //     << pBfmManager->m_vecCurrentBlendshape[3 * i + 2] << std::endl;
    // }
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
    // pBfmManager->setRotTransScParams(transform.rotation, transform.translation, transform.scale);
    // pBfmManager->transformShapeExprBFM(transform.rotation, transform.translation, transform.scale);
    // pBfmManager->genAvgFace();
    // pBfmManager->writePly("avg_face_proc.ply", ModelWriteMode_None);

    // std::cout << pBfmManager->getMatR() << "\n";
    // std::cout << pBfmManager->getVecT() << "\n";
    // std::cout << pBfmManager->getScale() << "\n";
    // for (size_t k = 0; k < N_EXT_PARAMS; ++k) {
    //     std::cout << pBfmManager->getExtParams()[k] << "\n";
    // }
    // pBfmManager->setRotTransScParams(transform.rotation, transform.translation, transform.scale);
    // std::cout << pBfmManager->getMatR() << "\n";
    // std::cout << pBfmManager->getVecT() << "\n";
    // std::cout << pBfmManager->getScale() << "\n";
    // for (size_t k = 0; k < N_EXT_PARAMS; ++k) {
    //     std::cout << pBfmManager->getExtParams()[k] << "\n";
    // }
    // pBfmManager->m_vecLandmarkCurrentBlendshape;
    // pBfmManager->writeLandmarkPly("out_landmarks.ply");
	// pBfmManager->writePly("rnd_face.ply", ModelWriteMode_None);

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