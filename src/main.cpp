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

const std::string LOG_PATH("(./log)");

int main(int argc, char *argv[])
{
    // logging
    boost::filesystem::path data_path("../Data");
    std::string sBfmH5Path = (data_path / "model2017-1_face12_nomouth.h5").string();
    std::string sLandmarkIdxPath = (data_path / "landmark_68.anl").string();

    bool isInitGlog = initGlog(argc, argv, LOG_PATH);
    if (!isInitGlog) {
        std::cout << "Glog problem\n";
        return 1;
    }
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
    Matrix4d initPose = procruster.estimatePose(pBfmManager->m_vecLandmarkCurrentBlendshape, imageLandmarks);

    // pBfmManager->m_vecLandmarkCurrentBlendshape;
    pBfmManager->writeLandmarkPly("out_landmarks.ply");
	pBfmManager->writePly("rnd_face.ply", ModelWriteMode_None);

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