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
#include "utils.h"

const std::string LOG_PATH("(./log)");

int main(int argc, char *argv[])
{
    // logging
    std::string sBfmH5Path, sLandmarkIdxPath;
    double dFx = 1744.327628674942, dFy = 1747.838275588676, dCx = 800., dCy = 600.;
    bool isInitGlog = initGlog(argc, argv, sBfmH5Path, sLandmarkIdxPath, LOG_PATH);
    if (!isInitGlog) {
        std::cout << "Glog problem\n";
        return 1;
    }
    // intrinsics parameters
	CameraProjection camProj(dFx, dFy, dCx, dCy);
	std::unique_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));

    // TODO initisalize image and landmarks
    cv::Mat image;
    std::vector<Eigen::Vector2i> imageLandmarks;

    pBfmManager->writeLandmarkPly("landmarks.ply");
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