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

using namespace std;

int main(int argc, char*argv[]) {
    std::string sBfmH5Path = "../Data/model2017-1_face12_nomouth.h5",
    sLandmarkIdxPath = "../example/example_landmark_68.anl";
    double dFx = 1744.327628674942, dFy = 1747.838275588676, dCx = 800., dCy = 600.;
    bool isInitGlog = initGlog(argc, argv);
    if (!isInitGlog) {
        std::cout << "Glog problem or just info help\n";
        return 1;
    }
    std::unique_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));
    std::cout << "H% file path " << sBfmH5Path << "\n";

    pBfmManager->genAvgFace();
    pBfmManager->writePly("avg_face.ply");

    pBfmManager->genRndFace(1.0);
    pBfmManager->writePly("rnd_face.ply");
}
