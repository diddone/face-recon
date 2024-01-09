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
    std::cout << "faces\n\n";
    pBfmManager->genAvgFace();
    pBfmManager->writePly("avg_face.ply");

    pBfmManager->genRndFace(1.0);
    pBfmManager->writePly("rnd_face.ply");
    int min_v = 10000000;
    int max_v = -1;
    std::cout << "Number of vertices " << pBfmManager->m_nVertices << "\n";
    std::cout << "Number of faces " << pBfmManager->m_nFaces << "\n";

    int cnt = 0;
    int blank = 28588;
    std::cout << "Blank";
    for (int iVertice = blank; iVertice < pBfmManager->m_nVertices; iVertice++) {
        float x, y, z;

        x = float(pBfmManager->m_vecCurrentBlendshape(iVertice * 3));
        y = float(pBfmManager->m_vecCurrentBlendshape(iVertice * 3 + 1));
        z = float(pBfmManager->m_vecCurrentBlendshape(iVertice * 3 + 2));

        // y = -y; z = -z;
        std::cout << x << " " << y << " " << z << "\n";
    }

}
