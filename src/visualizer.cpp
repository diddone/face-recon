#include "bfm_manager.h"
#include "utils.h"
#include "visualizer.h"
#include <iostream>

int main(int argc, char *argv[]) {

    Visualizer visualizer(argc, *argv);
    visualizer.setupImage("/home/david/Documents/tum/projects/3d/face-recon/Data/image.png");

    std::string sBfmH5Path = "../Data/model2017-1_face12_nomouth.h5",
            sLandmarkIdxPath = "../example/example_landmark_68.anl";

    // do we care what gets passed to initGlog?
    bool isInitGlog = initGlog(argc, argv);
    if (!isInitGlog) {
        std::cout << "Glog problem or just info help\n";
        return false;
    }
    std::unique_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));
    pBfmManager->genAvgFace();
//    pBfmManager->writePly("avg_face.ply");

    visualizer.setupFace(std::move(pBfmManager));

    // render loop
    // -----------
    while (visualizer.shouldRenderFrame()) {
        visualizer.setupFrame();
        visualizer.renderImage();
        visualizer.renderFaceMesh();
        visualizer.finishFrame();
    }

    visualizer.closeOpenGL();

    return 0;
}
