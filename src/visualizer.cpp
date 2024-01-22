#include "bfm_manager.h"
#include "utils.h"
#include "visualizer.h"
#include "optimizer_class.h"
#include <iostream>

int main(int argc, char *argv[]) {

    Visualizer visualizer(argc, *argv);
    visualizer.setupImage("/home/david/Documents/tum/projects/3d/face-recon/Data/image.png");

    std::string sBfmH5Path = "../Data/model2017-1_face12_nomouth.h5",
            sLandmarkIdxPath = "../Data/landmark_68.anl";

    // do we care what gets passed to initGlog?
    bool isInitGlog = initGlog(argc, argv);
    if (!isInitGlog) {
        std::cout << "Glog sparseProblem or just info help\n";
        return false;
    }
    std::shared_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));
//    pBfmManager->genAvgFace();

    std::shared_ptr<ImageUtilityThing> imageUtilityThing = std::make_shared<ImageUtilityThing>("/home/david/Documents/tum/projects/3d/face-recon/Data/camera_info.yaml");

    Optimizer optimizer(pBfmManager, imageUtilityThing);
    optimizer.solveSparse();

    visualizer.setupFace(pBfmManager);

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
