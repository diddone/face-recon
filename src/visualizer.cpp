#include "bfm_manager.h"
#include "utils.h"
#include "visualizer.h"
#include <iostream>

int main(int argc, char*argv[])
{

    Visualizer visualizer(argc, *argv);
    visualizer.setupImage("/home/david/Documents/tum/projects/3d/face-recon/Data/image.png");
    visualizer.setupFace();

    // render loop
    // -----------
    while (visualizer.shouldRenderFrame())
    {
        visualizer.setupFrame();
        visualizer.renderImage();
        visualizer.renderFaceMesh();
        visualizer.finishFrame();
    }

    visualizer.closeOpenGL();

    return 0;
}
