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
    pBfmManager->genAvgFace();
    //pBfmManager->writePly("avg_face_transformed.ply");
    //pBfmManager->writePlyNew("avg_face_transformed_neg.ply");

    // Sparse optimization
    // TODO: Initialize necessary parameters
    // double initCost=0.;
    // auto landmark_uv = imageUtility.getUVLandmarks();
    // for (size_t iLandmark = 0; iLandmark < pBfmManager->m_mapLandmarkIndices.size(); ++iLandmark) {
    //     Vector2i landmark(landmark_uv[2 * iLandmark], landmark_uv[2 * iLandmark+1]);
    //     auto costFunction = SparseCostFunction(pBfmManager, imageUtility.camera_matrix, iLandmark, landmark, 0.125);
    //     double residuals[2];
    //     bool t = costFunction(pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef, residuals);
    //     std::cout << iLandmark<< " " << residuals[0] << " " << residuals[1] << std::endl;
    //     initCost += residuals[0] * residuals[0] + residuals[1] * residuals[1];
    // }
    // std::cout << "My init cost" << initCost << "\n";
    Optimizer optimizer(pBfmManager, imageUtility);
    optimizer.setNumThreads(4);
    // after rescaling we need to make scale for shapePrior smaller
    optimizer.addPriorConstraints(1.0 / pBfmManager->m_dScale, 1., 1.);
    optimizer.addSparseConstraints(0.002);
    optimizer.solve();
    optimizer.printReport();

    // if you want to reset the problem and
    // only work with some of the constraints:

    optimizer.resetConstraints();
    optimizer.setNumIterations(15);
    optimizer.addPriorConstraints(1.0 / pBfmManager->m_dScale, 1., 1.);
    optimizer.addSparseConstraints(0.002);
    optimizer.addDepthConstraints(1.0);
    optimizer.solve();
    optimizer.printReport();

    // we need to generate blendshapes for color optimisations.

    // auto l = {
    //             16214, 16229, 16248, 16270, 16295,
    //             25899, 26351, 26776, 27064
    //         };
    // for (auto& x: l) {
    //     auto costFn = ColorCostFunction(pBfmManager, imageUtility, x, 1.0);
    //     double residual[3];
    //     costFn(pBfmManager->m_aTexCoef, residual);
    //     std::cout << "Residual " << residual[0] << " " << residual[1] << " " << residual[2] << std::endl;
    // }
    std::cout << "Blendshapes before depth\n";
    for (size_t t = 0; t < 3; ++t) {
        size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
        std::cout << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 1] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 2] << "\n";
    }

    pBfmManager->updateFaceUsingParams();
    std::cout << "Blendshapes after depth\n";
    for (size_t t = 0; t < 3; ++t) {
        size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
        std::cout << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 1] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 2] << "\n";
    }
    //std::cout << "starting color term" << std::endl;
    // optimizer.resetConstraints();
    // optimizer.setNumThreads(4);
    // optimizer.addPriorConstraints(1.0);
    // optimizer.addColorConstraints(0.5);
    // optimizer.solve();
    // optimizer.printReport();
    setCurrentTexAsImage(pBfmManager, imageUtility);
    std::cout << "Blendshapes after color\n";
    for (size_t t = 0; t < 3; ++t) {
        size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
        std::cout << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 1] << " "
        << pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 2] << "\n";
    }


    // Important, dont forget to regenerate face (using coefs and extr)
    // std::cout << "Ext params translation" << pBfmManager->m_aExtParams[3] << " " << pBfmManager->m_aExtParams[4] << " " << pBfmManager->m_aExtParams[5] << "\n";
    // std::cout << "Old color\n";
    // for (size_t t = 0; t < 3; ++t) {
    //     size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
    //     auto uvVec = imageUtility.getUVLandmarks();
    //     std::cout << "Image rgb " << imageUtility.UVtoColor(uvVec[2 * t], uvVec[2 * t + 1]) << std::endl;
    //     std::cout << pBfmManager->m_vecCurrentTex[3 * vertexInd] << " "
    //     << pBfmManager->m_vecCurrentTex[3 * vertexInd + 1] << " "
    //     << pBfmManager->m_vecCurrentTex[3 * vertexInd + 2] << "\n";
    // }
    // pBfmManager->updateFaceUsingParams();
    // std::cout << "New color\n";
    // for (size_t t = 0; t < 3; ++t) {
    //     size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
    //     auto uvVec = imageUtility.getUVLandmarks();
    //     std::cout << "Image rgb " << imageUtility.UVtoColor(uvVec[2 * t], uvVec[2 * t + 1]) << std::endl;
    //     std::cout << pBfmManager->m_vecCurrentTex[3 * vertexInd] << " "
    //     << pBfmManager->m_vecCurrentTex[3 * vertexInd + 1] << " "
    //     << pBfmManager->m_vecCurrentTex[3 * vertexInd + 2] << "\n";
    // }
    // std::cout << std::endl;
    // std::cout << "After upd Ext params translation" << pBfmManager->m_aExtParams[3] << " " << pBfmManager->m_aExtParams[4] << " " << pBfmManager->m_aExtParams[5] << "\n";
    // std::cout << "scale factor" << pBfmManager->m_dScale << std::endl;
	// std::cout << "Rotation matrix " << pBfmManager->m_matR << std::endl;
	// std::cout << "translation vector " << pBfmManager->m_vecT << std::endl;


    Visualizer visualizer(argc, *argv);
    visualizer.setupImage(imageFile);
    visualizer.setupFace(pBfmManager);
    while (visualizer.shouldRenderFrame()) {
        visualizer.setupFrame();
        //visualizer.renderImage();
        visualizer.renderFaceMesh();
        visualizer.finishFrame();
    }

    visualizer.closeOpenGL();
    google::ShutdownGoogleLogging();
	return 0;
}