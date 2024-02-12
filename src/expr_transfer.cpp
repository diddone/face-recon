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
const std::string WEIGHTS_FILE_PATH_SOURCE("../Data/weights.txt"); // Path to the weights file
const std::string WEIGHTS_FILE_PATH("../Data/smile_weights.txt"); // Path to the weights file

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
    std::shared_ptr<ImageUtilityThing> pImageUtility(new ImageUtilityThing(cameraInfoPath));
    std::shared_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));


    std::string imageFile = (data_path/"image.png").string();
    std::string cloudFile = (data_path/"cloud.pcd").string();
    std::string landmarkFile = (data_path / "image_landmarks_dlib.txt").string();
  
    // std::string imageFile = (data_path/"smile.png").string();
    // std::string cloudFile = (data_path/"smile_cloud.pcd").string();
    // std::string landmarkFile = (data_path / "landmarkdlib_smilingman.txt").string();


    bool weightsLoadeds = false;
    pImageUtility->input(imageFile, cloudFile, landmarkFile);

    // Load weights if they exist
    if (boost::filesystem::exists(WEIGHTS_FILE_PATH_SOURCE)) {
        try {
            pBfmManager->loadWeights(WEIGHTS_FILE_PATH_SOURCE);
            pBfmManager->updateFaceUsingParams(); // Apply the loaded parameters
            LOG(INFO) << "Weights loaded successfully from " << WEIGHTS_FILE_PATH_SOURCE;
            weightsLoadeds = true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error loading weights: " << e.what();
            return 1;
        }
    } else {
          LOG(INFO) << "No weights file found. Proceeding without loading weights.";

           // pImageUtility->input(imageFile, cloudFile, landmarkFile);
            VectorXd imageLandmarks = pImageUtility->getXYZLandmarks();

            ProcrustesAligner procruster;
            ExtrinsicTransform transform = procruster.estimatePose(pBfmManager->m_vecLandmarkCurrentBlendshape, imageLandmarks);
            pBfmManager->setRotTransScParams(transform.rotation, transform.translation, transform.scale);
        
            pBfmManager->computeVertexNormals();
            pImageUtility->computeNormals();

            {
            // this is for better starting initialisation
            pBfmManager->m_dScale = 0.001;
            pBfmManager->genExtParams();
            Optimizer optimizer(pBfmManager, pImageUtility);
            optimizer.setNumThreads(1);
            optimizer.setNumIterations(50);
            optimizer.addPriorConstraints(100., 200., 0.);
            optimizer.addSparseConstraints(0.0005);
            optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aShapeCoef);
            optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aExprCoef);
            optimizer.solve();
            optimizer.printReport();
            //optimizer.problem.SetParameterBlockVariable(pBfmManager->m_aShapeCoef);
            //optimizer.problem.SetParameterBlockVariable(pBfmManager->m_aExprCoef);
            pBfmManager->genTransforms();
            std::cout << "Rotation\n" << pBfmManager->m_matR << std::endl;
            std::cout << "Translation\n" << pBfmManager->m_vecT << std::endl;
            std::cout << "scale \n" << pBfmManager->m_dScale << std::endl;
            }
    }
    
    //apply texture to face mesh
    setCurrentTexAsImage(pBfmManager, pImageUtility);

    //save tex vector
    std::vector<double> textureVertexData(pBfmManager->m_vecCurrentTex.data(), pBfmManager->m_vecCurrentTex.data() + pBfmManager->m_nVertices * 3);


    // second face
    bool weightsLoaded = false;
    std::shared_ptr<BfmManager> pBfmManagerT(new BfmManager(sBfmH5Path, sLandmarkIdxPath));


    if (boost::filesystem::exists(WEIGHTS_FILE_PATH)) {
        try {
            pBfmManagerT->loadWeights(WEIGHTS_FILE_PATH);
            pBfmManagerT->updateFaceUsingParams(); // Apply the loaded parameters
            LOG(INFO) << "Weights loaded successfully from " << WEIGHTS_FILE_PATH;
            weightsLoaded = true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error loading weights: " << e.what();
            return 1;
        }
    } 
    else {
          LOG(INFO) << "No weights file found. Proceeding without loading weights.";

          // second face

          std::string imageFileT = (data_path/"smile.png").string();
          std::string landmarkFileT = (data_path / "landmarkdlib_smilingman.txt").string();
          std::shared_ptr<ImageRGBOnly> pImageUtilityT(new ImageRGBOnly);
          pImageUtilityT->input(imageFileT, landmarkFileT);
          
          {
            // this is for better starting initialisation
            pBfmManagerT->m_dScale = 0.001;
            pBfmManagerT->genExtParams();
            Optimizer optimizer(pBfmManagerT, pImageUtilityT);
            optimizer.setNumThreads(1);
            optimizer.setNumIterations(50);
            optimizer.addPriorConstraints(100., 100., 0.);
            optimizer.addSparseConstraints(0.0007);
          // optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aShapeCoef);
          //   optimizer.problem.SetParameterBlockConstant(pBfmManager->m_aExprCoef);
            optimizer.solve();
            optimizer.printReport();
            optimizer.problem.SetParameterBlockVariable(pBfmManagerT->m_aShapeCoef);
            optimizer.problem.SetParameterBlockVariable(pBfmManagerT->m_aExprCoef);
            pBfmManagerT->genTransforms();
            std::cout << "Rotation\n" << pBfmManagerT->m_matR << std::endl;
            std::cout << "Translation\n" << pBfmManagerT->m_vecT << std::endl;
            std::cout << "scale \n" << pBfmManagerT->m_dScale << std::endl;
          }
    }
    
    *pBfmManager->getMutableExprCoef() = *pBfmManagerT->m_aExprCoef;
    std::copy(pBfmManagerT->getMutableExprCoef(), pBfmManagerT->getMutableExprCoef() + pBfmManagerT->m_nExprPcs, pBfmManager->getMutableExprCoef());
    

    pBfmManager->updateFaceUsingParams();
   // setCurrentTexAsImage(pBfmManager, pImageUtility);

    // set tex vector to saved tex. 
    std::copy(textureVertexData.begin(),
              textureVertexData.end(),
              pBfmManager->m_vecCurrentTex.data());

    Visualizer visualizer(argc, *argv);
    visualizer.setupImage(imageFile);
    visualizer.setupFace(pBfmManager, pImageUtility);
    while (visualizer.shouldRenderFrame()) {
        visualizer.setupFrame();
        visualizer.renderImage();
        visualizer.renderFaceMesh();
        visualizer.finishFrame();
    }

    visualizer.closeOpenGL();
    google::ShutdownGoogleLogging();
    return 0;
}
