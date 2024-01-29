#pragma once

#include "glog/logging.h"
#include "constant.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <cmath>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <dlib/opencv/cv_image.h>

class ImageUtilityThing {
public:
    cv::Mat rgb_image;
    cv::Mat cloud_x;
    cv::Mat cloud_y;
    cv::Mat cloud_z;
    Eigen::VectorXd landmarks_xyz;
    Eigen::VectorXi landmarks_uv;

    Eigen::Matrix3d camera_matrix;
    cv::Mat dist_coeffs;
    cv::Size init_rgb_image_size;
    cv::Size init_depth_image_size;
    double maxDepth = 0.0;

    cv::Size image_size;
    double depth_init_scale = 0.5;
    double scale = 1.;
public:
    ImageUtilityThing(const std::string& yaml_file) {
        YAML::Node config = YAML::LoadFile(yaml_file);

        // Load intrinsic parameters
        camera_matrix <<
            config["K"][0].as<double>(), 0, config["K"][2].as<double>(),
            0, config["K"][4].as<double>(), config["K"][5].as<double>(),
            0, 0, 1.;

        // Assuming the distortion model is "plumb_bob", which typically has 5 coefficients
        dist_coeffs = (cv::Mat_<double>(5, 1) <<
            config["D"][0].as<double>(),
            config["D"][1].as<double>(),
            config["D"][2].as<double>(),
            config["D"][3].as<double>(),
            config["D"][4].as<double>());

        // Set the image size
        // depth_image_width = config["width"].as<int>() / 2;
        // depth_image_height = config["height"].as<int>() / 2;
        init_rgb_image_size = cv::Size(config["width"].as<int>(), config["height"].as<int>());
        init_depth_image_size = cv::Size(
            int(config["width"].as<int>() * depth_init_scale),
            int(config["height"].as<int>() * depth_init_scale)
        );
        // depth_image_size = cv::Size(config["width"].as<int>() / 2, config["height"].as<int>()/ 2);

        // Initialize the point cloud
        cloud_x = cv::Mat::zeros(init_depth_image_size.height, init_depth_image_size.width, CV_64F);
        cloud_y = cv::Mat::zeros(init_depth_image_size.height, init_depth_image_size.width, CV_64F);
        cloud_z = cv::Mat::zeros(init_depth_image_size.height, init_depth_image_size.width, CV_64F);

        // init landmarks vector
        landmarks_xyz = Eigen::VectorXd::Zero(3 * N_DLIB_LANDMARKS);
        landmarks_uv = Eigen::VectorXi::Zero(2 * N_DLIB_LANDMARKS);
    }

    cv::Size rescaleImageSize(const cv::Size& old_image_size) const {
        return cv::Size(int(old_image_size.width * scale), (old_image_size.height * scale));
    }
    void input(const std::string& image_file, const std::string& pcd_file, const std::string& landmarkFile, double new_scale=1.0) {
        if (new_scale <= 0) {
        std::cerr << "Error: Scale must be greater than zero." << std::endl;
        return;
    }
        // set new scale and image size
        scale = new_scale;

        // Reset the maxDepth to zero before loading new data
        maxDepth = 0.0;

        //cv::Size image_size = rescaleImageSize(init_rgb_image_size);
        this->image_size = rescaleImageSize(init_rgb_image_size); // Use this to set the class member


        // Load RGB image
        // cv::Mat rgb_image = cv::imread(image_file);
        rgb_image = cv::imread(image_file);
        if (rgb_image.empty()) {
            std::cerr << "Error: RGB image not loaded properly." << std::endl;
            return;
        }


        // Load PCD file
        auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1) {
            PCL_ERROR("Couldn't read pcd file \n");
            exit(EXIT_FAILURE);
        }

        // Check if point cloud size matches expected image size after scaling
        int expected_number_of_points = init_depth_image_size.width * init_depth_image_size.height;
        if (cloud->points.size() != expected_number_of_points) {
            std::cerr << "Warning: Point cloud data dimensions do not match the expected image size." << std::endl;
            std::cerr << "Expected number of points: " << expected_number_of_points << std::endl;
            std::cerr << "Actual number of points: " << cloud->points.size() << std::endl;
            // Handle the mismatch accordingly
        }

        // init matrcies with x, y, z coordinates
        uint idx = 0;
        for (auto& p: cloud->points) {
            cloud_x.at<double>(idx) = p.x;
            cloud_y.at<double>(idx) = p.y;
            cloud_z.at<double>(idx) = p.z;
            if (p.z > maxDepth) {
                maxDepth = p.z;
            }
            ++idx;
        }
        auto depthInterMode = (scale > depth_init_scale) ? cv::INTER_NEAREST : cv::INTER_LINEAR;
        cv::resize(cloud_x, cloud_x, image_size, 0., 0., depthInterMode);
        cv::resize(cloud_y, cloud_y, image_size, 0., 0., depthInterMode);
        cv::resize(cloud_z, cloud_z, image_size, 0., 0., depthInterMode);

        //landmarks

        dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
        dlib::shape_predictor landmarkDetector;
        dlib::deserialize(landmarkFile) >> landmarkDetector;

        dlib::cv_image<dlib::bgr_pixel> dlibImage(rgb_image);
        std::vector<dlib::rectangle> faces = faceDetector(dlibImage);

        if (!faces.empty()) {
            // only work with the largest face in the image
            dlib::rectangle largestFace = *std::max_element(faces.begin(), faces.end(),
                                                            [](const dlib::rectangle& a, const dlib::rectangle& b) {
                                                                return a.area() < b.area();
                                                            });

            // Find facial landmarks for the largest face
            dlib::full_object_detection landmarks = landmarkDetector(dlibImage, largestFace);
            std::cout<<"Landmarks:"<<std::endl;
            for (size_t i = 0; i < landmarks.num_parts(); ++i) {
                int uLandmark = std::round(landmarks.part(i).x() * scale);
                int vLandmark = std::round(landmarks.part(i).y() * scale);

                std::cout<<"x: "<<uLandmark<<"; y: "<<vLandmark<<std::endl;

                landmarks_uv[2 * i] = uLandmark;
                landmarks_uv[2 * i + 1] = vLandmark;
                auto xyz = UVtoXYZ(uLandmark, vLandmark);
                landmarks_xyz[3 * i] = xyz[0];
                landmarks_xyz[3 * i + 1] = xyz[1];
                landmarks_xyz[3 * i + 2] = xyz[2];
            }
        }


        // Resize and normalize
        cv::resize(rgb_image, rgb_image, image_size);
        rgb_image.convertTo(rgb_image, CV_64FC3);
        rgb_image /= 255.0f;
    }

    // Retrieves the color values at a given (u, v) coordinate as double values.
    // These values can later be converted to actual RGB values for plotting or visualization.
    template <typename T>
    Eigen::Vector3d UVtoColor(T u, T v) const {
        if (u >= 0 && u < rgb_image.cols && v >= 0 && v < rgb_image.rows) {
            // Access the pixel at (u, v) and convert the color values to double
            const cv::Vec3d& color = rgb_image.at<cv::Vec3d>(v, u);

            // The cv::Vec3d contains BGR values in the order of [0] = Blue, [1] = Green, [2] = Red
            // Convert them to RGB and return as Eigen::Vector3d
            return Eigen::Vector3d(color[2], color[1], color[0]);
        } else {
            // Return NaN for out-of-bounds
            return Eigen::Vector3d(std::nan(""), std::nan(""), std::nan(""));
        }
    }

    template <typename T>
    Eigen::Vector3d UVtoXYZ(T u, T v) const {
        if (u < 0 || u >= image_size.width || v < 0 || v >= image_size.height) {
            // Return NaN values if the coordinates are out of bounds
            return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN());
        }
        int idx = v * image_size.width + u;
        return Eigen::Vector3d(cloud_x.at<double>(v, u),
                            cloud_y.at<double>(v, u),
                            cloud_z.at<double>(v, u));
    }


    template <typename T>
    double UVtoDepth(T u, T v) const{
        return UVtoXYZ(std::round(u), std::round(v))[2];
    }

    // Method to project XYZ onto the image space(pixels) using intrinsic parameters
    Eigen::Vector2d XYZtoUV(const Eigen::Vector3d& xyz) const {

        Eigen::Vector3d image_coords = camera_matrix * xyz;

        // Apply camera intrinsics to project onto the image plane
        return Eigen::Vector2d(image_coords[0] / image_coords[2], image_coords[1] / image_coords[2]);
    }

    //TODO: How to get adj. vertices? (For smooth shading)
    //Currently it assumes that every triangle is lying on the same plane
    std::vector<Eigen::Vector3d> computeVertexNormals(const std::shared_ptr<BfmManager>& pBfmManager){
        std::vector<Eigen::Vector3d> face_vertices;
        std::vector<unsigned int> face_indices;
        std::vector<Eigen::Vector3d> vertexNormals;
        
        //gives the indices of vertices
        for (size_t iIndex = 0; iIndex < pBfmManager->m_nFaces; iIndex++) {
            face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3));
            face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3 + 1));
            face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3 + 2));
        }
        // std::cout << "Num face indices: " << face_indices.size() << std::endl;
        //coordinates of the vertices
        for (size_t iVertex = 0; iVertex < pBfmManager->m_nVertices; iVertex++) {
            float x, y, z;

            x = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3));
            y = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3 + 1));
            z = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3 + 2));

            face_vertices.push_back({x, y, z});
        }
        std::cout << face_indices.size();
        // Compute vertex normals
        for (size_t i = 0; i < face_indices.size() - 2; i += 3) {
            unsigned int index1 = face_indices[i];
            unsigned int index2 = face_indices[i + 1];
            unsigned int index3 = face_indices[i + 2];

            Eigen::Vector3d vertA = face_vertices[index1];
            Eigen::Vector3d vertB = face_vertices[index2];
            Eigen::Vector3d vertC = face_vertices[index3];


            Eigen::Vector3d faceNormal = computeVNormal(vertA, vertB, vertC);
            //if triangle is lying on single plane normal of every vertex of a triangle is the same
            vertexNormals.push_back(faceNormal);
            vertexNormals.push_back(faceNormal);
            vertexNormals.push_back(faceNormal);
        }
        //normalize every vertex normal
        for(size_t i = 0; i < vertexNormals.size(); i++){
            vertexNormals[i].normalize();
        }
        return vertexNormals;
    }
    
    //computes normal of the given vertex 
    Eigen::Vector3d computeVNormal(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c){
        Eigen::Vector3d norm_vec(0.0f, 0.0f, 0.0f);
        Eigen::Vector3d AB = b - a; 
        Eigen::Vector3d AC = c - a; 

        return AB.cross(AC);
    }


    int getWidth() const {
        return image_size.width;
    }

    int getHeight() const {
        return image_size.height;
    }

    double getMaxDepth() const {
        return maxDepth;
    }

    const Eigen::VectorXd getXYZLandmarks() const {
        return landmarks_xyz;
    }

    const Eigen::VectorXi& getUVLandmarks() const {
        return landmarks_uv;
    }

    const Eigen::Matrix3d& getIntMat() const {
        return camera_matrix;
    }
};

bool initGlog(
    int argc, char* argv[],
    const std::string& logPath=R"(./log)") {

    namespace fs = boost::filesystem;
    namespace po = boost::program_options;

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    // try {
    //     if(fs::exists(logPath))
    //         fs::remove_all(logPath);
    //     fs::create_directory(logPath);
    // } catch (const std::exception& e) {
    //     std::cerr << "Error creating Google logs directory: " << e.what() << std::endl;
    //     // Handle the error appropriately
    // }
    FLAGS_alsologtostderr = true;
    FLAGS_log_dir = logPath;
    FLAGS_log_prefix = true;
    FLAGS_colorlogtostderr =true;

    // command options
    po::options_description opts("Options");
    po::variables_map vm;

    // path to .h5 file containing bfm basis and landmarks
    // opts.add_options()
    //     ("bfm_h5_path", po::value<string>(&sBfmH5Path)->default_value(
    //         R"(../Data/model2017-1_face12_nomouth.h5)"),
    //         "Path of Basel Face Model.")
    //     ("landmark_idx_path", po::value<string>(&sLandmarkIdxPath)->default_value(
    //         R"(../example/example_landmark_68.anl)"),
    //         "Path of corresponding between dlib and model vertex index.")
    //     ("help,h", "Help message");
    // try
    // {
    //     po::store(po::parse_command_line(argc, argv, opts), vm);
    // }
    // catch(...)
    // {
    //     LOG(ERROR) << "These exists undefined command options.";
    //     return false;
    // }

    po::notify(vm);
    if(vm.count("help"))
    {
        LOG(INFO) << opts;
        return false;
    }

    return true;
}