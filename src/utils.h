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
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

class ImageUtilityThing {
public:
    cv::Mat rgb_image;
    cv::Mat cloud_x;
    cv::Mat cloud_y;
    cv::Mat cloud_z;
    Eigen::VectorXd landmarks_xyz;

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

        // Resize and normalize
        cv::resize(rgb_image, rgb_image, image_size);
        rgb_image.convertTo(rgb_image, CV_64FC3);
        rgb_image /= 255.0f;


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
        std::ifstream inFile;
        inFile.open(landmarkFile, std::ios::in);
        assert(inFile.is_open());
        int uLandmark, vLandmark;
        size_t landmarkCnt = 0;
        while (inFile >> uLandmark >> vLandmark) {
            uLandmark = std::round(uLandmark * scale);
            vLandmark = std::round(vLandmark * scale);
            auto xyz = UVtoXYZ(uLandmark, vLandmark);
            landmarks_xyz[3 * landmarkCnt] = xyz[0];
            landmarks_xyz[3 * landmarkCnt + 1] = xyz[1];
            landmarks_xyz[3 * landmarkCnt + 2] = xyz[2];
            ++landmarkCnt;
        }
        inFile.close();
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
    Eigen::Vector2d XYZtoUV(const Eigen::Vector3d& xyz) {

        Eigen::Vector3d image_coords = camera_matrix * xyz;

        // Apply camera intrinsics to project onto the image plane
        return Eigen::Vector2d(image_coords[0] / image_coords[2], image_coords[1] / image_coords[2]);
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

    Eigen::VectorXd getXYZLandmarks() {
        return landmarks_xyz;
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