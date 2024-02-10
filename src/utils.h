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

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

class ImageRGBOnly {
protected:
    cv::Mat rgb_image;
    Eigen::VectorXi landmarks_uv;
    Eigen::Matrix3d camera_matrix;

    cv::Size init_rgb_image_size;
    cv::Size image_size;
    double scale = 1.;
public:
    ImageRGBOnly() {
      landmarks_uv = Eigen::VectorXi::Zero(2 * N_DLIB_LANDMARKS);
    }

    cv::Size rescaleImageSize(const cv::Size& old_image_size) const {
        return cv::Size(int(old_image_size.width * scale), (old_image_size.height * scale));
    }

    void input(const std::string& image_file, const std::string& landmarkFile, double new_scale=1.0) {
        if (new_scale <= 0) {
          std::cerr << "Error: Scale must be greater than zero." << std::endl;
          return;
        }
        // set new scale and image size
        scale = new_scale;
        //cv::Size image_size = rescaleImageSize(init_rgb_image_size);

        // Load RGB image
        // cv::Mat rgb_image = cv::imread(image_file);
        rgb_image = cv::imread(image_file);
        init_rgb_image_size = rgb_image.size();
        image_size = rescaleImageSize(init_rgb_image_size);
        if (rgb_image.empty()) {
            std::cerr << "Error: RGB image not loaded properly." << std::endl;
            return;
        }

        // Resize and normalize
        cv::resize(rgb_image, rgb_image, image_size);
        rgb_image.convertTo(rgb_image, CV_32FC3);
        rgb_image /= 255.0f;

        // init intrinsics
        double Fx = 50.;
        double Fy = 50.;
        double W = 36.;
        double H = 24.;
        double fx = Fx * image_size.width / W;
        double fy = Fy * image_size.height / H;
        double cx = double(image_size.width) / 2;
        double cy = double(image_size.height) / 2;
        camera_matrix << fx, 0., cx,
                          0., fy, cy,
                          0., 0., 1.;

        std::ifstream inFile;
        inFile.open(landmarkFile, std::ios::in);
        assert(inFile.is_open());
        int uLandmark, vLandmark;
        size_t landmarkCnt = 0;
        while (inFile >> uLandmark >> vLandmark) {
            uLandmark = std::round(uLandmark * scale);
            vLandmark = std::round(vLandmark * scale);

            landmarks_uv[2 * landmarkCnt] = uLandmark;
            landmarks_uv[2 * landmarkCnt + 1] = vLandmark;
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

    // this function may be reimplemented
    // virtual Eigen::Vector3d UVtoXYZ(double u, double v) const;


    // template <typename T>
    // double UVtoDepth(T u, T v) const{
    //     return UVtoXYZ(std::round(u), std::round(v))[2];
    // }

    // Method to project XYZ onto the image space(pixels) using intrinsic parameters
    Eigen::Vector2d XYZtoUV(const Eigen::Vector3d& xyz) const {

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

    // virtual double getMaxDepth() const;

    // virtual const Eigen::VectorXd getXYZLandmarks() const;

    const Eigen::VectorXi& getUVLandmarks() const {
        return landmarks_uv;
    }

    const Eigen::Matrix3d& getIntMat() const {
        return camera_matrix;
    }

    // void writePly(std::string fn) const;
};

class ImageUtilityThing : public ImageRGBOnly {
public:
    // cv::Mat rgb_image;
    cv::Mat cloud_x;
    cv::Mat cloud_y;
    cv::Mat cloud_z;
    Eigen::VectorXd landmarks_xyz;
    // Eigen::VectorXi landmarks_uv;

    // Eigen::Matrix3d camera_matrix;
    cv::Mat dist_coeffs;
    // cv::Size init_rgb_image_size;
    cv::Size init_depth_image_size;
    double maxDepth = 0.0;

    // cv::Size image_size;
    double depth_init_scale = 0.5;
    // double scale = 1.;
    cv::Mat normalMap;
    bool normalsComputed = false;
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

    // cv::Size rescaleImageSize(const cv::Size& old_image_size) const {
    //     return cv::Size(int(old_image_size.width * scale), (old_image_size.height * scale));
    // }
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
        // auto depthInterMode = (scale > depth_init_scale) ? cv::INTER_NEAREST : cv::INTER_LINEAR;
        cv::resize(cloud_x, cloud_x, image_size, 0., 0., cv::INTER_LINEAR);
        cv::resize(cloud_y, cloud_y, image_size, 0., 0., cv::INTER_LINEAR);
        cv::resize(cloud_z, cloud_z, image_size, 0., 0., cv::INTER_LINEAR);

        //landmarks

        std::ifstream inFile;
        inFile.open(landmarkFile, std::ios::in);
        assert(inFile.is_open());
        int uLandmark, vLandmark;
        size_t landmarkCnt = 0;
        while (inFile >> uLandmark >> vLandmark) {
            uLandmark = std::round(uLandmark * scale);
            vLandmark = std::round(vLandmark * scale);

            landmarks_uv[2 * landmarkCnt] = uLandmark;
            landmarks_uv[2 * landmarkCnt + 1] = vLandmark;
            auto xyz = UVtoXYZ(uLandmark, vLandmark);
            landmarks_xyz[3 * landmarkCnt] = xyz[0];
            landmarks_xyz[3 * landmarkCnt + 1] = xyz[1];
            landmarks_xyz[3 * landmarkCnt + 2] = xyz[2];

            ++landmarkCnt;
        }
        inFile.close();

        // init normals
        normalMap = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC3);
    }

    // Retrieves the color values at a given (u, v) coordinate as double values.
    // These values can later be converted to actual RGB values for plotting or visualization.
    // template <typename T>
    // Eigen::Vector3d UVtoColor(T u, T v) const {
    //     if (u >= 0 && u < rgb_image.cols && v >= 0 && v < rgb_image.rows) {
    //         // Access the pixel at (u, v) and convert the color values to double
    //         const cv::Vec3d& color = rgb_image.at<cv::Vec3d>(v, u);

    //         // The cv::Vec3d contains BGR values in the order of [0] = Blue, [1] = Green, [2] = Red
    //         // Convert them to RGB and return as Eigen::Vector3d
    //         return Eigen::Vector3d(color[2], color[1], color[0]);
    //     } else {
    //         // Return NaN for out-of-bounds
    //         return Eigen::Vector3d(std::nan(""), std::nan(""), std::nan(""));
    //     }
    // }

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
    // already been overriden
    // Eigen::Vector2d XYZtoUV(const Eigen::Vector3d& xyz) const {

    //     Eigen::Vector3d image_coords = camera_matrix * xyz;

    //     // Apply camera intrinsics to project onto the image plane
    //     return Eigen::Vector2d(image_coords[0] / image_coords[2], image_coords[1] / image_coords[2]);
    // }

    // int getWidth() const {
    //     return image_size.width;
    // }

    // int getHeight() const {
    //     return image_size.height;
    // }

    double getMaxDepth() const {
        return maxDepth;
    }

    const Eigen::VectorXd getXYZLandmarks() const {
        return landmarks_xyz;
    }

    // const Eigen::VectorXi& getUVLandmarks() const {
    //     return landmarks_uv;
    // }

    // const Eigen::Matrix3d& getIntMat() const {
    //     return camera_matrix;
    // }

    void computeNormals() {
      Eigen::Vector3d v1, v2, curNormal;
      for (size_t v = 0; v < image_size.height; ++v) {
        for (size_t u = 0; u < image_size.width; ++u) {
            size_t idx = 3 * (u + v * image_size.width);
            if (v == 0 || v == image_size.height - 1 || u == 0 || u == image_size.width - 1) {
                // Eigen::Vector3d({std::nan, std::nan, std::nan});
                normalMap.at<cv::Vec3f>(v, u) = {0., 0., 0.};
                continue;
            }

            v1 = UVtoXYZ(u + 1, v) - UVtoXYZ(u - 1, v);
            v2 = UVtoXYZ(u, v + 1) - UVtoXYZ(u, v - 1);
            if (v1.hasNaN() || v2.hasNaN()) {
              normalMap.at<cv::Vec3f>(v, u) = {0., 0., 0.};
              continue;
            }
            curNormal = v1.cross(v2);
            if (curNormal.dot(UVtoXYZ(u, v)) > 0) {
                curNormal *= -1;
            }

            cv::Vec3f& pixelValue = normalMap.at<cv::Vec3f>(v, u);
            curNormal.normalize();
            // opencv stores in bgr.
            pixelValue[0] = float(curNormal[2]);
            pixelValue[1] = float(curNormal[1]);
            pixelValue[2] = float(curNormal[0]);
        }
      }

      auto newMap = normalMap.clone();
      cv::bilateralFilter(newMap, normalMap, 7, 50, 50);
      normalsComputed = true;
    }

    void findMinMax(const cv::Mat& image) const {
      std::vector<cv::Mat> channels;
      cv::split(image, channels);

      // Find the maximum value for each channel
      std::vector<double> vals;
      for (int i = 0; i < channels.size(); ++i) {
          double minVal;
          double maxVal;
          cv::minMaxLoc(channels[i], &minVal, &maxVal);
          std::cout << minVal << " " << maxVal << std::endl;
      }
      std::cout << "----------------\n";
    }

    void writeNormalsAsImage(std::string savePath) const {
        // Create a Mat object from the image data
        cv::Mat image(image_size.height, image_size.width, CV_8UC3); // 3 channels, unsigned char
        // cv::Mat newMap = 255. * (0.5 * normalMap + 0.5);
        // newMap.convertTo(image, CV_8UC3);
        // findMinMax(image);
        // Copy the image data from the vector to the Mat object
        size_t height = image_size.height;
        size_t width = image_size.width;
        uchar b,g,r;
        Eigen::Vector3d normal;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            normal = UVtoNormal(w, h);
            size_t idx = 3 * (h * width + w);
            if (normal.hasNaN()) {
              r = 128;
              g = 128;
              b = 128;
            } else {
              // linearly map [-1, 1] to [0, 255]
              r = std::round((0.5 * normal[0]+ 0.5) * 255);
              g = std::round((0.5 * normal[1]+ 0.5)* 255);
              b = std::round((0.5 * normal[2]+ 0.5)* 255);
            }
            // in opencv image stored in bgr
            image.at<cv::Vec3b>(h, w) = cv::Vec3b(
                b,  // Blue channel
                g,  // Green channel
                r   // Red channel
            );
          }
        }
        // Save the Mat object to a PNG file
        cv::imwrite(savePath, image);
    }

    template <typename T>
    Eigen::Vector3d UVtoNormal(T u, T v) const {
        if (!normalsComputed) {
          throw std::logic_error("Normals for imageUtility should be computed first");
        }
        if (u < 0 || u >= image_size.width || v < 0 || v >= image_size.height) {
            // Return NaN values if the coordinates are out of bounds
            return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN());
        }

        size_t idx = std::round(v) * image_size.width + std::round(u);
        // normal stored as a floats
        cv::Vec3f normal = normalMap.at<cv::Vec3f>(v, u);
        float norm = cv::norm(normal);
        // filter vectors with small norms, we store (0, 0, 0) for nans
        if (norm < 0.3) {
           return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN());
        }
        // open cv stores in bgr
        return Eigen::Vector3d(normal[2], normal[1], normal[0]).normalized();
    }

    void writePly(std::string fn) const {
        std::ofstream out;
        out.open(fn, std::ios::out | std::ios::binary);
        if (!out.is_open()) {
            std::string sErrMsg = "Creation of " + fn + " failed.";
            LOG(ERROR) << sErrMsg;
            throw std::runtime_error(sErrMsg);
            return;
        }

        // we are using this to print only near the face region
        // we are using our landmark annotaion to define boundaries
        int MIN_IDX_Y = 9;
        int MAX_IDX_Y = 25;
        int y1 = getUVLandmarks()[2 * (MIN_IDX_Y-1) + 1];
        int y2 = getUVLandmarks()[2 * (MAX_IDX_Y-1) + 1];
        int y_min = std::min(y1, y2) - 10;
        int y_max = std::max(y1, y2) + 10;
        size_t validCnt = 0;
        for (int v = 0; v < getHeight(); ++v) {
            for (int u = 0; u < getWidth(); ++u) {
                double depth = UVtoDepth(u, v);
                if (depth > 0.5 && !std::isnan(depth) && depth < 0.9 && v < y_max && v > y_min) {
                    ++validCnt;
                }
            }
        }

        out << "ply\n";
        out << "format binary_little_endian 1.0\n";
        out << "comment Made from the 3D Morphable Face Model of the Univeristy of "
                "Basel, Switzerland.\n";
        out << "element vertex " << validCnt << "\n";
        out << "property double x\n";
        out << "property double y\n";
        out << "property double z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "element face " << 0 << "\n";
        out << "property list uchar int vertex_indices\n";
        out << "end_header\n";

        std::cout << "ValidCnt " << validCnt << std::endl;
        double x, y, z;
        unsigned char r=0,g=0,b=200;
        for (int v = 0; v < getHeight(); ++v) {
            for (int u = 0; u < getWidth(); ++u) {
                double depth = UVtoDepth(u, v);
                if (!std::isnan(depth) && depth > 0.5 && depth < 0.9 && v < y_max && v > y_min) {
                    auto xyz = UVtoXYZ(u, v);
                    x = xyz.x();
                    y = xyz.y();
                    z = xyz.z();
                    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
                        std::cout << "NAN " << x << " " << y << " " << z << std::endl;
                    }
                    // std::cout << "Cloud " << x << " " << y << " " << z << " " << std::endl;
                    out.write((char *)&x, sizeof(x));
                    out.write((char *)&y, sizeof(y));
                    out.write((char *)&z, sizeof(z));
                    out.write((char *)&r, sizeof(r));
                    out.write((char *)&g, sizeof(g));
                    out.write((char *)&b, sizeof(b));
                }
            }
        }
        out.close();
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
