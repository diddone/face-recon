#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <string>
#include <limits>
#include <fstream>

class ImageUtilityThing {
public:
    cv::Mat rgb_image;
    cv::Mat cloud_x;
    cv::Mat cloud_y;
    cv::Mat cloud_z;

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
    }

    cv::Size rescaleImageSize(const cv::Size& old_image_size) const {
        return cv::Size(int(old_image_size.width * scale), (old_image_size.height * scale));
    }
    void input(const std::string& image_file, const std::string& pcd_file, double new_scale=1.0) {
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

    // // back-projection
    // template <typename T>
    // Eigen::Vector3d UVtoXYZ(T u, T v) const{
    //     // Fetch RGB value at (u, v)
    //     int idx = std::round(u) + image_size.width * std::round(v);

    //     return {
    //         cloud_x.at<double>(idx),
    //         cloud_y.at<double>(idx),
    //         cloud_z.at<double>(idx)
    //     };
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

    // Backproject UV coordinates to XYZ and store in point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr backprojectToPointCloud(double maxDepthFilter = 2.5) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (int v = 0; v < image_size.height; ++v) {
            for (int u = 0; u < image_size.width; ++u) {
                Eigen::Vector3d xyz = UVtoXYZ(u, v);

                // Skip points with NaN values
                if (!std::isnan(xyz[0]) && !std::isnan(xyz[1]) && !std::isnan(xyz[2]) && xyz[2] < maxDepthFilter) {
                    point_cloud->push_back(pcl::PointXYZ(xyz[0], xyz[1], xyz[2]));
                }
            }
        }

        return point_cloud;
    }

    // Reproject XYZ coordinates back to UV space and check consistency
    void checkBackprojectionReprojectionConsistency() {
        cv::Mat consistency_image(image_size, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int v = 0; v < image_size.height; ++v) {
            for (int u = 0; u < image_size.width; ++u) {
                Eigen::Vector3d xyz = UVtoXYZ(u, v);

                // Skip points with NaN values
                if (!std::isnan(xyz[0]) && !std::isnan(xyz[1]) && !std::isnan(xyz[2])) {
                    Eigen::Vector2d uv = XYZtoUV(xyz);

                    // Check if the reprojected UV coordinates match the original
                    if (uv[0] >= 0 && uv[0] < image_size.width && uv[1] >= 0 && uv[1] < image_size.height) {
                        consistency_image.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 255, 0); // Green for consistency
                    } else {
                        consistency_image.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 255); // Red for inconsistency
                    }
                }
            }
        }

        // Save the consistency image for visual inspection
        cv::imwrite("../Output/consistency_image.jpeg", consistency_image);
    }

};


int main() {
    // Example usage
    ImageUtilityThing instance("../Data/rgbd_face_dataset_training/camera_info.yaml");
    instance.input("../Data/image.png", "../Data/cloud.pcd");
    // instance.printDetails();

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>("../Data/cloud.pcd", *cloud) == -1) {
            PCL_ERROR("Couldn't read pcd file \n");
            exit(EXIT_FAILURE);
        }

    // size_t c = 0;
    // for (int v = 0; v < instance.getHeight() / 2; ++v) {
    //     for (int u = 0; u < instance.getWidth() / 2; ++u) {
    //         auto p = (*cloud)(u, v);

    //         if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
    //             ++c;
    //         }
    //     }
    // }
    // std::ofstream out;
    // out.open ("init_cloud.ply");

    // out << "ply\n";
    // out << "format binary_little_endian 1.0\n";
    // out << "comment Made from the 3D Morphable Face Model of the Univeristy of "
    //         "Basel, Switzerland.\n";
    // out << "element vertex " << c << "\n";
    // out << "property float x\n";
    // out << "property float y\n";
    // out << "property float z\n";
    // out << "property uchar red\n";
    // out << "property uchar green\n";
    // out << "property uchar blue\n";
    // out << "element face " << 0 << "\n";
    // out << "property list uchar int vertex_indices\n";
    // out << "end_header\n";

    // for (int v = 0; v < instance.getHeight() / 2; ++v) {
    //     for (int u = 0; u < instance.getWidth() / 2; ++u) {
    //         auto p = (*cloud)(u, v);
    //         if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
    //             uchar r, g, b;
    //             if (p.x < 2.5) {
    //                 r = 255, g = 0, b = 255;
    //             } else {
    //                 r = 100, g= 100, b = 100;
    //             }
    //             out.write((char *)&p.x, sizeof(p.x));
    //             out.write((char *)&p.y, sizeof(p.y));
    //             out.write((char *)&p.z, sizeof(p.z));
    //             out.write((char *)&r, sizeof(r));
    //             out.write((char *)&g, sizeof(g));
    //             out.write((char *)&b, sizeof(b));
    //             // out << p.x << " " << p.y << " " << p.z << " " << r << " " << g << " " << b << "\n";
    //         }
    //      }
    // }
    // out.close();

    if (instance.rgb_image.empty()) {
        std::cerr << "Failed to load RGB image." << std::endl;
        return 1;
    }

    // Test: Save RGB image using OpenCV's imwrite
    cv::Mat processedImage(instance.getHeight(), instance.getWidth(), CV_8UC3);

    for (int v = 0; v < instance.getHeight(); ++v) {
        for (int u = 0; u < instance.getWidth(); ++u) {
            Eigen::Vector3d rgb = instance.UVtoColor(u, v);

            // Convert double to uchar
            cv::Vec3b color(
                static_cast<uchar>(rgb[2] * 255.0),  // R
                static_cast<uchar>(rgb[1] * 255.0),  // G
                static_cast<uchar>(rgb[0] * 255.0)); // B

            processedImage.at<cv::Vec3b>(cv::Point(u, v)) = color;
        }
    }

    // Save the processed image
    // if (!cv::imwrite("../Output/output_image.jpeg", processedImage)) {
    //     std::cerr << "Failed to save the processed image." << std::endl;
    //     return 1;
    // } else {
    //     std::cout << "RGB image saved successfully." << std::endl;
    // }

    // Create a grayscale image
    cv::Mat depth_image(instance.getHeight(), instance.getWidth(), CV_64F);

    // Iterate over each pixel in the image
    size_t cnt = 0;
    for (int v = 0; v < instance.getHeight(); ++v) {
        for (int u = 0; u < instance.getWidth(); ++u) {
            Eigen::Vector3d p = instance.UVtoXYZ(u, v);
            if (!std::isnan(p[0]) && !std::isnan(p[1]) && !std::isnan(p[2])) {
                ++cnt;
            }
         }
    }
    std::ofstream myfile;
    myfile.open ("cloud.off");
    myfile << "OFF\n";
    myfile << cnt << " " << 0 << " " << 0 << "\n";
    for (int v = 0; v < instance.getHeight(); ++v) {
        for (int u = 0; u < instance.getWidth(); ++u) {
            Eigen::Vector3d p = instance.UVtoXYZ(u, v);
            if (!std::isnan(p[0]) && !std::isnan(p[1]) && !std::isnan(p[2])) {
                myfile << p[0] << " " << p[1] << " " << p[2] << "\n";
            }
         }
    }
    myfile.close();


    // Test 5: Backprojection to Point Cloud and Observation
    // Create a new point cloud object to store filtered points
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud = instance.backprojectToPointCloud();

  //  Save the point cloud to a PCD file for observation in MeshLab
    if (pcl::io::savePCDFile("../Output/output_cloud_maxepth.pcd", *point_cloud) < 0) {
        std::cerr << "Error saving point cloud to PCD file." << std::endl;
        return -1;
    }
    std::cout << "Saved " << point_cloud->points.size() << " valid data points to output_cloud.pcd" << std::endl;

    // Test 6: Backprojection and Reprojection Consistency Check
    instance.checkBackprojectionReprojectionConsistency();

    return 0;
}
