#include "transform.hpp"

#include "bfm_manager.h"
#include "glog/logging.h"

#include <fstream>
#include <iostream>
#include <array>
#include <memory>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


class CameraProjection {
public:

    CameraProjection(double fx, double fy, double cx, double cy)
    {
        intMat << fx, 0., cx,
                    0., fy, cy,
                    0., 0., 1.;
        intMaxInv = intMat.inverse();
    }

    Eigen::Vector2d project(const Eigen::Vector3d& p) const {
        Vector3d proj_p = intMat * p;
        double z = proj_p(2);
        return {proj_p(0) / z, proj_p(1) / z};
    }

    Eigen::Vector3d back_project(const Eigen::Vector2i& uv) {
        Vector3d p({uv[0], uv[1], 1.});
        return intMaxInv * p;
    }
private:
    Matrix3d intMat;
    Matrix3d intMaxInv;
};


bool initGlog(
    int argc, char* argv[],
    const std::string& logPath=R"(./log)") {

    namespace fs = boost::filesystem;
    namespace po = boost::program_options;

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    if(fs::exists(logPath))
        fs::remove_all(logPath);
    fs::create_directory(logPath);
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