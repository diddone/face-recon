# Face Reconstruction
Our Face Reconstruction Pipeline aims to obtain parameters P' for a face parametric model M(P) that match a given RGB-D image I. It utilizes PCA-based morphable face models encompassing pose, shape, albedo, illumination, and expression parameters. The reconstruction process follows an analysis-by-synthesis approach, updating parameters to minimize an overall energy function. The energy function includes dense and sparse terms, incorporating geometry and color comparisons between the rendered face and the actual image.
## Dependencies

- Eigen3
- OpenCV
- glog - Google's logging library
- yaml-cpp
- GLFW
- PCL
- Boost
- Ceres
-

## Building
```bash
mkdir build && cd build
cmake ..
make

## Method

You can read the final report where explained our method [here](link-to-your-report).

## Results
