# Face Reconstruction
Our Face Reconstruction Pipeline aims to obtain parameters P' for a face parametric model M(P) that match a given RGB-D image I. It utilizes PCA-based morphable face models encompassing pose, shape, albedo, illumination, and expression parameters. The reconstruction process follows an analysis-by-synthesis approach, updating parameters to minimize an overall energy function. The energy function includes dense and sparse terms, incorporating geometry and color comparisons between the rendered face and the actual image.

## Main Dependencies

- OpenCV
- Eigen3
- Glog
- GLFW
- PCL
- Boost
- Ceres

## Expression Transfer
![image](https://github.com/diddone/face-recon/assets/47386144/73fc5182-38b5-4b56-9c01-02b1c2a84251)

Transfer expression example. We find parameters independently using source (top left) and target (bottom left) actors. Then, we
transfer expression coefficients from the source into the target. Finally, we project the source image into the mesh and visualize the mask
above the source image.

## Results
![rgb_only_table](https://github.com/diddone/face-recon/assets/47386144/fba8fb91-ea2e-4ae6-9aee-13c077de306d)

Results of the optimization pipeline using RGB only.
In the center column, we apply a BFM model texture. The last
column obtains texture from the projection mesh into the image.


## Building
```bash
mkdir build && cd build
cmake ..
make
```

## Method

You can read the final report where explained our method [here](link-to-your-report).


## Team Members
- Gökçe Şengün
- Dmitrii Pozdev
- Biray Sütçüoğlu
- David Gichev

