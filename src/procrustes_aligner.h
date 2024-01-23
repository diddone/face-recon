#pragma once

// #include "utils.h"
#include <iostream>
#include "eigen.h"
using namespace Eigen;

//TODO: Add scaling

struct ExtrinsicTransform {
	Eigen::Matrix3d rotation;
	Eigen::Vector3d translation;
	double scale;
};

class ProcrustesAligner {
public:
	ExtrinsicTransform estimatePose(const VectorXd& sourcePoints_, const VectorXd& targetPoints_) {
		// ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");
		// Method from:
    	// Umeyama et al. Least-squares estimation of transformation parameters between two point patterns
    	// \\ 1991, Pattern Analysis and Machine Intelligence, IEEE Transactions on
    	// http://web.stanford.edu/class/cs273/refs/umeyama.pdf

		ASSERT((sourcePoints_.size() == targetPoints_.size()));
		size_t numPoints = sourcePoints_.size() / 3;
		size_t nValidPoints = 0;
		// there can be nans in image data
        for (size_t i = 0; i < numPoints; ++i) {
			// std::cout << "Source " << sourcePoints_[3 * i] << " "
            // << sourcePoints_[3 * i+1] << " "
            // << sourcePoints_[3 * i+2] << std::endl;
            // std::cout << "Target " << targetPoints_[3 * i] << " "
            // << targetPoints_[3 * i+1] << " "
            // << targetPoints_[3 * i+2] << std::endl;
            if (!std::isnan(targetPoints_[3 * i]) &&
				!std::isnan(targetPoints_[3 * i + 1]) &&
				!std::isnan(targetPoints_[3 * i + 2])) {
				++nValidPoints;
			}
		}
        std::cout << "Number of valid points in procrusters " << nValidPoints << std::endl;
        VectorXd sourcePoints(3 * nValidPoints), targetPoints(3 * nValidPoints);
        size_t cnt = 0;
        for (size_t i = 0; i < numPoints; ++i) {
			if (!std::isnan(targetPoints_[3 * i]) &&
				!std::isnan(targetPoints_[3 * i + 1]) &&
				!std::isnan(targetPoints_[3 * i + 2])) {
                for(size_t k = 0; k < 3; ++k) {
                    sourcePoints[3 * cnt + k] = sourcePoints_[3 * i + k];
                    targetPoints[3 * cnt + k] = targetPoints_[3 * i + k];
                }
                ++cnt;
			}
		}

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);

		Matrix3d covMat = getCovMatrix(sourcePoints, sourceMean, targetPoints, targetMean);
		JacobiSVD<Matrix3d> svd(covMat, ComputeFullV | ComputeFullU);
		Matrix3d U = svd.matrixU();
		Matrix3d V = svd.matrixV();

		double sigma_sq = getSigmaSquared(sourcePoints, sourceMean);

		double scale = 1 / sigma_sq * (svd.singularValues().sum());
		Matrix3d rotation = U * V.transpose();
		Vector3d translation = targetMean - scale * (rotation * sourceMean);

		std::cout << "Procrustes scale factor" << scale << std::endl;
		std::cout << "Procrustes Rotation matrix " << rotation << std::endl;
        std::cout << "Procrustes Rotation matrix Determinant" << rotation.determinant() << std::endl;
		std::cout << "Procrustes translation vector " << translation << std::endl;


		for (size_t i = 0; i < sourcePoints.size() / 3; ++i) {
			Vector3d x = {sourcePoints[3 * i], sourcePoints[3 * i + 1], sourcePoints[3 * i + 2]};
			Vector3d y = {targetPoints[3 * i], targetPoints[3 * i + 1], targetPoints[3 * i + 2]};
			Vector3d new_x = scale * rotation * x + translation;

            // std::cout << y[0] << " " << y[1] << " " << y[2] << "\n";
            // std::cout << "X Y New_x \n";
			// for (uint k = 0; k < 3; ++k) {
			// 	std::cout << " " << x[k] << " " << y[k] << " " << new_x[k] << "\n";
			// }
			// std::cout << "--------------------";
		}
		return ExtrinsicTransform{rotation, translation, scale};
	}

private:
	Vector3d computeMean(const VectorXd& points) {
		// Hint: You can use the .size() method to get the length of a vector.
		double meanX = 0.0f;
		double meanY = 0.0f;
		double meanZ = 0.0f;
		int numPoints = points.size()/3;

    	for (int i = 0; i < numPoints; ++i) {
        	// std::cout << "Element " << i << ": " << points(i) << std::endl;
			meanX += points(i * 3);
        	meanY += points(i * 3 + 1);
        	meanZ += points(i * 3 + 2);
    	}
		meanX /= numPoints;
    	meanY /= numPoints;
    	meanZ /= numPoints;

		Vector3d meanVector = {meanX, meanY, meanZ};
		return meanVector;
	}


	Matrix3d getCovMatrix(const VectorXd& source_points, const Vector3d& sourceMean, const VectorXd& target_points, const Vector3d& targetMean) {
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		// Hint: You can initialize an Eigen matrix with "MatrixXf m(num_rows,num_cols);" and access/modify parts of it using the .block() method (see above).

		auto nPoints = source_points.size() / 3;
		MatrixXd sourceMatrix(nPoints, 3);
		MatrixXd targetMatrix(nPoints, 3);

		for (unsigned int i = 0; i < nPoints; i++) {
        	// Starting from 0 col, put 3 coordinates (x, y, z) = mean-centered points
        	sourceMatrix.block(i, 0, 1, 3) = (source_points.segment(i * 3, 3) - sourceMean).transpose();
        	targetMatrix.block(i, 0, 1, 3) = (target_points.segment(i * 3, 3) - targetMean).transpose();
    	}

		return (1./ nPoints) * targetMatrix.transpose() * sourceMatrix;
	}

	double getSigmaSquared(const VectorXd& points, Vector3d mean){
		// std::cout << "Center of mass: " << centerOfMass[0] << " " << centerOfMass[1] << " " << centerOfMass[2] << std::endl;

		double avgDistance = 0.0;
		double totalDistance = 0.0;
		int numPoints = points.size() / 3;

		double x = 0.0f;
		double y = 0.0f;
		double z = 0.0f;
		for (int i = 0; i < numPoints; i++){
			x = points(i * 3);
			y = points(i * 3 + 1);
			z = points(i * 3 + 2);

			totalDistance += std::pow(x - mean[0], 2) +
								std::pow(y - mean[1], 2) +
								std::pow(z - mean[2], 2);

			// std::cout << totalDistance << std::endl;
		}

		avgDistance = totalDistance / numPoints;
		return avgDistance;
	}
};