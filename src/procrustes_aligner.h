#pragma once

// #include "utils.h"
#include <iostream>
#include "eigen.h"
using namespace Eigen;

//TODO: Add scaling
class ProcrustesAligner {
public:
	Matrix4d estimatePose(const VectorXd& sourcePoints_, const VectorXd& targetPoints_) {
		// ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");
		// TODO: Use scaling matrix
		// We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.

		ASSERT((sourcePoints_.size() == targetPoints_.size()));
		size_t numPoints = sourcePoints_.size() / 3;
		VectorXd sourcePoints(sourcePoints_.size()), targetPoints(targetPoints_.size());
		size_t cnt = 0;
		for (size_t i = 0; i < numPoints; ++i) {
			std::cout << "Source" << sourcePoints_[3 * i] << " " << sourcePoints_[3 * i + 1] << " " << sourcePoints_[3 * i + 2] << "\n";
			std::cout << "Target" << targetPoints_[3 * i] << " " << targetPoints_[3 * i + 1] << " " << targetPoints_[3 * i + 2] << "\n";
			if (!std::isnan(sourcePoints_[3 * i]) &&
				!std::isnan(sourcePoints_[3 * i + 1]) &&
				!std::isnan(sourcePoints_[3 * i + 2])) {
				for (size_t k = 0; k < 3; ++k) {
					sourcePoints[3 * cnt + k] = sourcePoints_[3 * i + k];
					targetPoints[3 * cnt + k] = targetPoints_[3 * i + k];
				}
				++cnt;
			}
		}
		sourcePoints.resize(3 * cnt);
		targetPoints.resize(3 * cnt);

		double scalingFactor = estimateScaling(sourcePoints, targetPoints);
		std::cout << "Scaling factor: " << scalingFactor << std::endl;

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);

		Matrix3d rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		std::cout << "rotation matrix " << rotation << std::endl;

		Vector3d translation = computeTranslation(sourceMean, targetMean, rotation);
		std::cout << "translation vector " << translation << std::endl;

		// You can access parts of the matrix with .block(start_row, start_col, num_rows, num_cols) = elements
		Matrix4d estimatedPose = Matrix4d::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = translation;
		std::cout << "pose matrix " << estimatedPose << std::endl;
		return estimatedPose;
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

	// Compute center of gravity for 2 objects
	// Scale one object to match the avg. distance from all vertices to the center of gravity
	double estimateScaling(const VectorXd& source_points, const VectorXd& target_points){
		//for source points
		Vector3d centerofMass_source = computeMean(source_points);
		double distancetoCenter_source = calculateAvgDistancetoCenter(source_points, centerofMass_source);

		//for target points
		Vector3d centerofMass_target = computeMean(target_points);
		double distancetoCenter_target = calculateAvgDistancetoCenter(target_points, centerofMass_target);

		double scaling_factor = distancetoCenter_source / distancetoCenter_target;

		// float s_x = distancetoCenter_source.x / distancetoCenter_target.x;
		// float s_y = distancetoCenter_source.y / distancetoCenter_target.y;
		// float s_z = distancetoCenter_source.z / distancetoCenter_target.z;

		// // Create the scaling matrix
    	// Matrix3f scalingMatrix;
    	// scalingMatrix.m[0][0] = s_x;
    	// scalingMatrix.m[1][1] = s_y;
    	// scalingMatrix.m[2][2] = s_z;

    	// // Set the rest of the elements to 0
    	// for (int i = 0; i < 3; ++i) {
        // 	for (int j = 0; j < 3; ++j) {
        //     	if (i != j) {
        //         	scalingMatrix.m[i][j] = 0.0f;
        //     	}
        // 	}
    	// }
		return scaling_factor;
	}

	Matrix3d estimateRotation(const VectorXd& source_points, const Vector3d& sourceMean, const VectorXd& target_points, const Vector3d& targetMean) {
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

		JacobiSVD<Matrix3d> svd(targetMatrix.transpose() * sourceMatrix, ComputeFullV | ComputeFullU);
		Matrix3d U = svd.matrixU();
		Matrix3d V = svd.matrixV();

		double det = (U * V.transpose()).determinant();
		Matrix3d D = Matrix3d::Identity();
		if (det < 0) {
			D(2,2) = -1.;
		}

		Matrix3d rotation = U * D * V.transpose();

		return rotation;
	}

	Vector3d computeTranslation(const Vector3d& sourceMean, const Vector3d& targetMean, const Matrix3d& rotation) {
		// Vector3d translation = Vector3d::Zero();
		Vector3d translation = targetMean - rotation * sourceMean;
        return translation;
	}

	double calculateAvgDistancetoCenter(const VectorXd& points, Vector3d centerOfMass){
		// std::cout << "Center of mass: " << centerOfMass[0] << " " << centerOfMass[1] << " " << centerOfMass[2] << std::endl;

		double avgDistance = 0.0;
		double totalDistance = 0.0;
		int numPoints = points.size() / 3;

		double x = 0.0f;
		double y = 0.0f;
		double z = 0.0f;
		for (int i = 0; i < numPoints; i++){
			x += points(i * 3);
			y += points(i * 3 + 1);
			z += points(i * 3 + 2);

			totalDistance +=  std::sqrt(std::pow(x - centerOfMass[0], 2) +
										std::pow(y - centerOfMass[1], 2) +
										std::pow(z - centerOfMass[2], 2));

			// std::cout << totalDistance << std::endl;
		}

		avgDistance = totalDistance / numPoints;
		return avgDistance;
	}
};