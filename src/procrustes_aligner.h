#pragma once

#include "utils.h"
#include "eigen.h"
using namespace Eigen;

//TODO rewrite procrusters aligner
class ProcrustesAligner {
public:
	Matrix4f estimatePose(const VectorXd& sourcePoints, const VectorXd& targetPoints) {
		// ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");

		// // We estimate the pose between source and target points using Procrustes algorithm.
		// // Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// // from source points to target points.

		// auto sourceMean = computeMean(sourcePoints);
		// auto targetMean = computeMean(targetPoints);

		// Matrix3d rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		// Vector3d translation = computeTranslation(sourceMean, targetMean, rotation);

		// // TODO: Compute the transformation matrix by using the computed rotation and translation.
		// // You can access parts of the matrix with .block(start_row, start_col, num_rows, num_cols) = elements

		// Matrix4f estimatedPose = Matrix4f::Identity();
		// estimatedPose.block(0, 0, 3, 3) = rotation;
		// estimatedPose.block(0, 3, 3, 1) = translation;
		// return estimatedPose;
	}

private:
	Vector3d computeMean(const VectorXd& points) {
		// TODO: Compute the mean of input points.
		// Hint: You can use the .size() method to get the length of a vector.

		// Vector3d mean = Vector3d::Zero();
		// auto size = points.size();
		// for (auto& p : points) {
		// 	mean += p / size;
		// }
		// return mean;
	}

	Matrix3d estimateRotation(const std::vector<Vector3d>& sourcePoints, const Vector3d& sourceMean, const std::vector<Vector3d>& targetPoints, const Vector3d& targetMean) {
		// TODO: Estimate the rotation from source to target points, following the Procrustes algorithm.
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		// Hint: You can initialize an Eigen matrix with "MatrixXf m(num_rows,num_cols);" and access/modify parts of it using the .block() method (see above).

		// MatrixXd x_cor(3, 3);
		// for (uint i = 0; i < sourcePoints.size(); ++i) {
		// 	x_cor += targetPoints[i] * sourcePoints[i].transpose();
		// }

		// Eigen::JacobiSVD<MatrixXd> svd(x_cor, ComputeThinV | ComputeThinU);
		// Matrix3d rotation = Matrix3d::Identity();

		// auto U = svd.matrixU();
		// auto V = svd.matrixV();
		// if ((U * V).determinant() < 0) {
		// 	rotation(2, 2) = -1.;
		// }

		// rotation = U * rotation * V.transpose();
        // return rotation;
	}

	Vector3d computeTranslation(const Vector3d& sourceMean, const Vector3d& targetMean, const Matrix3d& rotation) {
		// TODO: Compute the translation vector from source to target points.

		// Vector3d translation = Vector3d::Zero();
		// translation = -rotation * sourceMean + targetMean;
        // return translation;
	}
};
