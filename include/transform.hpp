// This file is part of BFM Manager (https://github.com/Great-Keith/BFM-tools).


#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP


#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Dynamic;
using Eigen::Ref;


namespace bfm_utils {
	/*
	 * @Function Euler2Mat
	 * 		Transform Euler angle into rotation matrix.
	 * @Usage
	 * 		Matrix3d matR = Euler2Mat(dRoll, dYaw, dPitch, false);
	 * @Parameters
	 * 		yaw: Euler angle (radian);
	 * 		pitch: Euler angle (radian);
	 * 		roll: Euler angle (radian);
	 * 		bIsLinearized: Choose to use linearized Euler angle transform or not. If true, be sure yaw, pitch and roll
	 * 						keep small.
	 * @Return
	* 		Rotation matrix R.
	* 		If linearized:
	* 			R = [[1.0,  -roll, yaw  ],
	* 				 [roll, 1.0,  -pitch],
	* 				 [-yaw, pitch, 1.0  ]]
	* 		Else
	*			R = [[c2*c1, s3*s2*c1-c3*s1, c3*s2*c1+s3*s1],
	*				 [c2*s1, s3*s2*s1+c3*c1, c3*s2*s1-s3*c1],
	*				 [-s2,   s3*c2,          c3*c2         ]];
	*			(c1=cos(roll), s1=sin(roll))
	*			(c2=cos(yaw), s2=sin(yaw))
	*			(c3=cos(pitch), s3=sin(pitch))
	*/

	template<typename _Tp>
	Matrix<_Tp, 3, 3> Euler2Mat(const _Tp &roll, const _Tp &yaw, const _Tp &pitch, bool bIsLinearized = false)
	{
		// Z1Y2X3
		Matrix<_Tp, 3, 3> matR;
		if(bIsLinearized)
		{
			matR << _Tp(1.0), -roll,    yaw,
				    roll,     _Tp(1.0), -pitch,
				    -yaw,     pitch,    _Tp(1.0);
		}
		else
		{
			_Tp c1 = cos(roll), s1 = sin(roll);
			_Tp c2 = cos(yaw), s2 = sin(yaw);
			_Tp c3 = cos(pitch), s3 = sin(pitch);
			matR << c2 * c1, s3 * s2 * c1 - c3 * s1, c3 * s2 * c1 + s3 * s1,
				    c2 * s1, s3 * s2 * s1 + c3 * c1, c3 * s2 * s1 - s3 * c1,
				    -s2,     s3 * c2,                c3 * c2;
		}
		return matR;
	}


	template<typename _Tp, typename _Ep>
	Matrix<_Tp, Dynamic, 1> TransPoints(
		const Matrix<_Tp, 3, 3> &matR,
		const Matrix<_Tp, 3, 1> &vecT,
		const Matrix<_Ep, Dynamic, 1> &vecPoints)
	{
		Matrix<_Tp, 3, 4> matTrans;
		matTrans << matR, vecT;
		Matrix<_Tp, Dynamic, 1> vecPointsTypeTurned = vecPoints.template cast<_Tp>();
		Map<Matrix<_Tp, Dynamic, 3, Eigen::RowMajor>> matPoints(vecPointsTypeTurned.data(), vecPointsTypeTurned.rows() / 3, 3);
		Matrix<_Tp, 4, Dynamic> matPointsTransposed = matPoints.rowwise().homogeneous().transpose();
		// Matrix<_Tp, 3, Dynamic> matPointsTransformed = matTrans * (matPoints.rowwise().homogeneous().transpose()); // Stuck!
		Matrix<_Tp, 3, Dynamic> matPointsTransformed = matTrans * matPointsTransposed;
		return Map<Matrix<_Tp, Dynamic, 1>>(matPointsTransformed.transpose().data(), vecPoints.rows(), 1);
	}


	template<typename _Tp, typename _Ep>
	Matrix<_Tp, Dynamic, 1> TransPoints(const _Tp * const aExtParams, const Matrix<_Ep, Dynamic, 1> &vecPoints, bool bIsLinearized = false)
	{
		Matrix<_Tp, 3, 3> matR;
		Matrix<_Tp, 3, 1> vecT;
		matR = bfm_utils::Euler2Mat(aExtParams[0], aExtParams[1], aExtParams[2], bIsLinearized);
		vecT << aExtParams[3], aExtParams[4], aExtParams[5];
		// vecT << 0.0, 0.0, 0.0;
		return bfm_utils::TransPoints(matR, vecT, vecPoints);
	}


	template<typename _Tp>
	Matrix<_Tp, Dynamic, 1> TransPoints(
		const Matrix<_Tp, 4, 4> &matTrans,
		const Matrix<_Tp, Dynamic, 1> &vecPoints)
	{
		Matrix<_Tp, Dynamic, 1> vecPointsTypeTurned = vecPoints.template cast<_Tp>();
		Map<Matrix<_Tp, Dynamic, 3, Eigen::RowMajor>> matPoints(vecPointsTypeTurned.data(), vecPointsTypeTurned.rows() / 3, 3);
		Matrix<_Tp, 4, Dynamic> matPointsTransposed = matPoints.rowwise().homogeneous().transpose();
		Matrix<_Tp, 3, Dynamic> matPointsTransformed = (matTrans * matPointsTransposed).topRows(3);

		// DEBUG
		// Map<Matrix<_Tp, Dynamic, 1>> tmp(matPointsTransformed.transpose().data(), vecPoints.rows(), 1);
		// std::cout << "QUICK CHECK" << std::endl;
		// std::cout << matTrans << std::endl;
		// std::cout << vecPointsTypeTurned(0) << " " << vecPointsTypeTurned(1) << " " << vecPointsTypeTurned(2) << std::endl;
		// std::cout << matPoints(0, 0) << " " << matPoints(0, 1) << " " << matPoints(0, 2) << std::endl;
		// std::cout << matPointsTransposed(0, 0) << " " << matPointsTransposed(1, 0) << " " << matPointsTransposed(2, 0) << " " << matPointsTransposed(3, 0) << std::endl;
		// std::cout << matPointsTransformed(0, 0) << " " << matPointsTransformed(1, 0) << " " << matPointsTransformed(2, 0) << std::endl;
		// std::cout << tmp(0) << " " << tmp(1) << " " << tmp(2) << std::endl;

		return Map<Matrix<_Tp, Dynamic, 1>>(matPointsTransformed.transpose().data(), vecPoints.rows(), 1);
	}
}

#endif	// TRANFORM_HPP