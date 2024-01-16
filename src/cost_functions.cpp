#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <math.h>
#include "bfm_manager.h"

template <typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T* const array) : m_array{ array } { }

    const T* getData() const {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Eigen::Matrix4d convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const double* pose = poseIncrement.getData();
        const double* rotation = pose;
        const double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Eigen::Matrix4d matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

        return matrix;
    }

private:
    const T* m_array;
};

struct SparseCostFunction
{
	SparseCostFunction(const std::unique_ptr<BfmManager>& _pBfmManager, const Matrix3d& _cameraMatrix, size_t vertexInd, const Eigen::Vector2d& landmarkUV):
	pBfmManager(_pBfmManager), cameraMatrix(_cameraMatrix), vertexInd(vertexInd), landmarkUV(landmarkUV) {}


	template<typename T>
	bool operator()(const T* const pose, const T* const shapeCoef, const T* const exprCoef, T* residual) const
	{
		// init coordinates of the vertex with mean
		T vXYZ[3] = {
			pBfmManager->m_vecShapeMu(3 * vertexInd),
			pBfmManager->m_vecShapeMu(3 * vertexInd + 1),
			pBfmManager->m_vecShapeMu(3 * vertexInd + 2)
		};

		for(size_t i = 0; i < pBfmManager->m_nIdPcs; ++i) {
			vXYZ[0] += pBfmManager->m_matShapePc(3 * vertexInd, i) * shapeCoef[i];
			vXYZ[1] += pBfmManager->m_matShapePc(3 * vertexInd + 1, i) * shapeCoef[i];
			vXYZ[2] += pBfmManager->m_matShapePc(3 * vertexInd + 2, i) * shapeCoef[i];
		}

		for(size_t i = 0; i < pBfmManager->m_nExprPcs; ++i) {
			vXYZ[0] += pBfmManager->m_matExprPc(3 * vertexInd, i) * exprCoef[i];
			vXYZ[1] += pBfmManager->m_matExprPc(3 * vertexInd + 1, i) * exprCoef[i];
			vXYZ[2] += pBfmManager->m_matExprPc(3 * vertexInd + 2, i) * exprCoef[i];
		}

		T transformed[3], projected[3];
		pose->increment(vXYZ, transformed);

		for (size_t i = 0; i < 3; ++i) {
			projected[i] = cameraMatrix(i, 0) * transformed[0] + cameraMatrix(i, 1) * transformed[1] + cameraMatrix(i, 2) * transformed[2];
		}

		residual[0] = T(landmarkUV[0]) - projected[0] / projected[2];
		residual[1] = T(landmarkUV[1]) - projected[1] / projected[2];
		return true;
	}

	private:
		const std::unique_ptr<BfmManager>& pBfmManager;
		const Matrix3d& cameraMatrix;
		size_t vertexInd;
		const Eigen::Vector2d& landmarkUV;
};

