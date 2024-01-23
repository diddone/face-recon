#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <math.h>
#include "bfm_manager.h"
#include "utils.h"


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
        // pose[6] is a scale
        const T* rotation = m_array;
        const T* translation = m_array + 3;
        const T* scale = m_array + 6;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = scale[0] * temp[0] + translation[0];
        outputPoint[1] = scale[0] * temp[1] + translation[1];
        outputPoint[2] = scale[0] * temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    // static Eigen::Matrix4d convertToMatrix(const PoseIncrement<double>& poseIncrement) {
    //     // pose[0,1,2] is angle-axis rotation.
    //     // pose[3,4,5] is translation.
    //     // pose[6] is a scale
    //     const double* pose = poseIncrement.getData();
    //     const double* rotation = pose;
    //     const double* translation = pose + 3;

    //     // Convert the rotation from SO3 to matrix notation (with column-major storage).
    //     double rotationMatrix[9];
    //     ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

    //     // Create the 4x4 transformation matrix.
    //     Eigen::Matrix4d matrix;
    //     matrix.setIdentity();
    //     matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
    //     matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
    //     matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

    //     return matrix;
    // }

private:
    const T* m_array;
};

struct SparseCostFunction
{
	SparseCostFunction(
        const std::shared_ptr<const BfmManager> _pBfmManager, const Matrix3d& _cameraMatrix,
        size_t vertexInd, const Eigen::Vector2d& landmarkUV, double weight=1.0
    ):
	pBfmManager(_pBfmManager), cameraMatrix(_cameraMatrix), vertexInd(vertexInd), landmarkUV(landmarkUV), weight(weight){}


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
		pose->apply(vXYZ, transformed);

		for (size_t i = 0; i < 3; ++i) {
			projected[i] = cameraMatrix(i, 0) * transformed[0] + cameraMatrix(i, 1) * transformed[1] + cameraMatrix(i, 2) * transformed[2];
		}

		residual[0] = T(weight) * T(landmarkUV[0]) - projected[0] / projected[2];
		residual[1] = T(weight) * T(landmarkUV[1]) - projected[1] / projected[2];
		return true;
	}

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager> _pBfmManager, const Matrix3d& _cameraMatrix, size_t vertexInd, const Eigen::Vector2d& landmarkUV, double weight) {
        return new ceres::AutoDiffCostFunction<SparseCostFunction, 2, 7, N_SHAPE_PARAMS, N_EXPR_PARAMS>(
            new SparseCostFunction(_pBfmManager, _cameraMatrix, vertexInd, landmarkUV, weight)
        );
    }
	private:
		const std::shared_ptr<const BfmManager> pBfmManager;
		const Matrix3d& cameraMatrix;
		size_t vertexInd;
		const Eigen::Vector2d& landmarkUV;
        const double weight;
};

struct PriorRegCostFunction {

    PriorRegCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager, double weight): pBfmManager(_pBfmManager), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const shapeCoefs, const T* const exprCoefs, const T* const texCoefs, T* residual) const  {
        for(size_t i = 0; i < pBfmManager->m_nIdPcs; ++i) {
            residual[0] += (
                ceres::pow(shapeCoefs[3 * i], 2) +
                ceres::pow(shapeCoefs[3 * i + 1], 2) +
                ceres::pow(shapeCoefs[3 * i + 2], 2)
            ) / pBfmManager->m_vecShapeEv[i];

            residual[0] += (
                ceres::pow(texCoefs[3 * i], 2) +
                ceres::pow(texCoefs[3 * i + 1], 2) +
                ceres::pow(texCoefs[3 * i + 2], 2)
            ) / pBfmManager->m_vecTexEv[i];
        }

        for(size_t i = 0; i < pBfmManager->m_nExprPcs; ++i) {
            residual[0] += (
                ceres::pow(exprCoefs[3 * i], 2) +
                ceres::pow(exprCoefs[3 * i + 1], 2) +
                ceres::pow(exprCoefs[3 * i + 2], 2)
            ) / pBfmManager->m_vecExprEv[i];
        }

        residual[0] *= T(weight);
        return true;
    }

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager> _pBfmManager, double weight) {

        return new ceres::AutoDiffCostFunction<PriorRegCostFunction, 1, N_SHAPE_PARAMS, N_EXPR_PARAMS, N_SHAPE_PARAMS>(
            new PriorRegCostFunction(_pBfmManager, weight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const double weight;
};


struct ColorCostFunction {
    ColorCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager, const ImageUtilityThing& _imageUtility, size_t vertexId, double weight):
        pBfmManager(_pBfmManager), imageUtility(_imageUtility), vertexId(vertexId), weight(weight) {
            Vector3d xyz(
                pBfmManager->m_vecCurrentBlendshape[3 * vertexId],
                pBfmManager->m_vecCurrentBlendshape[3 * vertexId + 1],
                pBfmManager->m_vecCurrentBlendshape[3 * vertexId + 2]
            );

            xyz = (pBfmManager->m_dScale * pBfmManager->m_matR) * xyz + pBfmManager->m_vecT;
            Eigen::Vector2d uv = imageUtility.XYZtoUV(xyz);
            int i = std::floor(uv[0]);
            int j = std::floor(uv[1]);

            double w1 = (i + 1 - uv[0]) * (j + 1 - uv[1]);
            double w2 = (uv[0] - i) * (j + 1 - uv[1]);
            double w3 = (i + 1 - uv[0]) * (uv[1] - j);
            double w4 = (uv[0] - i) * (uv[1] - j);

            true_color(
                w1 * imageUtility.UVtoColor(i, j) +
                w2 * imageUtility.UVtoColor(i + 1, j) +
                w3 * imageUtility.UVtoColor(i, j + 1) +
                w4 * imageUtility.UVtoColor(i + 1, j + 1)
            );
        }

    // assuming there is no transformation
    template<typename T>
	bool operator()(const T* const texCoefs, T* residual) const  {
        // iterating over xyz

        for(size_t k = 0; k < 3; ++k) {
            T color(pBfmManager->m_vecTexMu[3 * vertexId + k]);

            for (size_t i = 0; i < pBfmManager->m_nIdPcs; ++i) {
                color += T(pBfmManager->m_vecTexMu(3 * vertexId + k, i)) * texCoefs[i];
            }
            residual[k] = T(weight) * (color - T(true_color[k]));
        }

        return true;
    }

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager> _pBfmManager, const ImageUtilityThing& _imageUtility, size_t vertexId, double weight) {

        return new ceres::AutoDiffCostFunction<ColorCostFunction, 3, N_SHAPE_PARAMS>(
            new ColorCostFunction(_pBfmManager, _imageUtility, vertexId, weight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const ImageUtilityThing& imageUtility;
        size_t vertexId;
        const double weight;
        // target values of color
        const Vector3d true_color;
};
