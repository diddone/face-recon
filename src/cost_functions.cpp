#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <math.h>

#include <utility>
#include "utils.h"

template <typename T>
void applyExtTransform(const T* extParams, T* inputPoint, T* outputPoint) {
    const T* rotation = extParams;
    const T* translation = rotation + 3;
    const T* scale = rotation + 6;

    T temp[3];
    ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

    outputPoint[0] = scale[0] * temp[0] + translation[0];
    outputPoint[1] = scale[0] * temp[1] + translation[1];
    outputPoint[2] = scale[0] * temp[2] + translation[2];
}

struct SparseCostFunction
{
	SparseCostFunction(
        const std::shared_ptr<const BfmManager>& _pBfmManager, const Matrix3d& _cameraMatrix,
        size_t landmarkInd, Eigen::Vector2i  landmarkUV, double weight
    ):
	pBfmManager(_pBfmManager), cameraMatrix(_cameraMatrix), landmarkInd(landmarkInd), landmarkUV(std::move(landmarkUV)), weight(weight){}


	template<typename T>
	bool operator()(const T* const pose, const T* const shapeCoef, const T* const exprCoef, T* residual) const
	{
        T vXYZ[3] = {
			T(pBfmManager->m_vecLandmarkShapeMu(3 * landmarkInd) + pBfmManager->m_vecLandmarkExprMu(3 * landmarkInd)),
			T(pBfmManager->m_vecLandmarkShapeMu(3 * landmarkInd + 1) + pBfmManager->m_vecLandmarkExprMu(3 * landmarkInd + 1)),
			T(pBfmManager->m_vecLandmarkShapeMu(3 * landmarkInd + 2) + pBfmManager->m_vecLandmarkExprMu(3 * landmarkInd + 2))
		};

		for(size_t i = 0; i < pBfmManager->m_nIdPcs; ++i) {
			vXYZ[0] += pBfmManager->m_matLandmarkShapePc(3 * landmarkInd, i) * shapeCoef[i];
			vXYZ[1] += pBfmManager->m_matLandmarkShapePc(3 * landmarkInd + 1, i) * shapeCoef[i];
			vXYZ[2] += pBfmManager->m_matLandmarkShapePc(3 * landmarkInd + 2, i) * shapeCoef[i];
		}

		for(size_t i = 0; i < pBfmManager->m_nExprPcs; ++i) {
			vXYZ[0] += pBfmManager->m_matLandmarkExprPc(3 * landmarkInd, i) * exprCoef[i];
			vXYZ[1] += pBfmManager->m_matLandmarkExprPc(3 * landmarkInd + 1, i) * exprCoef[i];
			vXYZ[2] += pBfmManager->m_matLandmarkExprPc(3 * landmarkInd + 2, i) * exprCoef[i];
		}

		T transformed[3], projected[3];
		applyExtTransform(pose, vXYZ, transformed);

		for (size_t i = 0; i < 3; ++i) {
			projected[i] = T(cameraMatrix(i, 0)) * transformed[0] + T(cameraMatrix(i, 1)) * transformed[1] + T(cameraMatrix(i, 2)) * transformed[2];
		}

		residual[0] = T(sqrt(weight)) * (T(0.5) + T(landmarkUV[0]) - projected[0] / projected[2]);
		residual[1] = T(sqrt(weight)) * (T(0.5) + T(landmarkUV[1]) - projected[1] / projected[2]);
		return true;
	}

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager>& _pBfmManager, const Matrix3d& _cameraMatrix, size_t landmarkInd, const Eigen::Vector2i& landmarkUV, double weight=1.0) {
        return new ceres::AutoDiffCostFunction<SparseCostFunction, 2, 7, N_SHAPE_PARAMS, N_EXPR_PARAMS>(
            new SparseCostFunction(_pBfmManager, _cameraMatrix, landmarkInd, landmarkUV, weight)
        );
    }
	private:
		const std::shared_ptr<const BfmManager> pBfmManager;
		const Matrix3d& cameraMatrix;
		const size_t landmarkInd;
		const Eigen::Vector2i landmarkUV;
        const double weight;
};

struct PriorShapeCostFunction {

    PriorShapeCostFunction(size_t nIdPcs, double weight): nIdPcs(nIdPcs), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const shapeCoefs, T* residual) const  {
        for(size_t i = 0; i < nIdPcs; ++i) {
            residual[i] = T(sqrt(weight)) * shapeCoefs[i];
        }

        return true;
    }

    static ceres::CostFunction* create(size_t nIdPcs, double weight) {
        return new ceres::AutoDiffCostFunction<PriorShapeCostFunction, N_SHAPE_PARAMS, N_SHAPE_PARAMS>(
            new PriorShapeCostFunction(nIdPcs, weight)
        );
    }

    private:
        const size_t nIdPcs;
        const double weight;
};

struct PriorTexCostFunction {

    PriorTexCostFunction(size_t nIdPcs, double weight): nIdPcs(nIdPcs), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const texCoefs, T* residual) const  {
        for(size_t i = 0; i < nIdPcs; ++i) {
            residual[i] = T(sqrt(weight)) * texCoefs[i];
        }

        return true;
    }

    static ceres::CostFunction* create(size_t nIdPcs, double weight) {
        return new ceres::AutoDiffCostFunction<PriorTexCostFunction, N_SHAPE_PARAMS, N_SHAPE_PARAMS>(
            new PriorTexCostFunction(nIdPcs, weight)
        );
    }

    private:
        const size_t nIdPcs;
        const double weight;
};

struct PriorExprCostFunction {

    PriorExprCostFunction(size_t nExprPcs, double weight): nExprPcs(nExprPcs), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const exprCoefs, T* residual) const  {
        for(size_t i = 0; i < nExprPcs; ++i) {
            residual[i] = T(sqrt(weight)) * exprCoefs[i];
        }
        return true;
    }

    static ceres::CostFunction* create(const size_t nExprPcs, double weight) {
        return new ceres::AutoDiffCostFunction<PriorExprCostFunction, N_EXPR_PARAMS, N_EXPR_PARAMS>(
            new PriorExprCostFunction(nExprPcs, weight)
        );
    }

    private:
        const size_t nExprPcs;
        const double weight;
};


struct ColorCostFunction {
    ColorCostFunction(const std::shared_ptr<const BfmManager>& _pBfmManager, const ImageUtilityThing& _imageUtility, size_t vertexId, double weight):
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
            residual[k] = T(sqrt(weight)) * (color - T(true_color[k]));
        }

        return true;
    }

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager>& _pBfmManager, const ImageUtilityThing& _imageUtility, size_t vertexId, double weight) {

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
