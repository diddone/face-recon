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

template<typename SCALAR, int N>
int get_integer_part( const ceres::Jet<SCALAR, N>& x ) {
    return static_cast<int>(x.a);
}

int get_integer_part( double x ){
    return static_cast<int>(x);
}

struct DepthP2PCostFunction {
    DepthP2PCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager, const ImageUtilityThing& _imageUtility, const size_t vertexInd, double weight):
    pBfmManager(_pBfmManager), imageUtility(_imageUtility), vertexInd(vertexInd), weight(weight) {}


template<typename T>
 bool operator()(const T* const pose, const T* const shapeCoefs, const T* const exprCoefs, T* residual) const  {
        T vXYZ[3] = {
            T(pBfmManager->m_vecShapeMu(3 * vertexInd) + pBfmManager->m_vecExprMu(3 * vertexInd)),
            T(pBfmManager->m_vecShapeMu(3 * vertexInd + 1) + pBfmManager->m_vecExprMu(3 * vertexInd + 1)),
            T(pBfmManager->m_vecShapeMu(3 * vertexInd + 2) + pBfmManager->m_vecExprMu(3 * vertexInd + 2))
        };

        for(size_t i = 0; i < pBfmManager->m_nIdPcs; ++i) {
            vXYZ[0] += pBfmManager->m_matShapePc(3 * vertexInd, i) * shapeCoefs[i];
            vXYZ[1] += pBfmManager->m_matShapePc(3 * vertexInd + 1, i) * shapeCoefs[i];
            vXYZ[2] += pBfmManager->m_matShapePc(3 * vertexInd + 2, i) * shapeCoefs[i];
        }

        for(size_t i = 0; i < pBfmManager->m_nExprPcs; ++i) {
            vXYZ[0] += pBfmManager->m_matExprPc(3 * vertexInd, i) * exprCoefs[i];
            vXYZ[1] += pBfmManager->m_matExprPc(3 * vertexInd + 1, i) * exprCoefs[i];
            vXYZ[2] += pBfmManager->m_matExprPc(3 * vertexInd + 2, i) * exprCoefs[i];
        }

        T transformed[3], projected[3], backProjected[3];
        applyExtTransform(pose, vXYZ, transformed);

        auto cameraMatrix = imageUtility.camera_matrix;
        for (size_t i = 0; i < 3; ++i) {
            projected[i] = T(cameraMatrix(i, 0)) * transformed[0] + T(cameraMatrix(i, 1)) * transformed[1] + T(cameraMatrix(i, 2)) * transformed[2];
        }
        projected[0] = projected[0] / projected[2];
        projected[1] = projected[1] / projected[2];

        int uImage, vImage;
        // here we get value of ceres::Jet
        uImage = get_integer_part(projected[0]);
        vImage = get_integer_part(projected[1]);

        double depth_value = imageUtility.UVtoDepth(uImage, vImage);
        if (std::isnan(depth_value)) {
            residual[0] = T(0);
        } else {
            residual[0] =  T(sqrt(weight)) * (T(depth_value) - projected[2]);
        }
        // multiply by weights and divide by number of valid samples
        // multiply by real depth
        // projected[0] *= depth;
        // projected[1] *= depth;
        // auto invCameraMatrix = imageUtility.inv_camera_matrix;
        // for (size_t i = 0; i < 3; ++i) {
  //  backProjected[i] = T(invCameraMatrix(i, 0)) * projected[0] + T(invCameraMatrix(i, 1)) * projected[1] + T(invCameraMatrix(i, 2)) * projected[2];
  // }

        // // compare backProjected with initially trasnformed values
        // residual[0] = T(sqrt(weight)) * (transformed[0] - backProjected[0]);
        // residual[1] = T(sqrt(weight)) * (transformed[1] - backProjected[1]);
        // residual[2] = T(sqrt(weight)) * (transformed[2] - backProjected[2]);
        return true;
    }

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager> _pBfmManager, const ImageUtilityThing& _imageUtility, const size_t vertexInd, double weight) {
        return new ceres::AutoDiffCostFunction<DepthP2PCostFunction, 1, 7, N_SHAPE_PARAMS, N_EXPR_PARAMS>(
            new DepthP2PCostFunction(_pBfmManager, _imageUtility, vertexInd, weight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const ImageUtilityThing& imageUtility;
        const size_t vertexInd;
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
            // transform current blendshape
            xyz = (pBfmManager->m_dScale * pBfmManager->m_matR) * xyz + pBfmManager->m_vecT;
            Eigen::Vector2d uv = imageUtility.XYZtoUV(xyz);

            // we are using centers of the pixels
            int i = std::floor(uv[0]);
            int j= std::floor(uv[1]);

            double c_i = i + 0.5;
            double c_j = j + 0.5;

            double w1 = (c_i + 1 - uv[0]) * (c_j + 1 - uv[1]);
            double w2 = (uv[0] - c_i) * (c_j + 1 - uv[1]);
            double w3 = (c_i + 1 - uv[0]) * (uv[1] - c_j);
            double w4 = (uv[0] - c_i) * (uv[1] - c_j);

            // std::cout << w1 << " " << imageUtility.UVtoColor(i, j) << std::endl;
            // std::cout << w2 << " " << imageUtility.UVtoColor(i + 1, j) << std::endl;
            // std::cout << w3 << " " << imageUtility.UVtoColor(i, j + 1) << std::endl;
            // std::cout << w4 << " " << imageUtility.UVtoColor(i + 1, j + 1) << std::endl;

            trueColor =
            (
                w1 * imageUtility.UVtoColor(i, j) +
                w2 * imageUtility.UVtoColor(i + 1, j) +
                w3 * imageUtility.UVtoColor(i, j + 1) +
                w4 * imageUtility.UVtoColor(i + 1, j + 1)
            );

            auto l = {
                16214, 16229, 16248, 16270, 16295,
                25899, 26351, 26776, 27064
            };
            if (std::find(l.begin(), l.end(), vertexId) != l.end()) {
                std::cout << "ID i j " << vertexId << " " << i << " " << j << std::endl;
                std::cout << "true color " << trueColor << "\n-------------\n";
            }
        }

    // assuming there is no transformation
    template<typename T>
	bool operator()(const T* const texCoefs, T* residual) const  {
        // iterating over xyz

        for(size_t k = 0; k < 3; ++k) {
            T color(pBfmManager->m_vecTexMu[3 * vertexId + k]);

            for (size_t i = 0; i < pBfmManager->m_nIdPcs; ++i) {
                color += T(pBfmManager->m_matTexPc(3 * vertexId + k, i)) * texCoefs[i];
            }
            residual[k] = T(sqrt(weight)) * (color - T(trueColor[k]));
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
        Vector3d trueColor;
};
