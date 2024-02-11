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

    // we are using log parameterisation for the scale
    outputPoint[0] = ceres::exp(scale[0]) * temp[0] + translation[0];
    outputPoint[1] = ceres::exp(scale[0]) * temp[1] + translation[1];
    outputPoint[2] = ceres::exp(scale[0]) * temp[2] + translation[2];
}

struct SparseCostFunction
{
	SparseCostFunction(
        const std::shared_ptr<const BfmManager>& _pBfmManager, const Matrix3d& _cameraMatrix,
        size_t _landmarkInd, Eigen::Vector2i _landmarkUV, double _weight
    ):
	pBfmManager(_pBfmManager), cameraMatrix(_cameraMatrix), landmarkInd(_landmarkInd), landmarkUV(std::move(_landmarkUV)), weight(_weight){}


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

template<typename SCALAR, int N>
double get_double_part( const ceres::Jet<SCALAR, N>& x ) {
    return static_cast<double>(x.a);
}

double get_double_part( double x ){
    return static_cast<double>(x);
}


struct DepthP2PCostFunction {
    DepthP2PCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager, const std::shared_ptr<const ImageUtilityThing> _pImageUtility, const size_t _vertexInd, double _weight):
    pBfmManager(_pBfmManager), pImageUtility(_pImageUtility), vertexInd(_vertexInd), weight(_weight) {}


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

        auto cameraMatrix = pImageUtility->getIntMat();
        for (size_t i = 0; i < 3; ++i) {
            projected[i] = T(cameraMatrix(i, 0)) * transformed[0] + T(cameraMatrix(i, 1)) * transformed[1] + T(cameraMatrix(i, 2)) * transformed[2];
        }
        projected[0] = projected[0] / projected[2];
        projected[1] = projected[1] / projected[2];

        int uImage, vImage;
        // here we get value of ceres::Jet
        uImage = get_integer_part(projected[0]);
        vImage = get_integer_part(projected[1]);

        double depth_value = get_double_part(projected[2]);
        double gt_depth = pImageUtility->UVtoDepth(uImage, vImage);
        if (std::isnan(gt_depth) || std::fabs(gt_depth - depth_value) > 3e-1) {
            residual[0] = T(0.);
        } else {
            residual[0] =  T(sqrt(weight)) * (T(gt_depth) - projected[2]);
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

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager> _pBfmManager, const std::shared_ptr<const ImageUtilityThing> _pImageUtility, const size_t vertexInd, double weight) {
        return new ceres::AutoDiffCostFunction<DepthP2PCostFunction, 1, 7, N_SHAPE_PARAMS, N_EXPR_PARAMS>(
            new DepthP2PCostFunction(_pBfmManager, _pImageUtility, vertexInd, weight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const std::shared_ptr<const ImageUtilityThing> pImageUtility;
        const size_t vertexInd;
        const double weight;
};

struct DepthP2PlaneCostFunction {
    DepthP2PlaneCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager, const std::shared_ptr<const ImageUtilityThing> _pImageUtility, const size_t _vertexInd, double _p2PointWeight, double _p2PlaneWeight):
    pBfmManager(_pBfmManager), pImageUtility(_pImageUtility), vertexInd(_vertexInd), p2PointWeight(_p2PointWeight), p2PlaneWeight(_p2PlaneWeight), sourceNormal(pBfmManager->m_vecNormals.segment(3 * vertexInd, 3).array()) {}

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

        auto cameraMatrix = pImageUtility->getIntMat();
        for (size_t i = 0; i < 3; ++i) {
            projected[i] = T(cameraMatrix(i, 0)) * transformed[0] + T(cameraMatrix(i, 1)) * transformed[1] + T(cameraMatrix(i, 2)) * transformed[2];
        }
        projected[0] = projected[0] / projected[2];
        projected[1] = projected[1] / projected[2];

        int uImage, vImage;
        // here we get value of ceres::Jet
        uImage = get_integer_part(projected[0]);
        vImage = get_integer_part(projected[1]);

        Vector3d target = pImageUtility->UVtoXYZ(uImage, vImage);
        Vector3d targetNormal = pImageUtility->UVtoNormal(uImage, vImage);

        double depth_value = get_double_part(projected[2]);
        double gt_depth = target[2];
        // invalidate if has nans or angle between normals > 60 degrees
        if (target.hasNaN() || targetNormal.hasNaN() || targetNormal.dot(sourceNormal) < 0.5 || std::fabs(gt_depth - depth_value) > 3e-1) {
            residual[0] = T(0);
            residual[1] = T(0);
        } else {
            // point to point term
            residual[0] =  T(sqrt(p2PointWeight)) * (T(gt_depth) - projected[2]);
            residual[1] = T(0);
            // dot product between (target - transformed) and sourceNormal
            for (size_t k = 0; k < 3; ++k) {
              residual[1] += (T(target[k]) - transformed[k]) * T(sourceNormal[k]);
            }

            residual[1] *= T(sqrt(p2PlaneWeight));
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

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager> _pBfmManager, const std::shared_ptr<const ImageUtilityThing> _pImageUtility, const size_t vertexInd, double p2PointWeight, double p2PlaneWeight) {
        return new ceres::AutoDiffCostFunction<DepthP2PlaneCostFunction, 2, 7, N_SHAPE_PARAMS, N_EXPR_PARAMS>(
            new DepthP2PlaneCostFunction(_pBfmManager, _pImageUtility, vertexInd, p2PointWeight, p2PlaneWeight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const std::shared_ptr<const ImageUtilityThing> pImageUtility;
        const size_t vertexInd;
        const double p2PointWeight;
        const double p2PlaneWeight;
        const Vector3d sourceNormal;
};


struct PriorShapeCostFunction {

    PriorShapeCostFunction(size_t nIdPcs, const Eigen::VectorXd& sigmasRef, double weight): nIdPcs(nIdPcs), sigmasRef(sigmasRef), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const shapeCoefs, T* residual) const  {
        for(size_t i = 0; i < nIdPcs; ++i) {
            residual[i] = T(sqrt(weight / sigmasRef[i])) * shapeCoefs[i];
        }

        return true;
    }

    static ceres::CostFunction* create(size_t nIdPcs, const Eigen::VectorXd& sigmasRef, double weight) {
        return new ceres::AutoDiffCostFunction<PriorShapeCostFunction, N_SHAPE_PARAMS, N_SHAPE_PARAMS>(
            new PriorShapeCostFunction(nIdPcs, sigmasRef, weight)
        );
    }

    private:
        const size_t nIdPcs;
        const double weight;
        const Eigen::VectorXd& sigmasRef;
};

struct PriorTexCostFunction {

    PriorTexCostFunction(size_t nIdPcs, const Eigen::VectorXd& sigmasRef, double weight): nIdPcs(nIdPcs), sigmasRef(sigmasRef), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const texCoefs, T* residual) const  {
        for(size_t i = 0; i < nIdPcs; ++i) {
            residual[i] = T(sqrt(weight / sigmasRef[i])) * texCoefs[i];
        }

        return true;
    }

    static ceres::CostFunction* create(size_t nIdPcs, const Eigen::VectorXd& sigmasRef, double weight) {
        return new ceres::AutoDiffCostFunction<PriorTexCostFunction, N_SHAPE_PARAMS, N_SHAPE_PARAMS>(
            new PriorTexCostFunction(nIdPcs, sigmasRef, weight)
        );
    }

    private:
        const size_t nIdPcs;
        const double weight;
        const Eigen::VectorXd& sigmasRef;
};

struct PriorExprCostFunction {

    PriorExprCostFunction(size_t nExprPcs, const Eigen::VectorXd& sigmasRef, double weight): nExprPcs(nExprPcs), sigmasRef(sigmasRef), weight(weight)
    {}

    template<typename T>
	bool operator()(const T* const exprCoefs, T* residual) const  {
        for(size_t i = 0; i < nExprPcs; ++i) {
            residual[i] = T(sqrt(weight / sigmasRef[i])) * exprCoefs[i];
        }
        return true;
    }

    static ceres::CostFunction* create(const size_t nExprPcs, const Eigen::VectorXd& sigmasRef, double weight) {
        return new ceres::AutoDiffCostFunction<PriorExprCostFunction, N_EXPR_PARAMS, N_EXPR_PARAMS>(
            new PriorExprCostFunction(nExprPcs, sigmasRef, weight)
        );
    }

    private:
        const size_t nExprPcs;
        const double weight;
        const Eigen::VectorXd& sigmasRef;
};

Eigen::Vector3d projectVertexIntoMesh(const std::shared_ptr<const BfmManager>& pBfmManager, const std::shared_ptr<const ImageRGBOnly>& pImageUtility, size_t vertexInd) {
    Vector3d xyz(
    pBfmManager->m_vecCurrentBlendshape[3 * vertexInd],
    pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 1],
    pBfmManager->m_vecCurrentBlendshape[3 * vertexInd + 2]
    );
    // transform current blendshape
    xyz = (pBfmManager->m_dScale * pBfmManager->m_matR) * xyz + pBfmManager->m_vecT;
    Eigen::Vector2d uv = pImageUtility->XYZtoUV(xyz);

    // we are using centers of the pixels
    int i = std::floor(uv[0]);
    int j= std::floor(uv[1]);

    double c_i = i;
    double c_j = j;

    double w1 = (c_i + 1 - uv[0]) * (c_j + 1 - uv[1]);
    double w2 = (uv[0] - c_i) * (c_j + 1 - uv[1]);
    double w3 = (c_i + 1 - uv[0]) * (uv[1] - c_j);
    double w4 = (uv[0] - c_i) * (uv[1] - c_j);

    std::cout << w1 << " " << pImageUtility->UVtoColor(i, j) << std::endl;
    std::cout << w2 << " " << pImageUtility->UVtoColor(i + 1, j) << std::endl;
    std::cout << w3 << " " << pImageUtility->UVtoColor(i, j + 1) << std::endl;
    std::cout << w4 << " " << pImageUtility->UVtoColor(i + 1, j + 1) << std::endl;

    Eigen::Vector3d trueColor =
    (
        w1 * pImageUtility->UVtoColor(i, j) +
        w2 * pImageUtility->UVtoColor(i + 1, j) +
        w3 * pImageUtility->UVtoColor(i, j + 1) +
        w4 * pImageUtility->UVtoColor(i + 1, j + 1)
    );

    return trueColor;
}

struct ColorCostFunction {
    ColorCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager, const std::shared_ptr<const ImageRGBOnly> _pImageUtility, size_t _vertexId, double _weight):
        pBfmManager(_pBfmManager), pImageUtility(_pImageUtility), vertexId(_vertexId), weight(_weight) {
            trueColor = projectVertexIntoMesh(pBfmManager, pImageUtility, vertexId);
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

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager>& _pBfmManager, const std::shared_ptr<const ImageRGBOnly>  _pImageUtility, size_t vertexId, double weight) {

        return new ceres::AutoDiffCostFunction<ColorCostFunction, 3, N_SHAPE_PARAMS>(
            new ColorCostFunction(_pBfmManager, _pImageUtility, vertexId, weight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const std::shared_ptr<const ImageRGBOnly> pImageUtility;
        size_t vertexId;
        const double weight;
        // target values of color
        Vector3d trueColor;
};


void setCurrentTexAsImage(const std::shared_ptr<BfmManager>& pBfmManager, const std::shared_ptr<const ImageRGBOnly> pImageUtility) {
    for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {
        Vector3d trueColor = projectVertexIntoMesh(pBfmManager, pImageUtility, vertexInd);
        pBfmManager->m_vecCurrentTex[3 * vertexInd] = trueColor[0];
        pBfmManager->m_vecCurrentTex[3 * vertexInd + 1] = trueColor[1];
        pBfmManager->m_vecCurrentTex[3 * vertexInd + 2] = trueColor[2];
    }
}


VectorXd EvaluateSH(const Vector3d& dir) {
    VectorXd shBasis(9);
    double c0 = 0.5 * sqrt(M_1_PI);
    double c1 = 0.5 * sqrt(double(3.) * M_1_PI);

    shBasis[0] = c0;

    shBasis[1] = c1 * dir.y();
    shBasis[2] = c1 * dir.z();
    shBasis[3] = c1 * dir.x();

    double c2 = double(0.5) * sqrt(double(15.) * M_1_PI);
    shBasis[4] = c2 * dir.x() * dir.y();
    shBasis[5] = c2 * dir.y() * dir.z();
    shBasis[6] = 0.25 * sqrt(double(5.) * M_1_PI) * (3. * dir.z() * dir.z() - 1);
    shBasis[7] = c2 * dir.x() * dir.z();
    shBasis[8] = 0.25 * sqrt(double(15.) * M_1_PI) * (dir.x() * dir.x() - dir.y() * dir.y());

    return shBasis;
}

struct ColorWithLightningCostFunction {
    ColorWithLightningCostFunction(const std::shared_ptr<const BfmManager> _pBfmManager,  const std::shared_ptr<const ImageRGBOnly> _pImageUtility, size_t vertexId, double weight):
        pBfmManager(_pBfmManager), pImageUtility(_pImageUtility), vertexId(vertexId), weight(weight) {
            trueColor = projectVertexIntoMesh(pBfmManager, pImageUtility, vertexId);
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

    static ceres::CostFunction* create(const std::shared_ptr<const BfmManager>& _pBfmManager, const std::shared_ptr<const ImageRGBOnly> _pImageUtility, size_t vertexId, double weight) {

        return new ceres::AutoDiffCostFunction<ColorWithLightningCostFunction, 3, N_SHAPE_PARAMS>(
            new ColorWithLightningCostFunction(_pBfmManager, _pImageUtility, vertexId, weight)
        );
    }

    private:
        const std::shared_ptr<const BfmManager> pBfmManager;
        const std::shared_ptr<const ImageRGBOnly> pImageUtility;
        size_t vertexId;
        const double weight;
        // target values of color
        Vector3d trueColor;
        VectorXd shBasis;
};


struct ResultsDepthCostFunc {
  double pure_p2p;
  double p2p;
  double p2plane;
};

std::ostream& operator<<(std::ostream& os, const ResultsDepthCostFunc& result) {
    os << "Pure Point-to-Point: " << result.pure_p2p << "\n";
    os << "Point-to-Point: " << result.p2p << "\n";
    os << "Point-to-Plane: " << result.p2plane << std::endl;
    return os;
}

double computeSparseCostFunction(const std::shared_ptr<const BfmManager> pBfmManager, const std::shared_ptr<const ImageUtilityThing> pImageUtility) {
  double sparseCost = 0.;
  Eigen::VectorXi imageLandmarks = pImageUtility->getUVLandmarks();
  size_t numberOfLandmarks = pBfmManager->m_mapLandmarkIndices.size();

  for (size_t iLandmark = 0; iLandmark < numberOfLandmarks; ++iLandmark) {
      Eigen::Vector2i landmark(imageLandmarks[2 * iLandmark], imageLandmarks[2 * iLandmark + 1]);
      auto costFunc = SparseCostFunction{pBfmManager, pImageUtility->getIntMat(), iLandmark, landmark, 1.0};

      double residual[2];
      costFunc(pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef, residual);
      sparseCost += ((residual[0] * residual[0]) + (residual[1] * residual[1])) / numberOfLandmarks;
  }

  return sparseCost;
}

ResultsDepthCostFunc computeDepthCostFunction(const std::shared_ptr<const BfmManager> pBfmManager, const std::shared_ptr<const ImageUtilityThing> pImageUtility) {
  double pure_p2p = 0.;
  double p2p = 0.;
  double p2plane = 0.;

  // for (size_t t = 28;  t < 32; ++t) {
  //   size_t vertexInd = pBfmManager->m_mapLandmarkIndices[t];
  //   auto uvVec = pImageUtility->getUVLandmarks();
  //   auto landmarkVec = pImageUtility->UVtoXYZ(uvVec[2 * t], uvVec[2 * t + 1]);
  //   std::cout << "Lanmrk vec " << landmarkVec[0] << " " << landmarkVec[1] << " " << landmarkVec[2] << std::endl;

  //   auto p2pFunc = DepthP2PCostFunction{pBfmManager, pImageUtility, vertexInd, 1.0};
  //   auto p2PlaneFunc = DepthP2PlaneCostFunction{pBfmManager, pImageUtility, vertexInd, 1.0, 1.0};

  //   double p2pRes[1];
  //   double p2PlaneRes[2];
  //   p2pFunc(pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef, p2pRes);
  //   p2PlaneFunc(pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef, p2PlaneRes);

  //   pure_p2p += (p2pRes[0] * p2pRes[0]) / 4;
  //   p2p += (p2PlaneRes[0] * p2PlaneRes[0]) / 4;
  //   p2plane += (p2PlaneRes[1] * p2PlaneRes[1]) / 4;
  // }
  for (size_t vertexInd = 0; vertexInd < pBfmManager->m_nVertices; ++vertexInd) {

    auto p2pFunc = DepthP2PCostFunction{pBfmManager, pImageUtility, vertexInd, 1.0};
    auto p2PlaneFunc = DepthP2PlaneCostFunction{pBfmManager, pImageUtility, vertexInd, 1.0, 1.0};

    double p2pRes[1];
    double p2PlaneRes[2];
    p2pFunc(pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef, p2pRes);
    p2PlaneFunc(pBfmManager->m_aExtParams.data(), pBfmManager->m_aShapeCoef, pBfmManager->m_aExprCoef, p2PlaneRes);

    pure_p2p += (p2pRes[0] * p2pRes[0]) / pBfmManager->m_nVertices;
    p2p += (p2PlaneRes[0] * p2PlaneRes[0]) / pBfmManager->m_nVertices;
    p2plane += (p2PlaneRes[1] * p2PlaneRes[1]) / pBfmManager->m_nVertices;
  }

  return {pure_p2p, p2p, p2plane};
}
