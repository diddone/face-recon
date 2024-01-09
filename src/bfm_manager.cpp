#include "bfm_manager.h"
#include <cassert>
#include <cstdint>
#include <cmath>

BfmManager::BfmManager(const std::string &strModelPath,
                       const std::string &strLandmarkIdxPath)
    : m_strModelPath(strModelPath), m_strLandmarkIdxPath(strLandmarkIdxPath) {
  if (!fs::exists(strModelPath)) {
    LOG(ERROR) << "Path of Basel Face Model does not exist. Unexpected path:\t"
               << strModelPath;
    m_strModelPath = "";
    return;
  }
  fs::path modelPath(strModelPath);
  std::string strFn, strExt;

  strExt = modelPath.extension().string();

  if (strExt != ".h5") {
    LOG(ERROR) << "Data type must be hdf5. Unexpected tyoe: " << strExt;
  } else {
    strFn = modelPath.stem().string();
    if (strFn == "model2019_bfm") {
      m_strVersion = "2019";
      m_nVertices = 47439, m_nFaces = 94464, m_nIdPcs = 199, m_nExprPcs = 100,
      m_strShapeMuH5Path = R"(shape/model/mean)";
      m_strShapeEvH5Path = R"(shape/model/pcaVariance)";
      m_strShapePcH5Path = R"(shape/model/pcaBasis)";
      m_strTexMuH5Path = R"(color/model/mean)";
      m_strTexEvH5Path = R"(color/model/pcaVariance)";
      m_strTexPcH5Path = R"(color/model/pcaBasis)";
      m_strExprMuH5Path = R"(expression/model/mean)";
      m_strExprEvH5Path = R"(expression/model/pcaVariance)";
      m_strExprPcH5Path = R"(expression/model/pcaBasis)";
      m_strTriangleListH5Path = R"(shape/representer/cells)";
    } else if (strFn == "model2017-1_face12_nomouth") {
      m_strVersion = "2017-face12";
      m_nVertices = 28588, m_nFaces = 56572, m_nIdPcs = 199, m_nExprPcs = 100,
      m_strShapeMuH5Path = R"(shape/model/mean)";
      m_strShapeEvH5Path = R"(shape/model/pcaVariance)";
      m_strShapePcH5Path = R"(shape/model/pcaBasis)";
      m_strTexMuH5Path = R"(color/model/mean)";
      m_strTexEvH5Path = R"(color/model/pcaVariance)";
      m_strTexPcH5Path = R"(color/model/pcaBasis)";
      m_strExprMuH5Path = R"(expression/model/mean)";
      m_strExprEvH5Path = R"(expression/model/pcaVariance)";
      m_strExprPcH5Path = R"(expression/model/pcaBasis)";
      m_strTriangleListH5Path = R"(shape/representer/cells)";
    } else {
      LOG(ERROR) << "Unknown model " << strFn << "\n";
    }
  }
  m_bUseLandmark = strLandmarkIdxPath == "" ? false : true;
  if (m_bUseLandmark) {
    std::ifstream inFile;
    inFile.open(strLandmarkIdxPath, std::ios::in);
    assert(inFile.is_open());
    int dlibIdx, bfmIdx;
    while (inFile >> bfmIdx) {
      // dlibIdx--;
      m_mapLandmarkIndices.push_back(
          std::move(bfmIdx));
    }
    inFile.close();
  }

  this->alloc();
  this->load();
  this->extractLandmarks();

  unsigned int iTex = 0;
  while (m_bIsTexStd && iTex < m_vecTexMu.size()) {
    if (m_vecTexMu(iTex++) > 1.0)
      m_bIsTexStd = false;
  }

  this->genAvgFace();
  this->genLandmarkBlendshape();
}

void BfmManager::alloc() {
  LOG(INFO) << "Allocate memory for model.";

  m_aShapeCoef = new double[m_nIdPcs];
  std::fill(m_aShapeCoef, m_aShapeCoef + m_nIdPcs, 0.0);
  m_vecShapeMu.resize(m_nVertices * 3);
  m_vecShapeEv.resize(m_nIdPcs);
  m_matShapePc.resize(m_nVertices * 3, m_nIdPcs);

  m_aTexCoef = new double[m_nIdPcs];
  std::fill(m_aTexCoef, m_aTexCoef + m_nIdPcs, 0.0);
  m_vecTexMu.resize(m_nVertices * 3);
  m_vecTexEv.resize(m_nIdPcs);
  m_matTexPc.resize(m_nVertices * 3, m_nIdPcs);

  m_aExprCoef = new double[m_nExprPcs];
  std::fill(m_aExprCoef, m_aExprCoef + m_nExprPcs, 0.0);
  m_vecExprMu.resize(m_nVertices * 3);
  m_vecExprEv.resize(m_nExprPcs);
  m_matExprPc.resize(m_nVertices * 3, m_nExprPcs);

  m_vecTriangleList.resize(m_nFaces * 3);

  m_vecCurrentShape.resize(m_nVertices * 3);
  m_vecCurrentTex.resize(m_nVertices * 3);
  m_vecCurrentExpr.resize(m_nVertices * 3);
  m_vecCurrentBlendshape.resize(m_nVertices * 3);

  auto nLandmarks = m_mapLandmarkIndices.size();
  if (m_bUseLandmark) {
    m_vecLandmarkShapeMu.resize(nLandmarks * 3);
    m_matLandmarkShapePc.resize(nLandmarks * 3, m_nIdPcs);
    m_vecLandmarkExprMu.resize(nLandmarks * 3);
    m_matLandmarkExprPc.resize(nLandmarks * 3, m_nExprPcs);
  }
}

bool BfmManager::load() {
  LOG(INFO) << "Load model from disk.";

  try {
    std::unique_ptr<float[]> vecShapeMu(new float[m_nVertices * 3]);
    std::unique_ptr<float[]> vecShapeEv(new float[m_nIdPcs]);
    std::unique_ptr<float[]> matShapePc(new float[m_nVertices * 3 * m_nIdPcs]);
    std::unique_ptr<float[]> vecTexMu(new float[m_nVertices * 3]);
    std::unique_ptr<float[]> vecTexEv(new float[m_nIdPcs]);
    std::unique_ptr<float[]> matTexPc(new float[m_nVertices * 3 * m_nIdPcs]);
    std::unique_ptr<float[]> vecExprMu(new float[m_nVertices * 3]);
    std::unique_ptr<float[]> vecExprEv(new float[m_nExprPcs]);
    std::unique_ptr<float[]> matExprPc(new float[m_nVertices * 3 * m_nExprPcs]);
    std::unique_ptr<uint16_t[]> vecTriangleList(new uint16_t[m_nFaces * 3]);

    hid_t file = H5Fopen(m_strModelPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    bfm_utils::LoadH5Model(file, m_strShapeMuH5Path, vecShapeMu, m_vecShapeMu,
                           H5T_NATIVE_FLOAT);
    bfm_utils::LoadH5Model(file, m_strShapeEvH5Path, vecShapeEv, m_vecShapeEv,
                           H5T_NATIVE_FLOAT);
    bfm_utils::LoadH5Model(file, m_strShapePcH5Path, matShapePc, m_matShapePc,
                           H5T_NATIVE_FLOAT);

    bfm_utils::LoadH5Model(file, m_strTexMuH5Path, vecTexMu, m_vecTexMu,
                           H5T_NATIVE_FLOAT);
    bfm_utils::LoadH5Model(file, m_strTexEvH5Path, vecTexEv, m_vecTexEv,
                           H5T_NATIVE_FLOAT);
    bfm_utils::LoadH5Model(file, m_strTexPcH5Path, matTexPc, m_matTexPc,
                           H5T_NATIVE_FLOAT);

    bfm_utils::LoadH5Model(file, m_strExprMuH5Path, vecExprMu, m_vecExprMu,
                           H5T_NATIVE_FLOAT);
    bfm_utils::LoadH5Model(file, m_strExprEvH5Path, vecExprEv, m_vecExprEv,
                           H5T_NATIVE_FLOAT);
    bfm_utils::LoadH5Model(file, m_strExprPcH5Path, matExprPc, m_matExprPc,
                           H5T_NATIVE_FLOAT);

    bfm_utils::LoadH5Model(file, m_strTriangleListH5Path, vecTriangleList,
                           m_vecTriangleList, H5T_NATIVE_UINT16);
  } catch (std::bad_alloc &ba) {
    LOG(ERROR) << "Failed to alloc";
    return false;
  }
  auto tmp = m_vecTriangleList;
  assert(m_vecTriangleList.size() % 3 == 0);
  uint indent = m_vecTriangleList.size() / 3;
  for (uint i = 0; i < indent; ++i) {
    m_vecTriangleList[3 * i] = tmp[i];
    m_vecTriangleList[3 * i + 1] = tmp[indent + i];
    m_vecTriangleList[3 * i + 2] = tmp[2 * indent + i];
  }
  return true;
}

void BfmManager::extractLandmarks() {
  unsigned int iLandmark = 0;
  for (const auto& bfmIdx : m_mapLandmarkIndices) {
    m_vecLandmarkShapeMu(iLandmark * 3) = m_vecShapeMu(bfmIdx * 3);
    m_vecLandmarkShapeMu(iLandmark * 3 + 1) = m_vecShapeMu(bfmIdx * 3 + 1);
    m_vecLandmarkShapeMu(iLandmark * 3 + 2) = m_vecShapeMu(bfmIdx * 3 + 2);
    m_vecLandmarkExprMu(iLandmark * 3) = m_vecExprMu(bfmIdx * 3);
    m_vecLandmarkExprMu(iLandmark * 3 + 1) = m_vecExprMu(bfmIdx * 3 + 1);
    m_vecLandmarkExprMu(iLandmark * 3 + 2) = m_vecExprMu(bfmIdx * 3 + 2);

    for (unsigned int iIdPc = 0; iIdPc < m_nIdPcs; iIdPc++) {
      m_matLandmarkShapePc(iLandmark * 3, iIdPc) =
          m_matShapePc(bfmIdx * 3, iIdPc);
      m_matLandmarkShapePc(iLandmark * 3 + 1, iIdPc) =
          m_matShapePc(bfmIdx * 3 + 1, iIdPc);
      m_matLandmarkShapePc(iLandmark * 3 + 2, iIdPc) =
          m_matShapePc(bfmIdx * 3 + 2, iIdPc);
    }

    for (unsigned int iExprPc = 0; iExprPc < m_nExprPcs; iExprPc++) {
      m_matLandmarkExprPc(iLandmark * 3, iExprPc) =
          m_matExprPc(bfmIdx * 3, iExprPc);
      m_matLandmarkExprPc(iLandmark * 3 + 1, iExprPc) =
          m_matExprPc(bfmIdx * 3 + 1, iExprPc);
      m_matLandmarkExprPc(iLandmark * 3 + 2, iExprPc) =
          m_matExprPc(bfmIdx * 3 + 2, iExprPc);
    }

    ++iLandmark;
  }
}

void BfmManager::genRndFace(double dScale) {
  if (dScale == 0.0)
    LOG(INFO) << "Generate average face";
  else
    LOG(INFO) << "Generate random face (using the same scale)";

  m_aShapeCoef = bfm_utils::randn(m_nIdPcs, dScale);
  m_aTexCoef = bfm_utils::randn(m_nIdPcs, dScale);
  if (m_strVersion != "2009")
    m_aExprCoef = bfm_utils::randn(m_nExprPcs, dScale);

  this->genFace();
}

void BfmManager::genRndFace(double dShapeScale, double dTexScale,
                            double dExprScale) {
  LOG(INFO) << "Generate random face (using different scales)";
  m_aShapeCoef = bfm_utils::randn(m_nIdPcs, dShapeScale);
  m_aTexCoef = bfm_utils::randn(m_nIdPcs, dTexScale);
  if (m_strVersion != "2009")
    m_aExprCoef = bfm_utils::randn(m_nExprPcs, dExprScale);

  this->genFace();
}

void BfmManager::genFace() {
  LOG(INFO) << "Generate face with shape and expression coefficients";

  m_vecCurrentShape = this->coef2Object(m_aShapeCoef, m_vecShapeMu,
                                        m_matShapePc, m_vecShapeEv, m_nIdPcs);
  m_vecCurrentTex = this->coef2Object(m_aTexCoef, m_vecTexMu, m_matTexPc,
                                      m_vecTexEv, m_nIdPcs);
  if (m_strVersion != "2009") {
    m_vecCurrentExpr = this->coef2Object(m_aExprCoef, m_vecExprMu, m_matExprPc,
                                         m_vecExprEv, m_nExprPcs);
    m_vecCurrentBlendshape = m_vecCurrentShape + m_vecCurrentExpr;
  } else
    m_vecCurrentBlendshape = m_vecCurrentShape;
}

void BfmManager::genLandmarkBlendshape() {
  LOG(INFO) << "Generate landmarks with shape and expression coefficients";

  m_vecLandmarkCurrentShape =
      this->coef2Object(m_aShapeCoef, m_vecLandmarkShapeMu,
                        m_matLandmarkShapePc, m_vecShapeEv, m_nIdPcs);
  if (m_strVersion != "2009") {
    m_vecLandmarkCurrentExpr =
        this->coef2Object(m_aExprCoef, m_vecLandmarkExprMu, m_matLandmarkExprPc,
                          m_vecExprEv, m_nExprPcs);
    m_vecLandmarkCurrentBlendshape =
        m_vecLandmarkCurrentShape + m_vecLandmarkCurrentExpr;
  } else
    m_vecLandmarkCurrentBlendshape = m_vecLandmarkCurrentShape;
}
void BfmManager::genRMat() {
  LOG(INFO) << "Generate rotation matrix.";

  const double &roll = m_aExtParams[0];
  const double &yaw = m_aExtParams[1];
  const double &pitch = m_aExtParams[2];
  m_matR = bfm_utils::Euler2Mat(roll, yaw, pitch, false);
}

void BfmManager::genTVec() {
  LOG(INFO) << "Generate translation vector.";

  const double &tx = m_aExtParams[3];
  const double &ty = m_aExtParams[4];
  const double &tz = m_aExtParams[5];
  m_vecT << tx, ty, tz;
}

void BfmManager::genTransMat() {
  this->genRMat();
  this->genTVec();
}

void BfmManager::writePly(std::string fn, long mode) const {
  std::ofstream out;
  /* Note: In Linux Cpp, we should use std::ios::out as flag, which is not
   * necessary in Windows */
  out.open(fn, std::ios::out | std::ios::binary);
  if (!out.is_open()) {
    std::string sErrMsg = "Creation of " + fn + " failed.";
    LOG(ERROR) << sErrMsg;
    throw std::runtime_error(sErrMsg);
    return;
  }

  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "comment Made from the 3D Morphable Face Model of the Univeristy of "
         "Basel, Switzerland.\n";
  out << "element vertex " << m_nVertices << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  out << "element face " << m_nFaces << "\n";
  out << "property list uchar int vertex_indices\n";
  out << "end_header\n";

  int cnt = 0;
  for (int iVertice = 0; iVertice < m_nVertices; iVertice++) {
    float x, y, z;
    if (mode & ModelWriteMode_NoExpr) {
      x = float(m_vecCurrentShape(iVertice * 3));
      y = float(m_vecCurrentShape(iVertice * 3 + 1));
      z = float(m_vecCurrentShape(iVertice * 3 + 2));
    } else {
      x = float(m_vecCurrentBlendshape(iVertice * 3));
      y = float(m_vecCurrentBlendshape(iVertice * 3 + 1));
      z = float(m_vecCurrentBlendshape(iVertice * 3 + 2));
    }

    // TODO add translation and rotation?
    unsigned char r, g, b;
    auto d_to_ui = [](const double& x) { return uint(round(x * 255)); };
    if (mode & ModelWriteMode_PickLandmark) {
      bool bIsLandmark = false;
      for (const auto &bfmIdx : m_mapLandmarkIndices) {
        if (bfmIdx == iVertice) {
          bIsLandmark = true;
          break;
        }
      }
      if (bIsLandmark) {
        r = 255;
        g = 0;
        b = 0;
        cnt++;
      }
    } else {
      r = d_to_ui(m_vecCurrentTex(iVertice * 3));
      g = d_to_ui(m_vecCurrentTex(iVertice * 3 + 1));
      b = d_to_ui(m_vecCurrentTex(iVertice * 3 + 2));
    }

    out.write((char *)&x, sizeof(x));
    out.write((char *)&y, sizeof(y));
    out.write((char *)&z, sizeof(z));
    out.write((char *)&r, sizeof(r));
    out.write((char *)&g, sizeof(g));
    out.write((char *)&b, sizeof(b));
  }

  if ((mode & ModelWriteMode_PickLandmark) &&
      cnt != m_mapLandmarkIndices.size()) {
    LOG(ERROR) << "Pick too less landmarks.";
    LOG(ERROR) << "Number of picked points is " << cnt;
    throw std::runtime_error("Pick too less landmarks");
  }

  unsigned char N_VER_PER_FACE = 3;
  for (int iFace = 0; iFace < m_nFaces; iFace++) {
    out.write((char *)&N_VER_PER_FACE, sizeof(N_VER_PER_FACE));
    uint x = m_vecTriangleList(iFace * 3);
    uint y = m_vecTriangleList(iFace * 3 + 1);
    uint z = m_vecTriangleList(iFace * 3 + 2);

    out.write((char *)&x, sizeof(x));
    out.write((char *)&y, sizeof(y));
    out.write((char *)&z, sizeof(z));
  }

  out.close();
}

void BfmManager::writePlyNew(std::string fn, long mode) const {
  std::ofstream out;
  /* Note: In Linux Cpp, we should use std::ios::out as flag, which is not
   * necessary in Windows */
  out.open(fn, std::ios::out | std::ios::binary);
  if (!out.is_open()) {
    std::string sErrMsg = "Creation of " + fn + " failed.";
    LOG(ERROR) << sErrMsg;
    throw std::runtime_error(sErrMsg);
    return;
  }

  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "comment Made from the 3D Morphable Face Model of the Univeristy of "
         "Basel, Switzerland.\n";
  out << "element vertex " << m_nVertices << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "element face " << m_nFaces << "\n";
  out << "property list uchar int vertex_indices\n";
  out << "end_header\n";

  int cnt = 0;
  for (int iVertice = 0; iVertice < m_nVertices; iVertice++) {
    float x, y, z;
    if (mode & ModelWriteMode_NoExpr) {
      x = float(m_vecCurrentShape(iVertice * 3));
      y = float(m_vecCurrentShape(iVertice * 3 + 1));
      z = float(m_vecCurrentShape(iVertice * 3 + 2));
    } else {
      x = float(m_vecCurrentBlendshape(iVertice * 3));
      y = float(m_vecCurrentBlendshape(iVertice * 3 + 1));
      z = float(m_vecCurrentBlendshape(iVertice * 3 + 2));
    }

    out.write((char *)&x, sizeof(x));
    out.write((char *)&y, sizeof(y));
    out.write((char *)&z, sizeof(z));
  }

  unsigned char N_VER_PER_FACE = 3;
  for (int iFace = 0; iFace < m_nFaces; iFace++) {
    out.write((char *)&N_VER_PER_FACE, sizeof(N_VER_PER_FACE));
    int x = m_vecTriangleList(iFace * 3);
    int y = m_vecTriangleList(iFace * 3 + 1);
    int z = m_vecTriangleList(iFace * 3 + 2);

    out.write((char *)&x, sizeof(x));
    out.write((char *)&y, sizeof(y));
    out.write((char *)&z, sizeof(z));
  }

  out.close();
}

void BfmManager::writePlyPoints(std::string fn, long mode) const {
  std::ofstream out;
  /* Note: In Linux Cpp, we should use std::ios::out as flag, which is not
   * necessary in Windows */
  out.open(fn, std::ios::out | std::ios::binary);
  if (!out.is_open()) {
    std::string sErrMsg = "Creation of " + fn + " failed.";
    LOG(ERROR) << sErrMsg;
    throw std::runtime_error(sErrMsg);
    return;
  }
  int blank = 28588;
  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "comment Made from the 3D Morphable Face Model of the Univeristy of "
         "Basel, Switzerland.\n";
  out << "element vertex " << m_nVertices - blank << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "element face " << 0 << "\n";
  out << "property list uchar int vertex_indices\n";
  out << "end_header\n";

  int cnt = 0;
  for (int iVertice = blank; iVertice < m_nVertices; iVertice++) {
    float x, y, z;
    if (mode & ModelWriteMode_NoExpr) {
      x = float(m_vecCurrentShape(iVertice * 3));
      y = float(m_vecCurrentShape(iVertice * 3 + 1));
      z = float(m_vecCurrentShape(iVertice * 3 + 2));
    } else {
      x = float(m_vecCurrentBlendshape(iVertice * 3));
      y = float(m_vecCurrentBlendshape(iVertice * 3 + 1));
      z = float(m_vecCurrentBlendshape(iVertice * 3 + 2));
    }

    out.write((char *)&x, sizeof(x));
    out.write((char *)&y, sizeof(y));
    out.write((char *)&z, sizeof(z));
  }

  out.close();
}

void BfmManager::writeLandmarkPly(std::string fn) const {
  std::ofstream out;
  /* Note: In Linux Cpp, we should use std::ios::out as flag, which is not
   * necessary in Windows */
  out.open(fn, std::ios::out | std::ios::binary);
  if (!out.is_open()) {
    LOG(ERROR) << "Creation of " << fn << " failed.";
    return;
  }

  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "comment Made from the 3D Morphable Face Model of the Univeristy of "
         "Basel, Switzerland.\n";
  out << "element vertex " << m_mapLandmarkIndices.size() << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "end_header\n";

  int cnt = 0;
  for (int i = 0; i < m_mapLandmarkIndices.size(); i++) {
    float x, y, z;
    x = float(m_vecLandmarkCurrentBlendshape(i * 3));
    y = float(m_vecLandmarkCurrentBlendshape(i * 3 + 1));
    z = float(m_vecLandmarkCurrentBlendshape(i * 3 + 2));
    out.write((char *)&x, sizeof(x));
    out.write((char *)&y, sizeof(y));
    out.write((char *)&z, sizeof(z));
  }

  out.close();
}

void BfmManager::clrExtParams() {
  m_aExtParams.fill(0.0);
  this->genTransMat();
  this->genFace();
}
