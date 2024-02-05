#include "bfm_manager.h"
#include <cassert>
#include <cstdint>
#include <cmath>
#include <ceres/rotation.h>

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

  this->setIdExtParams();

  this->alloc();
  this->load();
  this->extractLandmarks();

  unsigned int iTex = 0;
  while (m_bIsTexStd && iTex < m_vecTexMu.size()) {
    if (m_vecTexMu(iTex++) > 1.0)
      m_bIsTexStd = false;
  }

  this->genAvgFace();
  // here landmarks blendshapes are set
  this->genLandmarkBlendshape();
  this->genLandmarkBlendshape();
}

// allocates memory for bfm parameters
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

  m_vecNormals.resize(m_nVertices * 3);
  auto nLandmarks = m_mapLandmarkIndices.size();
  if (m_bUseLandmark) {
    m_vecLandmarkShapeMu.resize(nLandmarks * 3);
    m_matLandmarkShapePc.resize(nLandmarks * 3, m_nIdPcs);
    m_vecLandmarkExprMu.resize(nLandmarks * 3);
    m_matLandmarkExprPc.resize(nLandmarks * 3, m_nExprPcs);
  }
}

// load bfm parameters
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
  // it is doing usefful thing
  // trinalges list is transposed from the start
  // but we want sequential order
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

// Save weights to file
void BfmManager::saveWeights(const std::string& filePath) {
    std::ofstream outFile(filePath, std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for saving weights: " + filePath);
    }

    // Save extrinsic parameters
    for (const auto& param : m_aExtParams) {
        outFile << param << " ";
    }
    outFile << "\n";

    // Save texture coefficients
    for (int i = 0; i < m_nIdPcs; ++i) {
        outFile << m_aTexCoef[i] << " ";
    }
    outFile << "\n";

    // Save shape coefficients
    for (int i = 0; i < m_nIdPcs; ++i) {
        outFile << m_aShapeCoef[i] << " ";
    }
    outFile << "\n";

    // Save expression coefficients
    for (int i = 0; i < m_nExprPcs; ++i) {
        outFile << m_aExprCoef[i] << " ";
    }
    outFile << "\n";

    outFile.close();
}

// Load weights from file
void BfmManager::loadWeights(const std::string& filePath) {
    std::ifstream inFile(filePath, std::ios::in);
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for loading weights: " + filePath);
    }

    // Load extrinsic parameters
    for (auto& param : m_aExtParams) {
        inFile >> param;
    }

    // Load texture coefficients
    for (int i = 0; i < m_nIdPcs; ++i) {
        inFile >> m_aTexCoef[i];
    }

    // Load shape coefficients
    for (int i = 0; i < m_nIdPcs; ++i) {
        inFile >> m_aShapeCoef[i];
    }

    // Load expression coefficients
    for (int i = 0; i < m_nExprPcs; ++i) {
        inFile >> m_aExprCoef[i];
    }

    inFile.close();
}

void BfmManager::setIdExtParams() {
	m_matR = Matrix3d::Identity();
	m_vecT.fill(0.);
	m_dScale = 1.;
    m_aExtParams.fill(0);
    this->genExtParams();
}
void BfmManager::setRotTransScParams(const Matrix3d& newR, const Vector3d& newT, const double& newScale) {
		// matrix may have negative determinant, then we need to change
    // bfm coefs inplace
    if (newR.determinant() < 0) {
        std::cout << "WARNING: Rotation matrix has negative determinant\n";
        this->setIdExtParams();
        Eigen::DiagonalMatrix<double, 3> diag({1., -1., 1.});
        m_matR = diag;
        // change bfm parameters to have rotation matrix with det=1
        transformShapeExprBFM();
        m_matR = newR * diag;
        this->genFace();
    } else {
      m_matR = newR;
    }

    m_vecT = newT;
	m_dScale = newScale;

    this->genExtParams();
}

void BfmManager::setExtParams(const double* const extParams) {
    for (size_t i = 0; i < m_aExtParams.size(); ++i) {
        m_aExtParams[i] = extParams[i];
    }
    genTransforms();
}

void BfmManager::genExtParams() {
    ceres::RotationMatrixToAngleAxis(m_matR.data(), &m_aExtParams[0]);
    m_aExtParams[3] = m_vecT[0];
    m_aExtParams[4] = m_vecT[1];
    m_aExtParams[5] = m_vecT[2];
    m_aExtParams[6] = m_dScale;
}

void BfmManager::genTransforms() {
  LOG(INFO) << "Generate rotation, tranlation and scale.";

  double* startPointer = &m_aExtParams[0];
  double* rotation = startPointer;
  double* translation = startPointer + 3;
  double* scale = startPointer + 6;

  double rotationMatrix[9];
  ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);
  m_matR(0, 0) = rotationMatrix[0];	m_matR(0, 1) = rotationMatrix[3];	m_matR(0, 2) = rotationMatrix[6];
  m_matR(1, 0) = rotationMatrix[1];	m_matR(1, 1) = rotationMatrix[4];	m_matR(1, 2) = rotationMatrix[7];
  m_matR(2, 0) = rotationMatrix[2];	m_matR(2, 1) = rotationMatrix[5];	m_matR(2, 2) = rotationMatrix[8];

  const double &tx = translation[0];
  const double &ty = translation[1];
  const double &tz = translation[2];
  m_vecT << tx, ty, tz;

  m_dScale = scale[0];
}

void BfmManager::transformShapeExprBFM() {
    Matrix3d R = m_dScale * m_matR;
    auto translation = m_vecT;
    Vector3d mu;
    Matrix3d M;
    for (int iVertice = 0; iVertice < m_nVertices; iVertice++) {
        Vector3d shapeMu(m_vecShapeMu[3 * iVertice], m_vecShapeMu[3 * iVertice + 1], m_vecShapeMu[3 * iVertice + 2]);
        m_vecShapeMu.segment(3 * iVertice, 3) = R * shapeMu + translation;
        Vector3d newPc;
        for(size_t i = 0; i < m_nIdPcs; ++i) {
            newPc = R * Vector3d(
              m_matShapePc(3 * iVertice, i),
              m_matShapePc(3 * iVertice + 1, i),
              m_matShapePc(3 * iVertice + 2, i)
            );
            m_matShapePc.block(3 * iVertice, i, 3, 1) = newPc;
        }

        Vector3d exprMu(m_vecExprMu[3 * iVertice], m_vecExprMu[3 * iVertice + 1], m_vecExprMu[3 * iVertice + 2]);
        m_vecExprMu.segment(3 * iVertice, 3) = R * exprMu;
        for(size_t i = 0; i < m_nExprPcs; ++i) {
            newPc = R * Vector3d(
              m_matExprPc(3 * iVertice, i),
              m_matExprPc(3 * iVertice + 1, i),
              m_matExprPc(3 * iVertice + 2, i)
            );
            m_matExprPc.block(3 * iVertice, i, 3, 1) = newPc;
        }
    }

    this->setIdExtParams();
}

void BfmManager::updateFaceUsingParams() {
    this->genFace();
    this->genTransforms();
}

void BfmManager::computeVertexNormals() {
    for (size_t iFace = 0; iFace < m_nFaces; ++iFace) {
        unsigned int indexA = m_vecTriangleList[3 * iFace];
        unsigned int indexB = m_vecTriangleList[3 * iFace + 1];
        unsigned int indexC = m_vecTriangleList[3 * iFace + 2];

        Eigen::Vector3d vertA(
            m_vecCurrentBlendshape[3 * indexA],
            m_vecCurrentBlendshape[3 * indexA + 1],
            m_vecCurrentBlendshape[3 * indexA + 2]
        );
        Eigen::Vector3d vertB(
            m_vecCurrentBlendshape[3 * indexB],
            m_vecCurrentBlendshape[3 * indexB + 1],
            m_vecCurrentBlendshape[3 * indexB + 2]
        );
        Eigen::Vector3d vertC(
            m_vecCurrentBlendshape[3 * indexC],
            m_vecCurrentBlendshape[3 * indexC + 1],
            m_vecCurrentBlendshape[3 * indexC + 2]
        );

        Eigen::Vector3d faceNormal = computeFNormal(vertA, vertB, vertC);
        if (vertA.dot(faceNormal) > 0) {
            faceNormal *= -1;
        }
        //if triangle is lying on single plane normal of every vertex of a triangle is the same
        m_vecNormals.segment(3 * indexA, 3) += faceNormal;
        m_vecNormals.segment(3 * indexB, 3) += faceNormal;
        m_vecNormals.segment(3 * indexC, 3) += faceNormal;
    }
    //normalize every vertex normal
    for(size_t iVertex = 0; iVertex < m_nVertices; ++iVertex) {
        m_vecNormals.segment(3 * iVertex, 3).normalize();
        // std::cout << m_vecNormals.segment(3 * iVertex, 3).dot(m_vecNormals.segment(3 * iVertex, 3)) << std::endl;
    }
}

Eigen::Vector3d BfmManager::computeFNormal(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c){
    Eigen::Vector3d norm_vec(0.0f, 0.0f, 0.0f);
    Eigen::Vector3d AB = b - a;
    Eigen::Vector3d AC = c - a;

    return AB.cross(AC).normalized();
}
// std::vector<Eigen::Vector3d> BfmManager::computeVertexNormals() {
//     std::vector<Eigen::Vector3d> face_vertices;
//     std::vector<unsigned int> face_indices;
//     std::vector<Eigen::Vector3d> vertex_normals;

//     //gives the indices of vertices
//     for (size_t iIndex = 0; iIndex < m_nFaces; iIndex++) {
//         face_indices.push_back(m_vecTriangleList(iIndex * 3));
//         face_indices.push_back(m_vecTriangleList(iIndex * 3 + 1));
//         face_indices.push_back(m_vecTriangleList(iIndex * 3 + 2));
//     }
//     // std::cout << "Num face indices: " << face_indices.size() << std::endl;
//     //coordinates of the vertices
//     for (size_t iVertex = 0; iVertex < m_nVertices; iVertex++) {
//         double x, y, z;

//         x = double(m_vecCurrentBlendshape(iVertex * 3));
//         y = double(m_vecCurrentBlendshape(iVertex * 3 + 1));
//         z = double(m_vecCurrentBlendshape(iVertex * 3 + 2));

//         face_vertices.push_back({x, y, z});
//       }
//     // std::cout << face_indices.size();
//       // Compute vertex normals
//     for (size_t i = 0; i < face_indices.size() - 2; i += 3) {
//         unsigned int index1 = face_indices[i];
//         unsigned int index2 = face_indices[i + 1];
//         unsigned int index3 = face_indices[i + 2];

//         Eigen::Vector3d vertA = face_vertices[index1];
//         Eigen::Vector3d vertB = face_vertices[index2];
//         Eigen::Vector3d vertC = face_vertices[index3];

//         Eigen::Vector3d faceNormal = computeVNormal(vertA, vertB, vertC);
//         //if triangle is lying on single plane normal of every vertex of a triangle is the same
//         vertex_normals.push_back(faceNormal);
//         vertex_normals.push_back(faceNormal);
//         vertex_normals.push_back(faceNormal);
//     }
//     //normalize every vertex normal
//     for(size_t i = 0; i < vertex_normals.size(); i++){
//         vertex_normals[i].normalize();
//     }
//     return vertex_normals;
// }

//computes normal of the given vertex
