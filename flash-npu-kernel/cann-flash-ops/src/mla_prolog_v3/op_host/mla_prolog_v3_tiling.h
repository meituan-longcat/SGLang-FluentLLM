/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_tiling.h
 * \brief
 */

#include "../op_kernel/mla_prolog_v3_policy.h"

#ifndef MLA_PROLOG_V3_TILING_H
#define MLA_PROLOG_V3_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#if MLA_PROLOG_V3_SFA1
#if __has_include("../../fluent_mla_prolog_v3/op_kernel/fluent_mla_prolog_v3_template_tiling_key.h")
#include "../../fluent_mla_prolog_v3/op_kernel/fluent_mla_prolog_v3_template_tiling_key.h"
#else
#include "../fluent_mla_prolog_v3/fluent_mla_prolog_v3_template_tiling_key.h"
#endif
#else
#include "../op_kernel/mla_prolog_v3_template_tiling_key.h"
#endif
#include "../op_kernel/mla_prolog_v3_tiling_data.h"

#ifdef ASCENDC_OP_TEST
#define MLA_EXTERN_C extern "C"
#else
#define MLA_EXTERN_C
#endif

namespace optiling {
#if MLA_PROLOG_V3_SFA1
namespace fluent_mla_prolog_v3 {
#endif

// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_DQ_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INPUT_INDEX = 2;
constexpr uint32_t WEIGHT_UK_INPUT_INDEX = 3;
constexpr uint32_t WEIGHT_DKV_KR_INPUT_INDEX = 4;
constexpr uint32_t RMSNORM_GAMMA_CQ_INPUT_INDEX = 5;
constexpr uint32_t RMS_NORM_GAMMA_CKV_INPUT_INDEX = 6;
constexpr uint32_t ROPE_SIN_INPUT_INDEX = 7;
constexpr uint32_t ROPE_COS_INPUT_INDEX = 8;
constexpr uint32_t KV_CACHE_INPUT_INDEX = 9;
constexpr uint32_t KR_CACHE_INPUT_INDEX = 10;

// INPUT(OPTION)
constexpr uint32_t CACHE_INDEX_INPUT_INDEX = 11;
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t DEQUANT_SCALE_W_DQ_INDEX = 13;
constexpr uint32_t DEQUANT_SCALE_W_UQ_QR_INDEX = 14;
constexpr uint32_t DEQUANT_SCALE_W_DKV_KR_INDEX = 15;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
constexpr uint32_t QUANT_SCALE_CKR_INDEX = 17;
constexpr uint32_t SMOOTH_SCALES_CQ_INDEX = 18;
constexpr uint32_t ACTUAL_SEQ_LEN_INDEX = 19;
constexpr uint32_t K_NOPE_CLIP_ALPHA_INDEX = 20;

// OUTPUT
constexpr uint32_t QUERY_OUTPUT_INDEX = 0;
constexpr uint32_t QUERY_ROPE_OUTPUT_INDEX = 1;
constexpr uint32_t KV_CACHE_OUT_OUTPUT_INDEX = 2;
constexpr uint32_t KR_CACHE_OUT_OUTPUT_INDEX = 3;
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX = 4;
constexpr uint32_t QUERY_NORM_OUTPUT_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_Q_NORM_OUTPUT_INDEX = 6;
#if MLA_PROLOG_V3_SFA1
constexpr uint32_t CUR_K_NOPE_OUTPUT_INDEX = 7;
constexpr uint32_t CUR_K_ROPE_OUTPUT_INDEX = 8;
#endif

// ATTR
constexpr uint32_t RMS_NORM_EPSILON_CQ_ATTR_INDEX = 0;
constexpr uint32_t RMS_NORM_EPSILON_CKV_ATTR_INDEX = 1;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 2;
constexpr uint32_t QUERY_NORM_ATTR_INDEX = 3;
constexpr uint32_t WEIGHT_QUANT_MODE_INDEX = 4;
constexpr uint32_t KV_QUANT_MODE_INDEX = 5;
constexpr uint32_t QUERY_QUANT_MODE_INDEX = 6;
constexpr uint32_t CKVKR_REPO_MODE_INDEX = 7;
constexpr uint32_t QUANT_SCALE_REPO_MODE_INDEX = 8;
constexpr uint32_t TILE_SIZE_INDEX = 9;
constexpr uint32_t QCQR_SCALE_INDEX = 10;
constexpr uint32_t KC_SCALE_INDEX = 11;
#if MLA_PROLOG_V3_SFA1
constexpr uint32_t OUTPUT_CUR_K_NOPE_ROPE_ATTR_INDEX = 12;
#else
constexpr uint32_t ENABLE_ROPE_INDEX = 12;
#endif

constexpr uint32_t MLA_PROLOG_V3_DIM_INDEX_0 = 0;
constexpr uint32_t MLA_PROLOG_V3_DIM_INDEX_1 = 1;
constexpr uint32_t MLA_PROLOG_V3_DIM_INDEX_2 = 2;
constexpr uint32_t MLA_PROLOG_V3_DIM_INDEX_3 = 3;

constexpr uint32_t MLA_PROLOG_V3_DIM_NUM_0 = 0;
constexpr uint32_t MLA_PROLOG_V3_DIM_NUM_1 = 1;
constexpr uint32_t MLA_PROLOG_V3_DIM_NUM_2 = 2;
constexpr uint32_t MLA_PROLOG_V3_DIM_NUM_3 = 3;
constexpr uint32_t MLA_PROLOG_V3_DIM_NUM_4 = 4;

constexpr char CACHE_MODE_BSND[] {"BSND"};
constexpr char CACHE_MODE_TND[] {"TND"};
constexpr char CACHE_MODE_PA_BSND[] {"PA_BSND"};
constexpr char CACHE_MODE_PA_NZ[] {"PA_NZ"};
constexpr char CACHE_MODE_PA_BLK_BSND[] {"PA_BLK_BSND"};
constexpr char CACHE_MODE_PA_BLK_NZ[] {"PA_BLK_NZ"};
constexpr char V1_OP_NAME[] {"MlaProlog"};
constexpr char V2_OP_NAME[] {"MlaPrologV2"};
#if MLA_PROLOG_V3_SFA1
constexpr char V3_OP_NAME[] {"FluentMlaPrologV3"};
#else
constexpr char V3_OP_NAME[] {"MlaPrologV3"};
#endif

constexpr uint32_t CACHE_MODE_BSND_LEN = sizeof(CACHE_MODE_BSND) - 1;
constexpr uint32_t CACHE_MODE_TND_LEN = sizeof(CACHE_MODE_TND) - 1;
constexpr uint32_t CACHE_MODE_PA_BSND_LEN = sizeof(CACHE_MODE_PA_BSND) - 1;
constexpr uint32_t CACHE_MODE_PA_NZ_LEN = sizeof(CACHE_MODE_PA_NZ) - 1;
constexpr uint32_t CACHE_MODE_PA_BLK_BSND_LEN = sizeof(CACHE_MODE_PA_BLK_BSND) - 1;
constexpr uint32_t CACHE_MODE_PA_BLK_NZ_LEN = sizeof(CACHE_MODE_PA_BLK_NZ) - 1;

struct MlaPrologV3BaseShapeInfo {
    uint32_t bSize = 0;     // B
    uint32_t s1Size = 0;      // S1
    uint32_t tSize = 0;     // T
    uint32_t s2Size = 0;      // S2
    uint32_t heSize = 0;     // He
    uint32_t hcqSize = 0;    // Hcq
    uint32_t hckvSize = 0;   // Hckv
    uint32_t headSizeQc = 0;    // N * D
    uint32_t headSizeQr = 0;    // N * Dr
    uint32_t headSizeUqQr = 0;  // N * (D + Dr)
    uint32_t nSize = 0;   // N
    uint32_t nkvSize = 0; // Nkv
    uint32_t dSize = 0; // D
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t drSize = 0;   // Dr
    uint32_t dtileSize = 0; // Dtile
};

enum class CACHE_MODE:uint8_t {
    BSND = 0,
    PA_BSND = 1,
    PA_NZ = 2,
    PA_BLK_BSND = 3,
    PA_BLK_NZ = 4,
    TND = 5
};

enum class EMPTY_TENSOR_MODE:uint8_t {
    NON_EMPTY = 0,
    EMPTY_CACHE = 1,
    EMPTY_QUERY = 2
};

enum class ACTUAL_SEQ_MODE:uint8_t {
    DISABLED = 0,
    EN_Q_LEN = 1
};

enum class QUANT_MODE:int8_t {
    NO_QUANT = -1,
    PARTIAL_QUANT_KV_NO_QUANT = 0,
    PARTIAL_QUANT_KV_QUANT_PER_CHANNEL = 1,
    FULL_QUANT_KV_NO_QUANT = 2,
    FULL_QUANT_KV_QUANT_PER_TENSOR = 3,
    PARTIAL_QUANT_KV_QUANT_PER_TILE = 4,
#if MLA_PROLOG_V3_SFA1
    FULL_QUANT_KV_QUANT_PER_TILE = 5
#else
    FULL_QUANT_KV_QUANT_PER_TILE = 5,
    NO_QUANT_KV_QUANT_PER_TILE = 9
#endif
};

enum class WEIGHT_QUANT_MODE:int32_t {
    NO_QUANT = 0,
    PARTIAL_QUANT = 1,
    FULL_QUANT = 2
};

enum class KV_QUANT_MODE:int32_t {
    NO_QUANT = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_TILE = 3
};

enum class QUERY_QUANT_MODE:int32_t {
    NO_QUANT = 0,
    PER_TOKEN_HEAD = 1
};

enum class CKV_KR_REPO_MODE:int32_t {
    DIVIDE = 0,
    COMBINE = 1
};

enum class QUANT_SCALE_REPO_MODE:int32_t {
    DIVIDE = 0,
    COMBINE = 1
};

struct MlaPrologV3ScenarioInfo {
    bool isV1Flag_;
    bool batchSeqFusedFlag_;
    uint32_t splitMFlag_ = 0;
    QUANT_MODE quantMode_;
    CACHE_MODE cacheMode_;
    EMPTY_TENSOR_MODE emptyTensorMode_;
    ACTUAL_SEQ_MODE actualSeqMode_;
};

struct MlaPrologV3CompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    const gert::Tensor *tensor;
};

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t NUM_BYTES_INT32 = 4;
constexpr uint32_t NUM_BYTES_FP32 = 4;

constexpr uint32_t GROUP_COMPUTE_CUBE_NUM_PER_GROUP = 8U;
constexpr uint32_t HIGH_THROUGHPUT__D_SIZE = 128U;
constexpr uint32_t GROUP_COMPUTE_T_SIZE = 1U;
constexpr uint32_t GROUP_COMPUTE_NKV_SIZE = 8U;
constexpr uint32_t GROUP_COMPUTE_MIN_AIC_NUM = 16U;
constexpr uint32_t GROUP_COMPUTE_MIN_AIV_NUM = 32U;
constexpr uint32_t GROUP_COMPUTE_N_SIZE = 8U;

constexpr uint64_t MLA_PROLOG_V3_TILINGKEY_BASE_OFFSET = uint64_t(10000000000000000UL);          // 10^16
constexpr uint64_t MLA_PROLOG_V3_TYPE_OFFSET = uint64_t(10UL);
constexpr uint64_t MLA_PROLOG_V3_QUANT_TYPE_OFFSET = uint64_t(100UL);
constexpr uint64_t MLA_PROLOG_V3_ENABLE_DEQUANT_OPT_OFFSET = uint64_t(1000UL);
constexpr uint64_t MLA_PROLOG_V3_ENABLE_GROUP_COMPUTE_OPT_OFFSET = uint64_t(10000UL);
constexpr uint64_t MLA_PROLOG_V3_EMPTY_TENSOR_MODE_OFFSET = uint64_t(100000UL);
constexpr uint64_t MLA_PROLOG_V3_ACTUAL_SEQ_MODE_OFFSET = uint64_t(1000000UL);
constexpr uint64_t MLA_PROLOG_V3_SPLIT_M_OFFSET = uint64_t(1000000000000000UL);

struct MlaPrologV3Context {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo tokenX;
    RequiredParaInfo weightDq;
    RequiredParaInfo weightUqQr;
    RequiredParaInfo weightUk;
    RequiredParaInfo weightDkvKr;
    RequiredParaInfo rmsnormGammaCq;
    RequiredParaInfo rmsnormGammaCkv;
    RequiredParaInfo ropeSin;
    RequiredParaInfo ropeCos;
    RequiredParaInfo kvCache;
    RequiredParaInfo krCache;
    OptionalParaInfo cacheIndex;
    OptionalParaInfo dequantScaleX;
    OptionalParaInfo dequantScaleWDq;
    OptionalParaInfo dequantScaleWUqQr;
    OptionalParaInfo dequantScaleWDkvKr;
    OptionalParaInfo quantScaleCkv;
    OptionalParaInfo quantScaleCkr;
    OptionalParaInfo smoothScalesCq;
    OptionalParaInfo actualSeqLen;
    OptionalParaInfo kNopeClipAlpha;
    RequiredParaInfo query;
    RequiredParaInfo queryRope;
    RequiredParaInfo kvCacheOut;
    RequiredParaInfo krCacheOut;
    OptionalParaInfo dequantScaleQNope;
    OptionalParaInfo queryNorm;
    OptionalParaInfo dequantScaleQNorm;
#if MLA_PROLOG_V3_SFA1
    RequiredParaInfo curKNope;
    RequiredParaInfo curKRope;
#endif

    const float *rmsNormEspilonCq;
    const float *rmsNormEspilonCkv;
    const char *cacheMode;
    const bool *queryNormFlag;
#if MLA_PROLOG_V3_SFA1
    const bool *outputCurKNopeRopeFlag;
#endif

    const int *weightQuantMode;
    const int *kvQuantMode;
    const int *queryQuantMode;
    const int *ckvkrRepoMode;
    const int *quantScaleRepoMode;
    const int *tileSize;

    const float *qcQrScale;
    const float *kcScale;
#if !MLA_PROLOG_V3_SFA1
    const bool *enableRope;
#endif

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class MlaPrologV3Tiling {
public:
    MlaPrologV3Tiling() = default;
    ~MlaPrologV3Tiling() = default;

#if MLA_PROLOG_V3_SFA1
    ge::graphStatus RunBigKernelTiling(MlaPrologV3Context &context, FluentMlaPrologV3TilingData* tilingData);
#else
    ge::graphStatus RunBigKernelTiling(MlaPrologV3Context &context, MlaPrologV3TilingData* tilingData);
#endif
    static ge::graphStatus ConvertContext(gert::TilingContext &context, MlaPrologV3Context &mlaPrologV3Context);

private:
    static void ConvertRequiredParams(gert::TilingContext &context, MlaPrologV3Context &mlaPrologV3Context);
    static void ConvertOptionalParams(gert::TilingContext &context, MlaPrologV3Context &mlaPrologV3Context);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetScenarioInfo();
    ge::graphStatus SetAttrInfo();
    QUANT_MODE GetQuantizationMode() const;
    ge::graphStatus SetShapeInfo();
    ge::graphStatus ProcessBaseInputs();
    ge::graphStatus FillTiling();
    ge::graphStatus FillMatmul1Tiling();
    ge::graphStatus FillMatmul2Tiling();
    ge::graphStatus FillMatmul3Tiling();
    ge::graphStatus FillMatmul4Tiling();
    uint32_t CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum = 16) const;
    bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType);
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus GenTilingKey() const;

    MlaPrologV3BaseShapeInfo baseShapeInfo_;
    MlaPrologV3ScenarioInfo scenarioInfo_;

    uint32_t stepBatchSize_ = 0;
    uint32_t stepNumHeadDequant_ = 0;
    uint32_t mSubSize_ = 0;
    uint32_t mSubCoreNum_ = 0;

    uint32_t mm1BlockNum_ = 0;
    uint32_t mm2BlockNum_ = 0;
    uint32_t mm3BlockNum_ = 0;
    uint32_t mm4BlockNum_ = 0;
    uint32_t vectorBlockNum_ = 0;

    uint32_t singlecoreHeadSizeCq_ = 0;
    uint32_t singlecoreHeadSizeQcQr_ = 0;
    uint32_t singlecoreHeadSizeCkvKr_ = 0;
    uint32_t singlecoreNumHeadSize_ = 0;

    float reciprocalCq_ = 0.00001f;
    float epsilonCq_ = 1.0;
    float reciprocalCkv_ = 0.00001f;
    float epsilonCkv_ = 1.0;
    bool queryNormFlag_ = false;
#if MLA_PROLOG_V3_SFA1
    bool outputCurKNopeRope_ = false;
#endif

    int32_t weightQuantMode_ = 0;
    KV_QUANT_MODE kvQuantMode_ = KV_QUANT_MODE::NO_QUANT;
    int32_t queryQuantMode_ = 0;
    int32_t ckvkrRepoMode_ = 0;
    int32_t quantScaleRepoMode_ = 0;
    int32_t tileSize_ = 128;
    float qcQrScale_ = 1.0f;
    float kcScale_ = 1.0f;
#if !MLA_PROLOG_V3_SFA1
    bool enableRope_ = true;
#endif

    ge::DataType mmDateType_ = ge::DT_BF16;
    bool enableDequantOpt_ = false;
    bool enableGroupComputeOpt_ = false;    // 低延时场景算例分组标记

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;

    MlaPrologV3Context *context_ = nullptr;
    TCubeTiling *bmm1TilingData_ = nullptr;
    TCubeTiling *bmm2TilingData_ = nullptr;
    TCubeTiling *bmm3TilingData_ = nullptr;
    TCubeTiling *bmm4TilingData_ = nullptr;
#if MLA_PROLOG_V3_SFA1
    FluentMlaPrologV3BaseParams *baseParams_ = nullptr;
#else
    MlaPrologV3BaseParams *baseParams_ = nullptr;
#endif
};

#if MLA_PROLOG_V3_SFA1
MLA_EXTERN_C ge::graphStatus TilingFluentMlaPrologV3(gert::TilingContext *context);
#else
MLA_EXTERN_C ge::graphStatus TilingMlaPrologV3(gert::TilingContext *context);
#endif
} // optiling
#if MLA_PROLOG_V3_SFA1
} // namespace optiling
#endif
#endif
