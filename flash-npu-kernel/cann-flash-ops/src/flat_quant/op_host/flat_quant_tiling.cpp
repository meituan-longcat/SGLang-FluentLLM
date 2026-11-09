/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flat_quant_tiling.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"
#include "flat_quant_tiling.h"

// Compatibility aliases for the host implementation.
#define OPS_LOGE OPS_LOG_E
#define OPS_LOGI OPS_LOG_I

namespace optiling {

// Target arch is fixed at compile time by BUILD_ARCH (A2/A3 -> 910b/910_93 non-regbase,
// A5 -> 950 regbase with FLOAT4_E2M1/FLOAT8_E8M0 support), so no runtime SoC probing is needed.
#if defined(A5_COMPATIBLE)
constexpr bool FLAT_QUANT_IS_REGBASE = true;
#else
constexpr bool FLAT_QUANT_IS_REGBASE = false;
#endif

constexpr uint8_t INPUT_TENSOR_NUM = 3;
constexpr uint8_t INDEX_TWO = 2;
constexpr uint8_t INDEX_ONE = 1;
constexpr uint8_t INDEX_ZERO = 0;
constexpr uint8_t INDEX_THREE = 3;
constexpr uint8_t INDEX_FOUR = 4;
constexpr uint8_t DIM_THREE = 3;
constexpr uint8_t DIM_TWO = 2;
constexpr uint8_t WORKSPACE_NUM = 1;

constexpr uint8_t BYTE_LEN_2 = 2;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t CEIL_SIZE = 16;
constexpr int32_t MAX_K_SIZE = 262144;
constexpr int32_t NUM_EIGHT = 8;
constexpr int32_t MAX_MN_SIZE = 256;
constexpr int32_t FACTOR_TWO = 2;
constexpr int32_t DATA_TYPE_SIZE = 2;
constexpr int32_t K_PER_VEC = 4;
constexpr int32_t BASE_K = 64;
constexpr int32_t BASE_SIZE = 128;
constexpr int32_t L1_SIZE = 512 * 1024;
constexpr int32_t L0_SIZE = 64 * 1024;
constexpr int32_t L0C_SIZE = 256 * 1024;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;
constexpr float SIX_FLOAT = 6.0f;
constexpr float TWELVE_FLOAT = 12.0f;

constexpr int64_t GROUP_LIST_TYPE_MIN = 0;
constexpr int64_t GROUP_LIST_TYPE_MAX = 2;
constexpr int64_t GROUP_LIST_TYPE_TWO_DIMS = 2;
constexpr int64_t MAX_GROUP_NUM = 1024;

constexpr uint8_t MM_BASE_MODE = 1;
constexpr uint8_t MM_DOUBLE_MODE = 2;
constexpr uint8_t MM_SPLIT_MODE = 3;
constexpr uint8_t MM_HIGH_MODE = 4;
constexpr uint8_t MM_ONE_MODE = 5;
constexpr uint8_t MM_HIGH_MODE_ALIGN = 6;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t WORK_SPACE_SIZE_APT = 48 * 1024 * 1024;

class FlatQuantTiling {
public:
    explicit FlatQuantTiling(gert::TilingContext* context) : tilingContext_(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    bool ValidateAll() const;
    bool CheckShapes() const;
    bool CheckClipRatio() const;
    bool CheckDstDtype(ge::DataType outDtype) const;
    bool CheckDstTypeMax(ge::DataType outDtype) const;
    void GetKernelMode(int64_t aivNum);
    uint8_t DetermineMmMode(ge::DataType outDtype) const;
    void CalculateWorkspace(int64_t aivNum, ge::DataType outDtype);
    ge::graphStatus GetTCubeTiling();
    ge::graphStatus InitializeInputsAndAttributes();
    ge::graphStatus SetBasicTilingData();
    bool CheckGroupList();
    ge::graphStatus CalculateIterBatch();
    ge::graphStatus SetTilingContextAndSaveData();

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    ge::DataType dataType_ = ge::DT_UNDEFINED;

    gert::TilingContext* tilingContext_ = nullptr;
    gert::Shape xShape_;
    gert::Shape p1Shape_;
    gert::Shape p2Shape_;
    gert::Shape groupListShape_;

    uint64_t mAlign_ = 0;
    uint64_t nAlign_ = 0;
    uint64_t iterBatch_ = 0;
    int64_t groupNum_ = 0;

    uint8_t mmMode_ = MM_BASE_MODE;
    bool hasP2_ = true;
    bool hasGroupList_ = false;

    const float* clipRatio_ = nullptr;
    const int64_t* dstDtype_ = nullptr;
    const float* dstTypeMax_ = nullptr;
    const int64_t* groupListType_ = nullptr;
    const bool* clearPadding_ = nullptr;
    FlatQuantTilingData tilingData_;
};

ge::graphStatus FlatQuantTiling::InitializeInputsAndAttributes()
{
    // 获取输入矩阵
    auto xTensor = tilingContext_->GetInputTensor(INDEX_ZERO);
    auto p1Tensor = tilingContext_->GetInputTensor(INDEX_ONE);
    auto p2Tensor = tilingContext_->GetInputTensor(INDEX_TWO);
    const gert::RuntimeAttrs* attrs = tilingContext_->GetAttrs();
    if (xTensor == nullptr || p1Tensor == nullptr || p2Tensor == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 获取输入的数据类型，输入Tensor的数据类型保持一致
    for (uint64_t i = 0; i < INPUT_TENSOR_NUM; i++) {
        auto temp = tilingContext_->GetInputDesc(i);
        OPS_LOG_E_IF_NULL(tilingContext_->GetNodeName(), temp, return ge::GRAPH_FAILED);
        if (dataType_ == ge::DT_UNDEFINED) {
            dataType_ = temp->GetDataType();
        } else if (dataType_ != temp->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
    }
    // 获取输入的shape和属性
    auto xShape = tilingContext_->GetInputShape(INDEX_ZERO);
    auto p1Shape = tilingContext_->GetInputShape(INDEX_ONE);
    auto p2Shape = tilingContext_->GetInputShape(INDEX_TWO);
    OPS_LOG_E_IF_NULL(tilingContext_->GetNodeName(), xShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(tilingContext_->GetNodeName(), p1Shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(tilingContext_->GetNodeName(), p2Shape, return ge::GRAPH_FAILED);
    xShape_ = xShape->GetOriginShape();
    p1Shape_ = p1Shape->GetOriginShape();
    p2Shape_ = p2Shape->GetOriginShape();
    hasP2_ = (p2Shape_.GetDim(INDEX_ZERO) > 0 && p2Shape_.GetDim(INDEX_ONE) > 0);
    auto groupListTensor = tilingContext_->GetOptionalInputTensor(INDEX_THREE);
    hasGroupList_ = groupListTensor != nullptr;
    if (hasGroupList_) {
        groupListShape_ = groupListTensor->GetShape().GetStorageShape();
    }
    clipRatio_ = attrs->GetAttrPointer<float>(INDEX_ZERO);
    dstDtype_ = attrs->GetAttrPointer<int64_t>(INDEX_ONE);
    dstTypeMax_ = attrs->GetAttrPointer<float>(INDEX_TWO);
    groupListType_ = attrs->GetAttrPointer<int64_t>(INDEX_THREE);
    clearPadding_ = attrs->GetAttrPointer<bool>(INDEX_FOUR);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::SetBasicTilingData()
{
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t N = xShape_.GetDim(INDEX_TWO);
    nAlign_ = CeilA2B(N, CEIL_SIZE) * CEIL_SIZE;
    mAlign_ = CeilA2B(M, CEIL_SIZE) * CEIL_SIZE;
    if (!ValidateAll()) {
        return ge::GRAPH_FAILED;
    }
    // 设置基本的tiling数据
    tilingData_.set_dataType(static_cast<uint8_t>(dataType_));
    tilingData_.set_hasP2(hasP2_ ? 1 : 0);
    tilingData_.set_K(xShape_.GetDim(INDEX_ZERO));
    if (!CheckGroupList()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_M(xShape_.GetDim(INDEX_ONE));
    tilingData_.set_N(xShape_.GetDim(INDEX_TWO));
    tilingData_.set_clipRatio(*clipRatio_);
    tilingData_.set_clearPadding(clearPadding_ != nullptr && *clearPadding_ ? 1 : 0);
    // 仅当dstTypeMax不为空时才设置相关字段，否则传默认值0
    if (dstTypeMax_ != nullptr) {
        tilingData_.set_dstTypeMax(*dstTypeMax_);
        tilingData_.set_invDstTypeMax(*dstTypeMax_ != ZERO_FLOAT ? ONE_FLOAT / *dstTypeMax_ : ZERO_FLOAT);
    } else {
        tilingData_.set_dstTypeMax(ZERO_FLOAT);
        tilingData_.set_invDstTypeMax(ZERO_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

bool FlatQuantTiling::CheckGroupList()
{
    OPS_ERR_IF(groupListType_ == nullptr,
                OPS_LOGE(tilingContext_->GetNodeName(), "group_list_type attribute is null."), return false);
    OPS_ERR_IF(*groupListType_ < GROUP_LIST_TYPE_MIN || *groupListType_ > GROUP_LIST_TYPE_MAX,
                OPS_LOGE(tilingContext_->GetNodeName(), "group_list_type[%ld] must be in range [0, 2].",
                         *groupListType_), return false);

    groupNum_ = 0;
    if (!hasGroupList_) {
        tilingData_.set_groupNum(groupNum_);
        tilingData_.set_groupListType(*groupListType_);
        return true;
    }

    const size_t expectedDims = (*groupListType_ == GROUP_LIST_TYPE_TWO_DIMS) ? DIM_TWO : INDEX_ONE;
    OPS_ERR_IF(groupListShape_.GetDimNum() != expectedDims,
                OPS_LOGE(tilingContext_->GetNodeName(),
                         "group_list must be %zuD when group_list_type is %ld, got %zuD.", expectedDims,
                         *groupListType_, groupListShape_.GetDimNum()), return false);
    if (*groupListType_ == GROUP_LIST_TYPE_TWO_DIMS) {
        OPS_ERR_IF(groupListShape_.GetDim(INDEX_ONE) != DIM_TWO,
                    OPS_LOGE(tilingContext_->GetNodeName(), "group_list shape must be [G, 2] for type 2."),
                    return false);
    }
    groupNum_ = groupListShape_.GetDim(INDEX_ZERO);
    OPS_ERR_IF(groupNum_ < 0 || groupNum_ > MAX_GROUP_NUM,
                OPS_LOGE(tilingContext_->GetNodeName(), "group_list dim0[%ld] must not exceed %ld.", groupNum_,
                         MAX_GROUP_NUM), return false);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_groupListType(*groupListType_);
    return true;
}

ge::graphStatus FlatQuantTiling::CalculateIterBatch()
{
    auto compileInfo = tilingContext_->GetCompileInfo<FlatQuantCompileInfo>();
    int64_t aicNum = compileInfo->aicNum;
    int64_t aivNum = compileInfo->aivNum;
    GetKernelMode(aivNum);
    if (mmMode_ == MM_HIGH_MODE && GetTCubeTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 计算迭代批次
    uint64_t p1Size = nAlign_ * nAlign_ * DATA_TYPE_SIZE;
    uint64_t p2Size = mAlign_ * mAlign_ * DATA_TYPE_SIZE;
    uint64_t xInputSize = mAlign_ * nAlign_ * DATA_TYPE_SIZE;
    uint64_t xL0CSize = mAlign_ * nAlign_ * sizeof(float);
    uint64_t iterBatchL1 = (L1_SIZE - p1Size - p2Size) / xInputSize / FACTOR_TWO;
    uint64_t iterBatchL0 = L0_SIZE / std::max(mAlign_, nAlign_) / BASE_K / DATA_TYPE_SIZE / FACTOR_TWO;
    uint64_t iterBatchL0C = L0C_SIZE / xL0CSize / FACTOR_TWO;
    uint64_t iterBatchK = CeilA2B(xShape_.GetDim(INDEX_ZERO), aicNum); // 根据K计算的batch
    iterBatch_ = std::max(std::min({iterBatchL1, iterBatchL0, iterBatchL0C, iterBatchK}), 1UL);
    tilingData_.set_iterBatch(iterBatch_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::SetTilingContextAndSaveData()
{
    auto compileInfo = tilingContext_->GetCompileInfo<FlatQuantCompileInfo>();
    tilingContext_->SetBlockDim(compileInfo->aicNum);
    tilingContext_->SetTilingKey(mmMode_);
    // 保存tiling数据
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::RunBigKernelTiling()
{
    // 初始化输入和属性
    if (InitializeInputsAndAttributes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 设置基本的tiling数据
    if (SetBasicTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 计算迭代批次
    if (CalculateIterBatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 设置tiling上下文并保存数据
    if (SetTilingContextAndSaveData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint8_t FlatQuantTiling::DetermineMmMode(ge::DataType outDtype) const
{
    int64_t N = xShape_.GetDim(INDEX_TWO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t K = xShape_.GetDim(INDEX_ZERO);

    if (outDtype == ge::DT_FLOAT4_E2M1) {
        if ((N % CEIL_SIZE == 0) && (M % CEIL_SIZE == 0)) {
            return MM_HIGH_MODE_ALIGN;
        }
        return MM_HIGH_MODE;
    }

    if (FLAT_QUANT_IS_REGBASE) {
        return MM_HIGH_MODE;
    }

    if (M == 1 && N % NUM_EIGHT == 0) {
        return MM_ONE_MODE;
    }
    if (mAlign_ <= BASE_SIZE / FACTOR_TWO && nAlign_ <= BASE_SIZE && K > 1) {
        return MM_DOUBLE_MODE;
    }
    if (mAlign_ * mAlign_ + nAlign_ * nAlign_ + FACTOR_TWO * FACTOR_TWO * mAlign_ * nAlign_ > L1_SIZE / BYTE_LEN_2) {
        return MM_HIGH_MODE;
    }
    if (mAlign_ > BASE_SIZE || nAlign_ > BASE_SIZE) {
        return MM_SPLIT_MODE;
    }
    return MM_BASE_MODE;
}

void FlatQuantTiling::CalculateWorkspace(int64_t aivNum, ge::DataType outDtype)
{
    int64_t N = xShape_.GetDim(INDEX_TWO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t K = xShape_.GetDim(INDEX_ZERO);

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(WORKSPACE_NUM);

    if (outDtype == ge::DT_FLOAT4_E2M1) {
        if (FLAT_QUANT_IS_REGBASE) {
            int64_t useAivNum = CeilA2B(K, K_PER_VEC) <= aivNum ? CeilA2B(K, K_PER_VEC) : aivNum;
            workspaces[0] = useAivNum *
                                (K_PER_VEC * M * N * BYTE_LEN_2 + FACTOR_TWO * K_PER_VEC * mAlign_ * N * BYTE_LEN_4) +
                            WORK_SPACE_SIZE_APT;
        }
        return;
    }

    int64_t alignedK = K;
    if (mmMode_ == MM_HIGH_MODE) {
        int64_t useAivNum = CeilA2B(K, K_PER_VEC) <= aivNum ? CeilA2B(K, K_PER_VEC) : aivNum;
        workspaces[0] = useAivNum *
                            (K_PER_VEC * M * N * BYTE_LEN_2 + FACTOR_TWO * K_PER_VEC * mAlign_ * N * BYTE_LEN_4) +
                        WORK_SPACE_SIZE;
    } else if (mmMode_ == MM_DOUBLE_MODE) {
        alignedK += (alignedK % FACTOR_TWO);
        workspaces[0] = (alignedK * mAlign_ * N + mAlign_ * mAlign_) * BYTE_LEN_2 + WORK_SPACE_SIZE;
    } else if (mmMode_ == MM_ONE_MODE) {
        workspaces[0] = alignedK * mAlign_ * nAlign_ * BYTE_LEN_2 + WORK_SPACE_SIZE;
    } else {
        workspaces[0] = (alignedK * mAlign_ * N) * BYTE_LEN_2 + WORK_SPACE_SIZE;
    }
}

void FlatQuantTiling::GetKernelMode(int64_t aivNum)
{
    auto outDesc = tilingContext_->GetOutputDesc(INDEX_ZERO);
    auto outDtype = outDesc->GetDataType();

    mmMode_ = DetermineMmMode(outDtype);
    CalculateWorkspace(aivNum, outDtype);
}

ge::graphStatus FlatQuantTiling::GetTCubeTiling()
{
    int64_t N = xShape_.GetDim(INDEX_TWO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t K = xShape_.GetDim(INDEX_ZERO);
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType_);

    matmul_tiling::MatmulApiTiling mmTilingR;
    mmTilingR.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetOrgShape(K * M, N, N);
    mmTilingR.SetShape(K_PER_VEC * M, N, N);
    if (hasP2_ && mmTilingR.GetTiling(tilingData_.matmulTilingR) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling mmTilingL;
    mmTilingL.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingL.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingL.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTilingL.SetOrgShape(M, N, M);
    mmTilingL.SetShape(M, N, M);
    if (mmTilingL.GetTiling(tilingData_.matmulTilingL) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FlatQuantTiling::CheckShapes() const
{
    OPS_ERR_IF(xShape_.GetDimNum() != DIM_THREE,
                OPS_LOGE(tilingContext_->GetNodeName(), "x shape dims must be 3, got %zu.", xShape_.GetDimNum()),
                return false);
    OPS_ERR_IF(p1Shape_.GetDimNum() != DIM_TWO,
                OPS_LOGE(tilingContext_->GetNodeName(), "p1 shape dims must be 2, got %zu.", p1Shape_.GetDimNum()),
                return false);
    OPS_ERR_IF(hasP2_ && p2Shape_.GetDimNum() != DIM_TWO,
                OPS_LOGE(tilingContext_->GetNodeName(), "p2 shape dims must be 2, got %zu.", p2Shape_.GetDimNum()),
                return false);

    int64_t K = xShape_.GetDim(INDEX_ZERO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t N = xShape_.GetDim(INDEX_TWO);
    OPS_ERR_IF(K > MAX_K_SIZE || M > MAX_MN_SIZE || N > MAX_MN_SIZE,
                OPS_LOGE(tilingContext_->GetNodeName(), "K[%ld]/M[%ld]/N[%ld] exceeds limit %d/%d/%d.", K, M, N,
                        MAX_K_SIZE, MAX_MN_SIZE, MAX_MN_SIZE),
                return false);
    OPS_ERR_IF(p1Shape_.GetDim(INDEX_ZERO) != M || p1Shape_.GetDim(INDEX_ONE) != M,
                OPS_LOGE(tilingContext_->GetNodeName(), "p1 shape must be [%ld,%ld], got [%ld,%ld].", M, M,
                        p1Shape_.GetDim(INDEX_ZERO), p1Shape_.GetDim(INDEX_ONE)),
                return false);
    bool isP2ZeroShape = (p2Shape_.GetDim(INDEX_ZERO) == 0 && p2Shape_.GetDim(INDEX_ONE) == 0);
    bool isP2NormalShape = (p2Shape_.GetDim(INDEX_ZERO) == N && p2Shape_.GetDim(INDEX_ONE) == N);
    OPS_ERR_IF(!isP2ZeroShape && !isP2NormalShape,
                OPS_LOGE(tilingContext_->GetNodeName(), "p2 shape must be [0,0] or [%ld,%ld], got [%ld,%ld].", N, N,
                        p2Shape_.GetDim(INDEX_ZERO), p2Shape_.GetDim(INDEX_ONE)),
                return false);

    return true;
}

bool FlatQuantTiling::CheckClipRatio() const
{
    OPS_ERR_IF(clipRatio_ == nullptr, OPS_LOGE(tilingContext_->GetNodeName(), "clip_ratio attribute is null."),
                return false);
    OPS_ERR_IF(*clipRatio_ <= ZERO_FLOAT || *clipRatio_ > ONE_FLOAT,
                OPS_LOGE(tilingContext_->GetNodeName(), "clip_ratio must be in (0, 1], got %f.", *clipRatio_),
                return false);
    return true;
}

bool FlatQuantTiling::CheckDstDtype(ge::DataType outDtype) const
{
    if (dstDtype_ != nullptr && outDtype == ge::DT_FLOAT4_E2M1) {
        OPS_ERR_IF(*dstDtype_ != ge::DT_FLOAT4_E2M1,
                    OPS_LOGE(tilingContext_->GetNodeName(), "dst_dtype mismatch for FLOAT4 output."), return false);
    }
    return true;
}

bool FlatQuantTiling::CheckDstTypeMax(ge::DataType outDtype) const
{
    if (dstTypeMax_ != nullptr && outDtype == ge::DT_FLOAT4_E2M1) {
        float localDstTypeMax = *dstTypeMax_;
        OPS_ERR_IF(
            localDstTypeMax != ZERO_FLOAT && (localDstTypeMax < SIX_FLOAT || localDstTypeMax > TWELVE_FLOAT),
            OPS_LOGE(tilingContext_->GetNodeName(), "dst_type_max[%f] must be 0 or in range [6, 12].", localDstTypeMax),
            return false);
    }
    return true;
}

bool FlatQuantTiling::ValidateAll() const
{
    if (!CheckShapes()) {
        return false;
    }
    if (!CheckClipRatio()) {
        return false;
    }
    auto outDesc = tilingContext_->GetOutputDesc(INDEX_ZERO);
    OPS_LOG_E_IF_NULL(tilingContext_->GetNodeName(), outDesc, return ge::GRAPH_FAILED);
    auto outDtype = outDesc->GetDataType();
    if (!CheckDstDtype(outDtype)) {
        return false;
    }
    if (!CheckDstTypeMax(outDtype)) {
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline auto FlatQuantTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

static ge::graphStatus Tiling4FlatQuantTiling(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOGE("FlatQuant", "FlatQuant context is null"),
                    return ge::GRAPH_FAILED);
    FlatQuantTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOGE("FlatQuant", "FlatQuant context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOGE("FlatQuant", "FlatQuant platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    auto compileInfo = context->GetCompiledInfo<FlatQuantCompileInfo>();
    OPS_LOG_E_IF_NULL(context->GetNodeName(), compileInfo, return ge::GRAPH_FAILED);

    compileInfo->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->npuArch = ascendcPlatform.GetCurNpuArch();

    OPS_LOGI("FlatQuant", "parse compile info success arch:%d, aicNum:%ld, aivNum:%ld",
            static_cast<int>(compileInfo->npuArch), compileInfo->aicNum, compileInfo->aivNum);
    OPS_ERR_IF(compileInfo->aicNum <= 0,
                OPS_LOGE(context->GetNodeName(), "FlatQuant is not supported for aicNum <=0 "),
                return ge::GRAPH_FAILED);
    if (FLAT_QUANT_IS_REGBASE) {
        OPS_ERR_IF(compileInfo->aivNum != (compileInfo->aicNum * 2), // aivNum must == aicNum*2
                    OPS_LOGE(context->GetNodeName(), "FlatQuantTiling is only supported for aivNum == aicNum*2 "),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlatQuant).Tiling(Tiling4FlatQuantTiling).TilingParse<FlatQuantCompileInfo>(TilingPrepareTiling);
} // namespace optiling
