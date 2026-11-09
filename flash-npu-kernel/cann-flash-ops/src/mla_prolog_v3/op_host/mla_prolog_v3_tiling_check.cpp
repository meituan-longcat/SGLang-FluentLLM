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
 * \file mla_prolog_v3_tiling_check.cpp
 * \brief
 */

#include "../op_kernel/mla_prolog_v3_policy.h"

#include "mla_prolog_v3_tiling_check.h"
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
#if MLA_PROLOG_V3_SFA1
namespace fluent_mla_prolog_v3 {
#endif

template <typename E>
std::string ElemToString(const E &elem)
{
    return std::to_string(elem);
}

std::string FormatToString(const ge::Format format)
{
    return std::string(ge::GetFormatName(format));
}

template <typename C, typename Func = std::string (*)(const typename C::value_type &)>
std::string ConvertContainerToString(const C &container, Func func = ElemToString<typename C::value_type>)
{
    if (container.empty() || func == nullptr) {
        return "[]";
    }
    std::stringstream ss;
    ss << "[";
    bool isFirst = true;
    for (const auto &elem : container) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << func(elem);
        isFirst = false;
    }
    ss << "]";
    return ss.str();
}

// =================================全量参数校验=================================
ge::graphStatus MlaPrologV3TilingCheck::CheckDims() const
{
    OPS_ERR_IF(baseShapeInfo_.bSize > MAX_B_SIZE,
        OPS_LOG_E(context_.opName, "B should not be greater than %u, got %u.",
            MAX_B_SIZE, baseShapeInfo_.bSize),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.s1Size > MAX_S1_SIZE,
        OPS_LOG_E(context_.opName, "S should not be greater than %u, got %u.",
            MAX_S1_SIZE, baseShapeInfo_.s1Size),
        return ge::GRAPH_FAILED);
#if MLA_PROLOG_V3_SFA1
    const std::set<uint32_t> supportedHeSize {7168U, 7680U, 3072U, 8192U};
#else
    const std::set<uint32_t> supportedHeSize {3072U, 4096U, 6144U, 7168U, 7680U, 8192U};
#endif
    OPS_ERR_IF(supportedHeSize.find(baseShapeInfo_.heSize) == supportedHeSize.end(),
        OPS_LOG_E(context_.opName, "He allows only %s, got %u.",
            ConvertContainerToString(supportedHeSize).c_str(), baseShapeInfo_.heSize),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.hcqSize != HCQ_SIZE,
        OPS_LOG_E(context_.opName, "Hcq allows only %u, got %u.",
            HCQ_SIZE, baseShapeInfo_.hcqSize),
        return ge::GRAPH_FAILED);
    const std::set<uint32_t> supportedNSize {1, 2, 4, 8, 16, 32, 64, 128};
    OPS_ERR_IF((supportedNSize.find(baseShapeInfo_.nSize) == supportedNSize.end()),
        OPS_LOG_E(context_.opName, "N allows only %s, but got %u.",
            ConvertContainerToString(supportedNSize).c_str(), baseShapeInfo_.nSize),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.hckvSize != HCKV_SIZE,
        OPS_LOG_E(context_.opName, "Hckv allows only %u, got %u.",
            HCKV_SIZE, baseShapeInfo_.hckvSize),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.dSize != D_SIZE,
        OPS_LOG_E(context_.opName, "D allows only %u, got %u.",
            D_SIZE, baseShapeInfo_.dSize),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.drSize != DR_SIZE,
        OPS_LOG_E(context_.opName, "Dr allows only %u, got %u.",
            DR_SIZE, baseShapeInfo_.drSize),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.nkvSize != NKV_SIZE,
        OPS_LOG_E(context_.opName, "Nkv allows only %u, got %u.",
            NKV_SIZE, baseShapeInfo_.nkvSize),
        return ge::GRAPH_FAILED);
    uint32_t supportedDtileSize = baseShapeInfo_.hckvSize;
    if (*(context_.ckvkrRepoMode) == static_cast<int>(CKV_KR_REPO_MODE::COMBINE)) {
#if MLA_PROLOG_V3_SFA1
        supportedDtileSize += baseShapeInfo_.drSize * 2;
#else
        supportedDtileSize += baseShapeInfo_.drSize *
            ge::GetSizeByDataType(context_.krCache.desc->GetDataType()) /
            ge::GetSizeByDataType(context_.kvCache.desc->GetDataType());
#endif
    }
    if (*(context_.quantScaleRepoMode) == static_cast<int>(QUANT_SCALE_REPO_MODE::COMBINE)) {
        supportedDtileSize += baseShapeInfo_.hckvSize / static_cast<uint32_t>(*(context_.tileSize)) * 4;
    }
#if MLA_PROLOG_V3_SFA1
    OPS_ERR_IF(baseShapeInfo_.dtileSize != supportedDtileSize,
        OPS_LOG_E(context_.opName, "dtileSize allows only %u, got %u.",
            supportedDtileSize, baseShapeInfo_.dtileSize),
#else
    // Packed C8 stores [512B INT8 C-KV, 128B BF16 RoPE, 16B FP32
    // per-tile scales].  Keep the 656-byte payload ABI, but also allow a
    // 32-byte-aligned 672-byte physical row.  The kernel already derives the
    // GM row stride from dtileSize and writes only the payload fields, so the
    // extra 16 bytes are tail padding.  Do not relax any other cache layout.
    const uint32_t alignedDtileSize =
        (supportedDtileSize + 31U) / 32U * 32U;
    const bool allowAlignedPackedC8 =
        supportedDtileSize == 656U &&
        alignedDtileSize == 672U &&
        *(context_.kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE) &&
        *(context_.ckvkrRepoMode) ==
            static_cast<int>(CKV_KR_REPO_MODE::COMBINE) &&
        *(context_.quantScaleRepoMode) ==
            static_cast<int>(QUANT_SCALE_REPO_MODE::COMBINE) &&
        baseShapeInfo_.dtileSize == alignedDtileSize;
    OPS_ERR_IF(baseShapeInfo_.dtileSize != supportedDtileSize &&
            !allowAlignedPackedC8,
        OPS_LOG_E(context_.opName,
            "dtileSize allows %u (payload) or %u (aligned packed C8), got %u.",
            supportedDtileSize, alignedDtileSize, baseShapeInfo_.dtileSize),
#endif
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(baseShapeInfo_.tSize > MAX_T_SIZE,
        OPS_LOG_E(context_.opName, "T should not be greater than %u, got %u.",
            MAX_T_SIZE, baseShapeInfo_.tSize),
        return ge::GRAPH_FAILED);
#if MLA_PROLOG_V3_SFA1
    // curKNope/curKRope 容量防御断言：
    // 新 dtype 契约下 cur_k dtype = 对应 cache 数据真实 dtype，行容量与数据字节恒等；
    // 正常路径恒真（CheckDims 上方已保证 dtileSize 与允许值一致），触发即实现缺陷信号，
    // 业务上不发生——仅 OPS_LOG_E 告警，不拒绝任何场景。
    const uint32_t nopeByteSize = baseShapeInfo_.hckvSize * NUM_BYTES_BF16;
    const uint32_t dtileByteSize = baseShapeInfo_.dtileSize * NUM_BYTES_BF16;
    if (dtileByteSize < nopeByteSize) {
        OPS_LOG_E(context_.opName,
            "curKNope capacity defensive assert failed: dtileBytes(%u) < nopeBytes(%u).",
            dtileByteSize, nopeByteSize);
    }
    if (baseShapeInfo_.drSize == 0) {
        OPS_LOG_E(context_.opName, "curKRope capacity defensive assert failed: drSize is 0.");
    }
#endif
    return ge::GRAPH_SUCCESS;
}

void MlaPrologV3TilingCheck::GenExpectedParamInfo()
{
    FillCommonParamInfo();
    FillScenarioParamInfo();
#if !MLA_PROLOG_V3_SFA1
    if (*(context_.ckvkrRepoMode) == static_cast<int>(CKV_KR_REPO_MODE::COMBINE)) {
        expectedParamInfo_.erase(KR_CACHE_NAME);
        expectedParamInfo_.erase(KR_CACHE_OUT_NAME);
    }
#endif
}

void MlaPrologV3TilingCheck::FillCommonParamInfo()
{
    FillRequiredParamShapeWithDims();

    if (context_.weightDq.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_DQ_NAME].dimNum = MLA_PROLOG_V3_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightDq.desc->GetDataType());
        expectedParamInfo_[WEIGHT_DQ_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.hcqSize) / weightAxisSize,
                static_cast<int64_t>(baseShapeInfo_.heSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }
    if (context_.weightUqQr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_UQ_QR_NAME].dimNum = MLA_PROLOG_V3_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightUqQr.desc->GetDataType());
        expectedParamInfo_[WEIGHT_UQ_QR_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.headSizeUqQr) / weightAxisSize,
                static_cast<int64_t>(baseShapeInfo_.hcqSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }
    if (context_.weightDkvKr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_DKV_KR_NAME].dimNum = MLA_PROLOG_V3_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightDkvKr.desc->GetDataType());
        expectedParamInfo_[WEIGHT_DKV_KR_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) / weightAxisSize,
                static_cast<int64_t>(baseShapeInfo_.heSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }

    expectedParamInfo_[WEIGHT_DQ_NAME].format = ge::FORMAT_FRACTAL_NZ;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].format = ge::FORMAT_FRACTAL_NZ;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].format = ge::FORMAT_FRACTAL_NZ;
}

void MlaPrologV3TilingCheck::FillRequiredParamShapeWithDims()
{
    if (scenarioInfo_.batchSeqFusedFlag_) {
        expectedParamInfo_.emplace(TOKEN_X_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.heSize});
        expectedParamInfo_.emplace(ROPE_SIN_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.drSize});
        expectedParamInfo_.emplace(ROPE_COS_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.drSize});
        if ((scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BSND) || (scenarioInfo_.cacheMode_ == CACHE_MODE::PA_NZ)) {
            expectedParamInfo_.emplace(CACHE_INDEX_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize});
        }

        expectedParamInfo_.emplace(QUERY_NAME,
            std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, baseShapeInfo_.hckvSize});
        expectedParamInfo_.emplace(QUERY_ROPE_NAME,
            std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, baseShapeInfo_.drSize});
    } else {
        expectedParamInfo_.emplace(TOKEN_X_NAME,
            std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.heSize});
        expectedParamInfo_.emplace(ROPE_SIN_NAME,
            std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.drSize});
        expectedParamInfo_.emplace(ROPE_COS_NAME,
            std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.drSize});
        if ((scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BSND) || (scenarioInfo_.cacheMode_ == CACHE_MODE::PA_NZ)) {
            expectedParamInfo_.emplace(CACHE_INDEX_NAME,
                std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size});
        }

        expectedParamInfo_.emplace(QUERY_NAME,
            std::vector<uint32_t>{baseShapeInfo_.bSize,
                baseShapeInfo_.s1Size, baseShapeInfo_.nSize, baseShapeInfo_.hckvSize});
        expectedParamInfo_.emplace(QUERY_ROPE_NAME,
            std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                baseShapeInfo_.nSize, baseShapeInfo_.drSize});
    }
    expectedParamInfo_.emplace(WEIGHT_DQ_NAME, std::vector<uint32_t>{baseShapeInfo_.heSize, baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(WEIGHT_UQ_QR_NAME,
        std::vector<uint32_t>{baseShapeInfo_.hcqSize, baseShapeInfo_.headSizeUqQr});
    expectedParamInfo_.emplace(WEIGHT_UK_NAME, std::vector<uint32_t>{baseShapeInfo_.nSize, baseShapeInfo_.dSize,
        baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(WEIGHT_DKV_KR_NAME, std::vector<uint32_t>{baseShapeInfo_.heSize,
        baseShapeInfo_.hckvSize + baseShapeInfo_.drSize});
    expectedParamInfo_.emplace(RMSNORM_GAMMA_CQ_NAME, std::vector<uint32_t>{baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(RMSNORM_GAMMA_CKV_NAME, std::vector<uint32_t>{baseShapeInfo_.hckvSize});
    if (scenarioInfo_.cacheMode_ == CACHE_MODE::TND) {
        expectedParamInfo_.emplace(KV_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize,
            baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
        expectedParamInfo_.emplace(KR_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize,
            baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    } else if (scenarioInfo_.cacheMode_ == CACHE_MODE::BSND) {
        expectedParamInfo_.emplace(KV_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
            baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
        expectedParamInfo_.emplace(KR_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
            baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    } else {
        expectedParamInfo_.emplace(KV_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.blockNum,
            baseShapeInfo_.blockSize, baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
        expectedParamInfo_.emplace(KR_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.blockNum,
            baseShapeInfo_.blockSize, baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    }
    expectedParamInfo_.emplace(KV_CACHE_OUT_NAME, expectedParamInfo_[KV_CACHE_NAME]);
    expectedParamInfo_.emplace(KR_CACHE_OUT_NAME, expectedParamInfo_[KR_CACHE_NAME]);
#if MLA_PROLOG_V3_SFA1
    // CANN 8.5.0 内置 GeOp def 仅注册 12 个 attr，索引越界 → nullptr，按 true 兜底（与 proto/tiling 一致）
    if (context_.outputCurKNopeRopeFlag == nullptr || *context_.outputCurKNopeRopeFlag) {
        if (scenarioInfo_.cacheMode_ == CACHE_MODE::TND) {
            expectedParamInfo_.emplace(CUR_K_NOPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
            expectedParamInfo_.emplace(CUR_K_ROPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
        } else if (scenarioInfo_.cacheMode_ == CACHE_MODE::BSND) {
            expectedParamInfo_.emplace(CUR_K_NOPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                    baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
            expectedParamInfo_.emplace(CUR_K_ROPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                    baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
        } else if (scenarioInfo_.batchSeqFusedFlag_) {
            // PA_* + BS 合轴（tokenX 2D）：按 token 展平，以 tSize x nkvSize x dtile/Dr 校验
            expectedParamInfo_.emplace(CUR_K_NOPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
            expectedParamInfo_.emplace(CUR_K_ROPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
        } else {
            // PA_*：BS 非合轴（tokenX 3D [B,S,He]），与 proto（isBsMerge=false）及 torchair 一致输出 4D。
            // 严格按 InferShape 分支 [B, S, Nkv, Dtile/Dr] 校验，勿按 token 展平。
            expectedParamInfo_.emplace(CUR_K_NOPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                    baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
            expectedParamInfo_.emplace(CUR_K_ROPE_NAME,
                std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                    baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
        }
    } else {
        expectedParamInfo_.emplace(CUR_K_NOPE_NAME, std::vector<uint32_t>{1});   // 占位 [1]
        expectedParamInfo_.emplace(CUR_K_ROPE_NAME, std::vector<uint32_t>{1});
    }
#endif
}

void MlaPrologV3TilingCheck::FillScenarioParamInfo()
{
    switch (scenarioInfo_.quantMode_) {
        case QUANT_MODE::NO_QUANT:
            FillNonQuantParamInfo();
            break;
        case QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT:
            FillPartialQuantParamInfo();
            break;
        case QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL:
            FillPartialKVQuantParamInfo();
            break;
        case QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE:
            FillPartialKVPertileQuantParamInfo();
            break;
        case QUANT_MODE::FULL_QUANT_KV_NO_QUANT:
            FillFullQuantParamInfo();
            break;
        case QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR:
            FillFullKVQuantParamInfo();
            break;
        case QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE:
            FillFullKVPertileQuantParamInfo();
#if !MLA_PROLOG_V3_SFA1
            break;
        case QUANT_MODE::NO_QUANT_KV_QUANT_PER_TILE:
            FillNoWeightKVPertileQuantParamInfo();
#endif
            break;
        default:
            break;
    }
    if (scenarioInfo_.cacheMode_ == CACHE_MODE::BSND || scenarioInfo_.cacheMode_ == CACHE_MODE::TND) {
        expectedParamInfo_.emplace(CACHE_INDEX_NAME, context_.cacheIndex);
        expectedParamInfo_[CACHE_INDEX_NAME].isValid = false;
    } else {
        expectedParamInfo_[CACHE_INDEX_NAME].isValid = true;
        expectedParamInfo_[CACHE_INDEX_NAME].dtype = ge::DT_INT64;
    }
}

void MlaPrologV3TilingCheck::FillNonQuantParamInfo()
{
    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_UK_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[RMSNORM_GAMMA_CQ_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[RMSNORM_GAMMA_CKV_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[ROPE_SIN_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[ROPE_COS_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KR_CACHE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[QUERY_ROPE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KR_CACHE_OUT_NAME].dtype = ge::DT_BF16;
#if MLA_PROLOG_V3_SFA1
    expectedParamInfo_[CUR_K_NOPE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[CUR_K_ROPE_NAME].dtype = ge::DT_BF16;
#endif

    if (!scenarioInfo_.isV1Flag_) {
        // 仅校验dequantScaleQNope有传入
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, context_.dequantScaleQNope);
        expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].isValid = true;
    }
}

void MlaPrologV3TilingCheck::FillPartialQuantParamInfo()
{
    FillNonQuantParamInfo();

    expectedParamInfo_.emplace(DEQUANT_SCALE_W_UQ_QR_NAME, std::vector<uint32_t>{1, baseShapeInfo_.headSizeUqQr});
    expectedParamInfo_.emplace(SMOOTH_SCALES_CQ_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hcqSize});

    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_INT8;

    expectedParamInfo_[DEQUANT_SCALE_W_UQ_QR_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].dtype = ge::DT_FLOAT;

    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].isValid = (context_.smoothScalesCq.desc != nullptr);
}

void MlaPrologV3TilingCheck::FillPartialKVQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(QUANT_SCALE_CKR_NAME, std::vector<uint32_t>{1, baseShapeInfo_.drSize});

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KR_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KR_CACHE_OUT_NAME].dtype = ge::DT_INT8;
#if MLA_PROLOG_V3_SFA1
    expectedParamInfo_[CUR_K_NOPE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[CUR_K_ROPE_NAME].dtype = ge::DT_INT8;
#endif

    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[QUANT_SCALE_CKR_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologV3TilingCheck::FillPartialKVPertileQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, std::vector<uint32_t>{1});
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].dtype = ge::DT_FLOAT;
#if MLA_PROLOG_V3_SFA1
    expectedParamInfo_[CUR_K_NOPE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[CUR_K_ROPE_NAME].dtype = ge::DT_BF16;
#else

    expectedParamInfo_.erase(KR_CACHE_NAME);
    expectedParamInfo_.erase(KR_CACHE_OUT_NAME);
}

void MlaPrologV3TilingCheck::FillNoWeightKVPertileQuantParamInfo()
{
    FillNonQuantParamInfo();

    expectedParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, std::vector<uint32_t>{1});
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].dtype = ge::DT_FLOAT;
#endif

    expectedParamInfo_.erase(KR_CACHE_NAME);
    expectedParamInfo_.erase(KR_CACHE_OUT_NAME);
}

void MlaPrologV3TilingCheck::FillFullQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(DEQUANT_SCALE_X_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, 1});
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME,
        std::vector<uint32_t>{1, baseShapeInfo_.hckvSize + baseShapeInfo_.drSize});

    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[DEQUANT_SCALE_X_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[DEQUANT_SCALE_W_DQ_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[DEQUANT_SCALE_W_DKV_KR_NAME].dtype = ge::DT_FLOAT;
#if MLA_PROLOG_V3_SFA1
    expectedParamInfo_[CUR_K_NOPE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[CUR_K_ROPE_NAME].dtype = ge::DT_BF16;
#endif
}

void MlaPrologV3TilingCheck::FillFullKVQuantParamInfo()
{
    FillFullQuantParamInfo();
    expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1});
    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME] =
        ParamInfo(std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, 1});

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].dtype = ge::DT_FLOAT;
#if MLA_PROLOG_V3_SFA1
    expectedParamInfo_[CUR_K_NOPE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[CUR_K_ROPE_NAME].dtype = ge::DT_BF16;
#endif

    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].isValid = true;
}

void MlaPrologV3TilingCheck::FillFullKVPertileQuantParamInfo()
{
    FillFullQuantParamInfo();
    expectedParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, std::vector<uint32_t>{1});
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].dtype = ge::DT_FLOAT;
#if MLA_PROLOG_V3_SFA1
    expectedParamInfo_[CUR_K_NOPE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[CUR_K_ROPE_NAME].dtype = ge::DT_BF16;
#endif

    expectedParamInfo_.erase(KR_CACHE_NAME);
    expectedParamInfo_.erase(KR_CACHE_OUT_NAME);
}

void MlaPrologV3TilingCheck::GenActualParamInfo()
{
    actualParamInfo_.emplace(TOKEN_X_NAME, context_.tokenX);
    actualParamInfo_.emplace(WEIGHT_DQ_NAME, context_.weightDq);
    actualParamInfo_.emplace(WEIGHT_UQ_QR_NAME, context_.weightUqQr);
    actualParamInfo_.emplace(WEIGHT_UK_NAME, context_.weightUk);
    actualParamInfo_.emplace(WEIGHT_DKV_KR_NAME, context_.weightDkvKr);
    actualParamInfo_.emplace(RMSNORM_GAMMA_CQ_NAME, context_.rmsnormGammaCq);
    actualParamInfo_.emplace(RMSNORM_GAMMA_CKV_NAME, context_.rmsnormGammaCkv);
    actualParamInfo_.emplace(ROPE_SIN_NAME, context_.ropeSin);
    actualParamInfo_.emplace(ROPE_COS_NAME, context_.ropeCos);
    actualParamInfo_.emplace(CACHE_INDEX_NAME, context_.cacheIndex);
    actualParamInfo_.emplace(KV_CACHE_NAME, context_.kvCache);
    actualParamInfo_.emplace(KR_CACHE_NAME, context_.krCache);
    actualParamInfo_.emplace(DEQUANT_SCALE_X_NAME, context_.dequantScaleX);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME, context_.dequantScaleWDq);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_UQ_QR_NAME, context_.dequantScaleWUqQr);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME, context_.dequantScaleWDkvKr);
    actualParamInfo_.emplace(QUANT_SCALE_CKV_NAME, context_.quantScaleCkv);
    actualParamInfo_.emplace(QUANT_SCALE_CKR_NAME, context_.quantScaleCkr);
    actualParamInfo_.emplace(SMOOTH_SCALES_CQ_NAME, context_.smoothScalesCq);
    actualParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, context_.kNopeClipAlpha);
    actualParamInfo_.emplace(QUERY_NAME, context_.query);
    actualParamInfo_.emplace(QUERY_ROPE_NAME, context_.queryRope);
    actualParamInfo_.emplace(KV_CACHE_OUT_NAME, context_.kvCacheOut);
    actualParamInfo_.emplace(KR_CACHE_OUT_NAME, context_.krCacheOut);
    actualParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, context_.dequantScaleQNope);
#if MLA_PROLOG_V3_SFA1
    // cur_k_nope / cur_k_rope outputs may not be instantiated in graph mode (torchair builds the
    // node from the torch schema which exposes only 7 outputs). Only check them when the output
    // descriptor actually exists, otherwise skip the check.
    if (context_.curKNope.desc != nullptr) {
        actualParamInfo_.emplace(CUR_K_NOPE_NAME, context_.curKNope);
    }
    if (context_.curKRope.desc != nullptr) {
        actualParamInfo_.emplace(CUR_K_ROPE_NAME, context_.curKRope);
    }
    if (scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE) {
#else
    if (*(context_.ckvkrRepoMode) == static_cast<int>(CKV_KR_REPO_MODE::COMBINE) ||
        scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT_KV_QUANT_PER_TILE) {
#endif
        actualParamInfo_.erase(KR_CACHE_NAME);
        actualParamInfo_.erase(KR_CACHE_OUT_NAME);
    }
}

ge::graphStatus MlaPrologV3TilingCheck::CheckParamByScenario()
{
    GenExpectedParamInfo();
    GenActualParamInfo();
    ge::graphStatus isCorrect {ge::GRAPH_SUCCESS};
    for (const auto &it : actualParamInfo_) {
        const auto &expectedParam {expectedParamInfo_[it.first]};
        if (__builtin_expect((expectedParam != it.second), 0)) {
            isCorrect = ge::GRAPH_FAILED;
            if (expectedParam.isValid != it.second.isValid) {
                OPS_LOG_E(context_.opName, "%s expected %s, got %s.",
                    it.first.c_str(),
                    expectedParam.isValid ? "not null" : "null",
                    it.second.isValid ? "not null" : "null");
                continue;
            }
            if (expectedParam.dtype != it.second.dtype) {
                OPS_LOG_E(context_.opName, "%s expected dtype %s, but got %s.",
                    it.first.c_str(),
                    TypeUtils::DataTypeToSerialString(expectedParam.dtype).c_str(),
                    TypeUtils::DataTypeToSerialString(it.second.dtype).c_str());
            }
            if (expectedParam.format != it.second.format) {
                OPS_LOG_E(context_.opName, "%s expected format %s, but got %s.",
                    it.first.c_str(),
                    ge::GetFormatName(expectedParam.format),
                    ge::GetFormatName(it.second.format));
            }
            if (expectedParam.shape != it.second.shape) {
                OPS_LOG_E(context_.opName, "%s expected shape %s, but got %s.",
                    it.first.c_str(),
                    ConvertContainerToString(expectedParam.shape).c_str(),
                    ConvertContainerToString(it.second.shape).c_str());
            }
        }
    }
    return isCorrect;
}

ge::graphStatus MlaPrologV3TilingCheck::CheckScenarParam()
{
    ge::graphStatus isCorrect {ge::GRAPH_SUCCESS};

    if (scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE ||
#if MLA_PROLOG_V3_SFA1
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE) {
#else
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT_KV_QUANT_PER_TILE) {
#endif
        if (*(context_.ckvkrRepoMode) != static_cast<int>(CKV_KR_REPO_MODE::COMBINE)) {
            OPS_LOG_E(context_.opName, "The ckvkrRepoMode expected %d, but got %d.",
                static_cast<int>(CKV_KR_REPO_MODE::COMBINE), *(context_.ckvkrRepoMode));
            isCorrect = ge::GRAPH_FAILED;
        }
        if (*(context_.quantScaleRepoMode) != static_cast<int>(QUANT_SCALE_REPO_MODE::COMBINE)) {
            OPS_LOG_E(context_.opName, "The quantScaleRepoMode expected %d, but got %d.",
                static_cast<int>(QUANT_SCALE_REPO_MODE::COMBINE), *(context_.quantScaleRepoMode));
            isCorrect = ge::GRAPH_FAILED;
        }
    } else {
#if MLA_PROLOG_V3_SFA1
        if (*(context_.ckvkrRepoMode) != static_cast<int>(CKV_KR_REPO_MODE::DIVIDE)) {
            OPS_LOG_E(context_.opName, "The ckvkrRepoMode expected %d, but got %d.",
                static_cast<int>(CKV_KR_REPO_MODE::DIVIDE), *(context_.ckvkrRepoMode));
#else
        const bool validNonQuantCombine = scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT &&
            scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BSND &&
            *(context_.ckvkrRepoMode) == static_cast<int>(CKV_KR_REPO_MODE::COMBINE);
        if (!validNonQuantCombine &&
            *(context_.ckvkrRepoMode) != static_cast<int>(CKV_KR_REPO_MODE::DIVIDE)) {
            OPS_LOG_E(context_.opName,
                "The ckvkrRepoMode COMBINE only supports per-tile KV quantization or non-quantized PA_BSND.");
#endif
            isCorrect = ge::GRAPH_FAILED;
        }
        if (*(context_.quantScaleRepoMode) != static_cast<int>(QUANT_SCALE_REPO_MODE::DIVIDE)) {
            OPS_LOG_E(context_.opName, "The quantScaleRepoMode expected %d, but got %d.",
                static_cast<int>(QUANT_SCALE_REPO_MODE::DIVIDE), *(context_.quantScaleRepoMode));
            isCorrect = ge::GRAPH_FAILED;
        }
#if !MLA_PROLOG_V3_SFA1
    }
    if (!*(context_.enableRope) && scenarioInfo_.quantMode_ != QUANT_MODE::NO_QUANT) {
        OPS_LOG_E(context_.opName, "Disabling RoPE only supports non-quantized BF16 inputs and weights.");
        isCorrect = ge::GRAPH_FAILED;
#endif
    }
    if (scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL) {
        if (*(context_.queryQuantMode) != static_cast<int>(QUERY_QUANT_MODE::PER_TOKEN_HEAD)) {
            OPS_LOG_E(context_.opName, "The queryQuantMode expected %d, but got %d.",
                static_cast<int>(QUERY_QUANT_MODE::PER_TOKEN_HEAD), *(context_.queryQuantMode));
            isCorrect = ge::GRAPH_FAILED;
        }
    } else {
        if (*(context_.queryQuantMode) != static_cast<int>(QUERY_QUANT_MODE::NO_QUANT)) {
            OPS_LOG_E(context_.opName, "The queryQuantMode expected %d, but got %d.",
                static_cast<int>(QUERY_QUANT_MODE::NO_QUANT), *(context_.queryQuantMode));
            isCorrect = ge::GRAPH_FAILED;
        }
    }
    return isCorrect;
}
// =================================全量参数校验=================================

// ==================================单参数校验==================================
bool MlaPrologV3TilingCheck::IsSingleParamValid(const RequiredParaInfo &param,
    const std::string &paramName, const std::set<ge::DataType> &expectedDtype,
    const std::set<ge::Format> &expectedFormat, const std::set<size_t> &expectedDimNum) const
{
    OPS_ERR_IF((param.shape == nullptr) || (param.desc == nullptr),
        OPS_LOG_E(context_.opName, "%s should not be null.", paramName.c_str()), return false);

    ge::DataType dtype = param.desc->GetDataType();
    OPS_ERR_IF((expectedDtype.find(dtype) == expectedDtype.end()),
        OPS_LOG_E(context_.opName, "%s datatype only supports %s, but got %s.",
            paramName.c_str(),
            ConvertContainerToString(expectedDtype, TypeUtils::DataTypeToSerialString).c_str(),
            TypeUtils::DataTypeToSerialString(dtype).c_str()),
        return false);

    ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(param.desc->GetStorageFormat()));
    OPS_ERR_IF((expectedFormat.find(format) == expectedFormat.end()),
        OPS_LOG_E(context_.opName, "%s format only supports %s, but got %s.",
            paramName.c_str(),
            ConvertContainerToString(expectedFormat, FormatToString).c_str(),
            ge::GetFormatName(format)),
        return false);

    size_t dimNum = param.shape->GetStorageShape().GetDimNum();
    OPS_ERR_IF((expectedDimNum.find(dimNum) == expectedDimNum.end()),
        OPS_LOG_E(context_.opName, "%s dim num supports only %s, but got %zu.",
            paramName.c_str(), ConvertContainerToString(expectedDimNum).c_str(), dimNum),
        return false);
    return true;
}

ge::graphStatus MlaPrologV3TilingCheck::CheckSingleRequiredParam() const
{
    if (!CheckTokenX() || !CheckWDq() || !CheckWUqQr() || !CheckWUk() || !CheckWDkvKr() || !CheckRmsnormGammaCq() ||
        !CheckRmsnormGammaCkv() || !CheckRopeSin() || !CheckRopeCos() || !CheckKvCache() ||
        !CheckKrCache() || !CheckActSeqLen()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool MlaPrologV3TilingCheck::CheckTokenX() const
{
    return IsSingleParamValid(context_.tokenX, TOKEN_X_NAME,
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologV3TilingCheck::CheckWDq() const
{
    return IsSingleParamValid(context_.weightDq, WEIGHT_DQ_NAME,
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_FRACTAL_NZ}, {2, 4});
}

bool MlaPrologV3TilingCheck::CheckWUqQr() const
{
    return IsSingleParamValid(context_.weightUqQr, WEIGHT_UQ_QR_NAME,
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_FRACTAL_NZ}, {2, 4});
}

bool MlaPrologV3TilingCheck::CheckWUk() const
{
    return IsSingleParamValid(context_.weightUk, WEIGHT_UK_NAME,
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3});
}

bool MlaPrologV3TilingCheck::CheckWDkvKr() const
{
    return IsSingleParamValid(context_.weightDkvKr, WEIGHT_DKV_KR_NAME,
#if MLA_PROLOG_V3_SFA1
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_FRACTAL_NZ}, {2, 3, 4, 5});
#else
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_FRACTAL_NZ}, {2, 4});
#endif
}

bool MlaPrologV3TilingCheck::CheckRmsnormGammaCq() const
{
    return IsSingleParamValid(context_.rmsnormGammaCq, RMSNORM_GAMMA_CQ_NAME,
        {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1});
}

bool MlaPrologV3TilingCheck::CheckRmsnormGammaCkv() const
{
    return IsSingleParamValid(context_.rmsnormGammaCkv, RMSNORM_GAMMA_CKV_NAME,
        {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1});
}

bool MlaPrologV3TilingCheck::CheckRopeSin() const
{
    return IsSingleParamValid(context_.ropeSin, ROPE_SIN_NAME,
        {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologV3TilingCheck::CheckRopeCos() const
{
    return IsSingleParamValid(context_.ropeCos, ROPE_COS_NAME,
        {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologV3TilingCheck::CheckKvCache() const
{
    return IsSingleParamValid(context_.kvCache, KV_CACHE_NAME,
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3, 4});
}

bool MlaPrologV3TilingCheck::CheckKrCache() const
{
#if MLA_PROLOG_V3_SFA1
    if (scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE) {
#else
    if (*(context_.ckvkrRepoMode) == static_cast<int>(CKV_KR_REPO_MODE::COMBINE) ||
        scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE ||
        scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT_KV_QUANT_PER_TILE) {
#endif
        return true;
    }
    return IsSingleParamValid(context_.krCache, KR_CACHE_NAME,
        {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3, 4});
}

ge::graphStatus MlaPrologV3TilingCheck::CheckCacheMode() const
{
    ge::graphStatus isCorrect {ge::GRAPH_SUCCESS};
    if (std::strncmp(context_.cacheMode, CACHE_MODE_TND, CACHE_MODE_TND_LEN) == 0) {
        if (context_.tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_3) {
            OPS_LOG_E(context_.opName,
                      "When tokenX dim is 3, Only support cacheMode (BSND, PA_BSND, PA_NZ), actually is %s.",
                      context_.cacheMode);
            isCorrect = ge::GRAPH_FAILED;
        }
        if (context_.kvCache.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_4) {
            OPS_LOG_E(context_.opName,
                      "When KVCache dim is 4, Only support cacheMode (BSND, PA_BSND, PA_NZ), actually is %s.",
                      context_.cacheMode);
            isCorrect = ge::GRAPH_FAILED;
        }
    }
    if (std::strncmp(context_.cacheMode, CACHE_MODE_BSND, CACHE_MODE_BSND_LEN) == 0) {
        if (context_.tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_2) {
            OPS_LOG_E(context_.opName,
                      "When tokenX dim is 2, Only support cacheMode (TND, PA_BSND, PA_NZ), actually is %s.",
                      context_.cacheMode);
            isCorrect = ge::GRAPH_FAILED;
        }
        if (context_.kvCache.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_3) {
            OPS_LOG_E(context_.opName,
                      "When KVCache dim is 3, Only support cacheMode (TND, PA_BSND, PA_NZ), actually is %s.",
                      context_.cacheMode);
            isCorrect = ge::GRAPH_FAILED;
        }
    }
    if ((std::strncmp(context_.cacheMode, CACHE_MODE_BSND, CACHE_MODE_BSND_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_TND, CACHE_MODE_TND_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BSND, CACHE_MODE_PA_BSND_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_NZ, CACHE_MODE_PA_NZ_LEN) != 0)) {
        OPS_LOG_E(context_.opName,
            "Only support cacheMode (BSND, TND, PA_BSND, PA_NZ), actually is %s.",
            context_.cacheMode);
        isCorrect = ge::GRAPH_FAILED;
    }
    return isCorrect;
}

ge::graphStatus MlaPrologV3TilingCheck::CheckPANZPerTile() const
{
    if ((std::strncmp(context_.cacheMode, CACHE_MODE_PA_NZ, CACHE_MODE_PA_NZ_LEN) == 0) &&
        *(context_.kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
        OPS_LOG_E(context_.opName, "Not support both cacheMode PA_NZ and pertile effective.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool MlaPrologV3TilingCheck::CheckActSeqLen() const
{
    if (context_.actualSeqLen.desc == nullptr) {
        return true;
    };
    ge::DataType dtype = context_.actualSeqLen.desc->GetDataType();
    OPS_ERR_IF((ge::DT_INT32 != dtype),
        OPS_LOG_E(context_.opName, "ActSeqLen datatype only supports %s, but got %s.",
           TypeUtils::DataTypeToSerialString(ge::DT_INT32).c_str(),
            TypeUtils::DataTypeToSerialString(dtype).c_str()),
        return false);
    return true;
}
// ==================================单参数校验==================================
} // namespace implementation
#if MLA_PROLOG_V3_SFA1
} // namespace optiling
#endif
