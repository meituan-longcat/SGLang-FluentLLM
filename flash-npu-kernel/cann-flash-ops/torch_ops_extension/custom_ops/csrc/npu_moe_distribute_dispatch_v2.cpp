/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <string>
#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

constexpr int64_t kNoQuant = 0;
constexpr int64_t kPerGroupQuant = 3;
constexpr int64_t kMxQuant = 4;
constexpr int64_t kPerGroupSize = 128;
constexpr int64_t kMxBlockSize = 32;
constexpr int64_t kCustomDtypeOffset = 256;

using DispatchOutputs =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

bool HasTensor(const c10::optional<at::Tensor>& tensor)
{
    return tensor.has_value() && tensor->defined();
}

aclDataType ResolveAclDtype(int64_t dtype)
{
    if (dtype >= kCustomDtypeOffset) {
        return static_cast<aclDataType>(dtype - kCustomDtypeOffset);
    }
    TORCH_CHECK(dtype >= 0 && dtype <= static_cast<int64_t>(at::ScalarType::NumOptions),
                "invalid dtype value: ", dtype);
    const aclDataType aclDtype = ConvertType(static_cast<at::ScalarType>(dtype));
    TORCH_CHECK(aclDtype != ACL_DT_UNDEFINED, "unsupported dtype value: ", dtype);
    return aclDtype;
}

at::ScalarType GetStorageScalarType(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return at::kFloat;
        case ACL_FLOAT16:
            return at::kHalf;
        case ACL_BF16:
            return at::kBFloat16;
        case ACL_INT8:
            return at::kChar;
        case ACL_UINT8:
            return at::kByte;
        case ACL_INT16:
            return at::kShort;
        case ACL_INT32:
            return at::kInt;
        case ACL_INT64:
            return at::kLong;
        case ACL_BOOL:
            return at::kBool;
        case ACL_FLOAT8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case ACL_FLOAT8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        case ACL_FLOAT8_E8M0:
        case ACL_HIFLOAT8:
            return at::kByte;
        default:
            TORCH_CHECK(false, "unsupported ACL dtype for output storage: ", static_cast<int64_t>(dtype));
    }
}

aclDataType GetAclDtype(const at::Tensor& tensor, const c10::optional<int64_t>& dtype)
{
    return dtype.has_value() ? ResolveAclDtype(*dtype) : ConvertType(tensor.scalar_type());
}

aclDataType GetDynamicScalesAclDtype(const at::Tensor& x, const c10::optional<at::Tensor>& scales,
                                     int64_t quantMode, const c10::optional<int64_t>& scalesDtype)
{
    if (quantMode == kMxQuant) {
        return ACL_FLOAT8_E8M0;
    }
    if (quantMode == kNoQuant && x.scalar_type() != at::kBFloat16 && x.scalar_type() != at::kHalf &&
        HasTensor(scales)) {
        return GetAclDtype(*scales, scalesDtype);
    }
    return ACL_FLOAT;
}

std::vector<int64_t> GetDynamicScalesShape(const c10::optional<at::Tensor>& scales, int64_t quantMode,
                                           int64_t a, int64_t h, int64_t tpWorldSize)
{
    if (tpWorldSize == 0) {
        return {a};
    }
    if (tpWorldSize != 1) {
        return {a * tpWorldSize};
    }
    if (quantMode == kNoQuant && HasTensor(scales)) {
        // Packed INT32/INT4 transport supplies one scale per token.  The
        // regular non-quantized scale path supplies [token, scale-group].
        if (scales->dim() == 1) {
            return {a};
        }
        TORCH_CHECK(scales->dim() >= 2, "scales must be at least 2-D for grouped scale transport");
        return {a * scales->size(1)};
    }
    if (quantMode == kPerGroupQuant) {
        return {a, (h + kPerGroupSize - 1) / kPerGroupSize};
    }
    if (quantMode == kMxQuant) {
        const int64_t blocks = (h + kMxBlockSize - 1) / kMxBlockSize;
        return {a, ((blocks + 1) / 2) * 2};
    }
    return {a};
}

struct DispatchShapeInfo {
    int64_t bs;
    int64_t h;
    int64_t k;
    int64_t a;
    int64_t localExpertNum;
    int64_t globalBs;
    int64_t epRecvCountNum;
};

DispatchShapeInfo InferDispatchShape(const at::Tensor& x, const at::Tensor& expertIds, int64_t epWorldSize,
                                     int64_t epRankId, int64_t moeExpertNum, int64_t tpWorldSize,
                                     int64_t expertShardType, int64_t sharedExpertNum,
                                     int64_t sharedExpertRankNum, int64_t globalBs,
                                     int64_t expertTokenNumsType, const c10::optional<at::Tensor>& elasticInfo)
{
    TORCH_CHECK(x.dim() == 2 && expertIds.dim() == 2, "x and expert_ids must be 2-D tensors");
    TORCH_CHECK(epWorldSize > 0, "ep_world_size must be greater than 0");
    TORCH_CHECK(epRankId >= 0 && epRankId < epWorldSize, "ep_rank_id must be in [0, ep_world_size)");
    TORCH_CHECK(tpWorldSize >= 0, "tp_world_size must be non-negative");
    TORCH_CHECK(sharedExpertRankNum >= 0 && sharedExpertRankNum < epWorldSize,
                "shared_expert_rank_num must be in [0, ep_world_size)");
    const bool sharedDefault = sharedExpertNum == 1 && sharedExpertRankNum == 0;
    const bool noShared = sharedExpertNum == 0 && sharedExpertRankNum == 0;
    const bool validShared = sharedExpertNum > 0 && sharedExpertRankNum / sharedExpertNum > 0 &&
                             sharedExpertRankNum % sharedExpertNum == 0;
    TORCH_CHECK(sharedDefault || noShared || validShared,
                "invalid shared_expert_num/shared_expert_rank_num combination");
    TORCH_CHECK(expertTokenNumsType == 0 || expertTokenNumsType == 1,
                "expert_token_nums_type must be 0 or 1");

    const int64_t bs = x.size(0);
    const int64_t h = x.size(1);
    const int64_t k = expertIds.size(1);
    const bool sharedFront = expertShardType == 0;
    const int64_t globalBsReal = globalBs == 0 ? bs * epWorldSize : globalBs;
    const bool isSharedExpert = sharedFront && epRankId < sharedExpertRankNum;
    int64_t localExpertNum = 1;
    int64_t a = 0;

    if (isSharedExpert) {
        const int64_t maxBs = globalBsReal / epWorldSize;
        const int64_t ranksPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        const int64_t maxSharedGroups = (epWorldSize + ranksPerSharedExpert - 1) / ranksPerSharedExpert;
        a = maxBs * maxSharedGroups;
    } else {
        localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
        a = globalBsReal * std::min(localExpertNum, k);
    }

    if (sharedFront && HasTensor(elasticInfo)) {
        const int64_t regularLocalExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
        if (sharedDefault || noShared) {
            localExpertNum = std::max(localExpertNum, regularLocalExpertNum);
            a = globalBsReal * std::min(localExpertNum, k);
        } else {
            const int64_t maxBs = globalBsReal / epWorldSize;
            const int64_t ranksPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
            const int64_t maxSharedGroups = (epWorldSize + ranksPerSharedExpert - 1) / ranksPerSharedExpert;
            a = std::max(maxBs * maxSharedGroups, globalBsReal * std::min(regularLocalExpertNum, k));
            localExpertNum = std::max(localExpertNum, regularLocalExpertNum);
        }
    }

    const int64_t epRecvCountNum =
        epWorldSize * localExpertNum * (tpWorldSize == 2 ? tpWorldSize : 1);
    return {bs, h, k, a, localExpertNum, globalBsReal, epRecvCountNum};
}

DispatchOutputs AllocateDispatchOutputs(const at::Tensor& x, const at::Tensor& expertIds,
                                        const c10::optional<at::Tensor>& scales,
                                        const c10::optional<at::Tensor>& expertScales,
                                        const c10::optional<at::Tensor>& elasticInfo, int64_t epWorldSize,
                                        int64_t epRankId, int64_t moeExpertNum, int64_t tpWorldSize,
                                        int64_t expertShardType, int64_t sharedExpertNum,
                                        int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
                                        int64_t expertTokenNumsType, const c10::optional<int64_t>& yDtype,
                                        const c10::optional<int64_t>& scalesDtype)
{
    auto shape = InferDispatchShape(x, expertIds, epWorldSize, epRankId, moeExpertNum, tpWorldSize,
                                    expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs,
                                    expertTokenNumsType, elasticInfo);
    at::ScalarType outputDtype = at::kChar;
    if (quantMode == kNoQuant) {
        outputDtype = x.scalar_type();
    } else if (yDtype.has_value()) {
        outputDtype = GetStorageScalarType(ResolveAclDtype(*yDtype));
    }
    const aclDataType dynamicScalesAclDtype =
        GetDynamicScalesAclDtype(x, scales, quantMode, scalesDtype);
    const at::ScalarType dynamicScalesDtype = GetStorageScalarType(dynamicScalesAclDtype);
    const int64_t expandedRows = std::max(shape.a, shape.a * tpWorldSize);
    auto options = x.options();
    at::Tensor expandX = at::empty({expandedRows, shape.h}, options.dtype(outputDtype));
    at::Tensor dynamicScales = at::empty(
        GetDynamicScalesShape(scales, quantMode, shape.a, shape.h, tpWorldSize),
        options.dtype(dynamicScalesDtype));
    at::Tensor assistInfo =
        at::empty({std::max(shape.bs * shape.k, shape.a * 128)}, options.dtype(at::kInt));
    at::Tensor expertTokenNums = at::empty({shape.localExpertNum}, options.dtype(at::kLong));
    int64_t epRecvCountNum = shape.epRecvCountNum;
    if (HasTensor(expertScales)) {
        epRecvCountNum = epWorldSize * shape.localExpertNum +
                         2 * shape.globalBs * shape.k * (epWorldSize / 8);
    }
    at::Tensor epRecvCounts = at::empty({epRecvCountNum}, options.dtype(at::kInt));
    at::Tensor tpRecvCounts = at::empty({tpWorldSize}, options.dtype(at::kInt));
    at::Tensor expandScales = at::empty({shape.a}, options.dtype(at::kFloat));
    return {expandX, dynamicScales, assistInfo, expertTokenNums, epRecvCounts, tpRecvCounts, expandScales};
}

}  // namespace

DispatchOutputs MoeDistributeDispatchV2Npu(
    const at::Tensor& x, const at::Tensor& expertIds, c10::string_view groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& xActiveMask, const c10::optional<at::Tensor>& expertScales,
    const c10::optional<at::Tensor>& elasticInfo, c10::string_view groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, c10::string_view commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    c10::optional<int64_t> yDtype, c10::optional<int64_t> xDtype, c10::optional<int64_t> scalesDtype)
{
    auto outputs = AllocateDispatchOutputs(x, expertIds, scales, expertScales, elasticInfo, epWorldSize,
                                           epRankId, moeExpertNum, tpWorldSize, expertShardType,
                                           sharedExpertNum, sharedExpertRankNum, quantMode, globalBs,
                                           expertTokenNumsType, yDtype, scalesDtype);
    std::string groupEpString(groupEp);
    std::string groupTpString(groupTp);
    std::string commAlgString(commAlg);
    char* groupEpPtr = groupEpString.data();
    char* groupTpPtr = groupTpString.data();
    char* commAlgPtr = commAlgString.data();
    at::Tensor scalesTensor = HasTensor(scales) ? *scales : at::Tensor();
    const aclDataType dynamicScalesAclDtype =
        GetDynamicScalesAclDtype(x, scales, quantMode, scalesDtype);
    TensorWrapper xWrapper{x, GetAclDtype(x, xDtype)};
    TensorWrapper scalesWrapper{scalesTensor,
                                HasTensor(scales) ? GetAclDtype(scalesTensor, scalesDtype) : ACL_FLOAT};
    TensorWrapper expandXWrapper{std::get<0>(outputs),
                                 yDtype.has_value() ? ResolveAclDtype(*yDtype)
                                                   : ConvertType(std::get<0>(outputs).scalar_type())};
    TensorWrapper dynamicScalesWrapper{std::get<1>(outputs), dynamicScalesAclDtype};
    const int64_t globalBsReal = globalBs == 0 ? x.size(0) * epWorldSize : globalBs;
    const c10::optional<at::Tensor> performanceInfo = c10::nullopt;

    EXEC_NPU_CMD_V1(aclnnMoeDistributeDispatchV4, xWrapper, expertIds, scalesWrapper, xActiveMask,
                    expertScales, elasticInfo, performanceInfo, groupEpPtr, epWorldSize, epRankId, moeExpertNum,
                    groupTpPtr, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                    sharedExpertRankNum, quantMode, globalBsReal, expertTokenNumsType, commAlgPtr,
                    zeroExpertNum, copyExpertNum, constExpertNum, expandXWrapper, dynamicScalesWrapper,
                    std::get<2>(outputs), std::get<3>(outputs), std::get<4>(outputs), std::get<5>(outputs),
                    std::get<6>(outputs));
    return outputs;
}

DispatchOutputs MoeDistributeDispatchV2Meta(
    const at::Tensor& x, const at::Tensor& expertIds, c10::string_view, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>& expertScales,
    const c10::optional<at::Tensor>& elasticInfo, c10::string_view, int64_t tpWorldSize, int64_t,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t globalBs, int64_t expertTokenNumsType, c10::string_view, int64_t, int64_t, int64_t,
    c10::optional<int64_t> yDtype, c10::optional<int64_t>, c10::optional<int64_t> scalesDtype)
{
    return AllocateDispatchOutputs(x, expertIds, scales, expertScales, elasticInfo, epWorldSize, epRankId,
                                   moeExpertNum, tpWorldSize, expertShardType, sharedExpertNum,
                                   sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, yDtype,
                                   scalesDtype);
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_moe_distribute_dispatch_v2", &custom::MoeDistributeDispatchV2Npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_moe_distribute_dispatch_v2", &custom::MoeDistributeDispatchV2Meta);
}
