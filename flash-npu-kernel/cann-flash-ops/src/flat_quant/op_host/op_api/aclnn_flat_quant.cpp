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
 * \file aclnn_flat_quant.cpp
 * \brief
 */

#include "aclnn_flat_quant.h"
#include "aclnn_flat_quant_v2.h"
#include "aclnn_flat_quant_v3.h"
#include "aclnn_flat_quant_v4.h"
#include "flat_quant.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace {
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr int32_t DIM_NUM = 3;
static constexpr int32_t P_DIM_NUM = 2;
static constexpr int32_t SCALE_DIM_NUM = 1;
static constexpr int32_t INDEX_0 = 0;
static constexpr int32_t INDEX_1 = 1;
static constexpr int32_t INDEX_2 = 2;
static constexpr int32_t K_INDEX = INDEX_0;
static constexpr int32_t M_INDEX = INDEX_1;
static constexpr int32_t N_INDEX = INDEX_2;
static constexpr int32_t MAX_K_SIZE = 262144;
static constexpr int32_t MAX_MN_SIZE = 256;
static constexpr int64_t MAX_GROUP_NUM = 1024;
static constexpr int64_t GROUP_LIST_TYPE_MIN = 0;
static constexpr int64_t GROUP_LIST_TYPE_MAX = 2;
static constexpr int64_t GROUP_LIST_TYPE_TWO_DIMS = 2;
static constexpr double ZERO = 0.0;
static constexpr double ONE = 1.0;
static constexpr double SIX = 6.0;
static constexpr double TWELVE = 12.0;

static const std::initializer_list<op::DataType> IN_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                          op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_INT4};

static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool IsDoubleEqual(double a, double b) { return std::abs(a - b) <= std::numeric_limits<double>::epsilon(); }

static aclnnStatus Int42Int32PackedTensor(const aclTensor* y, const aclTensor*& outTensor, const op::DataType& outDtype,
                                          aclOpExecutor* executor)
{
    OP_CHECK(outDtype == op::DataType::DT_INT32, OP_LOGD("aclnnFlatQuant output does not need to pack."),
             return ACLNN_SUCCESS);

    // if outType is int32, pack output
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("aclnnFlatQuant output real dtype is int4, pack to int32 to out.");

    return ACLNN_SUCCESS;
}

static bool CheckSpecialIOShape(const aclTensor* x, const aclTensor* out, const op::DataType& outDtype)
{
    // output with data type int4/int32
    auto dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    OP_CHECK(dimNum == DIM_NUM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor should be 3D."), return false);

    OP_CHECK(out->GetViewShape().GetDimNum() == DIM_NUM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out tensor should be 3D."),
             return false);

    // other dims should be same
    for (int64_t i = 0; i < dimNum - 1; ++i) {
        OP_CHECK(x->GetViewShape().GetDim(i) == out->GetViewShape().GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dim(%ld) of x must be same with out, x is (%ld), out is (%ld).",
                         i, x->GetViewShape().GetDim(i), out->GetViewShape().GetDim(i)),
                 return false);
    }

    // check last dim
    auto xLastDim = x->GetViewShape().GetDim(dimNum - 1);
    auto outLastDim = out->GetViewShape().GetDim(dimNum - 1);
    if (outDtype == op::DataType::DT_INT32) {
        OP_CHECK(xLastDim == outLastDim * INT4_NUMS_IN_INT32_SPACE,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "if outType is int32, the out last dim must be 1/8 of x last dim,"
                         " x last dim is (%ld), out last dim is (%ld).",
                         xLastDim, outLastDim),
                 return false);
    } else {
        OP_CHECK(xLastDim == outLastDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "if outType is int4, x last dim must be equal with out last dim. "
                         "x last dim is (%ld), out last dim is (%ld).",
                         xLastDim, outLastDim),
                 return false);
    }
    return true;
}

static bool CheckDim(const aclTensor* x, const aclTensor* kroneckerP1, const aclTensor* kroneckerP2,
                     const aclTensor* out, const aclTensor* quantScale)
{
    auto xShape = x->GetViewShape();
    auto p1Shape = kroneckerP1->GetViewShape();
    auto p2Shape = kroneckerP2->GetViewShape();
    auto outShape = out->GetViewShape();
    auto quantScaleShape = quantScale->GetViewShape();
    OP_CHECK(p1Shape.GetDimNum() == P_DIM_NUM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kroneckerP1 tensor should be 2D."),
             return false);
    OP_CHECK(p2Shape.GetDimNum() == P_DIM_NUM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kroneckerP2 tensor should be 2D."),
             return false);
    OP_CHECK(xShape.GetDimNum() == DIM_NUM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor should be 3D."), return false);
    OP_CHECK(outShape.GetDimNum() == DIM_NUM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out tensor should be 3D."),
             return false);
    OP_CHECK(quantScaleShape.GetDimNum() == SCALE_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantScale tensor should be 1D."), return false);
    int64_t K = xShape.GetDim(K_INDEX);
    int64_t M = xShape.GetDim(M_INDEX);
    int64_t N = xShape.GetDim(N_INDEX);
    OP_CHECK(K >= 0 && M > 0 && N > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "x dim0 must be non-negative and dim1/dim2 must be greater than 0."),
             return false);
    OP_CHECK(K <= MAX_K_SIZE && M <= MAX_MN_SIZE && N <= MAX_MN_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim0 should not exceed %ld, x dim1 and dim2 should not exceed %ld.",
                     MAX_K_SIZE, MAX_MN_SIZE),
             return false);
    if (out->GetDataType() == op::DataType::DT_INT32) {
        OP_CHECK(N % INT4_NUMS_IN_INT32_SPACE == 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "if outType is int32, x last dim must be divisible by %ld.",
                         INT4_NUMS_IN_INT32_SPACE),
                 return false);
    } else {
        OP_CHECK(N % P_DIM_NUM == 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x last dim must be divisible by %ld.", P_DIM_NUM), return false);
    }
    OP_CHECK(p1Shape.GetDim(INDEX_0) == M && p1Shape.GetDim(INDEX_1) == M,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kroneckerP1 dim0 and dim1 should be same with x dim1."), return false);
    OP_CHECK(p2Shape.GetDim(INDEX_0) == N && p2Shape.GetDim(INDEX_1) == N,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kroneckerP2 dim0 and dim1 should be same with x dim2."), return false);
    OP_CHECK(quantScaleShape.GetDim(INDEX_0) == K,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantScale dim0 should be same with x dim0."), return false);

    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* kroneckerP1, const aclTensor* kroneckerP2,
                       const aclTensor* out, const aclTensor* quantScale)
{
    // 维度限制校验
    OP_CHECK(CheckDim(x, kroneckerP1, kroneckerP2, out, quantScale),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check input / output shape failed."), return false);

    auto outputDtype = out->GetDataType();
    OP_CHECK(CheckSpecialIOShape(x, out, outputDtype), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check output shape failed."),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* out, const aclTensor* quantScale)
{
    auto xFormat = x->GetStorageFormat();
    auto outFormat = out->GetStorageFormat();
    auto quantScaleFormat = quantScale->GetStorageFormat();
    bool noSupportFormat = ((xFormat == Format::FORMAT_FRACTAL_NZ) || (outFormat == Format::FORMAT_FRACTAL_NZ) ||
                            (quantScaleFormat == Format::FORMAT_FRACTAL_NZ));
    if (noSupportFormat) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "op [flat_quant] currently only supports ND format.");
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* kroneckerP1, const aclTensor* kroneckerP2,
                         const aclTensor* out, const aclTensor* quantScale)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(kroneckerP1, return false);
    OP_CHECK_NULL(kroneckerP2, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(quantScale, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* kroneckerP1, const aclTensor* kroneckerP2,
                            const aclTensor* out, const aclTensor* quantScale)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(kroneckerP1, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(kroneckerP2, IN_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK(std::find(OUT_DTYPE_SUPPORT_LIST.begin(), OUT_DTYPE_SUPPORT_LIST.end(), out->GetDataType()) !=
                 OUT_DTYPE_SUPPORT_LIST.end(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor out should be in dtype support list %s",
                     op::ToString(OUT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(std::find(SCALE_DTYPE_SUPPORT_LIST.begin(), SCALE_DTYPE_SUPPORT_LIST.end(), quantScale->GetDataType()) !=
                 SCALE_DTYPE_SUPPORT_LIST.end(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor scale should be in dtype support list %s",
                     op::ToString(SCALE_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK_DTYPE_NOT_SAME(x, kroneckerP1, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, kroneckerP2, return false);
    return true;
}

static bool CheckRatioValid(double clipRatio)
{
    OP_CHECK(
        clipRatio > ZERO && clipRatio <= ONE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "clipRatio must be within the range of (0,1], clipRatio [%f].", clipRatio),
        return false);
    return true;
}

static bool CheckMaxValid(double dstTypeMax)
{
    OP_CHECK(IsDoubleEqual(dstTypeMax, ZERO) || (dstTypeMax >= SIX && dstTypeMax <= TWELVE),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstTypeMax must be 0 or in range [6, 12], dstTypeMax [%f].", dstTypeMax),
             return false);
    return true;
}

static bool CheckGroupList(const aclTensor* groupListOptional, int64_t groupListType)
{
    OP_CHECK(groupListType >= GROUP_LIST_TYPE_MIN && groupListType <= GROUP_LIST_TYPE_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupListType must be in range [0, 2], got %ld.", groupListType),
             return false);
    if (groupListOptional == nullptr) {
        return true;
    }
    OP_CHECK(groupListOptional->GetDataType() == op::DataType::DT_INT64,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupListOptional dtype must be int64."), return false);
    const auto& shape = groupListOptional->GetViewShape();
    const size_t expectedDims = groupListType == GROUP_LIST_TYPE_TWO_DIMS ? 2 : 1;
    OP_CHECK(shape.GetDimNum() == expectedDims,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupListOptional must be %zuD for groupListType %ld.", expectedDims,
                     groupListType), return false);
    if (groupListType == GROUP_LIST_TYPE_TWO_DIMS) {
        OP_CHECK(shape.GetDim(1) == 2,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupListOptional shape must be [G, 2] for groupListType 2."),
                 return false);
    }
    OP_CHECK(shape.GetDim(0) >= 0 && shape.GetDim(0) <= MAX_GROUP_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupListOptional dim0 must not exceed %ld.", MAX_GROUP_NUM),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* kroneckerP1, const aclTensor* kroneckerP2,
                               const aclTensor* out, const aclTensor* quantScale)
{
    CHECK_RET(CheckNotNull(x, kroneckerP1, kroneckerP2, out, quantScale), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, kroneckerP1, kroneckerP2, out, quantScale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, kroneckerP1, kroneckerP2, out, quantScale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x, out, quantScale), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
}; // namespace

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus FlatQuantGetWorkspaceSizeImpl(const aclTensor* x, const aclTensor* kroneckerP1,
                                                 const aclTensor* kroneckerP2, const aclTensor* groupListOptional,
                                                 double clipRatio, double dstTypeMax, int64_t groupListType,
                                                 bool clearPadding, aclTensor* out, aclTensor* quantScale,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, kroneckerP1, kroneckerP2, out, quantScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckRatioValid(clipRatio), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckMaxValid(dstTypeMax), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGroupList(groupListOptional, groupListType), ACLNN_ERR_PARAM_INVALID);

    // K is the token dimension. A legal K=0 returns empty out/scale without launching a kernel.
    if (x->GetViewShape().GetDim(K_INDEX) == 0) {
        OP_LOGD("aclnnFlatQuant, dealing with zero tokens.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto p1Contiguous = l0op::Contiguous(kroneckerP1, uniqueExecutor.get());
    CHECK_RET(p1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto p2Contiguous = l0op::Contiguous(kroneckerP2, uniqueExecutor.get());
    CHECK_RET(p2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* groupListContiguous = nullptr;
    if (groupListOptional != nullptr) {
        groupListContiguous = l0op::Contiguous(groupListOptional, uniqueExecutor.get());
        CHECK_RET(groupListContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    std::tuple<aclTensor*, aclTensor*> result = l0op::FlatQuant(
        selfContiguous, p1Contiguous, p2Contiguous, groupListContiguous, static_cast<float>(clipRatio),
        static_cast<int64_t>(op::DataType::DT_INT32), static_cast<float>(dstTypeMax), groupListType, clearPadding,
        out, quantScale, uniqueExecutor.get());

    const aclTensor* resultTensor = std::get<0>(result);
    const aclTensor* quantScaleTensor = std::get<1>(result);
    CHECK_RET(resultTensor != nullptr && quantScaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果输出是Int4，需要转成Int32，8个Int4输出拼成1个Int32
    auto outDtype = out->GetDataType();
    const aclTensor* outTensor = resultTensor;
    ret = Int42Int32PackedTensor(resultTensor, outTensor, outDtype, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto outResult = l0op::ViewCopy(outTensor, out, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_PARAM_INVALID);
    auto scaleResult = l0op::ViewCopy(quantScaleTensor, quantScale, uniqueExecutor.get());
    CHECK_RET(scaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlatQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* kroneckerP1,
                                           const aclTensor* kroneckerP2, double clipRatio, aclTensor* out,
                                           aclTensor* quantScale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFlatQuant, DFX_IN(x, kroneckerP1, kroneckerP2), DFX_OUT(out, quantScale));
    return FlatQuantGetWorkspaceSizeImpl(x, kroneckerP1, kroneckerP2, nullptr, clipRatio, ZERO, 0, true, out,
                                         quantScale, workspaceSize, executor);
}

aclnnStatus aclnnFlatQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlatQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlatQuantV2GetWorkspaceSize(const aclTensor* x, const aclTensor* kroneckerP1,
                                             const aclTensor* kroneckerP2, double clipRatio, double dstTypeMax,
                                             aclTensor* out, aclTensor* quantScale, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFlatQuantV2, DFX_IN(x, kroneckerP1, kroneckerP2), DFX_OUT(out, quantScale));
    return FlatQuantGetWorkspaceSizeImpl(x, kroneckerP1, kroneckerP2, nullptr, clipRatio, dstTypeMax, 0, true, out,
                                         quantScale, workspaceSize, executor);
}

aclnnStatus aclnnFlatQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlatQuantV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlatQuantV3GetWorkspaceSize(const aclTensor* x, const aclTensor* kroneckerP1,
                                             const aclTensor* kroneckerP2, const aclTensor* groupListOptional,
                                             double clipRatio, double dstTypeMax, int64_t groupListType,
                                             aclTensor* out, aclTensor* quantScale, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFlatQuantV3, DFX_IN(x, kroneckerP1, kroneckerP2, groupListOptional),
                   DFX_OUT(out, quantScale));
    return FlatQuantGetWorkspaceSizeImpl(x, kroneckerP1, kroneckerP2, groupListOptional, clipRatio, dstTypeMax,
                                         groupListType, true, out, quantScale, workspaceSize, executor);
}

aclnnStatus aclnnFlatQuantV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlatQuantV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlatQuantV4GetWorkspaceSize(const aclTensor* x, const aclTensor* kroneckerP1,
                                             const aclTensor* kroneckerP2, const aclTensor* groupListOptional,
                                             double clipRatio, double dstTypeMax, int64_t groupListType,
                                             bool clearPadding, aclTensor* out, aclTensor* quantScale,
                                             uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFlatQuantV4, DFX_IN(x, kroneckerP1, kroneckerP2, groupListOptional),
                   DFX_OUT(out, quantScale));
    return FlatQuantGetWorkspaceSizeImpl(x, kroneckerP1, kroneckerP2, groupListOptional, clipRatio, dstTypeMax,
                                         groupListType, clearPadding, out, quantScale, workspaceSize, executor);
}

aclnnStatus aclnnFlatQuantV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlatQuantV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
