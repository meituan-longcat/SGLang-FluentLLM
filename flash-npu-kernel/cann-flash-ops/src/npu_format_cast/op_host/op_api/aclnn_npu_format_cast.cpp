/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_npu_format_cast.h"

#include <algorithm>
#include <cstdint>
#include <new>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "op_error_check_local.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

constexpr int64_t NZ_N0 = 16;
constexpr int64_t BLOCK_BYTES = 32;
constexpr size_t MIN_ND_DIM = 2;
constexpr size_t MAX_ND_DIM = 6;
constexpr size_t MIN_NZ_DIM = 4;
constexpr size_t MAX_NZ_DIM = 8;

bool IsNzFormat(Format format)
{
    return format == Format::FORMAT_FRACTAL_NZ || format == Format::FORMAT_FRACTAL_NZ_C0_2 ||
           format == Format::FORMAT_FRACTAL_NZ_C0_4 || format == Format::FORMAT_FRACTAL_NZ_C0_8 ||
           format == Format::FORMAT_FRACTAL_NZ_C0_16 || format == Format::FORMAT_FRACTAL_NZ_C0_32;
}

bool IsNdLikeFormat(Format format)
{
    return format == Format::FORMAT_ND || format == Format::FORMAT_NCL || format == Format::FORMAT_NCHW ||
           format == Format::FORMAT_NCDHW;
}

bool IsSupportedDtype(DataType dtype, bool regbase)
{
    switch (dtype) {
        case DataType::DT_FLOAT:
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
        case DataType::DT_INT8:
        case DataType::DT_UINT8:
        case DataType::DT_INT32:
            return true;
        case DataType::DT_UINT32:
            return !regbase;
        case DataType::DT_HIFLOAT8:
        case DataType::DT_FLOAT8_E4M3FN:
        case DataType::DT_FLOAT4_E2M1:
        case DataType::DT_FLOAT4_E1M2:
            return regbase;
        default:
            return false;
    }
}

int64_t GetC0ByDtype(DataType dtype)
{
    if (dtype == DataType::DT_FLOAT4_E2M1 || dtype == DataType::DT_FLOAT4_E1M2) {
        return 64;
    }
    const int byteSize = ge::GetSizeByDataType(dtype);
    if (byteSize <= 0 || byteSize > BLOCK_BYTES) {
        return 0;
    }
    return BLOCK_BYTES / byteSize;
}

int64_t CeilDivPositive(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

bool ShapeHasZero(const Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

aclnnStatus CheckCommonTensor(const aclTensor* tensor, const char* name, bool requireContiguous)
{
    FORMAT_CAST_CHECK(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s is nullptr.", name);
    FORMAT_CAST_CHECK(IsSupportedDtype(tensor->GetDataType(), Ops::Transformer::AclnnUtil::IsRegbase()),
                      ACLNN_ERR_PARAM_INVALID, "The dtype of %s is not supported", name);
    if (requireContiguous) {
        FORMAT_CAST_CHECK(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID, "%s must be contiguous.", name);
    }
    FORMAT_CAST_CHECK(!ShapeHasZero(tensor->GetStorageShape()), ACLNN_ERR_PARAM_INVALID,
                      "%s cannot be empty, each axis of storageShape cannot be 0", name);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckCalculateSizeAndFormatInputs(const aclTensor* srcTensor, int dstFormat, int additionalDtype,
                                              int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat)
{
    FORMAT_CAST_CHECK(dstShape != nullptr && dstShapeSize != nullptr && actualFormat != nullptr,
                      ACLNN_ERR_PARAM_NULLPTR, "dstShape, dstShapeSize or actualFormat is nullptr.");
    auto ret = CheckCommonTensor(srcTensor, "srcTensor", false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const auto srcFormat = srcTensor->GetStorageFormat();
    const auto targetFormat = static_cast<Format>(dstFormat);
    const auto viewDim = srcTensor->GetViewShape().GetDimNum();
    const auto storageDim = srcTensor->GetStorageShape().GetDimNum();
    const bool ndToNz = IsNdLikeFormat(srcFormat) && targetFormat == Format::FORMAT_FRACTAL_NZ;
    const bool nzToNd = IsNzFormat(srcFormat) && targetFormat == Format::FORMAT_ND;
    FORMAT_CAST_CHECK(ndToNz || nzToNd, ACLNN_ERR_RUNTIME_ERROR,
                      "aclnnNpuFormatCastCalculateSizeAndFormat unsupported format transformation");

    if (ndToNz) {
        FORMAT_CAST_CHECK(viewDim >= MIN_ND_DIM && viewDim <= MAX_ND_DIM, ACLNN_ERR_PARAM_INVALID,
                          "The ViewShape dim of srcTensor must be between 2 and 6");
    } else {
        FORMAT_CAST_CHECK(storageDim >= MIN_NZ_DIM && storageDim <= MAX_NZ_DIM, ACLNN_ERR_PARAM_INVALID,
                          "The StorageShape dim of srcTensor must be between 4 and 8 for input tensor in Nz format");
    }

    if (!Ops::Transformer::AclnnUtil::IsRegbase() || !ndToNz) {
        return ACLNN_SUCCESS;
    }

    const auto srcDtype = srcTensor->GetDataType();
    const auto extraDtype = static_cast<DataType>(additionalDtype);
    FORMAT_CAST_CHECK(IsSupportedDtype(extraDtype, true), ACLNN_ERR_PARAM_INVALID,
                      "The current socVersion does not support additionalDtype.");
    if (srcDtype == DataType::DT_INT32) {
        FORMAT_CAST_CHECK(extraDtype == DataType::DT_FLOAT16 || extraDtype == DataType::DT_BF16 ||
                              extraDtype == DataType::DT_INT8,
                          ACLNN_ERR_PARAM_INVALID,
                          "The value of additionalDtype must be float16/bfloat16/int8 when the dtype of srcTensor is int32");
    }
    if (srcDtype == DataType::DT_FLOAT) {
        FORMAT_CAST_CHECK(extraDtype == DataType::DT_FLOAT16 || extraDtype == DataType::DT_BF16 ||
                              extraDtype == DataType::DT_FLOAT8_E4M3FN || extraDtype == DataType::DT_HIFLOAT8 ||
                              extraDtype == DataType::DT_UINT8,
                          ACLNN_ERR_PARAM_INVALID,
                          "additionalDtype must be float16, bfloat16, float8_e4m3fn, hifloat8 or uint8 when the dtype of srcTensor is float32");
    }
    if (extraDtype != srcDtype) {
        FORMAT_CAST_CHECK(viewDim == 2 || viewDim == 3, ACLNN_ERR_PARAM_INVALID,
                          "The srcTensor must be 2 or 3 when additionalDtype is not equal to srcTensor's dtype");
    } else {
        FORMAT_CAST_CHECK(viewDim >= MIN_ND_DIM && viewDim <= MAX_ND_DIM, ACLNN_ERR_PARAM_INVALID,
                          "The shape dim of srcTensor must be between 2 and 6 when additionalDtype equals the dtype of srcTensor");
        FORMAT_CAST_CHECK(srcTensor->GetViewShape().GetDim(viewDim - 1) != 1, ACLNN_ERR_PARAM_INVALID,
                          "The axis k of srcTensor cannot be equal to 1 when additionalDtype equals the dtype of srcTensor");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGetWorkSpaceSizeInputs(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    auto ret = CheckCommonTensor(srcTensor, "srcTensor", true);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckCommonTensor(dstTensor, "dstTensor", true);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    FORMAT_CAST_CHECK(srcTensor->GetViewShape() == dstTensor->GetViewShape(), ACLNN_ERR_PARAM_INVALID,
                      "The view shapes of srcTensor and dstTensor must be equal.");

    const auto srcFormat = srcTensor->GetStorageFormat();
    const auto dstFormat = dstTensor->GetStorageFormat();
    const bool ndToNz = IsNdLikeFormat(srcFormat) && IsNzFormat(dstFormat);
    const bool nzToNd = IsNzFormat(srcFormat) && IsNdLikeFormat(dstFormat);
    FORMAT_CAST_CHECK(ndToNz || nzToNd, ACLNN_ERR_PARAM_INVALID,
                      "Only ND to FRACTAL_NZ and FRACTAL_NZ to ND are supported.");

    const auto srcViewDim = srcTensor->GetViewShape().GetDimNum();
    const auto nzStorageDim = ndToNz ? dstTensor->GetStorageShape().GetDimNum()
                                    : srcTensor->GetStorageShape().GetDimNum();
    FORMAT_CAST_CHECK(srcViewDim >= MIN_ND_DIM && srcViewDim <= MAX_ND_DIM, ACLNN_ERR_PARAM_INVALID,
                      "The ViewShape dim must be between 2 and 6.");
    FORMAT_CAST_CHECK(nzStorageDim >= MIN_NZ_DIM && nzStorageDim <= MAX_NZ_DIM, ACLNN_ERR_PARAM_INVALID,
                      "The StorageShape dim of the NZ tensor must be between 4 and 8.");
    return ACLNN_SUCCESS;
}

const aclTensor* NormalizeTransInput(const aclTensor* srcTensor, aclOpExecutor* executor)
{
    const auto storageFormat = srcTensor->GetStorageFormat();
    // The common torch_npu path already carries canonical ND/FRACTAL_NZ
    // descriptors. Reusing it avoids an executor-side view allocation on
    // every format cast; only alias formats need a normalized descriptor.
    const bool canonicalFormat = storageFormat == Format::FORMAT_ND || storageFormat == Format::FORMAT_FRACTAL_NZ;
    if (canonicalFormat && srcTensor->GetOriginalShape() == srcTensor->GetViewShape()) {
        return srcTensor;
    }
    auto normalized = executor->CreateView(srcTensor, srcTensor->GetViewShape(), srcTensor->GetViewOffset());
    if (normalized == nullptr) {
        return nullptr;
    }
    if (IsNzFormat(storageFormat)) {
        // NZ -> ND keeps the physical storage shape, while TransData uses the
        // logical shape and logical base format carried by the view.
        normalized->SetStorageShape(srcTensor->GetStorageShape());
        normalized->SetOriginalShape(srcTensor->GetViewShape());
        normalized->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
    } else {
        // ND -> NZ normalizes NCL/NCHW/NCDHW descriptors to a logical ND view.
        normalized->SetStorageShape(srcTensor->GetViewShape());
        normalized->SetOriginalShape(srcTensor->GetViewShape());
        normalized->SetStorageFormat(Format::FORMAT_ND);
        normalized->SetOriginalFormat(Format::FORMAT_ND);
        normalized->SetViewFormat(Format::FORMAT_ND);
    }
    return normalized;
}

int64_t GetNzC0(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    const aclTensor* nzTensor = IsNzFormat(srcTensor->GetStorageFormat()) ? srcTensor : dstTensor;
    const auto& storageShape = nzTensor->GetStorageShape();
    if (storageShape.GetDimNum() < MIN_NZ_DIM) {
        return 0;
    }
    return storageShape.GetDim(storageShape.GetDimNum() - 1);
}

} // namespace

extern "C" {

aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNpuFormatCast, DFX_IN(srcTensor), DFX_OUT(dstTensor));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckGetWorkSpaceSizeInputs(srcTensor, dstTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* transInput = NormalizeTransInput(srcTensor, uniqueExecutor.get());
    CHECK_RET(transInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    dstTensor->SetViewShape(srcTensor->GetViewShape());
    if (IsNzFormat(srcTensor->GetStorageFormat())) {
        // torch_npu may represent a rank-3+ logical ND tensor as NCL/NCHW.
        // Preserve that logical format on the view, but force the physical
        // destination to ND for TransData.
        const auto logicalFormat = dstTensor->GetViewFormat();
        auto mutableInput = const_cast<aclTensor*>(transInput);
        mutableInput->SetOriginalFormat(logicalFormat);
        mutableInput->SetViewFormat(logicalFormat);
        dstTensor->SetStorageFormat(Format::FORMAT_ND);
    } else {
        dstTensor->SetViewFormat(srcTensor->GetViewFormat());
        dstTensor->SetOriginalFormat(srcTensor->GetOriginalFormat());
        dstTensor->SetOriginalShape(srcTensor->GetOriginalShape());
    }

    const auto dstFormat = IsNzFormat(dstTensor->GetStorageFormat()) ? Format::FORMAT_FRACTAL_NZ
                                                                    : dstTensor->GetStorageFormat();
    const int64_t c0 = GetNzC0(srcTensor, dstTensor);
    const aclTensor* transResult = nullptr;
    const int64_t inputDtypeSize = ge::GetSizeByDataType(srcTensor->GetDataType());
    const bool useTransDataSpecial = c0 == NZ_N0 && inputDtypeSize > 3 && !Ops::Transformer::AclnnUtil::IsRegbase();
    if (useTransDataSpecial) {
        transResult = l0op::TransDataSpecial(transInput, dstFormat, 1, uniqueExecutor.get());
    } else {
        transResult = l0op::TransData(transInput, dstFormat, 1, uniqueExecutor.get());
    }
    CHECK_RET(transResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mutableResult = const_cast<aclTensor*>(transResult);
    // Keep TransData's inferred logical shapes and align only the public
    // formats before ViewCopy, matching the installed implementation.
    mutableResult->SetStorageFormat(dstTensor->GetStorageFormat());
    mutableResult->SetOriginalFormat(dstTensor->GetOriginalFormat());
    mutableResult->SetViewFormat(dstTensor->GetViewFormat());
    const auto viewCopyResult = l0op::ViewCopy(transResult, dstTensor, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNpuFormatCast);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat,
                                                     int additionalDtype, int64_t** dstShape,
                                                     uint64_t* dstShapeSize, int* actualFormat)
{
    auto ret = CheckCalculateSizeAndFormatInputs(srcTensor, dstFormat, additionalDtype, dstShape, dstShapeSize,
                                                 actualFormat);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const auto srcFormat = srcTensor->GetStorageFormat();
    if (IsNzFormat(srcFormat)) {
        const auto& viewShape = srcTensor->GetViewShape();
        *dstShapeSize = viewShape.GetDimNum();
        auto shape = new (std::nothrow) int64_t[*dstShapeSize]();
        CHECK_RET(shape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        for (size_t i = 0; i < viewShape.GetDimNum(); ++i) {
            shape[i] = viewShape.GetDim(i);
        }
        *dstShape = shape;
        *actualFormat = static_cast<int>(Format::FORMAT_ND);
        return ACLNN_SUCCESS;
    }

    const auto& viewShape = srcTensor->GetViewShape();
    const bool regbase = Ops::Transformer::AclnnUtil::IsRegbase();
    const auto srcDtype = srcTensor->GetDataType();
    const auto extraDtype = static_cast<DataType>(additionalDtype);
    int64_t c0 = GetC0ByDtype(srcDtype);
    int format = static_cast<int>(Format::FORMAT_FRACTAL_NZ);
    if (regbase && extraDtype != srcDtype) {
        c0 = 16;
        if (extraDtype == DataType::DT_FLOAT16 || extraDtype == DataType::DT_BF16) {
            format = static_cast<int>(Format::FORMAT_FRACTAL_NZ_C0_16);
        } else if (extraDtype == DataType::DT_INT8 || extraDtype == DataType::DT_FLOAT8_E4M3FN) {
            c0 = 32;
            format = static_cast<int>(Format::FORMAT_FRACTAL_NZ_C0_32);
        }
    }
    FORMAT_CAST_CHECK(c0 > 0, ACLNN_ERR_PARAM_INVALID, "Failed to infer C0 from dtype.");

    *dstShapeSize = viewShape.GetDimNum() + 2;
    auto shape = new (std::nothrow) int64_t[*dstShapeSize]();
    CHECK_RET(shape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const size_t prefixDims = viewShape.GetDimNum() - 2;
    for (size_t i = 0; i < prefixDims; ++i) {
        shape[i] = viewShape.GetDim(i);
    }
    shape[prefixDims] = CeilDivPositive(viewShape.GetDim(viewShape.GetDimNum() - 1), c0);
    shape[prefixDims + 1] = CeilDivPositive(viewShape.GetDim(viewShape.GetDimNum() - 2), NZ_N0);
    shape[prefixDims + 2] = NZ_N0;
    shape[prefixDims + 3] = c0;
    *dstShape = shape;
    *actualFormat = format;
    return ACLNN_SUCCESS;
}

} // extern "C"
