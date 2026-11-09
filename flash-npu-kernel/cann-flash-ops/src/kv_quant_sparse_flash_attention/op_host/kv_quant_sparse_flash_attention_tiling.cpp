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
 * \file kv_quant_sparse_flash_attention_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "kv_quant_sparse_flash_attention_compat.h"
#include "register/op_def_registry.h"
#include "../op_kernel/kv_quant_sparse_flash_attention_template_tiling_key.h"
#include "kv_quant_sparse_flash_attention_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_INT8}},
    {VALUE_NAME,                  {ge::DT_INT8}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {SPARSE_INDICES_NAME,         {ge::DT_INT32}}
};

const std::map<std::string, std::vector<QSFALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {QSFALayout::BSND, QSFALayout::TND}},
    {KEY_NAME,               {QSFALayout::BSND, QSFALayout::TND, QSFALayout::PA_BSND}},
    {VALUE_NAME,             {QSFALayout::BSND, QSFALayout::TND, QSFALayout::PA_BSND}},
    {ATTEN_OUT_NAME,         {QSFALayout::BSND, QSFALayout::TND}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

struct KvQuantSparseFlashAttentionCompileInfo {
    int64_t coreNum;
};

static const std::map<QSFALayout, std::vector<QSFAAxis>> QSFA_LAYOUT_AXIS_MAP = {
    {QSFALayout::BSND, {QSFAAxis::B, QSFAAxis::S, QSFAAxis::N, QSFAAxis::D}},
    {QSFALayout::TND, {QSFAAxis::T, QSFAAxis::N, QSFAAxis::D}},
    {QSFALayout::PA_BSND, {QSFAAxis::Bn, QSFAAxis::Bs, QSFAAxis::N, QSFAAxis::D}},
};

static const std::map<QSFALayout, size_t> QSFA_LAYOUT_DIM_MAP = {
    {QSFALayout::BSND, DIM_NUM_FOUR},
    {QSFALayout::TND, DIM_NUM_THREE},
    {QSFALayout::PA_BSND, DIM_NUM_FOUR},
};

static std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static bool IsNonEmptyOutputShape(const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        return false;
    }
    const gert::Shape &storageShape = shape->GetStorageShape();
    return storageShape.GetDimNum() > 1 || (storageShape.GetDimNum() == 1 && storageShape.GetDim(0) > 0);
}

static std::string QSFADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("SparseFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

string QSFATensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << QSFAShape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << QSFAShape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
            static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
            .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string QSFADebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << QSFATensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << QSFATensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

std::string QSFALayoutToSerialString(QSFALayout layout)
{
    switch (layout) {
        case QSFALayout::BSND: return "BSND";
        case QSFALayout::TND: return "TND";
        case QSFALayout::PA_BSND: return "PA_BSND";
        default: return "UNKNOWN";
    }
}

ge::graphStatus QSFAMlaTiling::SetBlockDim(uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize) const
{
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetTilingData(TilingDef &tilingData) const
{
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::GetPlatformInfo()
{
    OP_CHECK_IF(sfaaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(sfaaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(sfaaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QSFAMlaTiling::GenTilingKey()
{
    uint32_t layoutQuery = static_cast<uint32_t>(sfaaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaaInfo_->kvLayout);

    tilingKey_ = GET_TPL_TILING_KEY(0U, layoutQuery, layoutKV, perfMode_ == QSFAPerfMode::V_TEMPLATE_MODE);

    OP_LOGI(sfaaInfo_->opName, "QSFA tilingKey_: %lu.", tilingKey_);
}

void QSFAMlaTiling::ZeroTensorProcess() const
{
    if (sfaaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        sfaaInfo_->s2Size = 1024;
    }
}

void QSFAMlaTiling::InitParams()
{
    perfMode_ = QSFAPerfMode::V_TEMPLATE_MODE;
    coreNum_ = aicNum_;

    headDimAlign_ = Align(sfaaInfo_->qHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void QSFAMlaTiling::CalcUbBmm()
{
    uint32_t cubeMSize = sfaaInfo_->gSize * sfaaInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = sfaaInfo_->gSize * (headDimAlign_ + 64U) * sfaaInfo_->s1Size;
}

void QSFAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void QSFAMlaTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaaInfo_->qLayout != QSFALayout::TND) {
        if (s2Size == 256) {   // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128; // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (s2Size > 256 && s2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2; // 2:减半
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
}

void QSFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaaInfo_->s2Size);

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = sfaaInfo_->gSize;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void QSFAMlaTiling::Split()
{
    SplitBalanced();
}

void QSFAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(sfaaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(sfaaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(sfaaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(sfaaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(sfaaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(sfaaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(sfaaInfo_->n1Size / sfaaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(sfaaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(sfaaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(sfaaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(sfaaInfo_->sparseMode);
    tilingData_.baseParams.set_sparseBlockSize(sfaaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(sfaaInfo_->sparseBlockCount);
    tilingData_.baseParams.set_returnSoftmaxLse(sfaaInfo_->returnSoftmaxLse);
    tilingData_.baseParams.set_softmaxLseTnd(sfaaInfo_->softmaxLseTnd);
}

// for flash decode
void QSFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * sfaaInfo_->n2Size * mBaseSize_ * headDimAlign_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaaInfo_->n2Size * mBaseSize_ *
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void QSFAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void QSFAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void QSFAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t QSFAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{
    return actCoreNum * 2 * sfaaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void QSFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum);
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaaInfo_->blockTypeSize); // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaaInfo_->blockTypeSize;
        if (sfaaInfo_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}

void QSFAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void QSFAMlaTiling::GetWorkspaceSize()
{
    uint32_t mmResElemSize = 4;         // 4:fp32
    uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;       // 4:fp32
    uint32_t qPreProcResElemSize = 0;   // 普通场景不涉及Q预处理
    uint32_t softmaxSumElemSize = 4;   // 4:int32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    uint32_t actCoreNum = coreNum_;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
        mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
        qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize; // vec2ResGm
    // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
    //              bufNum  s2Base   D   dRope  sizeOf(half)
    workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum; // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
    // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
    workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数

    CalcFDWorkSpace(actCoreNum);
}

void QSFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(sfaaInfo_->opName, "QSFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus QSFAMlaTiling::DoOpTiling(QSFATilingInfo *sfaaInfo)
{
    sfaaInfo_ = sfaaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingKvQuantSparseFlashAttention(gert::TilingContext *context)
{
    QSFATilingInfo sfaaInfo;
    QSFAInfoParser sfaaInfoParser(context);
    if (sfaaInfoParser.Parse(sfaaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QSFATilingCheck tilingChecker(sfaaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QSFAMlaTiling tiling(context);
    return tiling.DoOpTiling(&sfaaInfo);
}

ge::graphStatus TilingPrepareForKvQuantSparseFlashAttention(gert::TilingParseContext* const context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const QSFATilingShapeCompareParam &param, const QSFALayout &layout) const
{
    if (layout == QSFALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == QSFALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == QSFALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", QSFALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CompareShape(QSFATilingShapeCompareParam &param,
    const gert::Shape &shape, const QSFALayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OP_LOGE(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OP_LOGE(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), QSFALayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void QSFATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << QSFADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), QSFADataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus QSFATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void QSFATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void QSFATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus QSFATilingCheck::CheckDimNumInLayoutSupport(const QSFALayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = QSFA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
            QSFALayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


void QSFATilingCheck::LogErrorLayoutSupport(const std::vector<QSFALayout> &expectLayoutList,
    const QSFALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << QSFALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), QSFALayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus QSFATilingCheck::CheckLayoutSupport(const QSFALayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, keyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OP_LOGE(opName_, "sparseMode must == 0/3, but got: %ld.", *opParamInfo_.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF(((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 16) ||
        (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) & static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
        OP_LOGE(opName_, "sparseBlockSize should be in range [1, 16] and be a power of 2, but got: %ld.",
            *opParamInfo_.sparseBlockSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckDequantScaleNotExistence()
{
    if (quantScaleRepoMode_ == 1) {
        OP_CHECK_IF((opParamInfo_.keyDequantScale.tensor == nullptr || opParamInfo_.valueDequantScale.tensor == nullptr),
            OP_LOGE(opName_,
            "When quant_scale_repo_mode is 1(combine), key_dequant_scale and value_dequant_scale should be both null."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus QSFATilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OP_LOGE(opName_, "Attr %s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OP_LOGE(opName_,
                "%s value should be %s, but got %s",
                name.c_str(),
                ossExpect.str().c_str(),
                ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckParaExistenceMlaAntiquant() const
{
    if (kvLayout_ == QSFALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == QSFALayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_kv is TND, actualSeqLengthsKv must not be null"),
                   return ge::GRAPH_FAILED);
    } else if (kvLayout_ == QSFALayout::PA_BSND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_kv is PA_BSND, actualSeqLengthsKv must not be null"),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_kv is PA_BSND, blockTable must not be null"),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaAntiquant();
}

ge::graphStatus QSFATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleNotExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

ge::graphStatus QSFATilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const QSFALayout &layout, const std::string &name) const
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of query is %s, %s must be provided.",
            QSFALayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

void QSFATilingCheck::SetQSFAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
}

ge::graphStatus QSFATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
            OP_LOGE(opName_, "when the layout_kv is %s, %s should be null",
                QSFALayoutToSerialString(kvLayout_).c_str(), BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(blockTableBatch != bSize_,
        OP_LOGE(opName_, "%s's first dimension(%u) should be equal to batch size(%u)",
            BLOCK_TABLE_NAME.c_str(), blockTableBatch, bSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE(opName_, "%s dtype should be %s, but it's %s.", name.c_str(),
            QSFADataTypeToSerialString(expectDtype).c_str(),
            QSFADataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckTopkShape()
{
    QSFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = sparseBlockCount_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus QSFATilingCheck::CheckAttenOutShape()
{
    QSFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = 512; // 512:输出的head_dim
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
        inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckKVShapeForBatchContinuous()
{
    QSFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t QSFATilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus QSFATilingCheck::CheckKVShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    QSFATilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.T = kvTSize_;
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckKVShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckKVShapeForPageAttention();
    }

    OP_LOGE(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus QSFATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
        inputKvType_, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OP_LOGE(opName_, "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            QSFADataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OP_LOGE(opName_, "actualSeqLengthsQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OP_LOGE(opName_, "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            QSFADataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OP_LOGE(opName_, "actualSeqLengths shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckMultiParaConsistency()
{
    SetQSFAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
        OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == QSFALayout::TND),
            OP_LOGE(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ <= 0,
            OP_LOGE(opName_, "q_head_num should be greater than 0, but got %u", n1Size_),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
        OP_LOGE(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OP_LOGE(opName_, "group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qHeadDim_ != 576, // 576:当前不泛化
        OP_LOGE(opName_, "q_head_dim only support 576, but got %u", qHeadDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(kHeadDim_ != 656, // 656:当前不泛化
        OP_LOGE(opName_, "k_head_dim only support 656, but got %u", kHeadDim_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OP_LOGE(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantDtype() const
{
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "query dtype only support %s and %s, but got %s",
            QSFADataTypeToSerialString(ge::DT_BF16).c_str(), QSFADataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            QSFADataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantAttr() const
{
    OP_CHECK_IF(attentionMode_ != 2, // 2:MLA-absorb
        OP_LOGE(opName_, "attention_mode should be 2(MLA-absorb), but got %d",
        attentionMode_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyQuantMode_ != 2, // 2:per-tile
        OP_LOGE(opName_, "key_quant_mode should be 2(per-tile), but got %d",
        keyQuantMode_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(valueQuantMode_ != 2, // 2:per-tile
        OP_LOGE(opName_, "value_quant_mode should be 2(per-tile), but got %d",
        valueQuantMode_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(quantScaleRepoMode_ != 1, // 1:combine
        OP_LOGE(opName_, "quant_scale_repo_mode should be 1(combine), but got %d",
        quantScaleRepoMode_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(preTokens_ != INT64_MAX,
        OP_LOGE(opName_, "preTokens_ should be INT64_MAX, but got %ld",
        preTokens_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(nextTokens_ != INT64_MAX,
        OP_LOGE(opName_, "nextTokens_ should be INT64_MAX, but got %ld",
        nextTokens_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tileSize_ != 128, // 128:当前不泛化
        OP_LOGE(opName_, "tile_size should be 128, but got %ld",
        tileSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ != 64, // 64:当前不泛化
        OP_LOGE(opName_, "rope_head_dim should be 64, but got %d",
        ropeHeadDim_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantPa() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OP_LOGE(opName_, "when page attention is enabled, block_size(%ld) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % 16 > 0,
        OP_LOGE(opName_, "when page attention is enabled, block_size(%ld) should be 16-aligned.",
        blockSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % sparseBlockSize_ > 0,
        OP_LOGE(opName_,
            "when page attention is enabled, block_size(%ld) must be divided by sparse_block_size(%ld), but now the remainder is %ld.",
        blockSize_, sparseBlockSize_, blockSize_ % sparseBlockSize_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantAttr() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaAntiquant();
}

ge::graphStatus QSFATilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

void QSFATilingCheck::Init()
{
    opName_ = sfaaInfo_.opName;
    platformInfo_ = sfaaInfo_.platformInfo;
    opParamInfo_ = sfaaInfo_.opParamInfo;
    socVersion_ = sfaaInfo_.socVersion;

    bSize_ = sfaaInfo_.bSize;
    n1Size_ = sfaaInfo_.n1Size;
    n2Size_ = sfaaInfo_.n2Size;
    s1Size_ = sfaaInfo_.s1Size;
    s2Size_ = sfaaInfo_.s2Size;
    gSize_ = sfaaInfo_.gSize;
    qHeadDim_ = sfaaInfo_.qHeadDim;
    kHeadDim_ = sfaaInfo_.kHeadDim;
    vHeadDim_ = sfaaInfo_.vHeadDim;
    ropeHeadDim_ = sfaaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = sfaaInfo_.maxBlockNumPerBatch;
    qTSize_ = sfaaInfo_.qTSize;
    kvTSize_ = sfaaInfo_.kvTSize;
    blockSize_ = sfaaInfo_.blockSize;
    sparseBlockCount_ = sfaaInfo_.sparseBlockCount;
    sparseBlockSize_ = sfaaInfo_.sparseBlockSize;

    attentionMode_ = sfaaInfo_.attentionMode;
    keyQuantMode_ = sfaaInfo_.keyQuantMode;
    valueQuantMode_ = sfaaInfo_.valueQuantMode;
    quantScaleRepoMode_ = sfaaInfo_.quantScaleRepoMode;
    tileSize_ = sfaaInfo_.tileSize;
    preTokens_ = sfaaInfo_.preTokens;
    nextTokens_ = sfaaInfo_.nextTokens;

    inputQType_ = sfaaInfo_.inputQType;
    inputKvType_ = sfaaInfo_.inputKvType;
    outputType_ = sfaaInfo_.outputType;

    qLayout_ = sfaaInfo_.qLayout;
    topkLayout_ = sfaaInfo_.topkLayout;
    kvLayout_ = sfaaInfo_.kvLayout;
    outLayout_ = sfaaInfo_.outLayout;

    kvStorageMode_ = sfaaInfo_.kvStorageMode;
    l2CacheSize_ = sfaaInfo_.l2CacheSize;
}

ge::graphStatus QSFATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool QSFAInfoParser::HasAxis(const QSFAAxis &axis, const QSFALayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = QSFA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == QSFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<QSFAAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    const auto& dimIt = QSFA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == QSFA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t QSFAInfoParser::GetAxisIdx(const QSFAAxis &axis, const QSFALayout &layout) const
{
    const std::vector<QSFAAxis>& axes = QSFA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t QSFAInfoParser::GetAxisNum(const gert::Shape &shape, const QSFAAxis &axis, const QSFALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

ge::graphStatus QSFAInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor k is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor k is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr, OP_LOGE(opName_, "Shape of tensor value is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr, OP_LOGE(opName_, "Desc of tensor value is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndices.shape == nullptr,
        OP_LOGE(opName_, "Shape of tensor sparseIndices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndices.desc == nullptr,
        OP_LOGE(opName_, "Desc of tensor sparseIndices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor output is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor output is nullptr"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr, OP_LOGE(opName_, "attr layoutQuery is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE(opName_, "attr layoutKV is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr, OP_LOGE(opName_, "attr sparseBlockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr, OP_LOGE(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OP_LOGE(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    QSFALayout &layout, const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE(opName_, "when layout of query is %s, %s must be provided.",
            QSFALayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ");
}

ge::graphStatus QSFAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("KvQuantSparseFlashAttentionLse", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void QSFAInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.keyDequantScale.tensor = context_->GetOptionalInputTensor(KEY_DEQUANT_SCALE_INPUT_INDEX);
    opParamInfo_.valueDequantScale.tensor = context_->GetOptionalInputTensor(VALUE_DEQUANT_SCALE_INPUT_INDEX);
}

void QSFAInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void QSFAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
    const auto computeNodeInfo = context_->GetComputeNodeInfo();
    const size_t outputNum = computeNodeInfo == nullptr ? 0 : computeNodeInfo->GetOutputsNum();
    if (outputNum > SOFTMAXMAX_INDEX) {
        opParamInfo_.softmaxMax.desc = context_->GetOutputDesc(SOFTMAXMAX_INDEX);
        opParamInfo_.softmaxMax.shape = context_->GetOutputShape(SOFTMAXMAX_INDEX);
    }
    if (outputNum > SOFTMAXSUM_INDEX) {
        opParamInfo_.softmaxSum.desc = context_->GetOutputDesc(SOFTMAXSUM_INDEX);
        opParamInfo_.softmaxSum.shape = context_->GetOutputShape(SOFTMAXSUM_INDEX);
    }
}

ge::graphStatus QSFAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    opParamInfo_.keyQuantMode = attrs->GetAttrPointer<int64_t>(KEY_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.valueQuantMode = attrs->GetAttrPointer<int64_t>(VALUE_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
    opParamInfo_.quantScaleRepoMode = attrs->GetAttrPointer<int64_t>(QUANT_SCALE_REPO_MODE_ATTR_INDEX);
    opParamInfo_.tileSize = attrs->GetAttrPointer<int64_t>(TILE_SIZE_ATTR_INDEX);
    opParamInfo_.ropeHeadDim = attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);
    if (attrs->GetAttrNum() > RETURN_SOFTMAX_LSE_ATTR_INDEX) {
        opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
    }
    if (attrs->GetAttrNum() > SOFTMAX_LSE_TND_ATTR_INDEX) {
        opParamInfo_.softmaxLseTnd = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_TND_ATTR_INDEX);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == QSFALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, QSFAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QSFAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == QSFALayout::TND) ? GetAxisNum(queryShape_, QSFAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKVTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以key的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    kvTSize_ = (kvLayout_ == QSFALayout::TND) ? GetAxisNum(keyShape_, QSFAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetQHeadDim()
{
    // 获取qHeadDim基准值
    // 以query的D维度为基准
    qHeadDim_ = GetAxisNum(queryShape_, QSFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKHeadDim()
{
    // 获取kHeadDim基准值
    // 以key的D维度为基准
    kHeadDim_ = GetAxisNum(keyShape_, QSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == QSFALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, QSFAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, QSFAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == QSFALayout::PA_BSND) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    // kv存储模式基准值
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKvLayout()
{
    const map<string, QSFALayout> layoutKVMap = {
        {"BSND",        QSFALayout::BSND},
        {"PA_BSND",     QSFALayout::PA_BSND},
        {"TND",         QSFALayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != QSFALayout::PA_BSND && qLayout_ != kvLayout_) {
        OP_LOGE(opName_, "When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == QSFALayout::PA_BSND && keyDimNum != 4U) {
        OP_LOGE(opName_, "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %u.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == QSFALayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, QSFAAxis::S, kvLayout_);
    } else if (kvLayout_ == QSFALayout::TND) {
        s2Size_ = GetAxisNum(keyShape_, QSFAAxis::T, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, blockTable must be provided.",
            QSFALayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != DIM_NUM_TWO) {
        OP_LOGE(opName_, "the dim num of block_table is %u, it should be %lu.", dimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OP_LOGE(opName_, "%s's second dimension(%ld) should be greater than 0",
            BLOCK_TABLE_NAME.c_str(), opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, QSFAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, QSFAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus QSFAInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = GetAxisNum(valueShape_, QSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<QSFALayout, QSFALayout>> layoutMap = {
        {"BSND",        {QSFALayout::BSND,    QSFALayout::BSND}},
        {"TND",         {QSFALayout::TND,     QSFALayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQuery);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, QSFAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, QSFAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void QSFAInfoParser::SetQSFAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
}

ge::graphStatus QSFAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

void QSFAInfoParser::GenerateInfo(QSFATilingInfo &sfaaInfo)
{
    sfaaInfo.opName = opName_;
    sfaaInfo.platformInfo = platformInfo_;
    sfaaInfo.opParamInfo = opParamInfo_;
    sfaaInfo.socVersion = socVersion_;

    sfaaInfo.bSize = bSize_;
    sfaaInfo.n1Size = n1Size_;
    sfaaInfo.n2Size = n2Size_;
    sfaaInfo.s1Size = s1Size_;
    sfaaInfo.s2Size = s2Size_;
    sfaaInfo.gSize = gSize_;
    sfaaInfo.qHeadDim = qHeadDim_;
    sfaaInfo.kHeadDim = kHeadDim_;
    sfaaInfo.vHeadDim = vHeadDim_;
    sfaaInfo.qTSize = qTSize_;
    sfaaInfo.kvTSize = kvTSize_;
    sfaaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    sfaaInfo.sparseBlockCount = sparseBlockCount_;

    sfaaInfo.inputQType = inputQType_;
    sfaaInfo.inputKvType = inputKvType_;
    sfaaInfo.outputType = outputType_;

    sfaaInfo.kvStorageMode = kvStorageMode_;
    sfaaInfo.l2CacheSize = l2CacheSize_;

    sfaaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    sfaaInfo.scaleValue = *opParamInfo_.scaleValue;
    sfaaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    sfaaInfo.blockSize = blockSize_;
    sfaaInfo.blockTypeSize =  sizeof(float);
    sfaaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    sfaaInfo.actualLenDimsQ = actualLenDimsQ_;
    sfaaInfo.actualLenDimsKV = actualLenDimsKV_;
    sfaaInfo.maxActualseq = maxActualseq_;
    sfaaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    sfaaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    sfaaInfo.isSameActualseq = isSameActualseq_;

    sfaaInfo.sparseMode = *opParamInfo_.sparseMode;
    sfaaInfo.attentionMode = *opParamInfo_.attentionMode;
    sfaaInfo.keyQuantMode = *opParamInfo_.keyQuantMode;
    sfaaInfo.valueQuantMode = *opParamInfo_.valueQuantMode;
    sfaaInfo.quantScaleRepoMode = *opParamInfo_.quantScaleRepoMode;
    sfaaInfo.preTokens = *opParamInfo_.preTokens;
    sfaaInfo.nextTokens = *opParamInfo_.nextTokens;
    sfaaInfo.tileSize = *opParamInfo_.tileSize;
    sfaaInfo.ropeHeadDim = *opParamInfo_.ropeHeadDim;
    sfaaInfo.returnSoftmaxLse = IsNonEmptyOutputShape(opParamInfo_.softmaxMax.shape) ||
        IsNonEmptyOutputShape(opParamInfo_.softmaxSum.shape) ||
        (opParamInfo_.returnSoftmaxLse != nullptr && *opParamInfo_.returnSoftmaxLse);
    sfaaInfo.softmaxLseTnd = opParamInfo_.softmaxLseTnd != nullptr && *opParamInfo_.softmaxLseTnd;

    sfaaInfo.qLayout = qLayout_;
    sfaaInfo.topkLayout = topkLayout_;
    sfaaInfo.kvLayout = kvLayout_;
    sfaaInfo.outLayout = outLayout_;
}

ge::graphStatus QSFAInfoParser::Parse(QSFATilingInfo &sfaaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("KvQuantSparseFlashAttentionLse", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetQSFAShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQHeadDim() ||
        ge::GRAPH_SUCCESS != GetKHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(sfaaInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvQuantSparseFlashAttentionLse)
    .Tiling(TilingKvQuantSparseFlashAttention)
    .TilingParse<KvQuantSparseFlashAttentionCompileInfo>(TilingPrepareForKvQuantSparseFlashAttention);
} // namespace optiling
