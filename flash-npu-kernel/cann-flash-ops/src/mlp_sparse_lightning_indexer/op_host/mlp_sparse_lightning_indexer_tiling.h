/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_tiling.h
 * \brief
 */

#ifndef MLP_SPARSE_LIGHTNING_INDEXER_TILING_H_
#define MLP_SPARSE_LIGHTNING_INDEXER_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
// #include "platform/platform_info.h"

namespace optiling {
// ------------------公共定义--------------------------
struct LISparseTilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct LISparseTilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class LISparseDataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    BnBsND = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t CUR_SEQ_Q_INDEX = 3;
constexpr uint32_t CUR_SEQ_K_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t SPARSE_INDEX_INDEX = 6;
//Outputs Index
constexpr uint32_t LIGHTNING_INDEXER = 0;
constexpr uint32_t LIGHTNING_VALUES = 1;
// Attributes Index
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
constexpr uint32_t ATTR_INIT_NUM_INDEX = 3;
constexpr uint32_t ATTR_LOCAL_NUM_INDEX = 4;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 5;
constexpr uint32_t ATTR_PRE_TOKENS_INDEX = 6;
constexpr uint32_t ATTR_NEXT_TOKENS_INDEX = 7;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 8;
constexpr uint32_t ATTR_SPARSE_INDEX_BLOCK_SIZE_INDEX = 9;

// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t TOPK_2K = 2048;
constexpr uint32_t TOPK_8K = 8192;
constexpr uint32_t SPARSE_MODE_LOWER = 3;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(LISparseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)

TILING_DATA_FIELD_DEF(uint32_t, initNum)
TILING_DATA_FIELD_DEF(uint32_t, localNum)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(int8_t, returnValue)
TILING_DATA_FIELD_DEF(uint32_t, useSparseIndex)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockNum)

END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MlpSparseLightningIndexer, LISparseTilingData)

// -----------算子CompileInfo定义-------------------
struct LISparseCompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct LISparseParaInfo {
    LISparseTilingRequiredParaInfo query = {nullptr, nullptr};
    LISparseTilingRequiredParaInfo key = {nullptr, nullptr};
    LISparseTilingRequiredParaInfo weights = {nullptr, nullptr};
    LISparseTilingOptionalParaInfo curSeqLengthsQ = {nullptr, nullptr};
    LISparseTilingOptionalParaInfo curSeqLengths = {nullptr, nullptr};
    LISparseTilingOptionalParaInfo blockTable = {nullptr, nullptr};
    LISparseTilingOptionalParaInfo sparseIndex = {nullptr, nullptr};
    LISparseTilingRequiredParaInfo attenOut = {nullptr, nullptr};
    LISparseTilingRequiredParaInfo valuesOut = {nullptr, nullptr};

    const char *layOut = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *sparseMode = nullptr;
    const int32_t *sparseCount = nullptr;
    const int32_t *sparseIndexBlockSize = nullptr;

    const int32_t *initNum = nullptr;
    const int32_t *localNum = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const bool *returnValue = nullptr;

};

// -----------算子Tiling入参信息类---------------
class LISparseTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LISparseParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;

    uint32_t initNum = 0;
    uint32_t localNum = 0;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int8_t returnValue = 0;
    // SparseIndex
    bool useSparseIndex = false;
    uint32_t sparseBlockNum = 0;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    LISparseDataLayout inputQLayout = LISparseDataLayout::BSND;
    LISparseDataLayout inputKLayout = LISparseDataLayout::BnBsND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class LISparseInfoParser {
public:
    explicit LISparseInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~LISparseInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetCurSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &curSeqLenName);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetAndCheckSparseIndex();
    void GenerateInfo(LISparseTilingInfo &liInfo);
    ge::graphStatus ParseAndCheck(LISparseTilingInfo &liInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    LISparseParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    // Layout
    LISparseDataLayout qLayout_ = LISparseDataLayout::BSND;
    LISparseDataLayout kLayout_ = LISparseDataLayout::BnBsND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType valuesOutType_ = ge::DT_FLOAT16;
    // SparseIndex
    bool useSparseIndex_ = false;
    uint32_t sparseBlockNum_ = 0;
};

// ---------------算子Tiling类---------------
class LISparseIndexerTiling {
public:
    explicit LISparseIndexerTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoTiling(LISparseTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    LISparseTilingData tilingData_;
};

} // namespace optiling
#endif // MLP_SPARSE_LIGHTNING_INDEXER_TILING_H_