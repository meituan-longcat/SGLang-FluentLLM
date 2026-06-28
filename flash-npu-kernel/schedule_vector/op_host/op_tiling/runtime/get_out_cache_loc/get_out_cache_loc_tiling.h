/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file get_out_cache_loc_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GET_OUT_CACHE_LOC_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GET_OUT_CACHE_LOC_TILING_H

#include "register/op_compile_info_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"
#include "register/op_def_registry.h"

constexpr int64_t TILING_KEY = 10000;
constexpr int64_t DOUBLE_BUFF = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int32_t INPUT_IDX_REQ_TO_TOKEN = 0;
constexpr int32_t INPUT_IDX_REQ_POOL_INDICES = 1;
constexpr int32_t INPUT_IDX_NEW_COMPUTE_LENS = 2;
constexpr int32_t INPUT_IDX_CACHED_LENS = 3;
constexpr int32_t OUTPUT_IDX_OUT_CACHE_LOC = 0;

namespace optiling {
BEGIN_TILING_DATA_DEF(GetOutCacheLocTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);                 // tensor处理用了多少个核
  TILING_DATA_FIELD_DEF(uint32_t, blockFactor);                 // 每个核处理多少个batch
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockFactor);             // 尾核处理多少个batch
  TILING_DATA_FIELD_DEF(uint32_t, ubFactor);                    // UB 一次能处理多少个值
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);                   // batch大小
  TILING_DATA_FIELD_DEF(uint32_t, poolLen);                     // pool长度
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GetOutCacheLoc, GetOutCacheLocTilingData)

struct GetOutCacheLocCompileInfo {
};

#define CHECK_FAIL(context, cond, ...)                      \
    do {                                                    \
        if (cond) {                                         \
            OP_LOGE(context->GetNodeName(), ##__VA_ARGS__); \
            return ge::GRAPH_FAILED;                        \
        }                                                   \
    } while (0)

#define CHECK_NULL(context, ptr, ...)                                                                    \
    do {                                                                                                 \
        if ((ptr) == nullptr) {                                                                          \
            const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
            OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", ##__VA_ARGS__);                               \
            REPORT_CALL_ERROR("EZ9999", "op[%s], %s is nullptr!", name, ##__VA_ARGS__);               \
            return ge::GRAPH_FAILED;                                                                     \
        }                                                                                                \
    } while (0)

class GetOutCacheLocTiling {
public:
    explicit GetOutCacheLocTiling(gert::TilingContext *context) : context_(context)
    {}
    ~GetOutCacheLocTiling()
    {}
    ge::graphStatus DoOpTiling() {
        CHECK_NULL(context_, context_, "context_ is nullptr.\n");
        auto res = GetPlatformInfo();
        CHECK_FAIL(context_, res != ge::GRAPH_SUCCESS, "GetPlatformInfo Failed.\n");
        res = GetShapeAttrsInfo();
        CHECK_FAIL(context_, res != ge::GRAPH_SUCCESS, "GetShapeAttrsInfo Failed.\n");
        res = BlockTiling();
        CHECK_FAIL(context_, res != ge::GRAPH_SUCCESS, "BlockTiling Failed.\n");
        res = UbTiling();
        CHECK_FAIL(context_, res != ge::GRAPH_SUCCESS, "UbTiling Failed.\n");
        SetTilingData();
        res = PostTiling();
        CHECK_FAIL(context_, res != ge::GRAPH_SUCCESS, "PostTiling Failed.\n");
        return ge::GRAPH_SUCCESS;
    }
private:
    gert::TilingContext *context_ = nullptr;
    GetOutCacheLocTilingData tilingData_;
    int64_t coreNum_{0};
    int64_t ubSize_ = {0};
    int64_t blockFactor_ = {0};
    int64_t tailBlockFactor_ = {0};
    int64_t ubFactor_ = {0};
    int64_t usedCoreNum_ = {0};
    int64_t batchSize_ = {0};
    int64_t poolLen_ = {0};
    int64_t sysWorkSpaceSize_ = {0};
    const char* opName_ = "GetOutCacheLoc";

private:
    ge::graphStatus GetPlatformInfo()
    {
        auto platformInfo = context_->GetPlatformInfo();
        CHECK_NULL(context_, platformInfo, "platformInfo is nullptr.\n");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        sysWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        CHECK_FAIL(context_, ubSize_ == 0, "ubSize_ is 0.\n");
        CHECK_FAIL(context_, coreNum_ == 0, "coreNum_ is 0.\n");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo()
    {
        auto attrs = context_->GetAttrs();
        CHECK_NULL(context_, attrs, "attrs is nullptr.\n");
        auto batchSize = attrs->GetInt(0);
        CHECK_NULL(context_, batchSize, "batchSize is nullptr.\n");
        batchSize_ = static_cast<int64_t>(*batchSize);
        const gert::StorageShape* reqToTokenShapePtr = context_->GetInputShape(INPUT_IDX_REQ_TO_TOKEN);
        CHECK_NULL(context_, reqToTokenShapePtr, "reqToTokenShape is nullptr.\n");
        auto reqToTokenShape = reqToTokenShapePtr->GetStorageShape();
        poolLen_ = reqToTokenShape.GetDim(1);
        CHECK_FAIL(context_, batchSize_ == 0, "batchSize_ is 0.\n");
        CHECK_FAIL(context_, poolLen_ == 0, "poolLen_ is 0.\n");
        auto reqPoolIndicesDtype = context_->GetInputDesc(INPUT_IDX_REQ_POOL_INDICES)->GetDataType();
        auto reqToTokenDtype = context_->GetInputDesc(INPUT_IDX_REQ_TO_TOKEN)->GetDataType();
        auto newComputeLensDtype = context_->GetInputDesc(INPUT_IDX_NEW_COMPUTE_LENS)->GetDataType();
        auto cachedLensDtype = context_->GetInputDesc(INPUT_IDX_CACHED_LENS)->GetDataType();
        auto outCacheLocDtype= context_->GetOutputDesc(OUTPUT_IDX_OUT_CACHE_LOC)->GetDataType();
        CHECK_FAIL(context_, reqPoolIndicesDtype != reqToTokenDtype
                     || reqToTokenDtype != newComputeLensDtype
                     || newComputeLensDtype != cachedLensDtype
                     || cachedLensDtype != outCacheLocDtype,
                     "reqPoolIndices, reqToToken, newComputeLens, cachedLens, outCacheLoc dtype should be same.\n");
        CHECK_FAIL(context_, reqPoolIndicesDtype != ge::DT_INT64 && reqPoolIndicesDtype != ge::DT_INT32,
                 "reqPoolIndicesDtype only support int64 and int32.\n");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus BlockTiling() {
        auto blockFactor = ops::CeilDiv(batchSize_, coreNum_);
        usedCoreNum_ = ops::CeilDiv(batchSize_, blockFactor);
        blockFactor_ = ops::CeilDiv(batchSize_, usedCoreNum_);
        tailBlockFactor_ = batchSize_ % usedCoreNum_ == 0 ? blockFactor_ : batchSize_ % blockFactor_;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus UbTiling()
    {
        ge::DataType dataType = context_->GetInputDesc(INPUT_IDX_REQ_TO_TOKEN)->GetDataType();
        int64_t newComputeLen = ops::CeilAlign(static_cast<int64_t>(batchSize_ * sizeof(int64_t)), BLOCK_SIZE);
        int64_t reserverUbSize = ubSize_ - newComputeLen;
        ubFactor_ = ops::CeilAlign(reserverUbSize / DOUBLE_BUFF, BLOCK_SIZE) / ge::GetSizeByDataType(dataType);
        return ge::GRAPH_SUCCESS;
    }

    void SetTilingData()
    {
        tilingData_.set_usedCoreNum(usedCoreNum_);
        tilingData_.set_blockFactor(blockFactor_);
        tilingData_.set_tailBlockFactor(tailBlockFactor_);
        tilingData_.set_ubFactor(ubFactor_);
        tilingData_.set_batchSize(batchSize_);
        tilingData_.set_poolLen(poolLen_);
        return;
    }

    ge::graphStatus PostTiling() {
        auto workSpaceSize = context_->GetWorkspaceSizes(1);
        workSpaceSize[0] = sysWorkSpaceSize_;
        context_->SetBlockDim(usedCoreNum_);
        context_->SetTilingKey(TILING_KEY);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
};
}

#endif