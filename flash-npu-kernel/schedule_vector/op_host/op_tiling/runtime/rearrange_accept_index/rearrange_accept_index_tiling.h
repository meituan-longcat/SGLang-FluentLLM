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
 * \file rearrange_accept_index_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_REARRANGE_ACCEPT_INDEX_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_REARRANGE_ACCEPT_INDEX_TILING_H

#include "register/op_compile_info_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"
#include "register/op_def_registry.h"

constexpr int64_t TILING_KEY = 100;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int32_t INPUT_ACCEPT_INDEX = 0;
constexpr int32_t INPUT_ACCEPT_LENS = 1;
constexpr int32_t INPUT_IDX_NEW_COMPUTE_LENS = 2;
constexpr int32_t OUTPUT_IDX_OUTPUT = 0;

namespace optiling {
BEGIN_TILING_DATA_DEF(RearrangeAcceptIndexTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockFactor);
  TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, poolLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RearrangeAcceptIndex, RearrangeAcceptIndexTilingData)

struct RearrangeAcceptIndexCompileInfo {
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

class RearrangeAcceptIndexTiling {
  public:
    explicit RearrangeAcceptIndexTiling(gert::TilingContext *context) : context_(context){}
    ~RearrangeAcceptIndexTiling(){}
    ge::graphStatus DoOpTiling() {
      auto res = GetPlatformInfo();
      res = GetShapeAttrInfo();
      res = DoCoreSplit();
      res = DoInnerCore();
      SetTilingData();
      res = PostTiling();
      return ge::GRAPH_SUCCESS;
    }
  private:
    gert::TilingContext *context_ = nullptr;
    RearrangeAcceptIndexTilingData tilingData_;
    int64_t coreNum_{0};
    int64_t ubSize_{0};
    int64_t blockFactor_{0};
    int64_t ubFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t usedCoreNum_{0};
    int64_t batchSize_{0};
    int64_t poolLen_{0};
    int64_t sysWorkSpaceSize_{0};
    const char* opName_ = "RearrangeAcceptIndex";

  private:
    ge::graphStatus GetPlatformInfo()
    {
      auto platformInfo = context_->GetPlatformInfo();
      CHECK_NULL(context_, platformInfo, "platformInfo is nullptr.\n");
      auto ascendCPlatform = platform_ascendc::PlatformAscendC(platformInfo);
      sysWorkSpaceSize_ = ascendCPlatform.GetLibApiWorkSpaceSize();
      coreNum_ = ascendCPlatform.GetCoreNumAiv();
      uint64_t ubSizePlatform;
      ascendCPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
      ubSize_ = static_cast<int64_t>(ubSizePlatform);
      CHECK_FAIL(context_, ubSize_ == 0, "ubSize_ is 0.\n");
      CHECK_FAIL(context_, coreNum_ == 0, "coreNum_ is 0.\n");
      auto acceptIndexDataType = context_->GetInputDesc(INPUT_ACCEPT_INDEX)->GetDataType();
      auto acceptLensDataType = context_->GetInputDesc(INPUT_ACCEPT_LENS)->GetDataType();
      auto outDataType= context_->GetOutputDesc(OUTPUT_IDX_OUTPUT)->GetDataType();
      CHECK_FAIL(context_, acceptIndexDataType != acceptLensDataType
                    || acceptLensDataType != outDataType,
                    "accept_index, accept_lens, out dtype should be same.\n");
      CHECK_FAIL(context_, acceptIndexDataType != ge::DT_INT64 && acceptIndexDataType != ge::DT_INT32,
                "acceptIndexDataType only support int64 and int32.\n");
      return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetShapeAttrInfo()
    {
      auto attrs = context_->GetAttrs();
      CHECK_NULL(context_, attrs, "attrs is nullptr.\n");
      auto bs = attrs->GetInt(0);
      CHECK_NULL(context_, bs, "batchSize is nullptr.\n");
      batchSize_ = static_cast<int64_t>(*bs);
      const gert::StorageShape* xShape = context_->GetInputShape(0);
      CHECK_NULL(context_, xShape, "xShape is nullptr.\n");
      const gert::Shape inputShape = xShape->GetStorageShape();
      int32_t dim0 = inputShape.GetDim(0);
      int32_t dim1 = inputShape.GetDim(1);
      poolLen_ = static_cast<int64_t>(dim1);
      CHECK_FAIL(context_, batchSize_ == 0, "batchSize_ is 0.\n");
      CHECK_FAIL(context_, poolLen_ == 0, "poolLen_ is 0.\n");
      return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoCoreSplit()
    {
      auto blockFactor = ops::CeilDiv(batchSize_, coreNum_);
      usedCoreNum_ = ops::CeilDiv(batchSize_, blockFactor);
      blockFactor_ = ops::CeilDiv(batchSize_, usedCoreNum_);
      tailBlockFactor_ = batchSize_ % usedCoreNum_ == 0 ? blockFactor_ : batchSize_ % blockFactor_;
      return ge::GRAPH_SUCCESS;
    }
    
    ge::graphStatus DoInnerCore()
    {
      ge::DataType x0DType = context_->GetInputDesc(0)->GetDataType();
      ge::DataType x1DType = context_->GetInputDesc(1)->GetDataType();
      int64_t acceptLenSize = ops::CeilAlign(static_cast<int64_t>(batchSize_ * sizeof(int64_t)), BLOCK_SIZE);
      int64_t reserverUbSize = ubSize_ - acceptLenSize;
      ubFactor_ = CeilAlign(reserverUbSize / DOUBLE_BUFFER, BLOCK_SIZE) / ge::GetSizeByDataType(x0DType);
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

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_REARRANGE_ACCEPT_INDEX_TILING_H