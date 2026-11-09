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
 * \file moe_init_routing_v3_tiling_arch35.cpp
 * \brief
 */
#include <sstream>
#include <string>
#include <unordered_set>
#include "register/op_def_registry.h"
#include "moe_init_routing_v3_tiling.h"
#include "../op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h"
#include "tiling_base/tiling_util.h"

#define MIRV3_CHECK_GE_RET(expr)                                                                                       \
    if (ge::graphStatus ret = (expr); ret != ge::GRAPH_SUCCESS) {                                                      \
        return ret;                                                                                                    \
    }

using Ops::Transformer::OpTiling::TilingBaseClass;

namespace optiling {
const static int64_t SIMT_DCACHE_SIZE = 64 * 1024LL; // UB要给SIMT预留64k的DCache空间，然后要用SetLocalMemSize()
const static int64_t SORT_API_MAX_ELEM = 32 * 255LL; // AscendC::Sort全排序模式最多支持一次排序(32*255rep)个元素
const static int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
const static int64_t MX_QUANT_BLOCK_SIZE = 32LL;

const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_THREE = 3LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t MRG_LIST_NUM = 4LL;
const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
const static int64_t UB_BLOCK_SIZE = 32LL;
const static size_t DIM_ONE = 1ULL;
const static size_t DIM_TWO = 2ULL;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t KV_FACTOR = 2LL;
const static int64_t ONE_CORE_SORT_BUFFER = 6LL;
const static int64_t EXPERT_IDX_MAX = 10240LL;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;
const static int64_t RANK_ONE = 1LL;
const static int64_t RANK_TWO = 2LL;
const static int64_t BF16_TO_FP32_SIZE_FACTOR = 2LL;

// 输入输出的位置索引
const static int64_t INPUT_X_INDEX = 0LL;
const static int64_t INPUT_EXPERT_IDX_INDEX = 1LL;
const static int64_t INPUT_SCALE_INDEX = 2LL;
const static int64_t INPUT_OFFSET_INDEX = 3LL;
const static int64_t OUTPUT_EXPANDED_X_INDEX = 0LL;
const static int64_t OUTPUT_EXPANDED_ROW_IDX_INDEX = 1LL;
const static int64_t OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2LL;
const static int64_t OUTPUT_EXPANDED_SCALE_INDEX = 3LL;
const static int64_t ATTR_ACTIVE_NUM_INDEX = 0LL;
const static int64_t ATTR_EXPERT_CAPACITY_INDEX = 1LL;
const static int64_t ATTR_EXPERT_NUM_INDEX = 2LL;
const static int64_t ATTR_DROP_PAD_MODE_INDEX = 3LL;
const static int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 4LL;
const static int64_t ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX = 5LL;
const static int64_t ATTR_QUANT_MODE_INDEX = 6LL;
const static int64_t ATTR_EXPERT_RANGE_INDEX = 7LL;
const static int64_t ATTR_ROW_IDX_TYPE_INDEX = 8LL;

const static int64_t ACTIVE_NUM_MIN_VALUE = -1LL;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21LL;
const static int64_t HIF8_PERTENSOR_QUANT_COLS_BUFFER = 5LL;
const static int64_t HIF8_PERTOKEN_QUANT_COLS_BUFFER = 5LL;

// 输入attrs相关
const static int64_t ROW_IDX_GATHER = 0LL;
const static int64_t ROW_IDX_SCATTER = 1LL;
const static int64_t QUANT_MODE_UNQUANT = -1LL;
const static int64_t QUANT_MODE_STATIC = 0LL;
const static int64_t QUANT_MODE_DYNAMIC = 1LL;
const static int64_t QUANT_MODE_MXFP8_E5M2 = 2LL;
const static int64_t QUANT_MODE_MXFP8_E4M3FN = 3LL;
const static int64_t QUANT_MODE_HIF8_CAST = 6LL;
const static int64_t QUANT_MODE_HIF8_PERTENSOR = 7LL;
const static int64_t QUANT_MODE_HIF8_PERTOKEN = 8LL;
const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;

// TilingKey相关
const static int64_t TILINGKEY_BASE = 1000000LL;
const static int64_t SORT_CORE_TILINGKEY_BASE = 100000LL;
const static int64_t QUANT_MODE_TILINGKEY_BASE = 10000LL;
const static int64_t DROP_MODE_TILINGKEY_BASE = 1000LL;

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

struct MultipleParams
{
    int64_t colMultiple = 0;
    int64_t rowMultiple = 0;
};

struct PerLoopParams
{
    int64_t perLoopCols = 0;
    int64_t perLoopMaxIndicesElements = 0;
};

class MoeInitRoutingV3Arch35TilingClass : public TilingBaseClass {
public:
    explicit MoeInitRoutingV3Arch35TilingClass(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRoutingV3Arch35TilingClass() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // 1、获取INPUT/OUTPUT/ATTR信息：DelayedGetShapeAttrsInfo()，延后到DoOpTiling内执行，以便先检查IsCapable()
    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetShapeAttrsInfo()");
        return ge::GRAPH_SUCCESS;
    }
    // 2、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 3、判断此Tiling模板是否适配当前SOC
    bool IsCapable() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::IsCapable()");
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }
    // 4、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::DoLibApiTiling()");
        return ge::GRAPH_SUCCESS;
    }
    // 6、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 8、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset()
    {
        opName = nullptr;
    };

private:
    // 校验并设置必要的平台信息（把GetPlatformInfo内，校验和计算的逻辑延后至DoOpTiling处理）
    ge::graphStatus CheckSetPlatformInfo();
    // 获取输入输出ShapeAttrs信息（把GetShapeAttrsInfo的逻辑延后至DoOpTiling处理）
    ge::graphStatus DoGetShapeAttrsInfo();
    // 校验输入Attrs
    ge::graphStatus CheckSetAttrs();
    ge::graphStatus CheckSetListAttrs();
    // 校验输入Tensor的Shape、Dtype
    ge::graphStatus CheckSetInputs();
    // 校验输出Tensor的Shape、Dtype
    ge::graphStatus CheckOutputs();

    // DoGetShapeAttrsInfo使用的子函数
    ge::graphStatus GetInputTensorsInfo();
    ge::graphStatus GetOutputTensorsInfo();
    ge::graphStatus GetInputAttrsInfo();
    // CheckInputShape使用的子函数
    ge::graphStatus CheckInputX();
    ge::graphStatus CheckInputExpertIdx();
    ge::graphStatus CheckInputScale();
    // CheckOutShape使用的子函数
    ge::graphStatus CheckOutputExpandedX();
    ge::graphStatus CheckOutputExpandedRowIdx();
    ge::graphStatus CheckOutputExpertTokensCountOrCumsum();
    ge::graphStatus CheckOutputExpandedScale();

    // 各阶段TilingData计算函数
    MultipleParams GetMultipleParams();
    PerLoopParams GetPerLoopParams(MultipleParams& multipleParams);
    void Tiling4GatherOutCompute();
    void Tiling4GatherOutMxQuant();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4ExpertTokensCountCompute();
    void Tiling4VBSOneCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling);
    void Tiling4VBSMultiCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling);
    int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols);

    // LogTilingData
    void LogBaseTilingData();
    void LogVbsTilingData();
    void LogVmsMiddleTilingData();
    void LogSortOutTilingData();
    void LogExpertTokensCountTilingData();
    void LogGatherOutTilingData();

    // 辅助工具函数
    template <bool IS_INPUT_TENSOR = true, bool IS_OPTIONAL_INPUT = false>
    ge::graphStatus GetTensorShapeDtype(gert::Shape &shape, ge::DataType &dtype, int64_t index)
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetTensorShapeDtype(...)");
        const gert::StorageShape *shapePtr{nullptr};
        const gert::CompileTimeTensorDesc *descPtr{nullptr};
        if constexpr (IS_INPUT_TENSOR) {
            if constexpr (IS_OPTIONAL_INPUT) {
                shapePtr = context_->GetOptionalInputShape(index);
                descPtr = context_->GetOptionalInputDesc(index);
            } else {
                shapePtr = context_->GetInputShape(index);
                descPtr = context_->GetInputDesc(index);
            }
        } else {
            shapePtr = context_->GetOutputShape(index);
            descPtr = context_->GetOutputDesc(index);
        }
        OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
        shape = shapePtr->GetStorageShape();
        OP_CHECK_NULL_WITH_CONTEXT(context_, descPtr);
        dtype = descPtr->GetDataType();
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetOptionalInputShapeDtype(gert::Shape &shape, ge::DataType &dtype, int64_t &marker, int64_t index)
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetOptionalInputShapeDtype(...)");
        if (context_->GetOptionalInputShape(index) != nullptr) {
            marker = 1;
            return GetTensorShapeDtype<true, true>(shape, dtype, index);
        } else {
            // 该Tensor没有输入
            marker = 0;
            return ge::GRAPH_SUCCESS;
        }
    }
    template <typename ATTR_T>
    ge::graphStatus GetInputAttr(ATTR_T &attr, const gert::RuntimeAttrs *attrsPtr, int64_t index)
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetInputAttr(...)");
        const auto *attrPtr = attrsPtr->GetAttrPointer<ATTR_T>(index);
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
        attr = *attrPtr;
        return ge::GRAPH_SUCCESS;
    }

    // op variables
    const char *opName = "";
    MoeInitRoutingV3Arch35TilingData *tilingDataPtr_{nullptr};

    // platform infos
    int64_t aivCoreNum_ = 0LL;
    int64_t totalUbSize_ = 0LL;
    int64_t availUbSize_ = 0LL;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;

    // important values
    int64_t sortLoopMaxElement_ = 0LL;
    int64_t totalLength_ = 0LL;
    int64_t n_ = 0LL;
    int64_t k_ = 0LL;
    int64_t cols_ = 0LL;
    int64_t inputXDtypeSize_;
    int64_t isInputScale_ = 0LL;
    int64_t isInputOffset_ = 0LL;
    int64_t sortMode_ = 0LL;

    // input attrs
    int64_t activeNum_ = -1LL;
    int64_t expertCapacity_ = -1LL;
    int64_t expertNum_ = -1LL;
    int64_t dropPadMode_ = -1LL;
    int64_t expertTokensNumType_ = -1LL;
    bool expertTokensNumFlag_ = false;
    int64_t quantMode_ = -1LL;
    int64_t expertStart_ = -1LL;
    int64_t expertEnd_ = -1LL;
    int64_t rowIdxType_ = -1LL;

    // input tensors shape
    gert::Shape xShape_;
    gert::Shape expertIdxShape_;
    gert::Shape scaleShape_;
    gert::Shape offsetShape_;
    // output tensors shape
    gert::Shape expandedXShape_;
    gert::Shape expandedRowIdxShape_;
    gert::Shape expertTokensCountOrCumsumShape_;
    gert::Shape expandedScaleShape_;

    // input tensors dtype
    ge::DataType xDtype_;
    ge::DataType expertIdxDtype_;
    ge::DataType scaleDtype_;
    ge::DataType offsetDtype_;
    // output tensors dtype
    ge::DataType expandedXDtype_;
    ge::DataType expandedRowIdxDtype_;
    ge::DataType expertTokensCountOrCumsumDtype_;
    ge::DataType expandedScaleDtype_;
};

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::GetPlatformInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetPlatformInfo()");

    const auto *compileInfoPtr = reinterpret_cast<const MoeInitRoutingV3CompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "Failed to get compileInfo from tiling context."),
                return ge::GRAPH_FAILED);
    aivCoreNum_ = static_cast<int64_t>(compileInfoPtr->aivNum);
    totalUbSize_ = static_cast<int64_t>(compileInfoPtr->ubSize);
    socVersion_ = compileInfoPtr->socVersion;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckSetPlatformInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckSetPlatformInfo()");

    // check aivCoreNum
    OP_CHECK_IF(aivCoreNum_ <= 0, OP_LOGE(context_, "Failed to get valid aivCoreNum, current is %ld.", aivCoreNum_),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->coreNum = aivCoreNum_;
    // check availUbSize
    availUbSize_ = totalUbSize_ - SIMT_DCACHE_SIZE;
    OP_CHECK_IF(
        totalUbSize_ <= 0 || availUbSize_ <= 0,
        OP_LOGE(context_,
                "Got invalid totalUbSize = %ld bytes or availUbSize = %ld bytes (SIMT_DCACHE_SIZE = %ld bytes).",
                totalUbSize_, availUbSize_, SIMT_DCACHE_SIZE),
        return ge::GRAPH_FAILED);
    // log info
    OP_LOGD(context_,
            "Got platform info aivCoreNum = %ld, totalUbSize = %ld bytes, availUbSize = %ld bytes (SIMT_DCACHE_SIZE = "
            "%ld bytes).",
            aivCoreNum_, totalUbSize_, availUbSize_, SIMT_DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::DoGetShapeAttrsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetInputAttrsInfo()");

    MIRV3_CHECK_GE_RET(GetInputTensorsInfo());
    MIRV3_CHECK_GE_RET(GetOutputTensorsInfo());
    MIRV3_CHECK_GE_RET(GetInputAttrsInfo());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::DoOpTiling()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::DoOpTiling()");

    // 获取tilingData指针
    tilingDataPtr_ = context_->GetTilingData<MoeInitRoutingV3Arch35TilingData>();

    MIRV3_CHECK_GE_RET(CheckSetPlatformInfo());
    MIRV3_CHECK_GE_RET(DoGetShapeAttrsInfo());
    MIRV3_CHECK_GE_RET(CheckSetAttrs());
    MIRV3_CHECK_GE_RET(CheckSetListAttrs());
    MIRV3_CHECK_GE_RET(CheckSetInputs());
    MIRV3_CHECK_GE_RET(CheckOutputs());

    sortLoopMaxElement_ = availUbSize_ / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    sortLoopMaxElement_ =
        std::min(sortLoopMaxElement_, SORT_API_MAX_ELEM); // 限制单核排序的元素个数在AscendC::Sort全排序的能力范围内

    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4ExpertTokensCountCompute();
    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN) {
        Tiling4GatherOutMxQuant();
    } else {
        Tiling4GatherOutCompute();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingV3Arch35TilingClass::GetTilingKey() const
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetTilingKey()");

    int64_t quantModeFactor = quantMode_ + 1;
    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN) {
        // 对于MXFP8量化，两种模式在TilingKey体现的QuantMode都为3。
        // 其余非量化为0，静态量化为1，动态量化为2，即都是quantMode_+1
        // 可以用与最低的UNQUANT的数值的差值来作为quantModeFactor，这里值就为3
        quantModeFactor = QUANT_MODE_MXFP8_E5M2 - QUANT_MODE_UNQUANT;
    }
    return static_cast<uint64_t>(TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
                                 quantModeFactor * QUANT_MODE_TILINGKEY_BASE + rowIdxType_ * DROP_MODE_TILINGKEY_BASE);
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::GetWorkspaceSize()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetWorkspaceSize()");
    // 计算workspace大小
    workspaceSize_ = 0;
    int64_t sortWorkspaceSize = totalLength_ * static_cast<int64_t>(sizeof(float) * NUM_TWO * NUM_THREE); // 排序需要的空间
    int64_t coreSyncWorkspaceSize = tilingDataPtr_->coreNum * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    int64_t scatterWorkspaceSize = totalLength_ * static_cast<int64_t>(sizeof(int32_t));
    int64_t expertTokensCountWorkspaceSize = (expertEnd_ - expertStart_) * static_cast<int64_t>(sizeof(int32_t));
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, static_cast<int64_t>(sizeof(int32_t)));
    int64_t quantTempWorkspaceSize = aivCoreNum_ * cols_ * static_cast<int64_t>(sizeof(float));
    workspaceSize_ += sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize +
                      expertTokensCountWorkspaceSize + expertTokenTotalCountWorkspace;
    if (quantMode_ >= QUANT_MODE_DYNAMIC && quantMode_ != QUANT_MODE_HIF8_CAST &&
        quantMode_ != QUANT_MODE_HIF8_PERTENSOR) {
        // DYNAMIC_QUANT、MXFP8_E5M2_QUANT、MXFP8_E4M3FN_QUANT
        workspaceSize_ += quantTempWorkspaceSize;
    }
    // 这里workspaceSize_除了计算必要的，还会加上16M的AscendC框架用大小
    workspaceSize_ += SIZE_16 * LENGTH_1024 * LENGTH_1024;
    OP_LOGD(context_, "Computed workspace size to allocate is %u bytes", workspaceSize_);
    // 设置workspace
    auto *workspacePtr = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspacePtr);
    workspacePtr[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::PostTiling()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::PostTiling()");

    // 这个tilingKey_成员变量(TilingBaseClass)不能在GetTilingKey()方法里赋值的设计还挺抽象的
    tilingKey_ = GetTilingKey();
    LogBaseTilingData();

    // 设置启动核数（全核启动）
    context_->SetBlockDim(aivCoreNum_);
    // 设置UB可用大小（必须是减除SIMT用的DCACHE大小后的）
    auto ret = context_->SetLocalMemorySize(availUbSize_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "Failed to set local memory sizeof %ld bytes", availUbSize_),
                return ge::GRAPH_FAILED);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    ret = context_->SetScheduleMode(1);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "Failed to set schedule mode to 1 for kernel that needs sync cores."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::GetInputTensorsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetInputTensorsInfo()");

    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<true>(xShape_, xDtype_, INPUT_X_INDEX));
    inputXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(xDtype_));
    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<true>(expertIdxShape_, expertIdxDtype_, INPUT_EXPERT_IDX_INDEX));
    // 可选输入scale
    MIRV3_CHECK_GE_RET(GetOptionalInputShapeDtype(scaleShape_, scaleDtype_, isInputScale_, INPUT_SCALE_INDEX));
    tilingDataPtr_->isInputScale = isInputScale_;
    // 可选输入offset
    MIRV3_CHECK_GE_RET(GetOptionalInputShapeDtype(offsetShape_, offsetDtype_, isInputOffset_, INPUT_SCALE_INDEX));
    tilingDataPtr_->isInputOffset = isInputOffset_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::GetOutputTensorsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetOutputTensorsInfo()");

    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expandedXShape_, expandedXDtype_, OUTPUT_EXPANDED_X_INDEX));
    MIRV3_CHECK_GE_RET(
        GetTensorShapeDtype<false>(expandedRowIdxShape_, expandedRowIdxDtype_, OUTPUT_EXPANDED_ROW_IDX_INDEX));
    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expertTokensCountOrCumsumShape_, expertTokensCountOrCumsumDtype_,
                                                  OUTPUT_EXPERT_TOKENS_COUNT_INDEX));
    MIRV3_CHECK_GE_RET(
        GetTensorShapeDtype<false>(expandedScaleShape_, expandedScaleDtype_, OUTPUT_EXPANDED_SCALE_INDEX));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::GetInputAttrsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetInputAttrsInfo()");

    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);

    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(activeNum_, attrsPtr, ATTR_ACTIVE_NUM_INDEX));
    OP_LOGD(context_, "Get input attr activeNum = %ld.", activeNum_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertCapacity_, attrsPtr, ATTR_EXPERT_CAPACITY_INDEX));
    OP_LOGD(context_, "Get input attr expertCapacity = %ld.", expertCapacity_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertNum_, attrsPtr, ATTR_EXPERT_NUM_INDEX));
    OP_LOGD(context_, "Get input attr expertNum = %ld.", expertNum_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(dropPadMode_, attrsPtr, ATTR_DROP_PAD_MODE_INDEX));
    OP_LOGD(context_, "Get input attr dropPadMode = %ld.", dropPadMode_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertTokensNumType_, attrsPtr, ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX));
    OP_LOGD(context_, "Get input attr expertTokensNumType = %ld.", expertTokensNumType_);
    MIRV3_CHECK_GE_RET(GetInputAttr<bool>(expertTokensNumFlag_, attrsPtr, ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX));
    OP_LOGD(context_, "Get input attr expertTokensNumFlag = %ld.", expertTokensNumFlag_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(quantMode_, attrsPtr, ATTR_QUANT_MODE_INDEX));
    OP_LOGD(context_, "Get input attr quantMode = %ld.", quantMode_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(rowIdxType_, attrsPtr, ATTR_ROW_IDX_TYPE_INDEX));
    OP_LOGD(context_, "Get input attr rowIdxType = %ld.", rowIdxType_);
    // expertStart, expertEnd
    const auto *aerPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_EXPERT_RANGE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, aerPtr);
    int64_t aerLen = aerPtr->GetSize();
    OP_CHECK_IF(aerLen != 2,
                OP_LOGE(context_, "The list length of active_expert_range should be 2, current is %ld.", aerLen),
                return ge::GRAPH_FAILED);
    const int64_t *aerList = reinterpret_cast<const int64_t *>(aerPtr->GetData());
    expertStart_ = aerList[0];
    expertEnd_ = aerList[1];
    OP_LOGD(context_, "Extracted input attrs expertStart = %ld, expertEnd = %ld.", expertStart_, expertEnd_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckSetAttrs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckSetAttrs()");

    // activeNum 暂不使用，在取得n和k后进行校验activeNum==n*k
    // expertCapacity 暂不使用，也不校验
    // expertTokensNumType：expertNum的约束依赖expertTokensNumType，先校验expertTokensNumType
    OP_CHECK_IF((expertTokensNumType_ != EXPERT_TOKENS_TYPE_COUNT) &&
                    (expertTokensNumType_ != EXPERT_TOKENS_TYPE_KEY_VALUE),
                OP_LOGE(context_, "Attr expert_tokens_num_type currently supports: %ld or %ld, but got %ld.",
                        EXPERT_TOKENS_TYPE_COUNT, EXPERT_TOKENS_TYPE_KEY_VALUE, expertTokensNumType_),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->expertTokensNumType = expertTokensNumType_;
    // expertNum
    int64_t maxExpertNum = (expertTokensNumType_ == EXPERT_TOKENS_TYPE_COUNT) ? EXPERT_IDX_MAX : KV_MODE_EXPERT_IDX_MAX;
    OP_CHECK_IF(
        expertNum_ <= 0 || expertNum_ > maxExpertNum,
        OP_LOGE(context_, "Attr expert_num should be in range [1, %ld], current is %ld.", maxExpertNum, expertNum_),
        return ge::GRAPH_FAILED);
    tilingDataPtr_->expertNum = expertNum_;
    // drop_pad_mode 暂不使用，只校验
    OP_CHECK_IF(dropPadMode_ != DROP_PAD_MODE_DROPLESS,
                OP_LOGE(context_, "Attr drop_pad_mode currently supports %ld, but got %ld.", DROP_PAD_MODE_DROPLESS,
                        dropPadMode_),
                return ge::GRAPH_FAILED);
    // expertTokensNumFlag 暂不使用，只校验
    OP_CHECK_IF(expertTokensNumFlag_ != true,
                OP_LOGE(context_, "Attr expert_tokens_num_flag currently supports True, but got %s",
                        (expertTokensNumFlag_ ? "True" : "False")),
                return ge::GRAPH_FAILED);
    // quantMode
    OP_CHECK_IF(quantMode_ != QUANT_MODE_UNQUANT && quantMode_ != QUANT_MODE_DYNAMIC &&
                    quantMode_ != QUANT_MODE_MXFP8_E5M2 && quantMode_ != QUANT_MODE_MXFP8_E4M3FN &&
                    quantMode_ != QUANT_MODE_HIF8_CAST && quantMode_ != QUANT_MODE_HIF8_PERTENSOR &&
                    quantMode_ != QUANT_MODE_HIF8_PERTOKEN,
                OP_LOGE(context_, "Attr quant_mode currently supports (%ld, %ld, %ld, %ld, %ld, %ld, %ld), but got %ld",
                        QUANT_MODE_UNQUANT, QUANT_MODE_DYNAMIC, QUANT_MODE_MXFP8_E5M2, QUANT_MODE_MXFP8_E4M3FN,
                        QUANT_MODE_HIF8_CAST, QUANT_MODE_HIF8_PERTENSOR, QUANT_MODE_HIF8_PERTOKEN, quantMode_),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->quantMode = quantMode_;
    // rowIdxType
    OP_CHECK_IF(rowIdxType_ != ROW_IDX_SCATTER && rowIdxType_ != ROW_IDX_GATHER,
                OP_LOGE(context_, "row_idx_type currently supports %ld or %ld, but got %ld.", ROW_IDX_SCATTER,
                        ROW_IDX_GATHER, rowIdxType_),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->rowIdxType = rowIdxType_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckSetListAttrs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckSetListAttrs()");

    // expertStart, expertEnd
    OP_CHECK_IF(expertStart_ < 0,
                OP_LOGE(context_, "Extracted attr expert_start should be equal or greater than 0, current is %ld.",
                        expertStart_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertStart_ >= expertEnd_,
                OP_LOGE(context_,
                        "Extracted attr expert_start should be less than expert_end, current [expert_start, "
                        "expert_end) is [%ld, %ld).",
                        expertStart_, expertEnd_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertEnd_ > expertNum_,
                OP_LOGE(context_,
                        "Extracted attr expert_end should be equal or less than expert_num(%ld), current is %ld.",
                        expertNum_, expertEnd_),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->expertStart = expertStart_;
    tilingDataPtr_->expertEnd = expertEnd_;
    tilingDataPtr_->actualExpertNum = expertEnd_ - expertStart_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckInputX()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckInputX()");

    // rank
    auto rank = static_cast<int64_t>(xShape_.GetDimNum());
    OP_CHECK_IF(rank != 2, OP_LOGE(context_, "The rank of input x should be 2, current is %ld.", rank),
                return ge::GRAPH_FAILED);
    // dtype
    using ge::DataType;
    using std::unordered_set;
    static const unordered_set<DataType> UNQUANT_SUPPORTED_DTYPES = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                     DataType::DT_BF16, DataType::DT_INT8, DataType::DT_HIFLOAT8};
    static const unordered_set<DataType> DYNAMIC_QUANT_SUPPORTED_DTYPES = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                     DataType::DT_BF16, DataType::DT_INT8};
    static const std::unordered_set<DataType> MX_OR_HIF8_QUANT_SUPPORTED_DTYPES = {ge::DataType::DT_FLOAT16,
                                                                          ge::DataType::DT_BF16};
    unordered_set<DataType> supportedDtypes;
    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN || quantMode_ == QUANT_MODE_HIF8_CAST
        || quantMode_ == QUANT_MODE_HIF8_PERTENSOR || quantMode_ == QUANT_MODE_HIF8_PERTOKEN) {
        supportedDtypes = MX_OR_HIF8_QUANT_SUPPORTED_DTYPES;
    } else if (quantMode_ == QUANT_MODE_UNQUANT) {
        supportedDtypes = UNQUANT_SUPPORTED_DTYPES;
    } else {
        //! 出于历史调用的兼容性，这里不拦截quant_mode=1（动态量化）下输入x为int8类型，仅资料说明此时算子输出expandedX、expandedScale无意义
        supportedDtypes = DYNAMIC_QUANT_SUPPORTED_DTYPES;
    }
    OP_CHECK_IF(supportedDtypes.count(xDtype_) == 0,
                OP_LOGE(context_, "Unsupported dtype of input x: %d under quant_mode: %ld.", xDtype_, quantMode_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckInputExpertIdx()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckInputExpertIdx()");

    auto rank = static_cast<int64_t>(expertIdxShape_.GetDimNum());
    OP_CHECK_IF(rank != 2, OP_LOGE(context_, "The rank of input expert_idx should be 2, current is %ld.", rank),
                return ge::GRAPH_FAILED);
    int64_t expertIdxDim0 = expertIdxShape_.GetDim(0);
    int64_t xDim0 = xShape_.GetDim(0);
    OP_CHECK_IF(expertIdxDim0 != xDim0,
                OP_LOGE(context_, "Unsupported dim0 of input expert_idx: %ld, should be equal to dim0 of x: %ld",
                        expertIdxDim0, xDim0),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckInputScale()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckInputScale()");

    if (isInputScale_ == 0) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t expectedRankScale{-1};
    int64_t expectedDim0{-1};
    int64_t expectedDim1{-1};
    if (quantMode_ == QUANT_MODE_UNQUANT) {
        expectedRankScale = RANK_ONE;
        expectedDim0 = xShape_.GetDim(0);
    } else if (quantMode_ == QUANT_MODE_DYNAMIC) {
        expectedRankScale = RANK_TWO;
        expectedDim0 = expertEnd_ - expertStart_;
        expectedDim1 = xShape_.GetDim(1);
    }
    if (expectedRankScale != -1) {
        auto rankScale = static_cast<int64_t>(scaleShape_.GetDimNum());
        OP_CHECK_IF(rankScale != expectedRankScale,
                    OP_LOGE(context_, "The rank of input scale should be %ld under quant_mode %ld, current is %ld",
                            expectedRankScale, quantMode_, rankScale),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim0 != -1) {
        auto dim0 = scaleShape_.GetDim(0);
        OP_CHECK_IF(dim0 != expectedDim0,
                    OP_LOGE(context_, "The dim0 of input scale should be %ld under quant_mode %ld, current is %ld",
                            expectedDim0, quantMode_, dim0),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim1 != -1) {
        auto dim1 = scaleShape_.GetDim(1);
        OP_CHECK_IF(dim1 != expectedDim1,
                    OP_LOGE(context_, "The dim1 of input scale should be %ld under quant_mode %ld, current is %ld",
                            expectedDim1, quantMode_, dim1),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(scaleDtype_ != ge::DataType::DT_FLOAT,
                OP_LOGE(context_, "Unsupported dtype of input scale: %d, should be: DT_FLOAT(%d).", xDtype_,
                        ge::DataType::DT_FLOAT),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckSetInputs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckSetInputs()");

    MIRV3_CHECK_GE_RET(CheckInputX());
    MIRV3_CHECK_GE_RET(CheckInputExpertIdx());
    MIRV3_CHECK_GE_RET(CheckInputScale());

    n_ = xShape_.GetDim(0);
    k_ = expertIdxShape_.GetDim(1);
    cols_ = xShape_.GetDim(1);
    totalLength_ = n_ * k_;
    tilingDataPtr_->n = n_;
    tilingDataPtr_->k = k_;
    tilingDataPtr_->cols = cols_;

    if (activeNum_ != ACTIVE_NUM_MIN_VALUE) {
        //! 出于历史调用的兼容性，保留校验activeNum=n*k，但实际上不使用该属性
        OP_CHECK_IF(
            activeNum_ != totalLength_,
            OP_LOGE(context_, "Attr active_num should equal to bs*k(%ld), current is %ld.", totalLength_, activeNum_),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputExpandedX()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputExpandedX()");

    auto rank = static_cast<int64_t>(expandedXShape_.GetDimNum());
    OP_CHECK_IF(rank != RANK_TWO, OP_LOGE(context_, "The rank of output expanded_x should be 2, current is %ld.", rank),
                return ge::GRAPH_FAILED);
    int64_t dim0 = expandedXShape_.GetDim(0);
    OP_CHECK_IF(dim0 != totalLength_,
                OP_LOGE(context_, "The dim0 of output expanded_x should be %ld, current is %ld.", totalLength_, dim0),
                return ge::GRAPH_FAILED);
    int64_t dim1 = expandedXShape_.GetDim(1);
    OP_CHECK_IF(dim1 != cols_,
                OP_LOGE(context_, "The dim1 of output expanded_x should be %ld, current is %ld.", cols_, dim1),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputExpandedRowIdx()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputExpandedRowIdx()");

    auto rank = static_cast<int64_t>(expandedRowIdxShape_.GetDimNum());
    OP_CHECK_IF(rank != RANK_ONE,
                OP_LOGE(context_, "The rank of output expanded_row_idx should be 1, current is %ld.", rank),
                return ge::GRAPH_FAILED);
    int64_t dim0 = expandedRowIdxShape_.GetDim(0);
    OP_CHECK_IF(
        dim0 != totalLength_,
        OP_LOGE(context_, "The dim0 of output expanded_row_idx should be %ld, current is %ld.", totalLength_, dim0),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputExpertTokensCountOrCumsum()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputExpertTokensCountOrCumsum()");

    int64_t expectedRank{-1}, expectedDim0{-1}, expectedDim1{-1};
    if (expertTokensNumType_ == EXPERT_TOKENS_TYPE_COUNT) {
        expectedRank = RANK_ONE;
        expectedDim0 = expertEnd_ - expertStart_;
    } else if (expertTokensNumType_ == EXPERT_TOKENS_TYPE_KEY_VALUE) {
        expectedRank = RANK_TWO;
        expectedDim0 = expertNum_;
        expectedDim1 = DIM_TWO;
    }

    auto rank = static_cast<int64_t>(expertTokensCountOrCumsumShape_.GetDimNum());
    if (expectedRank != -1) {
        OP_CHECK_IF(
            rank != expectedRank,
            OP_LOGE(context_,
                    "The rank of output expert_tokens_count_or_cumsum should be %ld under expert_tokens_num_type "
                    "%ld, current is %ld.",
                    expectedRank, expertTokensNumType_, rank),
            return ge::GRAPH_FAILED);
    }
    if (expectedDim0 != -1) {
        int64_t dim0 = expertTokensCountOrCumsumShape_.GetDim(0);
        OP_CHECK_IF(dim0 != expectedDim0,
                    OP_LOGE(context_,
                            "The dim0 of output expert_tokens_count_or_cumsum should be %ld under "
                            "expert_tokens_num_type %ld, current is %ld.",
                            expectedDim0, expertTokensNumType_, dim0),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim1 != -1) {
        int64_t dim1 = expertTokensCountOrCumsumShape_.GetDim(1);
        OP_CHECK_IF(dim1 != expectedDim1,
                    OP_LOGE(context_,
                            "The dim1 of output expert_tokens_count_or_cumsum should be %ld under "
                            "expert_tokens_num_type %ld,, current is %ld.",
                            expectedDim1, expertTokensNumType_, dim1),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputExpandedScale()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputExpandedScale()");

    int64_t expectedRank{-1}, expectedDim0{-1}, expectedDim1{-1};
    if ((quantMode_ == QUANT_MODE_UNQUANT && isInputScale_ == 1) || (quantMode_ == QUANT_MODE_DYNAMIC)) {
        expectedRank = RANK_ONE;
        expectedDim0 = totalLength_;
    } else if ((quantMode_ == QUANT_MODE_MXFP8_E5M2) || (quantMode_ == QUANT_MODE_MXFP8_E4M3FN)) {
        expectedRank = RANK_TWO;
        expectedDim0 = totalLength_;
        expectedDim1 = Ops::Base::CeilAlign<int64_t>(Ops::Base::CeilDiv<int64_t>(cols_, MX_QUANT_BLOCK_SIZE), 2LL);
    } else if ((quantMode_ == QUANT_MODE_HIF8_PERTOKEN)) {
        expectedRank = RANK_ONE;
        expectedDim0 = totalLength_;
    } else if (quantMode_ == QUANT_MODE_HIF8_CAST) {
        return ge::GRAPH_SUCCESS;
    }
    auto rank = static_cast<int64_t>(expandedScaleShape_.GetDimNum());
    if (expectedRank != -1) {
        OP_CHECK_IF(rank != expectedRank,
                    OP_LOGE(context_,
                            "The rank of output expanded_scale should be %ld under quant_mode "
                            "%ld, current is %ld.",
                            expectedRank, quantMode_, rank),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim0 != -1) {
        int64_t dim0 = expandedScaleShape_.GetDim(0);
        OP_CHECK_IF(dim0 != expectedDim0,
                    OP_LOGE(context_,
                            "The dim0 of output expanded_scale should be %ld under "
                            "quant_mode %ld, current is %ld.",
                            expectedDim0, quantMode_, dim0),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim1 != -1) {
        int64_t dim1 = expandedScaleShape_.GetDim(1);
        OP_CHECK_IF(dim1 != expectedDim1,
                    OP_LOGE(context_,
                            "The dim1 of output expanded_scale should be %ld under "
                            "quant_mode %ld,, current is %ld.",
                            expectedDim1, quantMode_, dim1),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputs()");

    MIRV3_CHECK_GE_RET(CheckOutputExpandedX());
    MIRV3_CHECK_GE_RET(CheckOutputExpandedRowIdx());
    MIRV3_CHECK_GE_RET(CheckOutputExpertTokensCountOrCumsum());
    MIRV3_CHECK_GE_RET(CheckOutputExpandedScale());

    return ge::GRAPH_SUCCESS;
}

void MoeInitRoutingV3Arch35TilingClass::LogBaseTilingData()
{
    std::stringstream ss;
    ss << "\n[TilingKey]\n" << tilingKey_ << "\n[WorkspaceSize]\n" << workspaceSize_ << "\n";
    ss << "[MoeInitRoutingV3Arch35TilingData]\n";
    ss << "coreNum = " << tilingDataPtr_->coreNum << "\n";
    ss << "n = " << tilingDataPtr_->n << "\n";
    ss << "cols = " << tilingDataPtr_->cols << "\n";
    ss << "k = " << tilingDataPtr_->k << "\n";
    ss << "expertStart = " << tilingDataPtr_->expertStart << "\n";
    ss << "expertEnd = " << tilingDataPtr_->expertEnd << "\n";
    ss << "actualExpertNum = " << tilingDataPtr_->actualExpertNum << "\n";
    ss << "quantMode = " << tilingDataPtr_->quantMode << "\n";
    ss << "rowIdxType = " << tilingDataPtr_->rowIdxType << "\n";
    ss << "isInputScale = " << tilingDataPtr_->isInputScale << "\n";
    ss << "isInputOffset = " << tilingDataPtr_->isInputOffset << "\n";
    ss << "expertNum = " << tilingDataPtr_->expertNum << "\n";
    ss << "expertTokensNumType = " << tilingDataPtr_->expertTokensNumType << "\n";
    ss << "expertTokensNumFlag = " << tilingDataPtr_->expertTokensNumFlag << "\n";
    ss << "gatherFirstFullload = " << tilingDataPtr_->gatherFirstFullload << "\n";
    ss << "epFullload = " << tilingDataPtr_->epFullload << "\n";
    ss << "activeNum = " << tilingDataPtr_->activeNum << "\n";
    ss << "dropPadMode = " << tilingDataPtr_->dropPadMode << "\n";
    ss << "smoothType = " << tilingDataPtr_->smoothType << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3Arch35TilingClass::LogVbsTilingData()
{
    std::stringstream ss;
    auto vbsTiling = &(tilingDataPtr_->vbsComputeParamsOp);
    ss << "\n[MoeV3Arch35VBSComputeTilingData]\n";
    ss << "needCoreNum = " << vbsTiling->needCoreNum << "\n";
    ss << "perCoreElements = " << vbsTiling->perCoreElements << "\n";
    ss << "perCoreLoops = " << vbsTiling->perCoreLoops << "\n";
    ss << "perCorePerLoopElements = " << vbsTiling->perCorePerLoopElements << "\n";
    ss << "perCoreLastLoopElements = " << vbsTiling->perCoreLastLoopElements << "\n";
    ss << "lastCoreElements = " << vbsTiling->lastCoreElements << "\n";
    ss << "lastCoreLoops = " << vbsTiling->lastCoreLoops << "\n";
    ss << "lastCorePerLoopElements = " << vbsTiling->lastCorePerLoopElements << "\n";
    ss << "lastCoreLastLoopElements = " << vbsTiling->lastCoreLastLoopElements << "\n";
    ss << "oneLoopMaxElements = " << vbsTiling->oneLoopMaxElements << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3Arch35TilingClass::LogVmsMiddleTilingData()
{
    std::stringstream ss;
    auto vmsMiddleTiling = &(tilingDataPtr_->vmsMiddleComputeParamsOp);
    ss << "\n[MoeV3Arch35VMSMiddleComputeTilingData]\n";
    ss << "needCoreNum = " << vmsMiddleTiling->needCoreNum << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3Arch35TilingClass::LogSortOutTilingData()
{
    std::stringstream ss;
    auto sortOutTiling = &(tilingDataPtr_->sortOutComputeParamsOp);
    ss << "\n[MoeV3Arch35SortOutComputeTilingData]\n";
    ss << "oneLoopMaxElements = " << sortOutTiling->oneLoopMaxElements << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3Arch35TilingClass::LogExpertTokensCountTilingData()
{
    std::stringstream ss;
    auto expertTokensCountTiling = &(tilingDataPtr_->expertTokensCountTilingDataOp);
    ss << "\n[MoeV3Arch35ExpertTokensCountTilingData]\n";
    ss << "needCoreNum = " << expertTokensCountTiling->needCoreNum << "\n";
    ss << "perCoreElements = " << expertTokensCountTiling->perCoreElements << "\n";
    ss << "lastCoreElements = " << expertTokensCountTiling->lastCoreElements << "\n";
    ss << "perCoreLoops = " << expertTokensCountTiling->perCoreLoops << "\n";
    ss << "perCorePerLoopElements = " << expertTokensCountTiling->perCorePerLoopElements << "\n";
    ss << "perCoreLastLoopElements = " << expertTokensCountTiling->perCoreLastLoopElements << "\n";
    ss << "lastCoreLoops = " << expertTokensCountTiling->lastCoreLoops << "\n";
    ss << "lastCorePerLoopElements = " << expertTokensCountTiling->lastCorePerLoopElements << "\n";
    ss << "lastCoreLastLoopElements = " << expertTokensCountTiling->lastCoreLastLoopElements << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3Arch35TilingClass::LogGatherOutTilingData()
{
    std::stringstream ss;
    auto gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    ss << "\n[MoeV3Arch35GatherOutComputeTilingData]\n";
    ss << "needCoreNum = " << gatherOutTiling->needCoreNum << "\n";
    ss << "perCoreIndicesElements = " << gatherOutTiling->perCoreIndicesElements << "\n";
    ss << "lastCoreIndicesElements = " << gatherOutTiling->lastCoreIndicesElements << "\n";
    ss << "perCoreIndicesLoops = " << gatherOutTiling->perCoreIndicesLoops << "\n";
    ss << "perCorePerLoopIndicesElements = " << gatherOutTiling->perCorePerLoopIndicesElements << "\n";
    ss << "perCoreLastLoopIndicesElements = " << gatherOutTiling->perCoreLastLoopIndicesElements << "\n";
    ss << "lastCoreIndicesLoops = " << gatherOutTiling->lastCoreIndicesLoops << "\n";
    ss << "lastCorePerLoopIndicesElements = " << gatherOutTiling->lastCorePerLoopIndicesElements << "\n";
    ss << "colsLoops = " << gatherOutTiling->colsLoops << "\n";
    ss << "perLoopCols = " << gatherOutTiling->perLoopCols << "\n";
    ss << "lastLoopCols = " << gatherOutTiling->lastLoopCols << "\n";
    ss << "activeNum = " << gatherOutTiling->activeNum << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4VBSOneCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4VBSOneCoreCompute(...)");

    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = totalLength_;
    vbsTiling->perCoreLoops = 1;
    vbsTiling->perCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4VBSMultiCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4VBSMultiCoreCompute(...)");

    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, sortLoopMaxElement_); // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));      // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivCoreNum_);                            // 不能超过物理核数

    OP_CHECK_IF(needCoreNum == 0, OP_LOGE(opName, "Variale needCoreNum cannot be 0."), return;);
    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (totalLength_ / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength_ - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops =
            Ops::Base::CeilDiv(vbsTiling->perCoreElements, sortLoopMaxElement_); // 每个核处理的loop数
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, sortLoopMaxElement_);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = totalLength_ - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops),
                               SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
    OP_CHECK_IF(vbsTiling->lastCoreLastLoopElements <= 0, OP_LOGE(opName, "vbs tiling failed"), ;);
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4VBSCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4VBSCompute()");

    if (totalLength_ <= sortLoopMaxElement_) { // 排序只用到一个核排序
        sortMode_ = 0;
    } else {
        sortMode_ = 1;
    }

    auto *vbsTiling = &(tilingDataPtr_->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = sortLoopMaxElement_;
    if (sortMode_ == 0) { // 只用到一个核
        Tiling4VBSOneCoreCompute(vbsTiling);
    } else {
        Tiling4VBSMultiCoreCompute(vbsTiling);
    }

    LogVbsTilingData();
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4VMSMiddleCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4VMSMiddleCompute()");

    auto *vbsTiling = &(tilingDataPtr_->vbsComputeParamsOp);
    auto *vmsMiddleTiling = &(tilingDataPtr_->vmsMiddleComputeParamsOp);
    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        vmsMiddleTiling->needCoreNum = 0;         // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum; // 需要的核数

    LogVmsMiddleTilingData();
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4SortOutCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4SortOutCompute()");

    auto *sortOutTiling = &(tilingDataPtr_->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;

    LogSortOutTilingData();
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4ExpertTokensCountCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4ExpertTokensCountCompute()");

    auto *tokensCountTiling = &(tilingDataPtr_->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingDataPtr_->n * tilingDataPtr_->k;
    int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, aivCoreNum_);
    int64_t needCoreNum = Ops::Base::CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingDataPtr_->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE) ?
                                   tilingDataPtr_->actualExpertNum :
                                   (tilingDataPtr_->actualExpertNum + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (availUbSize_ -
         Ops::Base::CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = Ops::Base::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = Ops::Base::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;

    LogExpertTokensCountTilingData();
}

MultipleParams MoeInitRoutingV3Arch35TilingClass::GetMultipleParams()
{
    MultipleParams params;
    params.colMultiple = NUM_TWO * inputXDtypeSize_;
    params.rowMultiple = NUM_TWO;
    if (quantMode_ == QUANT_MODE_DYNAMIC) {
        params.colMultiple = DYNAMIC_QUANT_COLS_BUFFER;
        params.rowMultiple = NUM_FOUR;
    } else if (quantMode_ == QUANT_MODE_HIF8_CAST && xDtype_ == ge::DataType::DT_BF16) {
        // 当BF16->FP32->HIF8转换时，额外需要存储FP32的中间结果
        params.colMultiple = NUM_TWO * (inputXDtypeSize_ + inputXDtypeSize_ * BF16_TO_FP32_SIZE_FACTOR);
    } else if (quantMode_ == QUANT_MODE_HIF8_PERTOKEN) {
        params.colMultiple = HIF8_PERTOKEN_QUANT_COLS_BUFFER;
        params.rowMultiple = NUM_FOUR;
    } else if (quantMode_ == QUANT_MODE_HIF8_PERTENSOR) {
        params.colMultiple = HIF8_PERTENSOR_QUANT_COLS_BUFFER;
        params.rowMultiple = NUM_FOUR;
    }
    return params;
}

PerLoopParams MoeInitRoutingV3Arch35TilingClass::GetPerLoopParams(MultipleParams& multipleParams)
{
    PerLoopParams perLoopParams;
    perLoopParams.perLoopCols = tilingDataPtr_->cols;
    if (quantMode_ == QUANT_MODE_HIF8_PERTENSOR) {
        perLoopParams.perLoopMaxIndicesElements =
            (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple) /
            multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        while (perLoopParams.perLoopMaxIndicesElements <= 0) {
            perLoopParams.perLoopCols = Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO);
            perLoopParams.perLoopMaxIndicesElements =
                (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple) /
                multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        }
    } else {
        perLoopParams.perLoopMaxIndicesElements =
            (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple -
            UB_BLOCK_SIZE * NUM_TWO) / multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        while (perLoopParams.perLoopMaxIndicesElements <= 0) {
            perLoopParams.perLoopCols = Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO);
            perLoopParams.perLoopMaxIndicesElements =
                (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple -
                UB_BLOCK_SIZE * NUM_TWO) / multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        }
    }
    return perLoopParams;
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4GatherOutCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4GatherOutCompute()");

    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    MultipleParams multipleParams = GetMultipleParams();
    PerLoopParams perLoopParams = GetPerLoopParams(multipleParams);

    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopParams.perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopParams.perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopParams.perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopParams.perLoopMaxIndicesElements,
        lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;

    LogGatherOutTilingData();
}

int64_t MoeInitRoutingV3Arch35TilingClass::CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CalcMaxRowIdxPerLoopMxQuant(...)");

    // 输入x[cols]所占大小：cols*sizeof(dtypeX)+cols*sizeof(Byte)
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize_) + AlignBytes(perLoopCols, sizeof(int8_t));
    // 输出scale[cols]所占大小：scaleCols*sizeof(dtypeX)*2+scaleCols*sizeof(Byte)
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize_) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    // 输出xOut[cols]所占大小：
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    // 返回的是(availUbSize-每行输入x、输出scale、输出xOut所占的大小)/sizeof(int32)，应该是留给sortedRowIdx元素的数目
    return (availUbSize_ - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

void MoeInitRoutingV3Arch35TilingClass::Tiling4GatherOutMxQuant()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::Tiling4GatherOutMxQuant()");

    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = Ops::Base::CeilAlign(tilingDataPtr_->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols);
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = Ops::Base::CeilAlign(Ops::Base::CeilDiv(perLoopCols, NUM_TWO), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols);
    }
    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->perCoreIndicesElements =
        perCoreIndicesElements; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->lastCoreIndicesElements =
        lastCoreIndicesElements; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops =
        perCoreIndicesLoops; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements =
        perCoreLastLoopIndicesElements; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops =
        lastCoreIndicesLoops; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements =
        lastCoreLastLoopIndicesElements; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值

    LogGatherOutTilingData();
}

REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV3, MoeInitRoutingV3Arch35TilingClass,
                             1000); // If 950, use this tiling class.
} // namespace optiling
