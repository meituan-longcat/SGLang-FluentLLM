#include "mlp_split_swiglu_tiling.h"

#include <algorithm>
#include <cstdint>

#include "error/ops_error.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

namespace {
constexpr int64_t INPUT_X_INDEX = 0;
constexpr int64_t INPUT_GROUP_LIST_INDEX = 1;
constexpr int64_t ATTR_LOCAL_EXPERT_START_INDEX = 0;
constexpr int64_t ATTR_LOCAL_EXPERT_END_INDEX = 1;
constexpr int64_t DENSE_DIM_NUM = 2;
constexpr int64_t GROUP_LIST_DIM_NUM = 1;
constexpr int64_t HIDDEN_ALIGNMENT = 16;
constexpr int64_t DB = 2;
constexpr int64_t MAX_UB_BYTES = 184 * 1024;
constexpr int64_t UB_SPLIT_NUM = 3;
constexpr int64_t UB_SPLIT_BYTES = MAX_UB_BYTES / UB_SPLIT_NUM / 1024 * 1024;
constexpr int64_t UB_SPLIT_DB_BYTES = UB_SPLIT_BYTES / DB / 1024 * 1024;
constexpr uint64_t TILING_KEY_BF16 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;

template <typename TilingData>
ge::graphStatus FinalizeTiling(gert::TilingContext* context, TilingData& tiling, uint32_t coreNum)
{
    context->SetBlockDim(coreNum);
    OPS_ERR_IF(context->GetRawTilingData() == nullptr,
               OPS_LOG_E(context->GetNodeName(), "raw tiling data is null"),
               return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    auto workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetDtypeTilingKey(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OPS_ERR_IF(xDesc == nullptr,
               OPS_LOG_E(context->GetNodeName(), "x desc is null"),
               return ge::GRAPH_FAILED);
    auto xType = xDesc->GetDataType();
    if (xType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
        return ge::GRAPH_SUCCESS;
    }
    if (xType == ge::DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_FP16);
        return ge::GRAPH_SUCCESS;
    }
    OPS_LOG_E(context->GetNodeName(), "unsupported x dtype: %d", static_cast<int32_t>(xType));
    return ge::GRAPH_FAILED;
}

uint32_t GetVectorCoreNum(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
}
}  // namespace

namespace optiling {
struct SplitSwigluCompileInfo {};

static ge::graphStatus TilingPrepareForSplitSwiglu(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMlpSplitSwiglu(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("MlpSplitSwiglu", "tiling context is null"),
               return ge::GRAPH_FAILED);
    SplitSwigluTilingData tiling;

    const auto xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    const auto groupListStorageShape = context->GetInputShape(INPUT_GROUP_LIST_INDEX);
    OPS_ERR_IF(xStorageShape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "x shape is null"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(groupListStorageShape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "group_list shape is null"),
               return ge::GRAPH_FAILED);

    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    const gert::Shape& groupListShape = groupListStorageShape->GetStorageShape();
    OPS_ERR_IF(xShape.GetDimNum() != DENSE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "x dim num must be 2, got %zu", xShape.GetDimNum()),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(groupListShape.GetDimNum() != GROUP_LIST_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "group_list dim num must be 1, got %zu", groupListShape.GetDimNum()),
               return ge::GRAPH_FAILED);

    const int64_t hidden = xShape.GetDim(1);
    const int64_t expertNum = groupListShape.GetDim(0);
    OPS_ERR_IF(hidden <= 0 || (hidden % 2) != 0,
               OPS_LOG_E(context->GetNodeName(), "hidden must be positive and even, got %ld", hidden),
               return ge::GRAPH_FAILED);
    const int64_t outHidden = hidden / 2;
    OPS_ERR_IF((outHidden % HIDDEN_ALIGNMENT) != 0,
               OPS_LOG_E(context->GetNodeName(), "out hidden must be aligned to %ld, got %ld",
                         HIDDEN_ALIGNMENT, outHidden),
               return ge::GRAPH_FAILED);
    const int64_t tokenCapacity = UB_SPLIT_DB_BYTES / static_cast<int64_t>(sizeof(float)) / outHidden;
    OPS_ERR_IF(tokenCapacity < 1,
               OPS_LOG_E(context->GetNodeName(), "out hidden is too large: %ld", outHidden),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_E(context->GetNodeName(), "attrs is null"),
               return ge::GRAPH_FAILED);
    const int64_t* localExpertStartPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_EXPERT_START_INDEX);
    const int64_t* localExpertEndPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_EXPERT_END_INDEX);
    OPS_ERR_IF(localExpertStartPtr == nullptr || localExpertEndPtr == nullptr,
               OPS_LOG_E(context->GetNodeName(), "local expert attrs are null"),
               return ge::GRAPH_FAILED);

    const int64_t localExpertStart = *localExpertStartPtr;
    const int64_t localExpertEnd = *localExpertEndPtr;
    OPS_ERR_IF(localExpertStart < 0,
               OPS_LOG_E(context->GetNodeName(), "local_expert_start must be non-negative, got %ld",
                         localExpertStart),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(localExpertStart >= localExpertEnd,
               OPS_LOG_E(context->GetNodeName(), "local_expert_start must be less than local_expert_end"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(localExpertEnd > expertNum,
               OPS_LOG_E(context->GetNodeName(), "local_expert_end %ld exceeds expert num %ld",
                         localExpertEnd, expertNum),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(SetDtypeTilingKey(context) != ge::GRAPH_SUCCESS,
               OPS_LOG_E(context->GetNodeName(), "set tiling key failed"),
               return ge::GRAPH_FAILED);

    const uint32_t coreNum = std::max<uint32_t>(GetVectorCoreNum(context), 1U);
    tiling.set_vecCoreNum(coreNum);
    tiling.set_inHidden(static_cast<uint32_t>(hidden));
    tiling.set_outHidden(static_cast<uint32_t>(outHidden));
    tiling.set_expertNum(static_cast<uint32_t>(expertNum));
    tiling.set_localExpertStart(static_cast<uint32_t>(localExpertStart));
    tiling.set_localExpertEnd(static_cast<uint32_t>(localExpertEnd));

    return FinalizeTiling(context, tiling, coreNum);
}

IMPL_OP_OPTILING(MlpSplitSwiglu)
    .Tiling(TilingForMlpSplitSwiglu)
    .TilingParse<SplitSwigluCompileInfo>(TilingPrepareForSplitSwiglu);
}  // namespace optiling
