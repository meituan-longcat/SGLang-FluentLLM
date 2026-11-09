#ifndef MLP_SPLIT_SWIGLU_TILING_H_
#define MLP_SPLIT_SWIGLU_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SplitSwigluTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, inHidden);
  TILING_DATA_FIELD_DEF(uint32_t, outHidden);
  TILING_DATA_FIELD_DEF(uint32_t, expertNum);
  TILING_DATA_FIELD_DEF(uint32_t, localExpertStart);
  TILING_DATA_FIELD_DEF(uint32_t, localExpertEnd);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MlpSplitSwiglu, SplitSwigluTilingData)
}  // namespace optiling

#endif  // MLP_SPLIT_SWIGLU_TILING_H_
