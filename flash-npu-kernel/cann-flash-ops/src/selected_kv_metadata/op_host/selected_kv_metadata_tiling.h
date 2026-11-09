#pragma once
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(SelectedKvMetadataTilingData)
TILING_DATA_FIELD_DEF(uint32_t, requests);
TILING_DATA_FIELD_DEF(uint32_t, queries);
TILING_DATA_FIELD_DEF(uint32_t, pages);
TILING_DATA_FIELD_DEF(uint32_t, world);
TILING_DATA_FIELD_DEF(uint32_t, localRows);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF(uint32_t, phase);
TILING_DATA_FIELD_DEF(uint32_t, cores);
TILING_DATA_FIELD_DEF(uint32_t, span);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SelectedKvMetadata, SelectedKvMetadataTilingData)
struct SelectedKvMetadataCompileInfo {};
} // namespace optiling
