#ifndef FUSED_BLOCK_MMAD_HPP
#define FUSED_BLOCK_MMAD_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_tile_mmad.hpp"

namespace NpuArch::Gemm::Block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace NpuArch::Gemm::Block

#include "../../../attn_infra/gemm/block/fused_block_mmad_qk.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_qk_decode.hpp"
#include "../../../attn_infra/gemm/block/fused_block_mmad_pv.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_pv_decode.hpp"

#endif // GEMM_BLOCK_BLOCK_MMAD_HPP