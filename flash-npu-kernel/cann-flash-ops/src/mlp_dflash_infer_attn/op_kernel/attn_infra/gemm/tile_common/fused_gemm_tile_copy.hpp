#ifndef FUSED_TILE_COPY_HPP
#define FUSED_TILE_COPY_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_copy_gm_to_l1.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_copy_l0c_to_gm.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_copy_l1_to_l0a.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_copy_l1_to_l0b.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_copy_l1_to_bt.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_gemm_copy_gm_to_ub.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_gemm_copy_ub_to_gm.hpp"
#include "../../../attn_infra/gemm/fused_helper.hpp"


namespace NpuArch::Gemm::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag,
            typename BiasTypeSelector::GMBiasType,
            typename BiasTypeSelector::L1BiasType>>;
};

//////////////////////////////
} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_TILE_COPY_HPP