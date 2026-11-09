#ifndef EPILOGUE_TILE_TILE_ELEMWISE_MUL_HPP
#define EPILOGUE_TILE_TILE_ELEMWISE_MUL_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Tile 
{

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    class TileShape_
>
struct TileElemwiseMul {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    __aicore__ inline
    TileElemwiseMul() {}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1
    )
    {
        // Do the calculation
        AscendC::Mul(ubOut, ubIn0, ubIn1, TileShape::COUNT);
    }
};

} // namespace NpuArch::Epilogue::Tile

#endif