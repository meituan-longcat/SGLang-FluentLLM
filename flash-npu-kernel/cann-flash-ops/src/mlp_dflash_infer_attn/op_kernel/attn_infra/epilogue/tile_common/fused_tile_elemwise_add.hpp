#ifndef EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP
#define EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Tile 
{

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    uint32_t COMPUTE_LENGTH_
>
struct TileElemWiseAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    __aicore__ inline
    TileElemWiseAdd() {}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1
    )
    {
        // Do the calculation
        AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
    }
};

} // namespace NpuArch::Epilogue::Tile

#endif