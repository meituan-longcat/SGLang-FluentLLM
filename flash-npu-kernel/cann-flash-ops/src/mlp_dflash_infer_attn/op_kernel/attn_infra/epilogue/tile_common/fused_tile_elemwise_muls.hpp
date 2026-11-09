 #ifndef EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
 #define EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP

 #include "../../../attn_infra/gemm/fused_helper.hpp"

 namespace NpuArch::Epilogue::Tile
 {
 template<
     class ArchTag_,
     class ComputeType_,
     uint32_t COMPUTE_LENGTH_
 >
 struct TileElemWiseMuls{
     using ArchTag = ArchTag_;
     using ElementCompute = typename ComputeType_::Element;

     static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

     __aicore__ inline
     TileElemWiseMuls(){}

     __aicore__ inline
     void operator()(
         AscendC::LocalTensor<ElementCompute> dstLocal,
         AscendC::LocalTensor<ElementCompute> srcTensor,
         ElementCompute scalar
     ){
         AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
     }
 };
 }

 #endif // EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP