#ifndef FUSED_GEMM_TYPE_HPP
#define FUSED_GEMM_TYPE_HPP

#include "../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Gemm 
{
template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType 
{
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = POSITION_;
};

} // namespace NpuArch::Gemm

#endif // GEMM_GEMM_TYPE_HPP