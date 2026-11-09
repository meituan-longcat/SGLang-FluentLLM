#ifndef FUSED_EPILOGUE_DISPATCH_POLICY_HPP
#define FUSED_EPILOGUE_DISPATCH_POLICY_HPP

#include "../../attn_infra/fused_base_defs.hpp"
#include "../../attn_infra/arch/fused_arch.hpp"

namespace NpuArch::Epilogue 
{

enum class LseMode {NONE = 0, OUT_ONLY = 1};
enum class SinkMode {DISABLE = 0, ENABLE = 1};
enum class MaskMode {
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SPEC = 2,
    MASK_SWA = 4
};
// For AtlasA2, FA Infer online Softmax
template <LseMode LSE_MODE_, SinkMode SINK_MODE_, MaskMode MASK_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2OnlineSoftmax {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr SinkMode SINK_MODE = SINK_MODE_;
    static constexpr MaskMode MASK_MODE = MASK_MODE_;
};

// For AtlasA2, FA Infer RescaleO
template <LseMode LSE_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2RescaleO {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

// For AtlasA2, FA Infer Deal kv-len=0
template <LseMode LSE_MODE_>
struct EpilogueAtlasA2InitOutWhenZero {
    using ArchTag = Arch::AtlasA2;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};


}  // namespace NpuArch::Epilogue

#endif  // EPILOGUE_DISPATCH_POLICY_HPP