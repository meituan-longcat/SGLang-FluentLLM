#ifndef FUSED_BLOCK_EPILOGUE_HPP
#define FUSED_BLOCK_EPILOGUE_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace NpuArch::Epilogue::Block

#include "../../../attn_infra/epilogue/block/fused_block_epilogue_online_softmax.hpp"
#include "../../../attn_infra/epilogue/block/fused_block_epilogue_online_softmax_low_prec.hpp"
#include "../../../attn_infra/epilogue/block/fused_block_epilogue_rescale_o.hpp"
#include "../../../attn_infra/epilogue/block/CombineScale.hpp"
#include "../../../attn_infra/epilogue/block/fused_block_epilogue_rescale_o_low_prec.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_init_outputs.hpp"
#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP