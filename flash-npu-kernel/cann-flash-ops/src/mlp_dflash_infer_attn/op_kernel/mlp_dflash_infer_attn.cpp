/*!
 * \file mlp_dflash_infer_attn.cpp
 * \brief MlpDflashInferAttn kernel 入口：DFlash PageAttention（FIA sparse_mode=15 专用算子版，arch22）。
 * 仅 4 个 tiling key：fp16/bf16 × lse off/on；paged + TND + NO_MASK + 非 FD 固定。
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "mlp_dflash_infer_attn_tilingkey.h"
#include "flash_attention_interface.cpp"

// GM 顺序对齐 def.cpp：query/key/value/blockTable/actualSeqLengths/actualSeqLengthsKv
//   → attentionOut/softmaxLse → workspace/tiling
extern "C" __global__ __aicore__ void mlp_dflash_infer_attn(__gm__ uint8_t* query,
                                                            __gm__ uint8_t* key,
                                                            __gm__ uint8_t* value,
                                                            __gm__ uint8_t* block_table,
                                                            __gm__ uint8_t* actual_seq_lengths,
                                                            __gm__ uint8_t* actual_seq_lengths_kv,
                                                            __gm__ uint8_t* attention_out,
                                                            __gm__ uint8_t* softmax_lse,
                                                            __gm__ uint8_t* workspace,
                                                            __gm__ uint8_t* tiling)
{
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(FAInferTilingData, tiling_data_in, tiling);
    const FAInferTilingData *__restrict tiling_data = &tiling_data_in;

    if (TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING)) {
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, nullptr, nullptr, block_table, attention_out, softmax_lse,
            actual_seq_lengths, actual_seq_lengths_kv, user, tiling_data, nullptr);
    }
    if (TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING)) {
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, nullptr, nullptr, block_table, attention_out, softmax_lse,
            actual_seq_lengths, actual_seq_lengths_kv, user, tiling_data, nullptr);
    }
    if (TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING)) {
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, nullptr, nullptr, block_table, attention_out, softmax_lse,
            actual_seq_lengths, actual_seq_lengths_kv, user, tiling_data, nullptr);
    }
    if (TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING)) {
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, nullptr, nullptr, block_table, attention_out, softmax_lse,
            actual_seq_lengths, actual_seq_lengths_kv, user, tiling_data, nullptr);
    }
}
