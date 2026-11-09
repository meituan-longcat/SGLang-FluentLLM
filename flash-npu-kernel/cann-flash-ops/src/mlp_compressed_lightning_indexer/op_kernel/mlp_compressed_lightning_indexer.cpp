/*!
 * \file lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mlp_compressed_lightning_indexer_template_tiling_key.h"
#include "mlp_compressed_lightning_indexer_kernel.h"

using namespace LIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<LIType<__VA_ARGS__>> op;                                                                         \
        LI_COPY_TILING_DATA(LICompressedTilingData, tiling);                                                                     \
        op.Init(query_pos, query_neg, key_max, key_min, weights, curSeqLengthsQ, curSeqLengths, blocktable, sparseIndices, sparseValues, user,           \
                tiling_data, &tPipe);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

#define LI_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                  \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;


template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void mlp_compressed_lightning_indexer(__gm__ uint8_t *query_pos, __gm__ uint8_t *query_neg,
                                                            __gm__ uint8_t *key_max, __gm__ uint8_t *key_min,
                                                            __gm__ uint8_t *weights,
                                                            __gm__ uint8_t *curSeqLengthsQ, __gm__ uint8_t *curSeqLengths,
                                                            __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if constexpr (DT_Q == LI_TPL_FP16 && DT_K == LI_TPL_FP16 && DT_OUT == LI_TPL_INT32) {
        INVOKE_LI_NO_KFC_OP_IMPL(LIPreload, half, half, int32_t, PAGE_ATTENTION, 
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(LIPreload, bfloat16_t, bfloat16_t, int32_t, PAGE_ATTENTION, 
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
    }
#endif
}

