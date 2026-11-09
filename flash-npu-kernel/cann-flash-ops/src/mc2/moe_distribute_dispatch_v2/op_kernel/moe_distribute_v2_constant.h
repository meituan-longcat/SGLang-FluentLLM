/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_v2_constant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_V2_CONSTANT_H
#define MOE_DISTRIBUTE_V2_CONSTANT_H


namespace Mc2Kernel {
constexpr uint32_t STATE_SIZE = 1024 * 1024; // 1M
constexpr uint64_t TIMEOUT_OFFSET = 1000UL * 1024UL;

// distributev2base所需常量段
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t ZERONE_STATE_POS = 0U;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr uint32_t HCCL_DFX_POS = 8U;
constexpr uint32_t HCCL_DFX_NUM = 2U;
constexpr uint32_t HCCL_EPRANKId_POS = 0U;
constexpr uint32_t HCCL_WORLDSIZE_POS = 1U;
constexpr uint32_t UB_ALIGN = 32U;

// dispatchv2所需常量段
constexpr uint64_t DISPATCH_STATE_WIN_OFFSET = 768UL * 1024UL;
constexpr uint8_t BUFFER_NUM = 2;        // 多buf
constexpr uint8_t BUFFER_SINGLE = 1; 
constexpr uint32_t STATE_OFFSET = 32U;  // 状态空间偏移地址
constexpr uint8_t COMM_NUM = 2;  // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;
constexpr uint8_t QUANT_PADDING_VALUE = 0;
constexpr uint32_t FIRST_CORE = 0U;
// 先写死这个偏移，如果TP固定为2，可直接往起始数据偏移开始读写
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000UL;
constexpr uint64_t TIMEOUT_DETECTION_TX_UNITS = 8UL;
constexpr uint64_t STATE_HCCL_OFFSET = 32UL;
constexpr uint32_t TP_STATE_SIZE = 100U * 1024U;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t EXPAND_IDX_INFO = 3U;  // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t FLAG_AFTER_WAIT = 2U;
constexpr uint8_t EP_WORLD_SIZE_IDX = 1U;
constexpr uint8_t SHARE_RANK_NUM_IDX = 2U;
constexpr uint8_t MOE_NUM_IDX = 3U;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t MAX_UB_SIZE = 170U * 1024U;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint32_t UNQUANT = 0;
constexpr uint32_t STATIC_QUANT = 1;
constexpr uint32_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr uint32_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr uint32_t MX_QUANT = 4;
constexpr uint32_t CACHEWRITESIZE = 8U;

// combinev2额外所需常量段
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 818UL * 1024UL;   // combine 0/1状态区偏移，为dispatch预留50k大小
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U;  // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint8_t EP_DOMAIN = 0;
constexpr uint8_t TP_DOMAIN = 1;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint32_t ALIGNED_LEN = 256U;                  // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr float SCALE_PARAM = 127.0;                    // 计算量化参数所需的缩放倍数
constexpr uint32_t REDUCE_NUM = 8U;
constexpr uint32_t DIM_NUM = 2;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t BLOCK_NUM = ALIGNED_LEN / UB_ALIGN;  // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr uint32_t INT8_DIVIVE = 2;

// combineARN额外所需常量段
constexpr uint64_t COMBINE_ARN_STATE_SIZE = 1024UL * 1024UL; // 1M
constexpr uint64_t COMBINE_ARN_STATE_WIN_OFFSET = 818UL * 1024UL;
constexpr uint32_t NUM_PER_REP_FP32 = 64U;  // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr float ZERO = 0;
constexpr float ONE = 1;

// context额外所需常量段
constexpr uint64_t WINDOWS_IN_OFFSET = 780U;
constexpr uint64_t EP_WIN_SIZE_OFFSET = 1U;
constexpr uint64_t TP_WIN_SIZE_OFFSET = 2U;
constexpr uint64_t EP_STATUS_DATA_SPACE_OFFSET = 3U;
constexpr uint64_t EP_RANK_ID_OFFSET = 4U;
constexpr uint64_t EP_WORLD_SIZE_OFFSET = 5U;

// 其他
constexpr uint32_t FLAG_OFFSET = STATE_OFFSET / sizeof(float);
}
#endif // MOE_DISTRIBUTE_V2_CONSTANT_H
