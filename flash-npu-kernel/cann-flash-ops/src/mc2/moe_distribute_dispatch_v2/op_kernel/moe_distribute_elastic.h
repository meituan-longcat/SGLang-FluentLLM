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
 * \file moe_distribute_elastic.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_ELASTIC_H
#define MOE_DISTRIBUTE_ELASTIC_H

#include "moe_distribute_v2_constant.h"
#include "moe_distribute_v2_base.h"

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

class MoeDistributeElastic{
public:
    TPipe* tpipe_;
    TBuf<> elasticInfoBuf_;
    GlobalTensor<int32_t> elasticInfoGM_;

    __aicore__ inline MoeDistributeElastic() = default;

    __aicore__ inline void SetElasticInitParams(TPipe* tpipe, GlobalTensor<int32_t> elasticInfoGM) 
    {
        tpipe_ = tpipe;
        elasticInfoGM_ = elasticInfoGM;
    }

    __aicore__ inline void InitElasticInfoTensor(uint32_t epWorldSizeOriginal_, LocalTensor<int32_t> &elasticInfoTensor_)
    {
        uint32_t elasticInfoSizeAlign = GetElasticInfoSizeAlign(epWorldSizeOriginal_);
        tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
        elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
        DataCopyExtParams elasticInfoParams = {1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(elasticInfoTensor_, elasticInfoGM_, elasticInfoParams, elasticInfoCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }

    __aicore__ inline void InitElasticInfo(bool &isScalingDownFlag_, uint32_t &epWorldSize_, uint32_t &sharedExpertRankNum_, 
                                           uint32_t &moeExpertNum_, uint32_t &epRankId_, uint32_t &moeExpertNumPerRank_)
    {
        if (isScalingDownFlag_) {
            epWorldSize_ = elasticInfoGM_.GetValue(EP_WORLD_SIZE_IDX);
            sharedExpertRankNum_ = elasticInfoGM_.GetValue(SHARE_RANK_NUM_IDX);
            moeExpertNum_ = elasticInfoGM_.GetValue(MOE_NUM_IDX);
            epRankId_ = elasticInfoGM_.GetValue(ELASTIC_INFO_OFFSET + epRankId_);
            moeExpertNumPerRank_ = moeExpertNum_ / (epWorldSize_ - sharedExpertRankNum_);
        }
    }

    __aicore__ inline uint32_t GetElasticInfoSizeAlign(uint32_t epWorldSizeOriginal_)
    {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * static_cast<uint32_t>(sizeof(uint32_t));
        uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
        return elasticInfoSizeAlign;
    }
};
}
#endif // MOE_DISTRIBUTE_ELASTIC_H