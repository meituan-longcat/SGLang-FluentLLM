/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_to_mul.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_TO_MUL_H
#define MATMUL_BLOCK_BLOCK_TO_MUL_H
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "adv_api/reduce/reduce.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <
    class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_, class TileCopy_>
class BlockMmad<MatmulToMul<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseMN_{0};
    uint64_t tailMN_{0};
    uint64_t baseK_{0};
    uint64_t currentK_{0};
    uint64_t tailK_{0};
    uint64_t alignK_{0};
    uint64_t loopK_{1};
    uint64_t shapeMN_{0};
    bool hasBias_{false};
    bool dataCopyMode_{false};

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG2);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x1);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& shape, const TupleShape& blockInfo, int64_t loopK, bool hasBias, bool dataCopyMode)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        baseMN_ = Get<DIMENSION_BASE_MN>(blockInfo);
        tailMN_ = baseMN_;
        baseK_ = Get<DIMENSION_BASE_K>(blockInfo);
        currentK_ = baseK_;
        tailK_ = Get<DIMENSION_TAIL_K>(blockInfo) == 0 ? baseK_ : Get<DIMENSION_TAIL_K>(blockInfo);
        loopK_ = loopK;
        hasBias_ = hasBias;
        dataCopyMode_ = dataCopyMode;
        shapeMN_ = n_ * m_;
    }

    __aicore__ inline void SetTailMN(int64_t tailMN)
    {
        tailMN_ = tailMN;
    }

    // 用于搬运维度为1矩阵
    __aicore__ inline void CopyInA(
        const AscendC::GlobalTensor<float>& aGlobal, const AscendC::LocalTensor<float>& aubLocal)
    {
        if (dataCopyMode_) {
            AscendC::MultiCopyParams<float, DIMENSION> ndDmaParams;
            ndDmaParams.loopInfo.loopSrcStride[0] = 0;
            ndDmaParams.loopInfo.loopSrcStride[1] = 1;
            ndDmaParams.loopInfo.loopDstStride[0] = 1;
            ndDmaParams.loopInfo.loopDstStride[1] = static_cast<uint32_t>(baseMN_);
            ndDmaParams.loopInfo.loopSize[0] = static_cast<uint32_t>(tailMN_);
            ndDmaParams.loopInfo.loopSize[1] = static_cast<uint32_t>(currentK_);
            ndDmaParams.loopInfo.loopLpSize[0] = 0;
            ndDmaParams.loopInfo.loopLpSize[1] = 0;
            ndDmaParams.loopInfo.loopRpSize[0] = 0;
            ndDmaParams.loopInfo.loopRpSize[1] = 0;
            AscendC::DataCopy(aubLocal, aGlobal, ndDmaParams);
        } else {
            AscendC::MultiCopyParams<float, DIMENSION> ndDmaParams;
            ndDmaParams.loopInfo.loopSrcStride[0] = 1;
            ndDmaParams.loopInfo.loopSrcStride[1] = 0;
            ndDmaParams.loopInfo.loopDstStride[0] = 1;
            ndDmaParams.loopInfo.loopDstStride[1] = static_cast<uint32_t>(baseK_);
            ndDmaParams.loopInfo.loopSize[0] = static_cast<uint32_t>(currentK_);
            ndDmaParams.loopInfo.loopSize[1] = static_cast<uint32_t>(tailMN_);
            ndDmaParams.loopInfo.loopLpSize[0] = 0;
            ndDmaParams.loopInfo.loopLpSize[1] = 0;
            ndDmaParams.loopInfo.loopRpSize[0] = 0;
            ndDmaParams.loopInfo.loopRpSize[1] = 0;
            AscendC::DataCopy(aubLocal, aGlobal, ndDmaParams);
        }
    }

    // 用于搬运维度不为1矩阵
    __aicore__ inline void CopyInB(
        const AscendC::GlobalTensor<float>& bGlobal, const AscendC::LocalTensor<float>& bubLocal)
    {
        if (dataCopyMode_) {
            uint32_t alignMN = AscendC::CeilAlign(tailMN_, ALIGN_NUM);
            uint8_t rightPadding = static_cast<uint8_t>(alignMN - tailMN_);
            AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
            uint32_t srcStride = static_cast<uint32_t>((shapeMN_ - tailMN_) * sizeof(float));
            uint32_t dstStride = static_cast<uint32_t>((baseMN_ - alignMN) / ALIGN_NUM);
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(currentK_), static_cast<uint32_t>(tailMN_ * sizeof(float)), srcStride, dstStride,
                0};
            AscendC::DataCopyPad(bubLocal, bGlobal, copyParams, copyPadParams);
        } else {
            uint32_t alignK = AscendC::CeilAlign(currentK_, ALIGN_NUM);
            uint8_t rightPadding = static_cast<uint8_t>(alignK - currentK_);
            AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
            uint32_t srcStride = static_cast<uint32_t>((k_ - currentK_) * sizeof(float));
            uint32_t dstStride = static_cast<uint32_t>((baseK_ - alignK) / ALIGN_NUM);
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(tailMN_), static_cast<uint32_t>(currentK_ * sizeof(float)), srcStride, dstStride,
                0};
            AscendC::DataCopyPad(bubLocal, bGlobal, copyParams, copyPadParams);
        }
    }

    __aicore__ inline void CopyInBias(
        const AscendC::GlobalTensor<float>& biasGlobal, const AscendC::LocalTensor<float>& biasubLocal)
    {
        if (n_ == 1) {
            AscendC::DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(biasubLocal, biasGlobal, copyParams, copyPadParams);
        } else {
            AscendC::DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailMN_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(biasubLocal, biasGlobal, copyParams, copyPadParams);
        }
    }

    __aicore__ inline void AivProcess(uint64_t ubOffsetA, uint64_t ubOffsetB, uint64_t ubOffsetC)
    {
        AscendC::LocalTensor<float> aubLocal = ubLocal_[ubOffsetA];
        AscendC::LocalTensor<float> bubLocal = ubLocal_[ubOffsetB];
        AscendC::LocalTensor<float> cubLocal = ubLocal_[ubOffsetC];
        int32_t calCount = static_cast<int32_t>(baseMN_ * baseK_);
        AscendC::MulAddDst(cubLocal, aubLocal, bubLocal, calCount);
    }

    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<float>& cGlobal, const AscendC::LocalTensor<float>& outubLocal)
    {
        uint32_t blockLen = static_cast<uint32_t>(tailMN_ * sizeof(float));
        AscendC::DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        AscendC::DataCopyPad(cGlobal, outubLocal, copyParams);
    }

    __aicore__ inline void operator()(
        const AscendC::GlobalTensor<float>& cGlobal, const AscendC::GlobalTensor<float>& aGlobal,
        const AscendC::GlobalTensor<float>& bGlobal, const AscendC::GlobalTensor<float>& biasGlobal)
    {
        uint64_t ubOffsetAPing = 0;
        uint64_t ubOffsetBPing = baseMN_ * baseK_ + ubOffsetAPing;
        uint64_t ubOffsetAPong = baseMN_ * baseK_ + ubOffsetBPing;
        uint64_t ubOffsetBPong = baseMN_ * baseK_ + ubOffsetAPong;
        uint64_t ubOffsetC = baseMN_ * baseK_ + ubOffsetBPong;
        uint64_t ubOffsetOut = baseMN_ * baseK_ + ubOffsetC;
        uint64_t ubOffsetBias = CeilAlign(baseMN_, ALIGN_NUM) + ubOffsetOut;
        uint64_t ubOffsetA[] = {ubOffsetAPing, ubOffsetAPong};
        uint64_t ubOffsetB[] = {ubOffsetBPing, ubOffsetBPong};
        constexpr bool isReuse = true;

        // k累加地址 ubOffsetC 需要清0
        AscendC::Duplicate<float>(ubLocal_[ubOffsetC], 0, static_cast<int32_t>(baseMN_ * baseK_));
        for (uint64_t j = 0; j < loopK_; ++j) {
            currentK_ = (j + 1 == loopK_) ? tailK_ : baseK_;
            if (j == loopK_ - 1) {
                // 尾轮不允许脏数据污染
                AscendC::Duplicate<float>(ubLocal_[ubOffsetB[j & 0x1]], 0, static_cast<int32_t>(baseMN_ * baseK_));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG1);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
            if (m_ == 1) {
                CopyInA(aGlobal[j * baseK_], ubLocal_[ubOffsetA[j & 0x1]]);
            } else {
                CopyInA(bGlobal[j * baseK_], ubLocal_[ubOffsetA[j & 0x1]]);
            }
            uint64_t blockOffsetB = dataCopyMode_ ? j * shapeMN_ * baseK_ : j * baseK_;
            if (m_ == 1) {
                CopyInB(bGlobal[blockOffsetB], ubLocal_[ubOffsetB[j & 0x1]]);
            } else {
                CopyInB(aGlobal[blockOffsetB], ubLocal_[ubOffsetB[j & 0x1]]);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            AivProcess(ubOffsetA[j & 0x1], ubOffsetB[j & 0x1], ubOffsetC);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
        if (dataCopyMode_) {
            uint32_t shape[] = {static_cast<uint32_t>(baseK_), static_cast<uint32_t>(baseMN_)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(
                ubLocal_[ubOffsetOut], ubLocal_[ubOffsetC], shape, true);
        } else {
            uint32_t shape[] = {static_cast<uint32_t>(baseMN_), static_cast<uint32_t>(baseK_)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(
                ubLocal_[ubOffsetOut], ubLocal_[ubOffsetC], shape, true);
        }
        if (hasBias_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG2);
            CopyInBias(biasGlobal, ubLocal_[ubOffsetBias]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_FLAG1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_FLAG1);
            if (m_ == 1) {
                AscendC::Add(
                    ubLocal_[ubOffsetOut], ubLocal_[ubOffsetOut], ubLocal_[ubOffsetBias],
                    static_cast<int32_t>(tailMN_));
            } else {
                AscendC::Adds(
                    ubLocal_[ubOffsetOut], ubLocal_[ubOffsetOut], ubLocal_[ubOffsetBias][0],
                    static_cast<int32_t>(tailMN_));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG2);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
        CopyOut(cGlobal, ubLocal_[ubOffsetOut]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint16_t DIMENSION_BASE_MN = 0;
    constexpr static uint16_t DIMENSION_TAIL_MN = 1;
    constexpr static uint16_t DIMENSION_BASE_K = 2;
    constexpr static uint16_t DIMENSION_TAIL_K = 3;
    constexpr static uint64_t ALIGN_NUM = 8;
    constexpr static uint16_t SYNC_FLAG1 = 2;
    constexpr static uint16_t SYNC_FLAG2 = 3;
    constexpr static uint8_t DIMENSION = 2;
    AscendC::LocalTensor<float> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
