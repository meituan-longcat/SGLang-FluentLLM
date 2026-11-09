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
 * \file block_batch_matmul_to_mul.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_BATCH_MATMUL_TO_MUL_H
#define MATMUL_BLOCK_BLOCK_BATCH_MATMUL_TO_MUL_H
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<BatchMatmulToMul<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
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
    uint64_t alignM_{1};
    uint64_t alignN_{1};
    uint64_t batchNum_{1};
    uint64_t batchNumPing_{1};
    uint64_t batchNumPong_{1};
    uint64_t alignNum_{1};
    const uint64_t BUFFER_NUM = 2;
    const uint64_t ALIGN_BYTE = 32;
    AscendC::TEventID eventIdVMte2Ping;
    AscendC::TEventID eventIdVMte2Pong;
    AscendC::TEventID eventIdMte3VPing;
    AscendC::TEventID eventIdMte3VPong;
    AscendC::TEventID eventIdPing1;
    AscendC::TEventID eventIdPing2;
    AscendC::TEventID eventIdPing3;
    AscendC::TEventID eventIdPong1;
    AscendC::TEventID eventIdPong2;
    AscendC::TEventID eventIdPong3;

    __aicore__ inline BlockMmad()
    {
        eventIdVMte2Ping = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
        eventIdVMte2Pong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
        eventIdMte3VPing = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
        eventIdMte3VPong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
        eventIdPing1 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        eventIdPing2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_S>();
        eventIdPing3 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        eventIdPong1 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        eventIdPong2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_S>();
        eventIdPong3 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Ping);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Pong);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPing);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPong);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Ping);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Pong);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPing);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVMte2Ping);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVMte2Pong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(eventIdMte3VPing);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(eventIdMte3VPong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdPing1);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(eventIdPing2);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdPong1);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(eventIdPong2);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(eventIdPing3);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(eventIdPong3);
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, uint32_t batchNum, uint32_t alignNum)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        batchNum_ = batchNum;
        batchNumPing_ = AscendC::CeilDiv(batchNum_, BUFFER_NUM);
        batchNumPong_ = batchNum_ - batchNumPing_;
        alignNum_ = alignNum;
        alignM_ = AscendC::CeilAlign(m_, alignNum_);
        alignN_ = AscendC::CeilAlign(n_, alignNum_);
    }

    __aicore__ inline void CopyInA(const AscendC::GlobalTensor<A_T>& aGlobal, const AscendC::LocalTensor<A_T>& aubLocal,
                                   const uint64_t batchAub)
    {
        if (batchAub > 0) {
            uint8_t rightPadding =
                static_cast<uint8_t>(AscendC::CeilAlign(batchAub * m_, ALIGN_BYTE / sizeof(A_T)) - batchAub * m_);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(batchAub * m_ * sizeof(A_T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<A_T> copyPadParams{false, 0, rightPadding, 0};
            AscendC::DataCopyPad(aubLocal, aGlobal, copyParams, copyPadParams);
        }
    }

    // 用于特殊优化场景的A Tensor对齐搬入
    __aicore__ inline void CopyInAlignA(const AscendC::GlobalTensor<A_T>& aGlobal,
                                       const AscendC::LocalTensor<A_T>& aubLocal, const uint64_t batchAub)
    {
        if (batchAub > 0) {
            uint16_t blockCount = static_cast<uint16_t>(batchAub);
            uint32_t blockLen = static_cast<uint32_t>(m_ * sizeof(A_T));
            uint8_t rightPadding = static_cast<uint8_t>(alignM_ - m_);
            AscendC::DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
            AscendC::DataCopyPadExtParams<A_T> copyPadParams{false, 0, rightPadding, 0};
            AscendC::DataCopyPad(aubLocal, aGlobal, copyParams, copyPadParams);
        }
    }

    __aicore__ inline void CopyInB(const AscendC::GlobalTensor<B_T>& bGlobal, const AscendC::LocalTensor<B_T>& bubLocal,
                                  const uint64_t batchBub)
    {
        if (batchBub > 0) {
            uint16_t blockCount = static_cast<uint16_t>(batchBub);
            uint32_t blockLen = static_cast<uint32_t>(n_ * sizeof(B_T));
            uint8_t rightPadding = static_cast<uint8_t>(alignN_ - n_);
            AscendC::DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
            AscendC::DataCopyPadExtParams<B_T> copyPadParams{false, 0, rightPadding, 0};
            AscendC::DataCopyPad(bubLocal, bGlobal, copyParams, copyPadParams);
        }
    }

    static __simd_vf__ inline void MulsVfSmallN(__ubuf__ A_T* dstPtr, __ubuf__ A_T* srcPtr,
                                                __ubuf__ A_T* aubPtr, uint32_t oneRepeatSize, uint32_t countMPerVL,
                                                uint16_t repeatTimes, uint64_t mAub, uint64_t alignM_, uint64_t nBub,
                                                uint64_t batchNum)
    {
        AscendC::MicroAPI::RegTensor<B_T> vSrcReg0;
        AscendC::MicroAPI::RegTensor<C_T> vDstReg0;
        AscendC::MicroAPI::RegTensor<A_T> scalarReg;
        AscendC::MicroAPI::MaskReg maskReg;
        maskReg = AscendC::MicroAPI::UpdateMask<A_T>(oneRepeatSize);
        for (uint16_t batchIdx = 0; batchIdx < static_cast<uint16_t>(batchNum); ++batchIdx) {
            auto addrRegSrc = AscendC::MicroAPI::CreateAddrReg<B_T>(batchIdx, nBub);
            AscendC::MicroAPI::LoadAlign<A_T, AscendC::MicroAPI::LoadDist::DIST_BLK>(vSrcReg0, srcPtr, addrRegSrc);
            for (uint16_t mIdx = 0; mIdx < repeatTimes; ++mIdx) {
                auto addrRegScalar = AscendC::MicroAPI::CreateAddrReg<C_T>(batchIdx, alignM_, mIdx, countMPerVL);
                if constexpr (sizeof(A_T)==sizeof(float)) {
                    AscendC::MicroAPI::LoadAlign<A_T, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(scalarReg, aubPtr, addrRegScalar);
                } else {
                    AscendC::MicroAPI::LoadAlign<A_T, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(scalarReg, aubPtr, addrRegScalar);
                }
                AscendC::MicroAPI::Mul(vDstReg0, vSrcReg0, scalarReg, maskReg);
                auto addrRegDst =
                    AscendC::MicroAPI::CreateAddrReg<C_T>(batchIdx, mAub * nBub, mIdx, countMPerVL * nBub);
                AscendC::MicroAPI::StoreAlign(dstPtr, vDstReg0, addrRegDst, maskReg);
            }
        }
    }

    static __simd_vf__ inline void MulsVf(__ubuf__ A_T* dstPtr, __ubuf__ A_T* srcPtr,
                                          __ubuf__ A_T* aubPtr, uint32_t count, uint16_t repeatTimes,
                                          uint64_t mAub, uint64_t nBub, uint64_t batchNum, uint32_t oneRepeatSize)
    {
        AscendC::MicroAPI::RegTensor<B_T> vSrcReg0;
        AscendC::MicroAPI::RegTensor<C_T> vDstReg0;
        AscendC::MicroAPI::RegTensor<A_T> scalarReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t tmpCount = count;
        for (uint16_t batchIdx = 0; static_cast<uint16_t>(batchIdx) < batchNum; ++batchIdx) {
            for (uint16_t mIdx = 0; static_cast<uint16_t>(mIdx) < mAub; ++mIdx) {
                auto addrRegScalar = AscendC::MicroAPI::CreateAddrReg<C_T>(batchIdx, mAub, mIdx, 1);
                if constexpr (sizeof(A_T)==sizeof(float)) {
                    AscendC::MicroAPI::LoadAlign<B_T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(scalarReg, aubPtr, addrRegScalar);
                } else {
                    AscendC::MicroAPI::LoadAlign<B_T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(scalarReg, aubPtr, addrRegScalar);
                }
                tmpCount = count;
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    maskReg = AscendC::MicroAPI::UpdateMask<A_T>(tmpCount);
                    auto addrRegSrc = AscendC::MicroAPI::CreateAddrReg<B_T>(
                        batchIdx, nBub, mIdx, 0, i, oneRepeatSize);
                    auto addrRegDst =
                        AscendC::MicroAPI::CreateAddrReg<C_T>(batchIdx, mAub * nBub, mIdx, nBub, i, oneRepeatSize);
                    AscendC::MicroAPI::LoadAlign(vSrcReg0, srcPtr, addrRegSrc);
                    AscendC::MicroAPI::Mul(vDstReg0, vSrcReg0, scalarReg, maskReg);
                    AscendC::MicroAPI::StoreAlign(dstPtr, vDstReg0, addrRegDst, maskReg);
                }
            }
        }
    }

    __aicore__ inline void AivProcess(uint64_t ubOffsetA, uint64_t ubOffsetB, uint64_t ubOffsetC, uint64_t batchNum)
    {
        AscendC::LocalTensor<A_T> aubLocal = ubLocal_[ubOffsetA];
        AscendC::LocalTensor<B_T> bubLocal = ubLocal_[ubOffsetB];
        AscendC::LocalTensor<C_T> cubLocal = ubLocal_[ubOffsetC];

        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(A_T);
        if (alignN_ <= alignNum_) {
            uint32_t countMPerVL = oneRepeatSize / alignN_;
            uint16_t repeatTimes = AscendC::CeilDiv(alignM_, countMPerVL);
            __ubuf__ C_T* dstPtr = (__ubuf__ C_T*)cubLocal.GetPhyAddr();
            __ubuf__ B_T* srcPtr = (__ubuf__ B_T*)bubLocal.GetPhyAddr();
            __ubuf__ A_T* aubPtr = (__ubuf__ A_T*)aubLocal.GetPhyAddr();
            AscendC::VF_CALL<MulsVfSmallN>(dstPtr, srcPtr, aubPtr, oneRepeatSize, countMPerVL, repeatTimes, m_, alignM_,
                                  alignN_, batchNum);
        } else {
            uint32_t count = n_;
            uint16_t repeatTimes = AscendC::CeilDiv(count, oneRepeatSize);
            __ubuf__ C_T* dstPtr = (__ubuf__ C_T*)cubLocal.GetPhyAddr();
            __ubuf__ B_T* srcPtr = (__ubuf__ B_T*)bubLocal.GetPhyAddr();
            __ubuf__ A_T* aubPtr = (__ubuf__ A_T*)aubLocal.GetPhyAddr();
            AscendC::VF_CALL<MulsVf>(dstPtr, srcPtr, aubPtr, count, repeatTimes, m_, alignN_, batchNum, oneRepeatSize);
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<C_T>& cGlobal, const AscendC::LocalTensor<C_T>& cubLocal,
                                   const uint64_t batchCub)
    {
        if (batchCub > 0) {
            uint16_t blockCount = static_cast<uint16_t>(m_ * batchCub);
            uint32_t blockLen = static_cast<uint32_t>(n_ * sizeof(C_T));
            AscendC::DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
            AscendC::DataCopyPad(cGlobal, cubLocal, copyParams);
        }
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<C_T>& cGlobal,
                                      const AscendC::GlobalTensor<A_T>& aGlobal,
                                      const AscendC::GlobalTensor<B_T>& bGlobal)
    {
        uint64_t ubOffsetAPing = 0;
        uint64_t ubOffsetBPing = AscendC::CeilAlign(batchNumPing_ * m_, alignNum_) + ubOffsetAPing;
        if (alignN_ <= alignNum_) {
            ubOffsetBPing = batchNumPing_ * alignM_ + ubOffsetAPing;
        }
        uint64_t ubOffsetCPing = batchNumPing_ * alignN_ + ubOffsetBPing;
        uint64_t ubOffsetAPong =
            AscendC::CeilAlign(batchNumPing_ * (m_ * alignN_), alignNum_) + ubOffsetCPing;
        if (alignN_ <= alignNum_) {
            ubOffsetAPong = batchNumPing_ * alignM_ * alignN_ + ubOffsetCPing;
        }
        uint64_t ubOffsetBPong = AscendC::CeilAlign(batchNumPong_ * m_, alignNum_) + ubOffsetAPong;
        if (alignN_ <= alignNum_) {
            ubOffsetBPong = batchNumPong_ * alignM_ + ubOffsetAPong;
        }
        uint64_t ubOffsetCPong = batchNumPong_ * alignN_ + ubOffsetBPong;

        uint64_t aGmOffset = batchNumPing_ * m_;
        uint64_t bGmOffset = batchNumPing_ * n_;
        uint64_t cGmOffset = batchNumPing_ * m_ * n_;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Ping);
        if (alignN_ <= alignNum_) {
            CopyInAlignA(aGlobal, ubLocal_[ubOffsetAPing], batchNumPing_);
        } else {
            CopyInA(aGlobal, ubLocal_[ubOffsetAPing], batchNumPing_);
        }
        CopyInB(bGlobal, ubLocal_[ubOffsetBPing], batchNumPing_);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdPing1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdPing2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdPing1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdPing2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPing);
        AivProcess(ubOffsetAPing, ubOffsetBPing, ubOffsetCPing, batchNumPing_);
        pipe_barrier(PIPE_ALL);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Ping);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdPing3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdPing3);

        CopyOut(cGlobal, ubLocal_[ubOffsetCPing], batchNumPing_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPing);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Pong);
        if (alignN_ <= alignNum_) {
            CopyInAlignA(aGlobal[aGmOffset], ubLocal_[ubOffsetAPong], batchNumPong_);
        } else {
            CopyInA(aGlobal[aGmOffset], ubLocal_[ubOffsetAPong], batchNumPong_);
        }
        CopyInB(bGlobal[bGmOffset], ubLocal_[ubOffsetBPong], batchNumPong_);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdPong1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdPong2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdPong1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdPong2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPong);
        AivProcess(ubOffsetAPong, ubOffsetBPong, ubOffsetCPong, batchNumPong_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVMte2Pong);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdPong3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdPong3);

        CopyOut(cGlobal[cGmOffset], ubLocal_[ubOffsetCPong], batchNumPong_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3VPong);
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    AscendC::LocalTensor<A_T> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
