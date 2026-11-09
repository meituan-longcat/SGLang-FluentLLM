/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mlp_sparse_lightning_indexer_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef LIGHTNING_INDEXER_SERVICE_CUBE_H
#define LIGHTNING_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "mlp_sparse_lightning_indexer_common.h"

namespace LIKernel {
using namespace LICommon;
template <typename LIT>
class LIMatmul {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;

    __aicore__ inline LIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<float> &mm1ResGm,
                                               const GlobalTensor<int32_t> &sparseIndexGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LICommon::RunInfo &runInfo);
    __aicore__ inline void ComputeMm1Normal(const LICommon::RunInfo &runInfo);
    __aicore__ inline void ComputeMm1SparseIndex(const LICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;         // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const LICommon::RunInfo &runInfo);
    __aicore__ inline void ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const LICommon::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, const LICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForSparseIndex(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                   const LICommon::RunInfo &runInfo,
                                                   uint32_t tokenIdx);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<int32_t> sparseIndexGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void
LIMatmul<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                   const GlobalTensor<Q_T> &queryGm, const GlobalTensor<float> &mm1ResGm,
                                   const GlobalTensor<int32_t> &sparseIndexGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    sparseIndexGm_ = sparseIndexGm;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeMm1(const LICommon::RunInfo &runInfo)
{
    if (constInfo_.useSparseIndex) {
        ComputeMm1SparseIndex(runInfo);
    } else {
        ComputeMm1Normal(runInfo);
    }
}

// 原始路径：所有 token 共享一份 K
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeMm1Normal(const LICommon::RunInfo &runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
        } else {
            KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        // s1gProcessSize当前必定不会超过2倍的s1g basic block
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1RealSize ? s1gL1RealSize - s1gL1Offset : M_BASIC_BLOCK_L0;
                    LoadQueryToL0a(s1gGmOffset, s1gL1Offset, s1gL1RealSize, s1gL0RealSize, runInfo);
                    LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    ComuteL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    Fixp(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo);
                    l0BufIdx_++;
                }
            }
            if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

// sparse_index 路径：每个 token 的 K 不同，按 token 分别 load K + sub-matmul
// mm1ResGm 布局: [s1BaseSize * gSize, s2BaseSize]
//   token0: rows [0, gSize)
//   token1: rows [gSize, 2*gSize)
// Q 在首个 s2 chunk 时 load 一次全部 s1g 行，后续复用
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeMm1SparseIndex(const LICommon::RunInfo &runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    uint64_t gSize = constInfo_.gSize;

    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;

        // Load Q: 只在首个 s2 chunk 时 load 一次全部 s1g 行
        if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
                uint64_t s1gL1RealSize =
                    s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            }
        }

        // 对每个 token 分别 load K 并计算对应的 gSize 行
        for (uint32_t tokenIdx = 0; tokenIdx < constInfo_.s1BaseSize; tokenIdx++) {
            // 先计算当前 token 是否有有效行，无效则跳过（T1 不是 s1BaseSize 整数倍时尾块会出现）
            uint64_t s1gStart = tokenIdx * gSize;
            if (s1gStart >= s1gProcessSize) {
                break; // 后续 token 也不会有有效行
            }
            uint64_t tokenRows = gSize;
            if (s1gStart + tokenRows > s1gProcessSize) {
                tokenRows = s1gProcessSize - s1gStart;
            }

            // 检查当前 token 在当前 s2 chunk 的首个 sparse block 是否为 -1
            // 有效值在前 -1 在后的保证下，首个 block = -1 意味着整个 chunk 无有效数据
            uint64_t tokenSparseOff = runInfo.sparseIndexGmOffset + tokenIdx * constInfo_.sparseBlockNum;
            uint64_t firstSparseIdx = (s2GmBaseOffset + s2GmOffset) / constInfo_.kCacheBlockSize;
            int32_t firstBlkId = sparseIndexGm_.GetValue(tokenSparseOff + firstSparseIdx);
            if (firstBlkId < 0) {
                // 无有效数据，跳过 K load + matmul，但维持 KEY buffer event 配对
                WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                keyL1BufIdx_++;
                continue;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            KeyNd2NzForSparseIndex(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo, tokenIdx);

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

            // 确定当前 token 所在的 L1 block：Q 是按 M_BASIC_BLOCK 步长 load 到 L1 的，
            // 每个 L1 slot 对应 [l1BlockIdx * M_BASIC_BLOCK, (l1BlockIdx+1) * M_BASIC_BLOCK) 这段行
            uint64_t l1BlockIdx = s1gStart / M_BASIC_BLOCK;
            // L1 block 内的行偏移
            uint64_t s1gOffsetInL1Block = s1gStart % M_BASIC_BLOCK;
            // 该 L1 block 实际 load 的总行数
            uint64_t l1BlockStart = l1BlockIdx * M_BASIC_BLOCK;
            uint64_t l1BlockRealSize = (l1BlockStart + M_BASIC_BLOCK > s1gProcessSize) ?
                                        s1gProcessSize - l1BlockStart : M_BASIC_BLOCK;

            // 回溯到正确的 L1 buffer slot
            uint64_t totalL1Blocks = CeilDiv(s1gProcessSize, M_BASIC_BLOCK);
            queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_ - (totalL1Blocks - 1 - l1BlockIdx);

            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                // token 在 L1 block 内从 s1gOffsetInL1Block 开始，长度为 tokenRows
                // 按 M_BASIC_BLOCK_L0 步长切分到 L0
                for (uint64_t tokenL1Off = 0; tokenL1Off < tokenRows; tokenL1Off += M_BASIC_BLOCK_L0) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize = (tokenL1Off + M_BASIC_BLOCK_L0 > tokenRows) ?
                                              tokenRows - tokenL1Off : M_BASIC_BLOCK_L0;
                    // 与 Normal 路径保持一致的参数语义:
                    // 参数1(s1gGmOffset): 全局行偏移，函数体内未使用
                    // 参数2(s1gL1Offset): L1 block 内偏移 → mStartPt
                    // 参数3(s1gL1RealSize): L1 block 实际总行数 → l1H
                    // 参数4(s1gL0RealSize): 本次搬到 L0 的行数 → mExtension
                    uint64_t s1gL1Offset = s1gOffsetInL1Block + tokenL1Off;
                    LoadQueryToL0a(s1gStart + tokenL1Off, s1gL1Offset, l1BlockRealSize, s1gL0RealSize, runInfo);
                    LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    ComuteL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    // Fixp 写入位置: s1gStart + tokenL1Off 是在 mm1ResGm 中的绝对行偏移
                    Fixp(s1gStart + tokenL1Off, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize,
                         runInfo);
                    l0BufIdx_++;
                }
            }

            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }

        // 释放 Q L1 buffer
        if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const LICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t keyGmOffset = runInfo.tensorKeyOffset + (s2GmOffset + s2L1Offset) * constInfo_.headDim;
        // 搬运按照S2_BASIC_BLOCK_L0*D_BASIC_BLOCK_L0的方式在l1上排布, 方便后续mte1
        // 根据s2的offset判断当前属于前一个L0分型还是后一个L0分型，暂时只支持两个分型
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE)); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const LICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                   constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
                               s2BlkOffset * constInfo_.headDim;
        // 搬运按照S2_BASIC_BLOCK_L0*D_BASIC_BLOCK_L0的方式在l1上排布, 方便后续mte1
        // 根据s2的offset判断当前属于前一个L0分型还是后一个L0分型，暂时只支持两个分型
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE)); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// SparseIndex: sparse_index → key 地址
// PA 场景：两级间接寻址 sparse_index → blockTable → 物理 key
// 非 PA 连续场景：一级间接寻址 sparse_index → 连续 key 偏移
// tokenIdx: 基本块内的 token 序号 (0 或 1)，用于偏移 sparse_index 行
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::KeyNd2NzForSparseIndex(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                              const LICommon::RunInfo &runInfo,
                                                              uint32_t tokenIdx)
{
    // 每个 token 有自己的 sparse_index 行，行间偏移 = sparseBlockNum
    uint64_t tokenSparseOffset = runInfo.sparseIndexGmOffset + tokenIdx * constInfo_.sparseBlockNum;
    // 非 PA 连续场景：反推 batch+n2 基地址（去掉 tensorKeyOffset 中的 s2Idx 偏移）
    uint64_t keyBaseOffset = 0;
    if constexpr (!PAGE_ATTENTION) {
        keyBaseOffset = runInfo.tensorKeyOffset -
            static_cast<uint64_t>(runInfo.s2Idx) * constInfo_.s2BaseSize * constInfo_.kHeadNum * constInfo_.headDim;
    }
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        // 计算当前位置对应的 sparse block index 和 block 内偏移
        uint64_t s2AbsOffset = s2L1Offset + s2GmOffset;
        uint64_t sparseIdx = s2AbsOffset / constInfo_.kCacheBlockSize;
        uint64_t inBlockOffset = s2AbsOffset % constInfo_.kCacheBlockSize;

        // 第一级：sparse_index → 逻辑 block 编号
        int32_t logicalBlockIdx = sparseIndexGm_.GetValue(tokenSparseOffset + sparseIdx);

        // 有效值在前，-1 在后：遇到第一个 -1 即后续全部无效，提前退出
        if (logicalBlockIdx < 0) {
            break;
        }

        // 计算本次搬运大小：不能跨 block 边界
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = inBlockOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ?
                         constInfo_.kCacheBlockSize - inBlockOffset : s2Mte2Size;

        uint64_t keyGmOffset;
        if constexpr (PAGE_ATTENTION) {
            // PA 场景：第二级 block_table → 物理 block 地址
            keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + logicalBlockIdx) *
                              constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
                          inBlockOffset * constInfo_.headDim;
        } else {
            // 非 PA 连续场景：直接用逻辑 block 编号算出 batch 内 token 偏移
            keyGmOffset = keyBaseOffset +
                (static_cast<uint64_t>(logicalBlockIdx) * constInfo_.kCacheBlockSize + inBlockOffset) *
                    constInfo_.kHeadNum * constInfo_.headDim;
        }

        uint64_t l1DstOffset = (keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                               (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                    S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                                    s2L1Offset * BLOCK_CUBE);

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[l1DstOffset], keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                 const LICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * constInfo_.headDim], nd2nzPara);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadQueryToL0a(uint64_t s1gGmOffset, uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                     uint64_t s1gL0RealSize, const LICommon::RunInfo &runInfo)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE); // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                         // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;         // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = CeilAlign(s1gL0RealSize, BLOCK_CUBE); // M height维度目的
    loadData3DParams.kExtension = constInfo_.headDim;                   // K width维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
                                   queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                   const LICommon::RunInfo &runInfo)
{
    uint64_t keyL1Offset = s2L1Offset >= S2_BASIC_BLOCK_L0 ? S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                const LICommon::RunInfo &runInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                           uint64_t s2L0RealSize, const LICommon::RunInfo &runInfo)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign;
    intriParams.srcStride = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11; // 3 unitflag
    intriParams.reluPre = 1;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize * constInfo_.s2BaseSize +
                                s1gGmOffset * intriParams.dstStride + s2GmOffset],
                      cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}
} // namespace LIKernel
#endif