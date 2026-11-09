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
 * \file block_mmad_mx.h
 * \brief
 */

#ifndef MATMUL_BLOCK_MMAD_MX_QUANT_H
#define MATMUL_BLOCK_MMAD_MX_QUANT_H
#include "../utils/layout_utils.h"
#include "../utils/common_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "..//tile/tile_copy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
using namespace AscendC;
using namespace Cmct::Gemm::QuantBatchMatmul;

struct TileL1L0Param {
    uint64_t curM = 0;
    uint64_t curN = 0;
    uint64_t curAlignM = 0;
    uint64_t curAlignN = 0;
    uint64_t curGmAKL1 = 0;
    uint64_t curGmBKL1 = 0;
    uint64_t curPadAKL1 = 0;  // pad to 64 align
    uint64_t curPadBKL1 = 0;  // pad to 64 align
    uint64_t curKL0 = 0;
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_,
          class Enable = void>
class BlockMmadMx {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadMx<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                BiasType_, LayoutBias_, TileCopy_,
                AscendC::Std::enable_if_t<
                    AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_> ||
                    AscendC::Std::is_base_of_v<MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                               DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using MxL0AType = typename GetL0DataType<AType, true>::Type;
    using MxL0BType = typename GetL0DataType<BType, true>::Type;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t BLOCK_REDUCE_CUBE = 32UL;
    constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;
    constexpr static uint64_t EVEN_FACTOR = 2UL;
    // Set unitflag state: 3 = final accumulation, 2 = non-final accumulation
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmadMx()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmadMx()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

public:
    __aicore__ inline void Init(const TupleShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        bL1OneBuffer_ = baseN_ * kL1_;
        scaleBL1OneBuffer_ = baseN_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            aL1OneBuffer_ = baseM_ * Cmct::Gemm::Align(kL1_, MXFP_DIVISOR_SIZE);
            scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
                // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
                uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] =
                    l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        } else {
            uint64_t mAlign = Cmct::Gemm::Align(baseM_, transA ? C0_SIZE : BLOCK_CUBE);
            uint64_t kAlign = Cmct::Gemm::Align(k_, MXFP_DIVISOR_SIZE);
            aL1OneBuffer_ = mAlign * kAlign;
            scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
            // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
            l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_ + biasL1OneBuffer_;
            l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
            uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= (AscendC::TOTAL_L1_SIZE >> 1)
                                ? l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ : (AscendC::TOTAL_L1_SIZE >> 1);
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType> &aGlobal,
        const AscendC::LocalTensor<AType> &al1Local, TileL1L0Param &tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? tileL1L0Param.curGmAKL1 : tileL1L0Param.curM;
        uint64_t dDim = transA ? tileL1L0Param.curM : tileL1L0Param.curGmAKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        nd2nzParams.dstNzC0Stride = transA ? tileL1L0Param.curPadAKL1 : tileL1L0Param.curAlignM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType> &bGlobal,
        const AscendC::LocalTensor<BType> &bl1Local, TileL1L0Param &tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? tileL1L0Param.curN : tileL1L0Param.curGmBKL1;
        uint64_t dDim = transB ? tileL1L0Param.curGmBKL1 : tileL1L0Param.curN;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        nd2nzParams.dstNzC0Stride = transB ? tileL1L0Param.curAlignN : tileL1L0Param.curPadBKL1;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1WeightNz(const AscendC::GlobalTensor<BType> &bGlobal,
                                            const AscendC::LocalTensor<BType> &bl1Local,
                                            const TileL1L0Param &tileL1L0Param)
    {
        AscendC::GlobalTensor<uint8_t> bGlobalUInt8;
        bGlobalUInt8.SetGlobalBuffer((__gm__ uint8_t *)bGlobal.GetPhyAddr());
        auto bl1LocalUint8 = bl1Local.template ReinterpretCast<uint8_t>();

        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<uint8_t> padParams;
        if constexpr (transB) {
            dataCopyParams.blockCount = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE);
            dataCopyParams.blockLen = tileL1L0Param.curAlignN * C0_SIZE;
            dataCopyParams.srcStride =
                (Cmct::Gemm::CeilAlign(n_, AscendC::BLOCK_CUBE) - tileL1L0Param.curAlignN) * C0_SIZE;
            dataCopyParams.dstStride = 0;
        } else {
            int64_t curGmBKL1NZ = Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, AscendC::BLOCK_CUBE);
            dataCopyParams.blockCount = tileL1L0Param.curAlignN / C0_SIZE;
            dataCopyParams.blockLen = curGmBKL1NZ * C0_SIZE;
            dataCopyParams.srcStride = (Cmct::Gemm::CeilAlign(k_, AscendC::BLOCK_CUBE) - curGmBKL1NZ) * C0_SIZE;
            dataCopyParams.dstStride = tileL1L0Param.curPadBKL1 - curGmBKL1NZ;
        }
        AscendC::DataCopyPad(bl1LocalUint8, bGlobalUInt8, dataCopyParams, padParams);
    }

    __aicore__ inline void InitA1(const AscendC::LocalTensor<AType> &al1Local, TileL1L0Param &tileL1L0Param)
    {
        AscendC::LocalTensor<half> al1LocalHalf = al1Local.template ReinterpretCast<half>();
        AscendC::InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        if constexpr (!transA) {
            // nd2nz pading to 64 align
            if (tileL1L0Param.curPadAKL1 - tileL1L0Param.curGmAKL1 < BLOCK_REDUCE_CUBE) {
                return;
            }
            // pad m1, m0, 16 for half
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = tileL1L0Param.curAlignM;
            initConstValueParams.dstGap = 0;
            initConstValueParams.initValue = 0;
            uint64_t kAlign = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmAKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
            offset = tileL1L0Param.curAlignM * kAlign;
        } else {
            if (tileL1L0Param.curPadAKL1 == tileL1L0Param.curGmAKL1) {
                return;
            }
            uint64_t m1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curAlignM, C0_SIZE);
            offset = tileL1L0Param.curGmAKL1 * AscendC::BLOCK_CUBE;
            initConstValueParams.repeatTimes = m1;
            initConstValueParams.blockNum = tileL1L0Param.curPadAKL1 - tileL1L0Param.curGmAKL1;
            initConstValueParams.dstGap = tileL1L0Param.curGmAKL1;
            initConstValueParams.initValue = 0;
        }
        AscendC::InitConstValue(al1LocalHalf[offset], initConstValueParams);
    }

    __aicore__ inline void InitB1(const AscendC::LocalTensor<BType> &bl1Local, TileL1L0Param &tileL1L0Param)
    {
        // Equivalent to curKL1 % MXFP_DIVISOR_SIZE
        AscendC::LocalTensor<half> bl1LocalHalf = bl1Local.template ReinterpretCast<half>();
        AscendC::InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        if constexpr (transB) {
            // nd2nz pading to 64 align
            if (tileL1L0Param.curPadBKL1 - tileL1L0Param.curGmBKL1 < BLOCK_REDUCE_CUBE) {
                return;
            }
            // pad n1, n0, 16 for half
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = tileL1L0Param.curAlignN;
            initConstValueParams.dstGap = 0;
            initConstValueParams.initValue = 0;
            uint64_t kAlign = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
            offset = tileL1L0Param.curAlignN * kAlign;
        } else {
            if constexpr (formatB == CubeFormat::NZ) {
                if (tileL1L0Param.curPadBKL1 == Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, AscendC::BLOCK_CUBE)) {
                    return;
                }
            } else {
                if (tileL1L0Param.curPadBKL1 == tileL1L0Param.curGmBKL1) {
                    return;
                }
            }
            // when format of B is NZ, we reuse code for ND even though we initialize unnecessary extra space
            uint64_t n1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curAlignN, C0_SIZE);
            offset = tileL1L0Param.curGmBKL1 * AscendC::BLOCK_CUBE;
            initConstValueParams.repeatTimes = n1;
            initConstValueParams.blockNum = tileL1L0Param.curPadBKL1 - tileL1L0Param.curGmBKL1;
            initConstValueParams.dstGap = tileL1L0Param.curGmBKL1;
            initConstValueParams.initValue = 0;
        }
        AscendC::InitConstValue(bl1LocalHalf[offset], initConstValueParams);
    }

    __aicore__ inline void CopyInBias(const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                      const AscendC::LocalTensor<BiasType> &cl1Local, uint64_t curNL1)
    {
        // No need to add sync flag for bias L1 loading because bias loading operation can be covered by A/B/ScaleA/ScaleB load.
        AscendC::DataCopyPadParams padParams;
        // 单位为Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
    }

    __aicore__ inline void CopyInScaleA(const GlobalTensor<fp8_e8m0_t> &aScaleGlobal,
                                        const LocalTensor<fp8_e8m0_t> &aScaleL1Local, uint64_t curML1, uint64_t curKL1,
                                        uint64_t kL1Offset)
    {
        if (DispatchPolicy::fullLoadMode != 0 && kL1Offset != 0) {
            return;
        }
        uint64_t curScaleKL1 = curKL1;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = transA ? Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) : curML1;
        uint64_t dDim = transA ? curML1 : Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

        uint64_t offsetScaleAGM = !transA ? kL1Offset / MXFP_DIVISOR_SIZE : (kL1Offset / MXFP_DIVISOR_SIZE) * m_;

        GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        if constexpr (!transA) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            dn2nzParams.dValue = nDim;
            dn2nzParams.nValue = dDim;
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], dn2nzParams);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = m_;
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], nd2nzParams);
        }
    }

    __aicore__ inline void CopyInScaleB(const GlobalTensor<fp8_e8m0_t> &bScaleGlobal,
                                        const LocalTensor<fp8_e8m0_t> &bScaleL1Local, uint64_t curNL1,
                                        uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = transB ? curNL1 : Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        uint64_t dDim = transB ? Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) : curNL1;

        GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        uint64_t offsetScaleBGM = transB ? kL1Offset / MXFP_DIVISOR_SIZE : (kL1Offset / MXFP_DIVISOR_SIZE) * n_;

        if constexpr (transB) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            dn2nzParams.dValue = nDim;
            dn2nzParams.nValue = dDim;
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], dn2nzParams);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = n_;
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], nd2nzParams);
        }
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<BiasType> &biasL1Local,
                                    const AscendC::LocalTensor<float> &biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Cmct::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void CopyInL0A(
        const AscendC::LocalTensor<MxL0AType>& l0aLocal, const AscendC::LocalTensor<AType>& al1Local,
        const AscendC::LocalTensor<fp8_e8m0_t>& scaleAl1Local, uint64_t iter, TileL1L0Param& tileL1L0Param,
        uint64_t curScaleKL1)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t m1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        if constexpr (!transA) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = m1;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if ((m1 & 1) == 0) {
                loadDataParams.mStep = Cmct::Gemm::CeilAlign(
                    Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), MXFP_MULTI_BASE_SIZE);
            } else {
                loadDataParams.mStep = B8_MIN_STEP;
            }
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curM, C0_SIZE);
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(tileL1L0Param.curPadAKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = m1;
            loadDataParams.ifTranspose = true;
        }
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = m1;
        loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (transA) {
            if ((m1 & 1) != 0) {
                PipeBarrier<PIPE_MTE1>();
                LocalTensor<AType> l0a = l0aLocal.template ReinterpretCast<AType>();
                uint64_t loadTimes =
                    Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, BLOCK_CUBE), B8_MIN_STEP);
                for (uint64_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition = B8_MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                    AscendC::LoadData(l0a[i * m1 * BLOCK_CUBE * C0_SIZE], al1Local, loadDataParams);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<MxL0BType> &l0bLocal,
                                     const AscendC::LocalTensor<BType> &bl1Local,
                                     const AscendC::LocalTensor<fp8_e8m0_t> &scaleBl1Local, uint64_t iter,
                                     TileL1L0Param &tileL1L0Param)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t n1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
        if constexpr (transB) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = n1;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if ((n1 & 1) == 0) {
                loadDataParams.mStep =
                    Cmct::Gemm::CeilAlign(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), EVEN_FACTOR);
            } else {
                loadDataParams.mStep = B8_MIN_STEP;
            }
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curN, C0_SIZE);
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(tileL1L0Param.curPadBKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = n1;
            loadDataParams.ifTranspose = true;
        }
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = n1;
        loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (!transB) {
            if ((n1 & 1) != 0) {
                PipeBarrier<PIPE_MTE1>();
                LocalTensor<BType> l0b = l0bLocal.template ReinterpretCast<BType>();
                uint64_t loadTimes =
                    Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, BLOCK_CUBE), B8_MIN_STEP);
                for (uint64_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition = B8_MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                    AscendC::LoadData(l0b[i * n1 * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, loadDataParams);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType> &cGlobal, AscendC::LocalTensor<float> &c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        // set mode according to dtype
        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<CType, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<CType, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = FINAL_ACCUMULATION;  // 3 unitflag
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void UpdateKL1(TileL1L0Param &tileL1L0Param, uint64_t iter0)
    {
        tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
        tileL1L0Param.curPadBKL1 = Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE);
        tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
        tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1;  // pad to 64 align
    }

    __aicore__ inline void UpdateKL0(TileL1L0Param &tileL1L0Param, uint64_t iter1)
    {
        if (iter1 * baseK_ + baseK_> tileL1L0Param.curPadBKL1) {
            tileL1L0Param.curKL0 = tileL1L0Param.curPadBKL1 - iter1 * baseK_;
        } else {
            tileL1L0Param.curKL0 = baseK_;
        }
    }

    __aicore__ inline void GetAlignMN(TileL1L0Param &tileL1L0Param)
    {
        if constexpr (transA) {
            tileL1L0Param.curAlignM = Cmct::Gemm::CeilAlign(tileL1L0Param.curM, C0_SIZE);
        } else {
            tileL1L0Param.curAlignM = Cmct::Gemm::CeilAlign(tileL1L0Param.curM, BLOCK_CUBE);
        }
        if constexpr (!transB) {
            tileL1L0Param.curAlignN = Cmct::Gemm::CeilAlign(tileL1L0Param.curN, C0_SIZE);
        } else {
            tileL1L0Param.curAlignN = Cmct::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE);
        }
    }

    __aicore__ inline void CopyScalesInL1(AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                          AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal, TileL1L0Param &tileL1L0Param,
                                          uint64_t l1Iter, uint64_t scaleL1BufId)
    {
        uint64_t kL1Offset = l1Iter * kL1_;
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            if (l1Iter % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]], tileL1L0Param.curM,
                             scaleKL1_, kL1Offset);
                CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
                             kL1Offset);
            }
        } else {
            if (l1Iter % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
                             kL1Offset);
            }
            if (abL1LoopCnt_ == 0) {
                CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[0]], tileL1L0Param.curM, k_, kL1Offset);
            }
        }
    }

    __aicore__ inline void CopyAInL1(AscendC::GlobalTensor<AType> aGlobal, TileL1L0Param tileL1L0Param, uint64_t offsetA,
                                     uint64_t offsetAL1, uint64_t l1Iter)
    {
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            InitA1(aL1Local_[offsetAL1], tileL1L0Param);
            CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
        } else {
            offsetAL1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
            if (abL1LoopCnt_ < kL1Iter_) {
                InitA1(aL1Local_[offsetAL1], tileL1L0Param);
                CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
            }
        }
    }

    __aicore__ inline void CopyBInL1(AscendC::GlobalTensor<BType> bGlobal, TileL1L0Param tileL1L0Param, uint64_t l1BufId,
                                     uint64_t l1Iter)
    {
        InitB1(bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        if constexpr (formatB == CubeFormat::NZ) {
            uint64_t offsetB =
                transB ? l1Iter * kL1_ * Cmct::Gemm::CeilAlign(n_, AscendC::BLOCK_CUBE) : l1Iter * kL1_ * C0_SIZE;
            CopyInB1WeightNz(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        } else {
            uint64_t offsetB = transB ? l1Iter * kL1_ : l1Iter * kL1_ * n_;
            CopyInB1(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        }
    }

    __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, MmadParams &mmadParams, uint64_t l1Iter, uint64_t l1BufId,
                                   uint64_t scaleL1BufId, uint64_t offsetAl1, uint64_t l0cOffset)
    {
        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            UpdateKL0(tileL1L0Param, iter1);
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            uint64_t offsetScaleL1 = BLOCK_CUBE * (l1Iter % (scaleKL1_ / kL1_)) * (kL1_ / MXFP_GROUP_SIZE);
            if constexpr (DispatchPolicy::fullLoadMode == 0) {
                CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                          scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
                          scaleKL1_);
            } else {
                offsetAl1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
                uint64_t offsetScaleAL1 = BLOCK_CUBE * l1Iter * (kL1_ / MXFP_GROUP_SIZE);
                CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                          scaleAL1Local_[l1BufferScaleAOffset_[0] + offsetScaleAL1], iter1, tileL1L0Param, k_);
            }
            // copy bias to bt
            CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
                     Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE), NeedBias(l1Iter, iter1));
            CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId]],
                      scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = Cmct::Gemm::CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            mmadParams.unitFlag =
                (l1Iter + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0 && !isBias_);
            Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId_, NeedBias(l1Iter, iter1));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<AType> aGlobal,
                                      AscendC::GlobalTensor<BType> bGlobal,
                                      AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal,
                                      AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal,
                                      AscendC::GlobalTensor<BiasType> biasGlobal,
                                      AscendC::GlobalTensor<CType> cGlobal,
                                      BlockShape singleShape)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = Get<IDX_N_TILEIDX>(singleShape);
        GetAlignMN(tileL1L0Param);
        AscendC::MmadParams mmadParams;
        mmadParams.m = tileL1L0Param.curM;
        mmadParams.n = tileL1L0Param.curN;
        mmadParams.disableGemv = true;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            // Load data to L1 and open DB
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t offsetA = transA ? iter0 * kL1_ * m_ : iter0 * kL1_;
            uint64_t offsetAl1 = l1BufferAOffset_[l1BufId];
            CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, iter0, scaleL1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            biasBufId_ = abL1LoopCnt_ & 1;
            UpdateKL1(tileL1L0Param, iter0);
            CopyAInL1(aGlobal, tileL1L0Param, offsetA, offsetAl1, iter0);
            if (isBias_ && iter0 == 0) {
                CopyInBias(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)],
                           tileL1L0Param.curN);
            }
            CopyBInL1(bGlobal, tileL1L0Param, l1BufId, iter0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1, l0cOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1Iter_ - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }
        // Copy out to GM
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        // 数据搬出到GM或ub
        CopyOut(cGlobal, c1Local, mmadParams.m, mmadParams.n);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    __aicore__ inline void Mmad(
        AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset, uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(
                c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset], mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

private:
    constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
    constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;
    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL}; // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL}; // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL}; // default 2 buffer
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(BiasType)};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
}  // namespace Block
}  // namespace Gemm
}  // namespace Cmct
#endif