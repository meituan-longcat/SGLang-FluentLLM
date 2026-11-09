/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_v3_kernel_split_n.h
 * \brief
 */

#include "mla_prolog_v3_policy.h"

#ifndef MLA_PROLOG_V3_KERNEL_SPLIT_N_H
#define MLA_PROLOG_V3_KERNEL_SPLIT_N_H

#include "mla_prolog_v3_comm.h"
#include "mla_prolog_v3_vector_comm.h"
#include "mla_prolog_v3_matmul.h"
#include "mla_prolog_v3_rms_norm.h"
#include "mla_prolog_v3_gather_sin_cos.h"
#include "mla_prolog_v3_rotary_position_embedding.h"
#include "mla_prolog_v3_scatter_cache.h"
#include "mla_prolog_v3_dequant.h"
#include "mla_prolog_v3_dynamic_quant_qn_mul_qr.h"
#include "mla_prolog_v3_tiling_data.h"
#if MLA_PROLOG_V3_SFA1
#if __has_include("../../fluent_mla_prolog_v3/op_kernel/fluent_mla_prolog_v3_template_tiling_key.h")
#include "../../fluent_mla_prolog_v3/op_kernel/fluent_mla_prolog_v3_template_tiling_key.h"
#else
#include "../fluent_mla_prolog_v3/fluent_mla_prolog_v3_template_tiling_key.h"
#endif
#else
#include "mla_prolog_v3_template_tiling_key.h"
#endif

namespace MlaPrologV3 {

template <typename MLAPT>
class MlaPrologV3SplitN {
public:
    static constexpr bool isPertile = MLAPT::isPertile;
#if !MLA_PROLOG_V3_SFA1
    static constexpr uint32_t cvMode = MLAPT::cvRatio;
#endif

    using mmInputType = typename MLAPT::mmInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmQnInputType = typename MLAPT::mmQnInputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using mmQnOutputType = typename MLAPT::mmQnOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using queryOutputType = typename std::conditional<isPertile, bfloat16_t, typename MLAPT::kvCacheType>::type;
    using kvCacheType = typename MLAPT::kvCacheType;
    using krCacheType = typename MLAPT::krCacheType;
    using dequantScaleQNopeType = typename MLAPT::dequantScaleQNopeType;
    using dequantScaleQNormType = typename MLAPT::dequantScaleQNormType;

    MMParams mmCqParam_;
    MMParams mmCkvKrParam_;
    MMParams mmQcQrParam_;
    MMParams mmQnParam_;

#if MLA_PROLOG_V3_SFA1
    __aicore__ inline MlaPrologV3SplitN(TPipe* pipe, const optiling::FluentMlaPrologV3TilingData* __restrict tilingData,
                                          const optiling::FluentMlaPrologV3BaseParams* __restrict baseParams)
#else
    __aicore__ inline MlaPrologV3SplitN(TPipe* pipe, const optiling::MlaPrologV3TilingData* __restrict tilingData,
                                          const optiling::MlaPrologV3BaseParams* __restrict baseParams)
#endif
        : pipe_(pipe), tilingData_(tilingData), baseParams_(baseParams) {}

    __aicore__ inline void Init(__gm__ uint8_t *tokenX,
                                __gm__ uint8_t *weightDq,
                                __gm__ uint8_t *weightUqQr,
                                __gm__ uint8_t *weightUk,
                                __gm__ uint8_t *weightDkvKr,
                                __gm__ uint8_t *rmsnormGammaCq,
                                __gm__ uint8_t *rmsnormGammaCkv,
                                __gm__ uint8_t *ropeSin,
                                __gm__ uint8_t *ropeCos,
                                __gm__ uint8_t *kvCache,
                                __gm__ uint8_t *krCache,
                                __gm__ uint8_t *cacheIndex,
                                __gm__ uint8_t *dequantScaleX,
                                __gm__ uint8_t *dequantScaleWDq,
                                __gm__ uint8_t *deqScaleQcQrW,
                                __gm__ uint8_t *dequantScaleWDkvkr,
                                __gm__ uint8_t *quantScaleCkv,
                                __gm__ uint8_t *quantScaleCkr,
                                __gm__ uint8_t *smoothScaleCq,
                                __gm__ uint8_t *actualSeqLen,
                                __gm__ uint8_t *kNopeClipAlpha,
                                __gm__ uint8_t *queryOut,
                                __gm__ uint8_t *queryRopeOut,
                                __gm__ uint8_t *dequantScaleQNopeOut,
                                __gm__ uint8_t *queryNormOut,
                                __gm__ uint8_t *dequantScaleQNormOut,
#if MLA_PROLOG_V3_SFA1
                                __gm__ uint8_t *curKNopeOut,
                                __gm__ uint8_t *curKRopeOut,
#endif
                                __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyGlobalParams();
    __aicore__ inline void ScaleInit(__gm__ uint8_t *dequantScaleX,
        __gm__ uint8_t *dequantScaleWDq, __gm__ uint8_t *deqScaleQcQrW,
        __gm__ uint8_t *dequantScaleWDkvkr, __gm__ uint8_t *quantScaleCkv,
        __gm__ uint8_t *quantScaleCkr, __gm__ uint8_t *smoothScaleCq,
        __gm__ uint8_t *kNopeClipAlpha);
    __aicore__ inline void WorkspaceInit(__gm__ uint8_t *workspace);
    // __aicore__ inline void UpdateActualSeq(int64_t fusedStartOffset);
    __aicore__ inline void MmParamInit();
    __aicore__ inline void MmCqParamInit();
    __aicore__ inline void MmCkvKrParamInit();
    __aicore__ inline void MmQcQrParamInit();
    __aicore__ inline void MmQnParamInit();
    __aicore__ inline void CubeBufferInit();
    __aicore__ inline void VectorBufferInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curStepBatchSize);
    __aicore__ inline void ComputeAicOffset(AicOffset &aicOffset, int64_t numHeadOffset);
    __aicore__ inline void GetBatchTokenIndex(int64_t &batchTokenIndex,
                                              int64_t &batchIndex, int64_t &batchSeqSize,
                                              int64_t &batchIndexOffset, int64_t tokenIndex);
    __aicore__ inline void GetCacheOffset(GlobalTensor<int64_t> indexGm,
                                          int64_t batchIndex,
                                          int64_t batchTokenIndex,
                                          int64_t batchSeqSize,
                                          int64_t batchIndexOffset,
                                          int64_t rows,
                                          int64_t headSize,
                                          int64_t &rowsInCurBatch,
                                          int64_t &cacheOffset,
                                          int64_t &nextBatchOffset);
    __aicore__ inline void ComputeAivOffset(AivOffset &aivOffset, int64_t batchOffset);
    template<bool needQnDynamicQuant>
    __aicore__ inline void AicProcess(AicOffset &aicOffset,
        int64_t batchOffset, int64_t mmQnLoops);
    template<bool needQnDynamicQuant>
    __aicore__ inline void AivProcess(AivOffset &aivOffset,
        int64_t batchOffset, int64_t curStepBatchSize, int64_t numHeadOffset, int64_t mmQnLoops);
    template <typename T, typename O, bool needCheckEmptyTensor = false,
              bool needCheckAFullLoad = false, bool isContinuousCopy = true>
    __aicore__ inline void MatmulSplitN(const GlobalTensor<O> &tensorResGm,
                                        const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,
                                        const MMParams &mmPara, const UsedBlockParams &mmBlockParams);
    __aicore__ inline void MatmulAndSyncQcQr(AicOffset &aicOffset);
    __aicore__ inline void MatmulQcQrSyncDequant(int64_t rmsNormCqResOffset,
                                                 int64_t weightUqQrOffset,
                                                 int64_t qcQrResOffset);
    __aicore__ inline void PreloadQnAndSync(AicOffset &aicOffset, int64_t mmQnLoops);
    __aicore__ inline void MatmulQnWeightPreload(int64_t weightUkOffset, int64_t mmQnLoops);
    template<bool needQnDynamicQuant>
    __aicore__ inline void MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset,
                                                            int64_t weightUkOffset,
                                                            int64_t qnResOffset,
                                                            int64_t mmQnLoops);
    __aicore__ inline void CopyInSinCos(int64_t tokenIndex,
                                        int64_t curVecToken,
                                        int64_t batchOffset,
                                        int64_t curStepBatchSize);
    __aicore__ inline void RmsNormCq(int64_t tokenIndex,
                                     int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset,
                                     int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void RopeAndScatterKr(LocalTensor<float>& dequantScaleXLocal,
                                            LocalTensor<uint8_t>& shareTmpUb,
                                            LocalTensor<typename MLAPT::ropeComputType>& cosLocalCkvKr,
                                            LocalTensor<typename MLAPT::ropeComputType>& sinLocalCkvKr,
                                            CkvkrParams ropeAndScatterKrParams);
    __aicore__ inline void RmsNormAndScatterCkv(LocalTensor<float>& dequantScaleXLocal,
                                                LocalTensor<uint8_t>& shareTmpUb,
                                                LocalTensor<typename MLAPT::ropeComputType>& cosLocalCkvKr,
                                                LocalTensor<typename MLAPT::ropeComputType>& sinLocalCkvKr,
                                                CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void RmsNormRopeScatterCkvKr(int64_t tokenIndex,
                                                   int64_t rmsNormCkvOffset,
                                                   int64_t ropeKrOffset,
                                                   int64_t curVecToken);
    __aicore__ inline void RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset,
        int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void DequantQc(int64_t mmQnPreDequantOffset,
                                    int64_t mmQnPreDequantResOffset,
                                    int64_t curVecToken,
                                    int64_t curBlockTokenOffset);
    // 低时延算力分组场景
    __aicore__ inline void DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset,
        int64_t mmQnPreDequantResOffset, int64_t qcQrScaleOffset);
    __aicore__ inline void RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantQcQrSplitN(const DequantQcQrSplitNParams& dequantQcQrSplitN);
    __aicore__ inline void RopeQrSplitN(const RopeQrSplitNParams& ropeQrSplitNParams);
    __aicore__ inline void DequantAndRopeSplitNSyncMMQcQr(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                                          int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset,
        int64_t curStepBatchSize, int64_t numHeadOffset, int64_t mmQnLoops);

    TPipe* pipe_;
#if MLA_PROLOG_V3_SFA1
    const optiling::FluentMlaPrologV3TilingData* __restrict tilingData_;
    const optiling::FluentMlaPrologV3BaseParams* __restrict baseParams_;
#else
    const optiling::MlaPrologV3TilingData* __restrict tilingData_;
    const optiling::MlaPrologV3BaseParams* __restrict baseParams_;
#endif
    uint32_t blockIdx_ = 0U;
#if !MLA_PROLOG_V3_SFA1
    uint32_t cvRatio_ = 2U;
#endif
    uint32_t cubeBlockIdx_ = 0U; // AIV上使用AIC的blockIdx
    int64_t vectorRow_ = 1;
    int64_t curVectorBlockNum_;
    int64_t vectorCoreNum_;
    uint32_t curStepVecFrontToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecBackToken_;
    uint32_t curVecTokenMax_;
    bool enableSmoothScalesCq_;

    struct DequantTool {
        GlobalTensor<float> deQuantScaleCqGm_;
        TBuf<TPosition::VECCALC> deQuantScaleCqBuffer_;  // 用于临时存储每一行的Scale，以及汇总最终每一行的Scale参数
        LocalTensor<float> deQuantScaleCqLocal_;
        __aicore__ inline DequantTool() {}
    };

    // 算子分组开关
    DequantTool dequantTool_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmQnInputType> weightUkGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<int64_t> cacheIndexGm_;
    GlobalTensor<kvCacheType> kvCacheGm_;
    GlobalTensor<krCacheType> krCacheGm_;
    GlobalTensor<ropeOutputType> qrOutGm_;

    GlobalTensor<float> dequantScaleXGm_;
    GlobalTensor<float> dequantScaleWDqGm_;
    GlobalTensor<float> dequantScaleWDkvkrGm_;
    GlobalTensor<float> smoothScaleCqGm_;
    GlobalTensor<float> deqScaleQcQrW_; // per-channel反量化参数
    GlobalTensor<float> quantScaleCkvGm_;
    GlobalTensor<float> quantScaleCkrGm_;

    GlobalTensor<int32_t> actualSeqLenGm_;
    GlobalTensor<float> kNopeClipAlphaGm_;

    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<mmQnInputType> mmQcQrResDequantGm_;
    GlobalTensor<mmQnOutputType> mmQnResGm_;
    GlobalTensor<dequantScaleQNopeType> dequantScaleQNopeGm_;
    GlobalTensor<queryOutputType> queryOutGm_;
    GlobalTensor<dequantScaleQNormType> dequantScaleQNormGm_;
#if MLA_PROLOG_V3_SFA1
    GlobalTensor<uint8_t> curKNopeGm_;    // 字节视图：cur_k_nope 输出（行 stride = Dtile * sizeof(kvCacheType)）
    GlobalTensor<uint8_t> curKRopeGm_;    // 字节视图：cur_k_rope 输出（行 stride = Dr * sizeof(krCacheType)）
#endif

    // UB
    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> shareBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDqBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDkvKrBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCqBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCkvBuffer_;
    TBuf<TPosition::VECCALC> smoothScaleCqBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkvBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkrBuffer_;
    TBuf<TPosition::VECCALC> stepActualSeqBuffer_;

    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<ropeComputType> sinLocal_;
    LocalTensor<float> dequantScaleWDqLocal_;
    LocalTensor<float> dequantScaleWDkvKrLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCqLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCkvLocal_;
    LocalTensor<float> smoothScaleCqLocal_;
    LocalTensor<float> quantScaleCkvLocal_;
    LocalTensor<float> quantScaleCkrLocal_;
    LocalTensor<int64_t> stepActualSeqLocal_;

    struct ActSeqState {
        bool inited = false;        // whether state is initialized for the run
        int64_t curBatch = 0;       // current batch index for the running token index
        int64_t prevPrefix = 0;     // prefix sum of seq lengths up to (curBatch - 1)
        int64_t curPrefix = 0;      // prefix sum of seq lengths up to curBatch (exclusive high bound)
        int64_t prevIndexOffset = 0;   // prefix sum of blocks up to (curBatch - 1)
        int64_t curIndexOffset = 0;    // prefix sum of blocks up to curBatch
    } actSeqState_;
    int64_t blockSpanPerBatch_ = 0; // blockNum * blockSize

    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    LocalTensor<mmInputType> aL1Tensor_;
    LocalTensor<mmInputType> bL1Tensor_;
    MMBufParams bufParam_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
};

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::Init(
    __gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq,
    __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk,
    __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq,
    __gm__ uint8_t *rmsnormGammaCkv, __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos, __gm__ uint8_t *kvCache,
    __gm__ uint8_t *krCache,  __gm__ uint8_t *cacheIndex,
    __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
    __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *actualSeqLen,
    __gm__ uint8_t *kNopeClipAlpha, __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut,
    __gm__ uint8_t *dequantScaleQNopeOut,
    __gm__ uint8_t *queryNormOut,
    __gm__ uint8_t *dequantScaleQNormOut,
#if MLA_PROLOG_V3_SFA1
    __gm__ uint8_t *curKNopeOut,
    __gm__ uint8_t *curKRopeOut,
#endif
    __gm__ uint8_t *workspace) {

    blockIdx_ = GetBlockIdx(); // cube:0-23  vec:0-47
    if ASCEND_IS_AIV {
#if MLA_PROLOG_V3_SFA1
        cubeBlockIdx_ = blockIdx_ >> 1;
#else
        cvRatio_ = GetSubBlockNum();
        cubeBlockIdx_ = blockIdx_ / cvRatio_;
#endif
    } else {
        cubeBlockIdx_ = blockIdx_;
    }
    curVectorBlockNum_ = static_cast<int64_t>(baseParams_->stepBatchSize);
    vectorCoreNum_ = static_cast<int64_t>(baseParams_->vectorBlockNum); // aivNum 48
#if !MLA_PROLOG_V3_SFA1
    if (cvMode == 1 && cvRatio_ == 2) {
        if (vectorCoreNum_ < curVectorBlockNum_) {
            vectorCoreNum_ *= 2;
        }
    }
#endif
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;
    enableSmoothScalesCq_ = smoothScaleCq == nullptr ? false : true;
    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType *)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDq);   // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)weightUqQr);   // NZ
    weightUkGm_.SetGlobalBuffer((__gm__ mmQnInputType *)weightUk);
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDkvKr);  // NZ
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCq);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCkv);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeCos);
    if constexpr (MLAPT::cacheMode != CACHE_MODE::ND) {
        cacheIndexGm_.SetGlobalBuffer((__gm__ int64_t *)cacheIndex);
    }
    kvCacheGm_.SetGlobalBuffer((__gm__ kvCacheType *)kvCache);
    krCacheGm_.SetGlobalBuffer((__gm__ krCacheType *)krCache);
    qrOutGm_.SetGlobalBuffer((__gm__ ropeOutputType *)queryRopeOut);
    if constexpr (std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value && !isPertile) {
        dequantScaleQNopeGm_.SetGlobalBuffer((__gm__ dequantScaleQNopeType *)dequantScaleQNopeOut);
        queryOutGm_.SetGlobalBuffer((__gm__ queryOutputType *)queryOut);
    } else {
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)queryOut);
    }
    if (baseParams_->queryNormFlag) {
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)queryNormOut);
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            dequantScaleQNormGm_.SetGlobalBuffer((__gm__ dequantScaleQNormType *)dequantScaleQNormOut);
        }
    }
#if MLA_PROLOG_V3_SFA1
    // 字节视图 GM 挂载
    curKNopeGm_.SetGlobalBuffer((__gm__ uint8_t *)curKNopeOut);
    curKRopeGm_.SetGlobalBuffer((__gm__ uint8_t *)curKRopeOut);
#endif
    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        actualSeqLenGm_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLen);
    }

    ScaleInit(dequantScaleX, dequantScaleWDq, deqScaleQcQrW,
        dequantScaleWDkvkr, quantScaleCkv, quantScaleCkr, smoothScaleCq, kNopeClipAlpha);
    MmParamInit();
    WorkspaceInit(workspace);
    if ASCEND_IS_AIV {
        VectorBufferInit();
    } else {
        CubeBufferInit();
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::ScaleInit(
    __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
    __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *kNopeClipAlpha) {
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        dequantScaleXGm_.SetGlobalBuffer((__gm__ float *)dequantScaleX);
        dequantScaleWDqGm_.SetGlobalBuffer((__gm__ float *)dequantScaleWDq);
        dequantScaleWDkvkrGm_.SetGlobalBuffer((__gm__ float *)dequantScaleWDkvkr);
    }
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        smoothScaleCqGm_.SetGlobalBuffer((__gm__ float *)smoothScaleCq);
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ float *)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkv);
        quantScaleCkrGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkr);
    }

    if constexpr (isPertile) {
        kNopeClipAlphaGm_.SetGlobalBuffer((__gm__ float *)kNopeClipAlpha);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MmParamInit() {
    MmCqParamInit();
    MmCkvKrParamInit();
    MmQcQrParamInit();
    MmQnParamInit();
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MmCqParamInit() {
    mmCqParam_.m = baseParams_->stepBatchSize; // 32
    if (cubeBlockIdx_ == baseParams_->mm1BlockNum - 1) {
        mmCqParam_.n = baseParams_->headSizeCq - baseParams_->mm1SingleCoreN * cubeBlockIdx_;
    } else {
        mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
    }
    mmCqParam_.k = baseParams_->headSizeX; // 7168
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgKa = mmCqParam_.k;
    mmCqParam_.orgKb = mmCqParam_.k;
    mmCqParam_.orgKc = baseParams_->headSizeCq;  // 1536
    mmCqParam_.baseK = (sizeof(mmInputType) == sizeof(int8_t)) ?
        256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCqParam_.baseN = 128;
    mmCqParam_.stepK = 4;
    mmCqParam_.kL1StepSize = mmCqParam_.baseK * mmCqParam_.stepK;
    mmCqParam_.tailStepK = (mmCqParam_.k % mmCqParam_.kL1StepSize != 0) ? (
        (mmCqParam_.k % mmCqParam_.kL1StepSize) / mmCqParam_.baseK) : mmCqParam_.stepK;
    mmCqParam_.kL1TailStepSize = mmCqParam_.tailStepK * mmCqParam_.baseK;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MmCkvKrParamInit() {
    mmCkvKrParam_.m = baseParams_->stepBatchSize; // 32
    if (cubeBlockIdx_ == baseParams_->mm2BlockNum - 1) {
        mmCkvKrParam_.n = baseParams_->headSizeCkv +
                          baseParams_->headSizeKr - baseParams_->mm2SingleCoreN * cubeBlockIdx_;
    } else {
        mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
    }
    mmCkvKrParam_.k = baseParams_->headSizeX; // 7168
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKc = (baseParams_->headSizeCkv + baseParams_->dimHeadRope);  // 576
    mmCkvKrParam_.baseK = (sizeof(mmInputType) == sizeof(int8_t)) ?
        256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCkvKrParam_.baseN = 128;
    mmCkvKrParam_.stepK = 4;
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;
    mmCkvKrParam_.tailStepK = (mmCkvKrParam_.k % mmCkvKrParam_.kL1StepSize != 0) ? (
        (mmCkvKrParam_.k % mmCkvKrParam_.kL1StepSize) / mmCkvKrParam_.baseK) : mmCkvKrParam_.stepK;
    mmCkvKrParam_.kL1TailStepSize = mmCkvKrParam_.tailStepK * mmCkvKrParam_.baseK;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MmQcQrParamInit() {
    mmQcQrParam_.m = baseParams_->stepBatchSize; // 32
    if constexpr (MLAPT::enableGroupComputeOpt) {
        // 算力分组仅考虑G8 Qc Qr(8+4核), n固定128, 这里不处理尾核
        mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    } else {
        if (cubeBlockIdx_ == baseParams_->mm3BlockNum - 1) {
            mmQcQrParam_.n = baseParams_->headSizeQc + baseParams_->headSizeQr -
                baseParams_->mm3SingleCoreN * cubeBlockIdx_;
        } else {
            mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
        }
    }

    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQc + baseParams_->headSizeQr); // (128 * 32 + 64 * 32)
    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? 128 : 64;
    if constexpr (MLAPT::enableGroupComputeOpt) {
        mmQcQrParam_.baseN = 128;
    } else {
        mmQcQrParam_.baseN = 128;
    }
    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
    mmQcQrParam_.tailStepK = (mmQcQrParam_.k % mmQcQrParam_.kL1StepSize != 0) ? (
        (mmQcQrParam_.k % mmQcQrParam_.kL1StepSize) / mmQcQrParam_.baseK) : mmQcQrParam_.stepK;
    mmQcQrParam_.kL1TailStepSize = mmQcQrParam_.tailStepK * mmQcQrParam_.baseK;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MmQnParamInit() {
    mmQnParam_.m = baseParams_->stepBatchSize; // 32
    mmQnParam_.n = baseParams_->headSizeCkv; // 512
    mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
    mmQnParam_.needSetOrgShape = 1;
    mmQnParam_.orgM = mmQnParam_.m;
    mmQnParam_.orgN = mmQnParam_.n;
    if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
        mmQnParam_.orgKa = baseParams_->headSizeQc;
    } else {
        mmQnParam_.orgKa = baseParams_->headSizeQc + baseParams_->headSizeQr;
    }
    mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
    mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;
    mmQnParam_.kL1StepSize = mmQnParam_.k;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::VectorBufferInit() {

    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        pipe_->InitBuffer(dequantScaleWDqBuffer_, baseParams_->headSizeCq * sizeof(float)); // [1, 1536]
        dequantScaleWDqLocal_ = dequantScaleWDqBuffer_.Get<float>();

        pipe_->InitBuffer(dequantScaleWDkvKrBuffer_,
            (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(float)); // [1, 512 + 64]
        dequantScaleWDkvKrLocal_ = dequantScaleWDkvKrBuffer_.Get<float>();
    }

    pipe_->InitBuffer(rmsnormGammaCqBuffer_, baseParams_->headSizeCq * sizeof(rmsNormGammaType)); // [1, 1536] bf16
    rmsnormGammaCqLocal_ = rmsnormGammaCqBuffer_.Get<rmsNormGammaType>();

    pipe_->InitBuffer(rmsnormGammaCkvBuffer_, baseParams_->headSizeCkv * sizeof(rmsNormGammaType)); // [1, 512] bf16
    rmsnormGammaCkvLocal_ = rmsnormGammaCkvBuffer_.Get<rmsNormGammaType>();

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        if (enableSmoothScalesCq_) {
            pipe_->InitBuffer(smoothScaleCqBuffer_, baseParams_->headSizeCq * sizeof(float)); // [1, 1536]
            smoothScaleCqLocal_ = smoothScaleCqBuffer_.Get<float>();
        }
    }

    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        pipe_->InitBuffer(quantScaleCkrBuffer_, baseParams_->dimHeadRope * sizeof(float)); // [1, 64]
        quantScaleCkrLocal_ = quantScaleCkrBuffer_.Get<float>();
    }

    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            pipe_->InitBuffer(quantScaleCkvBuffer_, ALIGN_BLOCK_SIZE);
        } else {
            pipe_->InitBuffer(quantScaleCkvBuffer_, baseParams_->headSizeCkv * sizeof(float)); // [1, 512]
        }
        quantScaleCkvLocal_ = quantScaleCkvBuffer_.Get<float>();
    }

    // 预留brcb的空间
    pipe_->InitBuffer(dequantTool_.deQuantScaleCqBuffer_, (baseParams_->stepBatchSize + 7) * ALIGN_BLOCK_SIZE);
    dequantTool_.deQuantScaleCqLocal_ = dequantTool_.deQuantScaleCqBuffer_.template Get<float>();

    if constexpr (MLAPT::enableDequantOpt) {
        // 在ropeQr进行切N处理后，会复用shareBuffer的内存，不需要额外申请
        // 开启开关后会按照head切分rope qr，此时需要加载一半batchsize数量的sin和cos值
        // 需要2倍的空间分别存储sin和cos
        pipe_->InitBuffer(sincosBuffer_, 2 * baseParams_->dimHeadRope *
#if MLA_PROLOG_V3_SFA1
                          sizeof(ropeComputType) * ((baseParams_->stepBatchSize + 1) >> 1));
#else
                          sizeof(ropeComputType) * ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_));
#endif
    } else {
        // 需要2倍的空间分别存储sin和cos
        pipe_->InitBuffer(sincosBuffer_, 2 * baseParams_->dimHeadRope *
                          sizeof(ropeComputType) * curVecTokenMax_); // [2, 64] float
    }

    uint64_t usedAddr;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
#if MLA_PROLOG_V3_SFA1
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + 1) >> 1)];
#else
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_)];
#endif
        usedAddr = reinterpret_cast<uint64_t>(
#if MLA_PROLOG_V3_SFA1
            sinLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + 1) >> 1)].GetPhyAddr());
#else
            sinLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_)].GetPhyAddr());
#endif
    } else {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
        usedAddr =  reinterpret_cast<uint64_t>(sinLocal_[baseParams_->dimHeadRope * curVecTokenMax_].GetPhyAddr());
    }

    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        pipe_->InitBuffer(stepActualSeqBuffer_, baseParams_->stepBatchSize * sizeof(int64_t)); // [1, stepBatchSize]
        stepActualSeqLocal_ = stepActualSeqBuffer_.Get<int64_t>();
        usedAddr += reinterpret_cast<uint64_t>(stepActualSeqLocal_[curVecTokenMax_].GetPhyAddr());
    }

    // 由于shareBuffer属于各个vector操作临时申请内存的区域内存使用不固定，建议shareBuffer始终放在最后
    // 防止写入shareBuffer越界导致前面固定申请的UB内存被踩。
    pipe_->InitBuffer(shareBuffer_, MAX_UB_SIZE - usedAddr); // 除 sincos 外的 sharebuffer 共享，不会同时使用
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::CubeBufferInit() {
    // cube相关Buffer初始化
    pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
    pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);

    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
    aL1Tensor_ = aBufL1_.Get<mmInputType>();
    bL1Tensor_ = bBufL1_.Get<mmInputType>();
    bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
    bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());

    pipe_->InitBuffer(aBufL0_, L0A_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(bBufL0_, L0B_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(cBufL0_, L0C_PP_SIZE * 2); // 128K

    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

    bufParam_.aL0BufAddr = aBufL0_.GetBufferAddr(aBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.bL0BufAddr = bBufL0_.GetBufferAddr(bBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.cL0BufAddr = cBufL0_.GetBufferAddr(cBufL0_.Get<float>().GetBufferHandle());
}

/*
 * workspace管理
 * 1. 常驻：dequantTool_.deQuantScaleCqGm_ stepBs * 32 Byte
 * 2. 中间结果：
 *    tokenXGm_───>mmCkvKrResGm_
 *          |      [stepBS, HCkv + Dr]
 *          |      (bf16 | int32)
 *          └─────>mmCqResGm_─────>rmsNormCqResGm_─────>mmQcQrResGm_──────>mmQcQrResDequantGm_──────>mmQnResGm_
 *                 [stepBS, HCq]   [stepBS, HCq]        [stepBS, N1, D + Dr]   [stepBS, N1, D]       [stepBS, N1, HCkv]
 *                 (bf16 | int32)  (bf16 | int8)        (bf16 | int32)         (bf16)                (bf16)
 */
template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::WorkspaceInit(__gm__ uint8_t *workspace) {
    int64_t workspaceOffset = 0;
    if constexpr (MLAPT::enableGroupComputeOpt || MLAPT::enableDequantOpt) {
        dequantTool_.deQuantScaleCqGm_.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffset));
        workspaceOffset += baseParams_->stepBatchSize * ALIGN_BLOCK_SIZE;
    }

    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType *)(workspace + workspaceOffset));
    workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                       static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                       sizeof(mmCkvKrOutputType);

    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType *)(workspace + workspaceOffset));
    if constexpr (!std::is_same<rmsNormCqOutputType, mmCqOutputType>::value) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeCq) * sizeof(mmCqOutputType);
    }

    if (!baseParams_->queryNormFlag) {
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType *)(workspace + workspaceOffset));
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeCq) * sizeof(rmsNormCqOutputType);
    } else if constexpr (std::is_same<rmsNormCqOutputType, mmCqOutputType>::value) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeCq) * sizeof(mmCqOutputType);
    }

    mmQcQrResGm_.SetGlobalBuffer((__gm__ mmQcQrOutputType *)(workspace + workspaceOffset));
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeQc + baseParams_->headSizeQr) *
                           sizeof(mmQcQrOutputType);
    }

    mmQcQrResDequantGm_.SetGlobalBuffer((__gm__ mmQnInputType *)(workspace + workspaceOffset));
    if constexpr (std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value && !isPertile) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->numHeadSize) *
                           static_cast<int64_t>(baseParams_->dimHeadSizeQc) * sizeof(mmQnInputType);
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)(workspace + workspaceOffset));
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::UpdateStepBatchParams(int64_t curStepBatchSize) {
    mmCqParam_.m = curStepBatchSize;
    mmCkvKrParam_.m = curStepBatchSize;
    mmQcQrParam_.m = curStepBatchSize;
    mmQnParam_.m = curStepBatchSize;
    curVectorBlockNum_ = curStepBatchSize;
}

/*
 * MlaPrologV3算子计算&CV流水同步流程
 *                    ┌───────────────── token_x ─────────────────┐
 *                    |                                           ▼
 *                    |                                      MatmulCkvKr
 *                    ▼                                           | wait mm CkvKr(0x1)
 *                MatmulCq                                        ▼
 *                    | wait mm Cq(0x1)                  ┌─────────────────┐
 *                    ▼                                  ▼                 ▼
 *               RmsNorm(Cq)                         RmsNorm(Ckv)       Rope(Kr)
 *                    | wait rmsNorm cq(0x1)             |                 |
 *                    ▼                                  ▼                 ▼
 *        ┌───────MatmulQcQr───────┐                 Scatter(Ckv)      Scatter(Kr)
 *        | wait mm Qc(0x1)        |                     |                 |
 *        ▼                        |                     ▼                 ▼
 *    DequantQc                    |                 kv_cache_out      kr_cache_out
 *        | wait dequant qc(0x1)   | wait mm Qr(0x2)
 *        ▼                        ▼
 *     MatmulQn                 Rope(Qr)
 *        | wait mm Qn(0x1)        |
 *   DynamicQuantQn──┐             |
 *        |          ▼             |
 *        ▼   dequant_scale_out    ▼
 *    query_out               query_rope_out
 * 注：仅为表明基本计算与CV同步流程，仅包含了影响CV同步的量化分支，其余量化分支应参考设计文档。
 */
template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::Process() {
    constexpr bool needQnDynamicQuant =
        std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value && !isPertile;
    int64_t numHeadOffset = cubeBlockIdx_ * baseParams_->mm4SingleCoreBatch;
    int64_t mmQnLoops;
    if (cubeBlockIdx_ == baseParams_->mm4BlockNum - 1) {
        mmQnLoops = baseParams_->numHeadSize - numHeadOffset;
    } else {
        mmQnLoops = baseParams_->mm4SingleCoreBatch;
    }

    // AIC的offset参数
    AicOffset aicOffset;
    ComputeAicOffset(aicOffset, numHeadOffset);

    // AIV的offset参数
    AivOffset aivOffset;
    ComputeAivOffset(aivOffset, 0);

    int64_t bsSize = static_cast<int64_t>(baseParams_->tokenSize);
    // 需要考虑BS合轴的尾块情况
    for (int64_t batchOffset = 0; batchOffset < bsSize; batchOffset += baseParams_->stepBatchSize) {
        if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
            if (!actSeqState_.inited) {
                actSeqState_.curBatch = 0;
                actSeqState_.prevPrefix = 0;
                actSeqState_.curPrefix = (baseParams_->batchSize > 0) ? actualSeqLenGm_(0) : 0;
                actSeqState_.inited = true;
                actSeqState_.prevIndexOffset = 0;
                actSeqState_.curIndexOffset = CeilDivT(actSeqState_.curPrefix,
                                                       static_cast<int64_t>(baseParams_->blockSize));
            }
        }
        int64_t curStepBatchSize = bsSize - batchOffset;
        if (curStepBatchSize < baseParams_->stepBatchSize) {
            UpdateStepBatchParams(curStepBatchSize); // 320 - 256
            if (batchOffset != 0) {
                ComputeAivOffset(aivOffset, batchOffset);
            }
        } else {
            curStepBatchSize = baseParams_->stepBatchSize;
        }
        if ASCEND_IS_AIC {
            AicProcess<needQnDynamicQuant>(aicOffset, batchOffset, mmQnLoops);
        }
        if ASCEND_IS_AIV {
            AivProcess<needQnDynamicQuant>(aivOffset, batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
        }
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
    }
}

template<typename MLAPT>
template<bool needQnDynamicQuant>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::AicProcess(AicOffset &aicOffset,
                                                                int64_t batchOffset, int64_t mmQnLoops) {
    int64_t tokenXOffset = batchOffset * baseParams_->headSizeX;
    // MatmulCq ──> RmsNorm(Cq)
    // [32, 7168] * [7168, 1536] = [32, 1536]
    MatmulSplitN<mmInputType, mmCqOutputType>(mmCqResGm_[aicOffset.cqResOffset],
        tokenXGm_[tokenXOffset], weightDqGm_[aicOffset.weightDqOffset],
        mmCqParam_, UsedBlockParams{0, baseParams_->mm1BlockNum});
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);
    // MatmulCkvKr ──> RmsNorm(Ckv)
    //            └──> Rope(Kr)
    // [32, 7168] * [7168, 512+64] = [32, 576]
    MatmulSplitN<mmInputType, mmCkvKrOutputType, true>(mmCkvKrResGm_[aicOffset.ckvKrResOffset],
        tokenXGm_[tokenXOffset], weightDkvKrGm_[aicOffset.weightDkvKrOffset],
        mmCkvKrParam_, UsedBlockParams{0, baseParams_->mm2BlockNum});
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);
    CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);
    MatmulAndSyncQcQr(aicOffset);
    PreloadQnAndSync(aicOffset, mmQnLoops);
    MatmulQnSyncDynamicQuantAndMulQr<needQnDynamicQuant>(aicOffset.qcOffset,
        aicOffset.weightUkOffset, aicOffset.qnResOffset, mmQnLoops);
    if constexpr (!needQnDynamicQuant) {
        // MatmulQn的结果直接输出到 queryOut, qnOffset需要按Batch轴偏移
        aicOffset.qnResOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
            static_cast<int64_t>(baseParams_->headSizeCkv) *
            static_cast<int64_t>(baseParams_->numHeadSize);
    }
    if (unlikely (baseParams_->queryNormFlag)) {
        aicOffset.rmsNormCqResOffset += static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->headSizeCq);
    }
}

template<typename MLAPT>
template<bool needQnDynamicQuant>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::AivProcess(AivOffset &aivOffset,
    int64_t batchOffset, int64_t curStepBatchSize, int64_t numHeadOffset, int64_t mmQnLoops) {
    int64_t tokenIndex = batchOffset + aivOffset.curBlockTokenOffset;
    int64_t rmsNormCqResOffset;
    if (baseParams_->queryNormFlag) {
        rmsNormCqResOffset = tokenIndex * baseParams_->headSizeCq;
    } else {
        rmsNormCqResOffset = aivOffset.rmsNormCqOffset;
    }
    if (batchOffset == 0) {
        // 只需要搬运一次
        CopyGlobalParams();
    }
    CopyInSinCos(tokenIndex, aivOffset.curVecToken, batchOffset, curStepBatchSize);
    CrossCoreWaitFlag(FINISH_MM_CQ);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    RmsNormCq(tokenIndex, aivOffset.rmsNormCqOffset, rmsNormCqResOffset,
        aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
    // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
    // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);

    // 聚合全部scale结果
    if constexpr (MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) {
        DataCopy(dequantTool_.deQuantScaleCqLocal_, dequantTool_.deQuantScaleCqGm_,
            ALIGN_BLOCK_SIZE / sizeof(float) * baseParams_->stepBatchSize);
    }
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
    CrossCoreWaitFlag(FINISH_MM_CKVKR);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    RmsNormRopeScatterCkvKr(tokenIndex, aivOffset.rmsNormCkvOffset, aivOffset.ropeKrOffset, aivOffset.curVecToken);

    // 根据不同分支条件处理
    if constexpr (MLAPT::enableGroupComputeOpt) {
        CrossCoreWaitFlag(FINISH_MM_QC);
        DequantQcSplitNGroupCase(aivOffset.mmQnPreDequantOffset,
            aivOffset.mmQnPreDequantResOffset, aivOffset.qcScaleOffsetSplitN);
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
        CrossCoreWaitFlag(FINISH_MM_QR);
        RopeQrSplitNGroupCase(aivOffset.ropeQrSplitNOffset, aivOffset.ropeQrResSplitNOffset);
    } else {
        if constexpr (MLAPT::enableDequantOpt) {
            DequantAndRopeSplitNSyncMMQcQr(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
                aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset);
        } else if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            CrossCoreWaitFlag(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            DequantQc(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
                aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset,
                aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
        } else {
            CrossCoreWaitFlag(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset,
                aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
        }

        if constexpr (needQnDynamicQuant && !(MLAPT::enableDequantOpt)) {
            // 非切N场景，需要等待全部rope的结果做完并搬到GM
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        }
        if constexpr (needQnDynamicQuant) {
            DynamicQuantQnAndMulQrSyncMMQn(batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
        }
        aivOffset.ropeQrResOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
            static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::ComputeAicOffset(AicOffset &aicOffset, int64_t numHeadOffset) {
    aicOffset.cqResOffset = baseParams_->mm1SingleCoreN * blockIdx_;  // 64 * idx
    aicOffset.weightDqOffset = static_cast<int64_t>(baseParams_->headSizeX) * aicOffset.cqResOffset;  // 7168 * 64 * idx

    aicOffset.ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_;  //  (512 + 64) / 9 * idx  = 64 * idx
    aicOffset.weightDkvKrOffset = static_cast<int64_t>(baseParams_->headSizeX) *
        aicOffset.ckvKrResOffset;  // 7168 * (512 + 64) / 9 * idx = 7168 * 64 * idx

    if constexpr (MLAPT::enableGroupComputeOpt) {
        aicOffset.weightUqOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                               (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                               blockIdx_;  // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
        aicOffset.weightQrOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                               (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * 2 * (blockIdx_ - QC_CORE_NUM) +
                               static_cast<int64_t>(baseParams_->headSizeCq) * baseParams_->dimHeadSizeQc;
        aicOffset.qCResOffset = baseParams_->dimHeadSizeQc * blockIdx_;
        aicOffset.qRResOffset = baseParams_->stepBatchSize * baseParams_->dimHeadSizeQc * QC_CORE_NUM +
                      baseParams_->dimHeadRope * 2 * (blockIdx_ - QC_CORE_NUM); // m=baseParams_->stepBatchSize
    } else {
        aicOffset.qcQrResOffset = baseParams_->mm3SingleCoreN *
            blockIdx_;  // (32 * 128 + 32 * 64) / 24 * idx = 256 * idx
        aicOffset.weightUqQrOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
            aicOffset.qcQrResOffset;  // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
    }

    if (cubeBlockIdx_ < baseParams_->mm4BlockNum) {
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) * numHeadOffset; // 128 * idx
        } else {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                numHeadOffset; // (128 + 64) * idx
        }
        aicOffset.weightUkOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
            static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset;   // (128 * 512) * idx
        aicOffset.qnResOffset = static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset; // BS, N, Dq
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::ComputeAivOffset(AivOffset &aivOffset, int64_t batchOffset) {
    curStepVecBackToken_ = curVectorBlockNum_ / static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontListNum_ = curVectorBlockNum_ % static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontToken_ = curStepVecFrontListNum_ == 0 ? curStepVecBackToken_ : curStepVecBackToken_ + 1;

    aivOffset.curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    aivOffset.curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ? blockIdx_ *
        aivOffset.curVecToken : blockIdx_ * aivOffset.curVecToken + curStepVecFrontListNum_;
    aivOffset.rmsNormCqOffset = baseParams_->headSizeCq * aivOffset.curBlockTokenOffset;  //  1536 * idx
    aivOffset.rmsNormCkvOffset = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
        aivOffset.curBlockTokenOffset;  // (512 + 64) * idx
    aivOffset.ropeKrOffset = baseParams_->headSizeCkv + aivOffset.rmsNormCkvOffset;  // 512 + (512 + 64) * idx
    if constexpr (!MLAPT::enableDequantOpt) {
        aivOffset.mmQnPreDequantOffset = (baseParams_->headSizeQc + baseParams_->headSizeQr) *
            aivOffset.curBlockTokenOffset;
        aivOffset.mmQnPreDequantResOffset = baseParams_->headSizeQc * aivOffset.curBlockTokenOffset;
        aivOffset.ropeQrOffset = baseParams_->dimHeadSizeQc + static_cast<int64_t>(baseParams_->numHeadSize) *
                        static_cast<int64_t>(baseParams_->dimHeadSizeQc +
                        baseParams_->dimHeadRope) *
                        aivOffset.curBlockTokenOffset; // 128 + 32 * (128 + 64) * idx
        aivOffset.ropeQrResOffset = static_cast<int64_t>(baseParams_->headSizeQr) *
            (aivOffset.curBlockTokenOffset + batchOffset); // 32 * 64 * idx; 按BS合轴切分step
    }

    // 以下的Offset均是与batchOffset无关的，仅初始化一次即可。
    if (batchOffset == 0) {
        if constexpr (MLAPT::enableDequantOpt) {
            aivOffset.mmQnPreDequantOffset = baseParams_->mm3SingleCoreN * cubeBlockIdx_; // qcQrResOffset
            aivOffset.mmQnPreDequantResOffset = baseParams_->mm3SingleCoreN / (baseParams_->dimHeadSizeQc +
                baseParams_->dimHeadRope) * baseParams_->dimHeadSizeQc * cubeBlockIdx_;
            aivOffset.ropeQrOffset = aivOffset.mmQnPreDequantOffset + baseParams_->dimHeadSizeQc;
            aivOffset.ropeQrResOffset = (baseParams_->mm3SingleCoreN / (baseParams_->dimHeadSizeQc +
                baseParams_->dimHeadRope)) * baseParams_->dimHeadRope * cubeBlockIdx_;
        }

        if constexpr (MLAPT::enableGroupComputeOpt) {
            aivOffset.qcScaleOffsetSplitN = (baseParams_->headSizeQc + baseParams_->headSizeQr) /
#if MLA_PROLOG_V3_SFA1
                QC_CORE_NUM * (blockIdx_ >> 1);
            aivOffset.mmQnPreDequantResOffset = (baseParams_->headSizeQc) / QC_CORE_NUM * (blockIdx_ >> 1);
#else
                QC_CORE_NUM * (blockIdx_ / cvRatio_);
            aivOffset.mmQnPreDequantResOffset = (baseParams_->headSizeQc) / QC_CORE_NUM * (blockIdx_ / cvRatio_);
#endif
            aivOffset.mmQnPreDequantOffset = aivOffset.mmQnPreDequantResOffset;
#if MLA_PROLOG_V3_SFA1
            aivOffset.ropeQrResSplitNOffset = (blockIdx_ - QC_CORE_NUM * 2) * baseParams_->dimHeadRope;
#else
            aivOffset.ropeQrResSplitNOffset = (blockIdx_ - QC_CORE_NUM * cvRatio_) * baseParams_->dimHeadRope;
#endif
            aivOffset.ropeQrSplitNOffset = baseParams_->headSizeQc + aivOffset.ropeQrResSplitNOffset;
        }
    }
}

// Mlaprolog 支持int8进int32出, 参考MatmulQcQr
template<typename MLAPT>
template<typename T, typename O, bool needCheckEmptyTensor, bool needCheckAFullLoad, bool isContinuousCopy>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MatmulSplitN(
    const GlobalTensor<O> &tensorResGm,
    const GlobalTensor<T> &tensorAGm,
    const GlobalTensor<T> &tensorBGm,
    const MMParams &mmPara, const UsedBlockParams &mmBlockParams) {
    if constexpr (needCheckEmptyTensor) {
        if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
            return;
        }
    }
    if (blockIdx_ < mmBlockParams.blockStartIdx || blockIdx_ >= mmBlockParams.blockEndIdx) {
        return;
    }
    // 用于enableGroupComputeOpt场景
    if constexpr (needCheckAFullLoad) {
        constexpr uint32_t mSize = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ?
            INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
        bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;
        if (isAFullLoad) {
            MatmulGroupComputeAFullLoad<T, O,
                isContinuousCopy>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_);
            return;
        }
    }
    uint32_t nInput = mmPara.n;
    uint32_t nL1SplitSize = mmPara.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        MatmulSplitK<T, O>(tensorResGm, tensorAGm, tensorBGm,
            mmPara, bufParam_, nL1 * nL1SplitSize, subNL1SplitSize);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MatmulAndSyncQcQr(AicOffset &aicOffset) {
    if constexpr (MLAPT::enableGroupComputeOpt) {
        // MatmulQc
        // 复用mmCqResGm_ workspace
        MatmulSplitN<rmsNormCqOutputType, mmQcQrOutputType, false, true>(
                mmQcQrResGm_[aicOffset.qCResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                weightUqQrGm_[aicOffset.weightUqOffset],
                mmQcQrParam_, UsedBlockParams{0, QC_CORE_NUM});
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QC);
        // MatmulQr
        MatmulSplitN<rmsNormCqOutputType, mmQcQrOutputType, false, true, false>(
                mmQcQrResGm_[aicOffset.qRResOffset],
                rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                weightUqQrGm_[aicOffset.weightQrOffset],
                mmQcQrParam_, UsedBlockParams{QC_CORE_NUM, QC_CORE_NUM + QR_CORE_NUM});
    } else {
        MatmulQcQrSyncDequant(aicOffset.rmsNormCqResOffset, aicOffset.weightUqQrOffset, aicOffset.qcQrResOffset);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MatmulQcQrSyncDequant(
        int64_t rmsNormCqResOffset, int64_t weightUqQrOffset, int64_t qcQrResOffset) {
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn
    //                           └──> Rope(Qr)
    // [32, 1536] * [1536, 32*(128+64)] = [32, 32*192]
    constexpr uint32_t mSize = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ?
        INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
    bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;

    uint32_t nInput = mmQcQrParam_.n;
    uint32_t nL1SplitSize = mmQcQrParam_.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;

    if (isAFullLoad) {
        LoadL1A(rmsNormCqResGm_[rmsNormCqResOffset], mmQcQrParam_.m, mmQcQrParam_.k,
            mmQcQrParam_.k, bufParam_);
        WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
    }

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        if (isAFullLoad) {
            MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, true>(
                mmQcQrResGm_[qcQrResOffset], rmsNormCqResGm_[rmsNormCqResOffset],
                weightUqQrGm_[weightUqQrOffset], mmQcQrParam_, bufParam_,
                nL1 * nL1SplitSize, subNL1SplitSize);
        } else {
            MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType>(
                mmQcQrResGm_[qcQrResOffset], rmsNormCqResGm_[rmsNormCqResOffset],
                weightUqQrGm_[weightUqQrOffset], mmQcQrParam_, bufParam_,
                nL1 * nL1SplitSize, subNL1SplitSize);
        }
        if constexpr (MLAPT::enableDequantOpt) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_N);
        }
    }

    if (isAFullLoad) {
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
        bufParam_.aL1BufIter++;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::PreloadQnAndSync(AicOffset &aicOffset, int64_t mmQnLoops) {
    MatmulQnWeightPreload(aicOffset.weightUkOffset, mmQnLoops);
    if constexpr (MLAPT::enableGroupComputeOpt) {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QR);
    } else if constexpr (MLAPT::enableDequantOpt) {
        // enableDequantOpt分支的CV同步由更细粒度的子函数控制
        return;
    } else {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QCQR);
    }

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC);
    } else {
        WaitAllCore<SYNC_MODE_ALL_CUBE, PIPE_FIX>(FINISH_MM_ALL);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MatmulQnWeightPreload(
        int64_t weightUkOffset, int64_t subLoopTimes) {
    if constexpr (MLAPT::enableGroupComputeOpt) {
        if (blockIdx_ >= QC_CORE_NUM) {
            return;
        }
    } else {
        if (blockIdx_ >= baseParams_->mm4BlockNum) {
            return;
        }
    }
    int64_t weightOffset = weightUkOffset;
    for (int32_t i = 0; i < subLoopTimes; ++i) {
        if (i < 2) { // preload double buffer
            LoadL1B<mmQnInputType, DataFormat::ND, false>(weightUkGm_[weightOffset],
                mmQnParam_.n, mmQnParam_.k, mmQnParam_.k, bufParam_);
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
            bufParam_.bL1BufIter++;
            weightOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
                static_cast<int64_t>(baseParams_->headSizeCkv);
        }
    }
    if (subLoopTimes == 1) {
        bufParam_.bL1BufIter--;
    }
}

template<typename MLAPT>
template<bool needQnDynamicQuant>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::MatmulQnSyncDynamicQuantAndMulQr(
        int64_t qcOffset, int64_t weightUkOffset,
        int64_t qnResOffset, int64_t subLoopTimes) {
    if constexpr (MLAPT::enableGroupComputeOpt) {
        if (blockIdx_ >= QC_CORE_NUM) {
            return;
        }
    } else {
        if (blockIdx_ >= baseParams_->mm4BlockNum) {
            return;
        }
    }
    // MatmulQcQr ──> MatmulQn ──> query_out
    // [32, 128] * [128, 512] = [32, 512]
    // [32, 2, 128] * [2, 128, 512] = [32, 2, 512]
    for (int64_t i = 0; i < subLoopTimes; i++) {
        if constexpr (MLAPT::enableDequantOpt) {
            CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC_SPLIT_N);
        }
        if (i < 2) {
            MatmulFullLoad<mmQnInputType, mmQnOutputType, true, true>(mmQnResGm_[qnResOffset],
                mmQcQrResDequantGm_[qcOffset],
                weightUkGm_[weightUkOffset], mmQnParam_, bufParam_);
        } else {
            MatmulFullLoad<mmQnInputType, mmQnOutputType, false, true>(mmQnResGm_[qnResOffset],
                mmQcQrResDequantGm_[qcOffset],
                weightUkGm_[weightUkOffset], mmQnParam_, bufParam_);
        }
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        } else {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        }
        weightUkOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
            static_cast<int64_t>(baseParams_->headSizeCkv);
        qnResOffset += baseParams_->headSizeCkv;

        if constexpr (needQnDynamicQuant) {
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
        }
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::CopyInSinCos(int64_t tokenIndex,
    int64_t curVecToken, int64_t batchOffset, int64_t curStepBatchSize) {
#if !MLA_PROLOG_V3_SFA1
    if (!baseParams_->enableRope) {
        return;
    }
#endif
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt) {
        // 如果是切N场景，mm3的每个C核都会做rope
        if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
            return;
        }
        // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
        // >> 1 是将curStepBatchSize分到每个vec核上；
#if MLA_PROLOG_V3_SFA1
        uint32_t subBlockIdx_ = blockIdx_ & 1u;
        int64_t offset = (curStepBatchSize >> 1) * subBlockIdx_ + batchOffset;
#else
        uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
        int64_t offset = (curStepBatchSize / cvRatio_) * subBlockIdx_ + batchOffset;
#endif
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_,
#if MLA_PROLOG_V3_SFA1
            ropeCosGm_, ropeSinGm_, offset, (curStepBatchSize + 1) >> 1,
#else
            ropeCosGm_, ropeSinGm_, offset, (curStepBatchSize + cvRatio_ - 1) / cvRatio_,
#endif
            shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    } else {
        if (blockIdx_ >= curVectorBlockNum_) {
            return;
        }
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_,
            ropeCosGm_, ropeSinGm_, tokenIndex, curVecToken,
            shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::CopyGlobalParams() {
    // dequantScaleWDq
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        DataCopyExtParams dequantCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDqLocal_, dequantScaleWDqGm_, dequantCopyParams, dequantPadParams);
    }

    // rmsnormGammaCq
    DataCopy(rmsnormGammaCqLocal_, rmsnormGammaCqGm_, baseParams_->headSizeCq);

    // rmsnormGammaCkv
    DataCopy(rmsnormGammaCkvLocal_, rmsnormGammaCkvGm_, baseParams_->headSizeCkv);

    // smoothScaleCq
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        if (enableSmoothScalesCq_) {
            DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq *
                sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
            DataCopyPad(smoothScaleCqLocal_, smoothScaleCqGm_, smoothCopyParams, smoothPadParams);
        }
    }

    // dequantScaleWDkvKr
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        DataCopyExtParams dequantCopyParams{1,
            static_cast<uint32_t>((baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
            sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDkvKrLocal_, dequantScaleWDkvkrGm_,
            dequantCopyParams, dequantPadParams);
    }

    // quantScaleCkv
    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value && !isPertile) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            DataCopyExtParams quantCopyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
            DataCopyPad(quantScaleCkvLocal_, quantScaleCkvGm_,
                quantCopyParams, quantPadParams);
        } else {
            DataCopy(quantScaleCkvLocal_, quantScaleCkvGm_, baseParams_->headSizeCkv);
        }
    }

    // quantScaleCkr
    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        DataCopyExtParams quantCopyParams{1, static_cast<uint32_t>(baseParams_->dimHeadRope *
            sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
        DataCopyPad(quantScaleCkrLocal_, quantScaleCkrGm_, quantCopyParams, quantPadParams);
    }
}


/**
 * @brief RmsNormCq流程，融合了dynamicquant
          内部所需空间约为 curVecToken(128) * 8*4 + 8*4 + (4*vectorRow_*baseParams_->headSizeCq + 8)*4 = 28.0625K
 */
template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RmsNormCq(int64_t tokenIndex,
    int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset,
    int64_t curVecToken, int64_t curBlockTokenOffset) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    int64_t stepTokenIndex = tokenIndex;
    LocalTensor<rmsNormCqOutputType> outputLocal = shareBuffer_.Get<rmsNormCqOutputType>();
    LocalTensor<float> dequantScaleQcQr = outputLocal[baseParams_->headSizeCq].template ReinterpretCast<float>();
    LocalTensor<float> dequantScaleXLocal = dequantScaleQcQr[curVecToken * FP32_BLOCK_ELEMENT_NUM];
    LocalTensor<uint8_t> shareTmpUb = dequantScaleXLocal[FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<uint8_t>();
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish

        // dequantScaleXGm_  [BS , 1] 每个每个token对应一个系数，此处扩展为一个DataBlock
        if constexpr (std::is_same<mmCqOutputType, int32_t>::value) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        uint64_t scaleOffset = curVecTokenIdx * FP32_BLOCK_ELEMENT_NUM;
        RmsNormParam rmsNormParams ={baseParams_->reciprocalCq, // reciprocal
            baseParams_->epsilonCq,  // epsilon
            (uint32_t)vectorRow_, // row
            baseParams_->headSizeCq,  // col
            baseParams_->qcQrScale, // scale
#if !MLA_PROLOG_V3_SFA1
            baseParams_->isQcQrScaleEnable // isScaleEnable
#endif
        };
        if constexpr (std::is_same<rmsNormCqOutputType, int8_t>::value) {
            RmsNormDynamicQuant<mmCqOutputType, rmsNormGammaType, float, rmsNormComputType>(
                                outputLocal, dequantScaleQcQr[scaleOffset],
                                mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_,
                                smoothScaleCqLocal_, dequantScaleWDqLocal_,
                                dequantScaleXLocal,  shareTmpUb,
                                rmsNormParams, enableSmoothScalesCq_);
        } else {
            RmsNormNormal<mmCqOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType>(
                outputLocal, mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_, dequantScaleWDqLocal_,
                dequantScaleXLocal, shareTmpUb, rmsNormParams);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
        DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, baseParams_->headSizeCq);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        rmsNormCqOffset += baseParams_->headSizeCq;
        rmsNormCqResOffset += baseParams_->headSizeCq;
        tokenIndex++;
    }
    Brcb(dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
        dequantScaleQcQr, curVecToken, {1, 1});
    if constexpr (MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(dequantTool_.deQuantScaleCqGm_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                 dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                 FP32_BLOCK_ELEMENT_NUM * curVecToken);
    }
    if (unlikely (baseParams_->queryNormFlag)) {
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            DataCopyPad(dequantScaleQNormGm_[stepTokenIndex], dequantScaleQcQr,
            {static_cast<uint16_t>(curVecToken), sizeof(dequantScaleQNormType), 0, 0});
        }
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::GetBatchTokenIndex(int64_t &batchTokenIndex,
    int64_t &batchIndex, int64_t &batchSeqSize, int64_t &batchIndexOffset, int64_t tokenIndex)
{
    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::DISABLED) {
        batchIndex = tokenIndex / baseParams_->seq1Size;
        batchTokenIndex = tokenIndex % baseParams_->seq1Size;
        batchSeqSize = baseParams_->seq1Size;
        batchIndexOffset = batchIndex * batchSeqSize;    // cacheIndex的offset
    }
    // Advance batch cursor until the token index falls in [prevPrefix, curPrefix)
    // curPrefix is an exclusive upper bound for the current batch
    else {
        while (actSeqState_.curBatch + 1 < static_cast<int64_t>(baseParams_->batchSize) &&
            tokenIndex >= actSeqState_.curPrefix) {
            actSeqState_.curBatch += 1;
            actSeqState_.prevPrefix = actSeqState_.curPrefix;
            actSeqState_.curPrefix = actualSeqLenGm_(actSeqState_.curBatch);
            actSeqState_.prevIndexOffset = actSeqState_.curIndexOffset;
            actSeqState_.curIndexOffset += CeilDivT(actSeqState_.curPrefix -
                actSeqState_.prevPrefix, static_cast<int64_t>(baseParams_->blockSize));
        }
        batchIndex = actSeqState_.curBatch;
        batchTokenIndex = tokenIndex - actSeqState_.prevPrefix;
        batchSeqSize = actSeqState_.curPrefix - actSeqState_.prevPrefix;
        batchIndexOffset =  actSeqState_.prevIndexOffset;
    }
}


template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::GetCacheOffset(GlobalTensor<int64_t> indexGm,
                                                                    int64_t batchIndex,
                                                                    int64_t batchTokenIndex,
                                                                    int64_t batchSeqSize,
                                                                    int64_t batchIndexOffset,
                                                                    int64_t rows,
                                                                    int64_t headSize,
                                                                    int64_t &rowsInCurBatch,
                                                                    int64_t &cacheOffset,
                                                                    int64_t &nextBatchOffset)
{
    int64_t indexOffset = batchIndexOffset + batchTokenIndex / baseParams_->blockSize;
    int64_t paBlkId = indexGm(indexOffset); // 取cacheIdx
    int64_t tokenOffsetInCurrentPage = batchTokenIndex % baseParams_->blockSize;
    cacheOffset = paBlkId * baseParams_->blockSize * headSize + tokenOffsetInCurrentPage * headSize;
    int64_t leftRowsInCurBatch = batchSeqSize - batchTokenIndex;
    if (leftRowsInCurBatch >= rows) {
        rowsInCurBatch = rows;
    } else {
        rowsInCurBatch = rows - leftRowsInCurBatch;
        indexOffset++;
        paBlkId = indexGm(indexOffset);
        nextBatchOffset = paBlkId * headSize;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RmsNormAndScatterCkv(LocalTensor<float>& dequantScaleXLocal,
                                                                        LocalTensor<uint8_t>& shareTmpUb,
                                                                        LocalTensor<ropeComputType>& cosLocalCkvKr,
                                                                        LocalTensor<ropeComputType>& sinLocalCkvKr,
                                                                        CkvkrParams rmsNormAndScatterCkvParams) {
    // MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv)
    LocalTensor<kvCacheType> outputLocal = shareTmpUb.ReinterpretCast<kvCacheType>();
    int32_t outputLocalSize = vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType);
    if (baseParams_->kvQuantMode == 3) {    // pertile量化场景，按照concat的最长长度申请内存
        int32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
        outputLocalSize = vectorRow_ * (baseParams_->headSizeCkv * sizeof(kvCacheType) + tileNum * sizeof(float));
        outputLocalSize = Align(outputLocalSize, static_cast<int32_t>(BYTE_BLOCK));
    }
    LocalTensor<uint8_t> rmsNormShareTmpUb = shareTmpUb[outputLocalSize].template ReinterpretCast<uint8_t>();

    RmsNormParam rmsNormParams = {
        baseParams_->reciprocalCkv, // reciprocal
        baseParams_->epsilonCkv,  // epsilon
        (uint32_t)vectorRow_, // row
        baseParams_->headSizeCkv, // col
        baseParams_->kcScale, // scale
#if !MLA_PROLOG_V3_SFA1
        baseParams_->isKcScaleEnable // isScaleEnable
#endif
    };

    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        // row = vectorRow_ = 1     col = baseParams_->headSizeCkv
        LocalTensor<float> tmpOut = rmsNormShareTmpUb.ReinterpretCast<float>();
        LocalTensor<uint8_t> sharedBuf = tmpOut[vectorRow_ *
            baseParams_->headSizeCkv].template ReinterpretCast<uint8_t>();
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, float>(
            tmpOut, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_,
            dequantScaleXLocal, sharedBuf, rmsNormParams);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        float kNopeClipAlpha = kNopeClipAlphaGm_.GetValue(0);

        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            if (baseParams_->kvQuantMode == 3) {
                PerTileQuantParams perTileQuantParams = {
                    static_cast<int32_t>(baseParams_->tileSize),                                // tileSize
                    static_cast<int32_t>(baseParams_->headSizeCkv / baseParams_->tileSize),     // tileNum
                    kNopeClipAlpha,         // alpha
                    static_cast<uint32_t>(vectorRow_),   // row
                    baseParams_->headSizeCkv             // col
                };
                QuantPerTile(outputLocal, tmpOut, sharedBuf, perTileQuantParams);
            } else {
                Rectangle rectangleParams {
                    (uint32_t)vectorRow_,    // row
                    (uint32_t)baseParams_->headSizeCkv, // col
                    (uint32_t)baseParams_->headSizeCkv // columnStride
                };
                QuantPerTensor(outputLocal, tmpOut, quantScaleCkvLocal_, sharedBuf, rectangleParams);
                pipe_barrier(PIPE_V);
            }
        } else {
            if (baseParams_->kvQuantMode == 3) {
                PerTileQuantParams perTileQuantParams = {
                    static_cast<int32_t>(baseParams_->tileSize),                                // tileSize
                    static_cast<int32_t>(baseParams_->headSizeCkv / baseParams_->tileSize),     // tileNum
                    kNopeClipAlpha,         // alpha
                    static_cast<uint32_t>(vectorRow_),   // row
                    baseParams_->headSizeCkv             // col
                };
                QuantPerTile(outputLocal, tmpOut, sharedBuf, perTileQuantParams);
            } else {
                Rectangle rectangleParams {
                    (uint32_t)vectorRow_,    // row
                    (uint32_t)baseParams_->headSizeCkv, // col
                    (uint32_t)baseParams_->headSizeCkv // columnStride
                };
                QuantPerChannel(outputLocal, tmpOut, quantScaleCkvLocal_, sharedBuf, rectangleParams);
            }
        }

    } else {
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType>(
            outputLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset],
            rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_,
            dequantScaleXLocal, rmsNormShareTmpUb, rmsNormParams);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#if MLA_PROLOG_V3_SFA1
    // cur_k_nope 输出：写 cache 的同一份数据复制到独立输出 curKNopeGm_
    // 行号 = 全局 tokenIndex；行 stride = Dtile * sizeof(kvCacheType)（字节视图）
    // 源行偏移：ckv 数据在 outputLocal 基址（既有 ScatterCache 同定位）
    if (baseParams_->outputCurKNopeRopeFlag) {
        LocalTensor<uint8_t> outputLocalByte = outputLocal.template ReinterpretCast<uint8_t>();
        if (baseParams_->kvQuantMode == 3) {
            // per-tile（tiling_check 强制 ckvkrRepoMode=1 + quantScaleRepoMode=1 → Dtile=656）：
            // 行 stride=656B 非 32B 对齐 → DataCopyPad（dst GM 支持非对齐；UB 源 32B 对齐 ✓）
            // nope 段（Hckv*sizeof(kvCacheType) = 512B，UB 源偏移 0 对齐）
            DataCopyPad(curKNopeGm_[rmsNormAndScatterCkvParams.tokenIndex * baseParams_->dtileSize],
                        outputLocalByte,
                        DataCopyExtParams{1, static_cast<uint32_t>(baseParams_->headSizeCkv * sizeof(kvCacheType)), 0, 0, 0});
            // scale 段（quantScaleRepoMode==1 恒与 per-tile 并存）：数据源 outputLocal[headSizeCkv]
            // （与既有 ScatterCacheUnAligned 同源）；行内偏移 = Hckv*1 + Dr*2 = 640B
            int32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
            LocalTensor<uint8_t> quantScaleCkvByte = outputLocal[vectorRow_ *
                baseParams_->headSizeCkv].template ReinterpretCast<uint8_t>();
            DataCopyPad(curKNopeGm_[rmsNormAndScatterCkvParams.tokenIndex * baseParams_->dtileSize +
                                     baseParams_->headSizeCkv * sizeof(kvCacheType) +
                                     baseParams_->headSizeKr * sizeof(krCacheType)],
                        quantScaleCkvByte,
                        DataCopyExtParams{1, static_cast<uint32_t>(tileNum * sizeof(float)), 0, 0, 0});
        } else {
            // 非 per-tile：Dtile=Hckv（DIVIDE+DIVIDE 强制），行 stride 32B 对齐 → DataCopy
            // 仅 k_nope 段（Hckv 元素，字节 = Hckv*sizeof(kvCacheType)）
            DataCopy(curKNopeGm_[rmsNormAndScatterCkvParams.tokenIndex * baseParams_->dtileSize *
                                  sizeof(kvCacheType)],
                     outputLocalByte,
                     baseParams_->headSizeCkv * sizeof(kvCacheType));
        }
    }
#endif
    // Scatter(Ckv)
    // RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) ||
        (MLAPT::cacheMode == CACHE_MODE::PA_BSND) || (MLAPT::cacheMode == CACHE_MODE::ND)) {
        int64_t paTokenIndex = cacheIndexGm_(rmsNormAndScatterCkvParams.tokenIndex);
        if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
            paTokenIndex = rmsNormAndScatterCkvParams.tokenIndex;
        }
        ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
                baseParams_->headSizeCkv, baseParams_->dtileSize});
        // 刷新量化scale
        if (baseParams_->kvQuantMode == 3 && baseParams_->quantScaleRepoMode == 1) {
            // BSND:
            int32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
            LocalTensor<kvCacheType> quantScaleCkvInt8Tensor = outputLocal[vectorRow_ *
                baseParams_->headSizeCkv].template ReinterpretCast<kvCacheType>();
            int64_t startOffset = 0;
            int64_t startColOffset = baseParams_->headSizeCkv;
            if (baseParams_->ckvkrRepoMode == 1) {
                startColOffset += baseParams_->headSizeKr * sizeof(krCacheType);
            }
            if constexpr ((MLAPT::cacheMode != CACHE_MODE::PA_NZ)) {
                startOffset = startColOffset;
            }

            ScatterCacheUnAligned<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[startOffset], quantScaleCkvInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
                static_cast<int64_t>(tileNum * sizeof(float)), baseParams_->dtileSize});
        }
    } else {
        int64_t batchTokenIndex = 0;
        int64_t batchIndex = 0;
        int64_t batchSeqSize = 0;
        int64_t batchIndexOffset = 0;
        int64_t rowsInCurBatch = 0;
        int64_t cacheOffset = 0;
        int64_t nextBatchOffset = 0;
        GetBatchTokenIndex(batchTokenIndex,
                           batchIndex,
                           batchSeqSize,
                           batchIndexOffset,
                           rmsNormAndScatterCkvParams.tokenIndex);
        GetCacheOffset(cacheIndexGm_,
                       batchIndex,
                       batchTokenIndex,
                       batchSeqSize,
                       batchIndexOffset,
                       vectorRow_,
                       baseParams_->headSizeCkv,
                       rowsInCurBatch,
                       cacheOffset, nextBatchOffset);

        ScatterCacheMultiRows<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, cacheOffset, vectorRow_, baseParams_->headSizeCkv,
                               baseParams_->seq1Size, rmsNormAndScatterCkvParams.tokenIndex},
                               rowsInCurBatch, cacheOffset, nextBatchOffset);
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RopeAndScatterKr(
                                                            LocalTensor<float>& dequantScaleXLocal,
                                                            LocalTensor<uint8_t>& shareTmpUb,
                                                            LocalTensor<ropeComputType>& cosLocalCkvKr,
                                                            LocalTensor<ropeComputType>& sinLocalCkvKr,
                                                            CkvkrParams ropeAndScatterKrParams) {
    // MatmulCkvKr ──> Rope(Ckv) ──> Scatter(Kr)
    LocalTensor<krCacheType> outputKrLocal = shareTmpUb.ReinterpretCast<krCacheType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputKrLocal[baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();
    int64_t stride = baseParams_->headSizeCkv + baseParams_->headSizeKr; // 512 + 64

    LocalTensor<ropeComputType> cosLocal;
    LocalTensor<ropeComputType> sinLocal;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal = cosLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    } else {
        cosLocal = cosLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    }
    Rectangle ropeParams{
        (uint32_t)vectorRow_, // row
        (uint32_t)baseParams_->dimHeadRope, // col
        (uint32_t)stride  // stride
    };
#if MLA_PROLOG_V3_SFA1
    if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value && std::is_same<krCacheType, bfloat16_t>::value) {
#else
    if (!baseParams_->enableRope) {
        if constexpr (std::is_same<mmCkvKrOutputType, krCacheType>::value) {
            DataCopyExtParams copyParams {
                static_cast<uint16_t>(vectorRow_),
                static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(krCacheType)),
                static_cast<uint32_t>((stride - baseParams_->dimHeadRope) * sizeof(mmCkvKrOutputType)),
                0,
                0
            };
            DataCopyPadExtParams<krCacheType> padParams {false, 0, 0, 0};
            DataCopyPad(outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], copyParams, padParams);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        }
    } else if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value &&
                         std::is_same<krCacheType, bfloat16_t>::value) {
#endif
        LocalTensor<uint8_t> sharedBuf = ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope *
            sizeof(ropeSinCosType)];
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType>(
            outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal,
            sharedBuf, ropeParams,
            dequantScaleWDkvKrLocal_[baseParams_->headSizeCkv], dequantScaleXLocal);
    } else if constexpr (std::is_same<krCacheType, int8_t>::value) {
        LocalTensor<ropeSinCosType> tmpOut = ropeShareTmpUb.ReinterpretCast<ropeSinCosType>();
        LocalTensor<uint8_t> sharedBuf = ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope *
            sizeof(ropeSinCosType)];
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, ropeSinCosType>(
            tmpOut, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal,
            sharedBuf, ropeParams);
        RopePostQuantPerChannel(outputKrLocal, tmpOut, quantScaleCkrLocal_, sharedBuf,
            vectorRow_ * baseParams_->dimHeadRope);
    } else {
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType>(outputKrLocal,
            mmCkvKrResGm_[ropeAndScatterKrParams.offset],
            cosLocal, sinLocal, ropeShareTmpUb, ropeParams);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#if MLA_PROLOG_V3_SFA1
    if (baseParams_->outputCurKNopeRopeFlag) {
        LocalTensor<uint8_t> outputKrLocalByte = outputKrLocal.template ReinterpretCast<uint8_t>();
        // 全部场景 kr 行 stride（64B int8 / 128B bf16）32B 对齐 → DataCopy
        DataCopy(curKRopeGm_[ropeAndScatterKrParams.tokenIndex * baseParams_->dimHeadRope * sizeof(krCacheType)],
                 outputKrLocalByte,
                 baseParams_->dimHeadRope * sizeof(krCacheType));
        if (baseParams_->ckvkrRepoMode == 1) {
            // per-tile 强制 COMBINE：kr 段并入 cur_k_nope 行（行起址非 32B 对齐 → DataCopyPad）
            // 行内偏移 = Hckv*sizeof(kvCacheType)（与既有 ScatterCache 合并段偏移一致）
            DataCopyPad(curKNopeGm_[ropeAndScatterKrParams.tokenIndex * baseParams_->dtileSize +
                                     baseParams_->headSizeCkv * sizeof(kvCacheType)],
                        outputKrLocalByte,
                        DataCopyExtParams{1, static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(krCacheType)), 0, 0, 0});
        }
    }
#endif
    // scatter(Kr)
    // Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
    int64_t paTokenIndex;
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) ||
        (MLAPT::cacheMode == CACHE_MODE::PA_BSND) || (MLAPT::cacheMode == CACHE_MODE::ND)) {
        paTokenIndex = cacheIndexGm_(ropeAndScatterKrParams.tokenIndex);
        if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
            paTokenIndex = ropeAndScatterKrParams.tokenIndex;
        }
#if MLA_PROLOG_V3_SFA1
        if (baseParams_->kvQuantMode == 3 && baseParams_->ckvkrRepoMode == 1) {
#else
        if (baseParams_->ckvkrRepoMode == 1) {
#endif
            int64_t startOffset;
            if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ)) {
                constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(kvCacheType);
                startOffset = CeilDiv(baseParams_->headSizeCkv, col0) * col0 * baseParams_->blockSize; // 列方向的偏移
            } else {
                startOffset = baseParams_->headSizeCkv;
            }
            LocalTensor<kvCacheType> outputKrInt8Tensor = outputKrLocal.template ReinterpretCast<kvCacheType>();
            ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(kvCacheGm_[startOffset],
                outputKrInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
#if MLA_PROLOG_V3_SFA1
                    static_cast<int64_t>(baseParams_->dimHeadRope * sizeof(krCacheType)), baseParams_->dtileSize});
#else
                    static_cast<int64_t>(baseParams_->dimHeadRope * sizeof(krCacheType) / sizeof(kvCacheType)),
                    baseParams_->dtileSize});
#endif
        } else {
            ScatterCache<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(krCacheGm_, outputKrLocal,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex,
                    vectorRow_, baseParams_->dimHeadRope, baseParams_->dimHeadRope});
        }
    } else {
        int64_t batchTokenIndex = 0;
        int64_t batchIndex = 0;
        int64_t batchSeqSize = 0;
        int64_t batchIndexOffset = 0;
        int64_t rowsInCurBatch = 0;
        int64_t cacheOffset = 0;
        int64_t nextBatchOffset = 0;
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RmsNormRopeScatterCkvKr(int64_t tokenIndex,
                                                                           int64_t rmsNormCkvOffset,
                                                                           int64_t ropeKrOffset,
                                                                           int64_t curVecToken) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    LocalTensor<float> dequantScaleXLocal = shareBuffer_.Get<float>();
    LocalTensor<ropeComputType> cosLocalCkvKr = dequantScaleXLocal[FP32_BLOCK_ELEMENT_NUM].
        template ReinterpretCast<ropeComputType>();
    LocalTensor<ropeComputType> sinLocalCkvKr = cosLocalCkvKr[baseParams_->dimHeadRope * curVecToken];
    LocalTensor<uint8_t> shareTmpUb = sinLocalCkvKr[baseParams_->dimHeadRope *
        curVecToken].template ReinterpretCast<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt) {
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocalCkvKr, sinLocalCkvKr,
                    ropeCosGm_, ropeSinGm_, tokenIndex, curVecToken,
                    shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
    if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return;
    }

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        if constexpr (std::is_same<mmCqOutputType, int32_t>::value) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        CkvkrParams rmsNormAndScatterCkvParams {
            tokenIndex,
            rmsNormCkvOffset,
            curVecTokenIdx
        };

        RmsNormAndScatterCkv(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, rmsNormAndScatterCkvParams);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        CkvkrParams ropeAndScatterKrParams {
            tokenIndex,
            ropeKrOffset,
            curVecTokenIdx
        };
        RopeAndScatterKr(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, ropeAndScatterKrParams);

        tokenIndex += 1;
        rmsNormCkvOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
        ropeKrOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset,
    int64_t curVecToken, int64_t curBlockTokenOffset)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    uint64_t stride = baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc;

    LocalTensor<uint8_t> ropeShareTmpUb;
    LocalTensor<ropeOutputType> outputLocal;
    LocalTensor<float> channelDeqScaleLocal = shareBuffer_.Get<float>();

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        uint64_t row = baseParams_->numHeadSize;
        uint64_t col = baseParams_->dimHeadRope;
        DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)),
                        static_cast<uint32_t>((stride - col) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[baseParams_->dimHeadSizeQc], copyParams, padParams); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        outputLocal = channelDeqScaleLocal [row * col].template ReinterpretCast<ropeOutputType>();
        ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();
    } else {
        outputLocal = shareBuffer_.Get<ropeOutputType>();
        ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();
    }

    Rectangle ropeParams{
        (uint32_t)baseParams_->numHeadSize,  // row
        (uint32_t)baseParams_->dimHeadRope,  // col
        (uint32_t)stride                     // stride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr ──> Rope(Qr) ──> query_rope_out

#if MLA_PROLOG_V3_SFA1
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
#else
        if (!baseParams_->enableRope) {
            if constexpr (std::is_same<mmQcQrOutputType, ropeOutputType>::value) {
                DataCopyExtParams copyParams {
                    static_cast<uint16_t>(baseParams_->numHeadSize),
                    static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(ropeOutputType)),
                    static_cast<uint32_t>((stride - baseParams_->dimHeadRope) * sizeof(mmQcQrOutputType)),
                    0,
                    0
                };
                DataCopyPadExtParams<ropeOutputType> padParams {false, 0, 0, 0};
                DataCopyPad(outputLocal, mmQcQrResGm_[ropeQrOffset], copyParams, padParams);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            }
        } else if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
#endif
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocal,
                mmQcQrResGm_[ropeQrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams,
                channelDeqScaleLocal,
                dequantTool_.deQuantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM]);
        } else {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocal,
                mmQcQrResGm_[ropeQrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams);
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, baseParams_->headSizeQr);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->numHeadSize) *
            static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

// 用于enableGroupComputeOpt场景 BS = 1 ， G = 8
template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RopeQrSplitNGroupCase(int64_t ropeQrOffset,
    int64_t ropeQrResOffset)
{
#if MLA_PROLOG_V3_SFA1
    if (blockIdx_ < QC_CORE_NUM * 2 || blockIdx_ >= (QC_CORE_NUM + QR_CORE_NUM) * 2) {
#else
    if (blockIdx_ < QC_CORE_NUM * cvRatio_ || blockIdx_ >= (QC_CORE_NUM + QR_CORE_NUM) * cvRatio_) {
#endif
        return;
    }

    uint32_t row = baseParams_->numHeadSize / QC_CORE_NUM;
    uint32_t col = baseParams_->dimHeadRope;
    int64_t stride = col;
    int64_t strideScale = baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc;
    int64_t deqScaleOffset = baseParams_->dimHeadSizeQc +
#if MLA_PROLOG_V3_SFA1
        row * (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * (blockIdx_ - QC_CORE_NUM * 2);
#else
        row * (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * (blockIdx_ - QC_CORE_NUM * cvRatio_);
#endif
    LocalTensor<float> channelDeqScaleLocal = shareBuffer_.Get<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)),
                    static_cast<uint32_t>((strideScale - col) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[deqScaleOffset], copyParams, padParams); // 复用内存
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    LocalTensor<ropeOutputType> outputLocal =
        channelDeqScaleLocal[row * col].template ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();

    Rectangle ropeParams{
        (uint32_t)col,    // row
        (uint32_t)col,    // col
        (uint32_t)stride  // stride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVectorBlockNum_; curVecTokenIdx++) {

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        // sharetmp共用，待V算完再执行MTE2搬运规避风险
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_,
            curVecTokenIdx * baseParams_->dimHeadRope, 1, ropeShareTmpUb, vectorRow_, baseParams_->dimHeadRope);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocal, mmQcQrResGm_[ropeQrOffset],
            cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams, channelDeqScaleLocal, dequantTool_.deQuantScaleCqLocal_);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, row * col);

                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->numHeadSize) * baseParams_->dimHeadRope;
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::RopeQrSplitN(const RopeQrSplitNParams& ropeQrSplitNParams) {
#if MLA_PROLOG_V3_SFA1
    uint32_t subBlockIdx_ = blockIdx_ & 1;
    uint32_t ropeCnt = (mmQcQrParam_.m + subBlockIdx_) >> 1;
#else
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
    uint32_t ropeCnt = (mmQcQrParam_.m + subBlockIdx_) / cvRatio_;
#endif
    uint32_t colQr = baseParams_->dimHeadRope;
#if MLA_PROLOG_V3_SFA1
    uint32_t ropeCntDown= mmQcQrParam_.m >> 1; // 向下取整，处理奇数核的偏移
#else
    uint32_t ropeCntDown= mmQcQrParam_.m / cvRatio_; // 向下取整，处理奇数核的偏移
#endif
    uint32_t deQuantScaleCqOffset = ropeCntDown * subBlockIdx_ * FP32_BLOCK_ELEMENT_NUM;

    DataCopyParams outputRopeParams {
    static_cast<uint16_t>(ropeCnt),
    static_cast<uint16_t>(colQr * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE),
    0,
    static_cast<uint16_t>(ropeQrSplitNParams.ropeDstStride * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE)};

    Rectangle ropeParams{
        ropeCnt,                                 // row
        colQr,                                   // col
        (uint32_t)ropeQrSplitNParams.ropeStride  // stride
    };

    GlobalTensor<mmQcQrOutputType> inputGmRope = mmQcQrResGm_[ropeQrSplitNParams.ropeQrOffset];
    GlobalTensor<float> deqScaleRope = deqScaleQcQrW_[ropeQrSplitNParams.ropeQrOffset];
    GlobalTensor<ropeOutputType> outputGmRope = qrOutGm_[ropeQrSplitNParams.ropeQrResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb.ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocalRope[ropeCnt * colQr].template ReinterpretCast<uint8_t>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocalRope,
        inputGmRope[ropeQrSplitNParams.inputOffsetRope], cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams,
        ropeQrSplitNParams.ropeStride,  deqScaleRope[ropeQrSplitNParams.deqScaleOffset],
        dequantTool_.deQuantScaleCqLocal_[deQuantScaleCqOffset]);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

    DataCopy(outputGmRope[ropeQrSplitNParams.outputOffsetRope], outputLocalRope, outputRopeParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::DequantQcQrSplitN(const DequantQcQrSplitNParams& dequantQcQrSplitN)
{
    uint32_t row = mmQcQrParam_.m;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
#if MLA_PROLOG_V3_SFA1
    uint32_t colQcSingle = colQc >> 1;
#else
    uint32_t colQcSingle = colQc / cvRatio_;
#endif
    uint32_t count = row * colQcSingle;

    DataCopyParams inputCopyParams {
    static_cast<uint16_t>(row),
    static_cast<uint16_t>(colQcSingle * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
    static_cast<uint16_t>(dequantQcQrSplitN.srcStride * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
    0};

    DataCopyParams outputCopyParams {
    static_cast<uint16_t>(row),
    static_cast<uint16_t>(colQcSingle * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE),
    0,
    static_cast<uint16_t>(dequantQcQrSplitN.dstStride * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE)};


    GlobalTensor<mmQcQrOutputType> inputGm = mmQcQrResGm_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<float> scale1Gm = deqScaleQcQrW_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[dequantQcQrSplitN.mmQnPreDequantResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<float> scale2Local = dequantTool_.deQuantScaleCqLocal_;
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    // 可以和inputLocal共享内存地址，减少UB使用
    LocalTensor<float> computeLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> scaleLocal =  inputLocal[count + FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<float>();
    // outputLocal比scaleLocal占用UB少，且不会同时使用，故可以复用UB内存
    LocalTensor<mmQnInputType> outputLocal = scaleLocal.template ReinterpretCast<mmQnInputType>();

    Rectangle dequantParams{
        row, //  row
        colQcSingle, // col
        colQcSingle  // columnStride
    };
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    DataCopy(inputLocal, inputGm[dequantQcQrSplitN.inputOffset], inputCopyParams);
#if MLA_PROLOG_V3_SFA1
    DataCopy(scaleLocal, scale1Gm[dequantQcQrSplitN.inputOffset], colQc >> 1);
#else
    DataCopy(scaleLocal, scale1Gm[dequantQcQrSplitN.inputOffset], colQc / cvRatio_);
#endif
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // cast
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Dequant(computeLocal, inputLocal, scaleLocal, scale2Local, dequantParams);
    PipeBarrier<PIPE_V>();
    // cast
    Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    // copy out
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(outputGm[dequantQcQrSplitN.outputOffset], outputLocal, outputCopyParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::DequantAndRopeSplitNSyncMMQcQr(
    int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t ropeQrOffset, int64_t ropeQrResOffset)
{
    if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // mmQcQr一个C核算stepBatchSize * singleN，每两个V核做对应C核输出的Qc部分的dequant
    // singleN一定整除(dimHeadSizeQc + dimHeadSizeQc)，保证dequant能找到dimHeadSizeQc部分的起始位置
#if MLA_PROLOG_V3_SFA1
    uint32_t subBlockIdx_ = blockIdx_ & 1;
#else
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
#endif
    uint32_t colCube = mmQcQrParam_.baseN;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t colQr = baseParams_->dimHeadRope;
    uint32_t oriCol = mmQcQrParam_.n;
    uint32_t colOffsetCube = 0;

    // DequantSplitN参数
#if MLA_PROLOG_V3_SFA1
    uint32_t srcStride = baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->dimHeadSizeQc / 2;
    uint32_t dstStride = baseParams_->headSizeQc - baseParams_->dimHeadSizeQc / 2;
    uint32_t colQcSingle = colQc >> 1;
#else
    uint32_t srcStride = baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->dimHeadSizeQc / cvRatio_;
    uint32_t dstStride = baseParams_->headSizeQc - baseParams_->dimHeadSizeQc / cvRatio_;
    uint32_t colQcSingle = colQc / cvRatio_;
#endif
    uint32_t colOffsetVec = 0;
    uint32_t inputOffset = subBlockIdx_ * colQcSingle;
    uint32_t outputOffset = inputOffset;

    // RopeQrSplitN参数
    uint32_t ropeDstStride = baseParams_->headSizeQr - colQr;
#if MLA_PROLOG_V3_SFA1
    uint32_t ropeCntDown= mmQcQrParam_.m >> 1; // 向下取整，处理奇数核的偏移
#else
    uint32_t ropeCntDown= mmQcQrParam_.m / cvRatio_; // 向下取整，处理奇数核的偏移
#endif
    int64_t ropeStride = static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(colQc + colQr);
    uint32_t inputOffsetRope = ropeCntDown * subBlockIdx_ * ropeStride;
    uint32_t outputOffsetRope = ropeCntDown * subBlockIdx_ * baseParams_->headSizeQr;
    uint32_t deqScaleOffset =0;
    uint32_t colOffsetRope = 0;

    // cube一次处理row*colCube，对应的两个vec一次处理row*colQc，两vec之间切colQc
    // 等cube生产足够数据了以后，vec开始消费
    while (colOffsetCube < oriCol) {    // 循环CeilDiv(oriCol, colCube)次
        colOffsetCube += colCube;
        if (colOffsetCube > oriCol) {   // 当oriCol不被colCube整除时，mm最后一个base块需要刷新col end
            colOffsetCube = oriCol;
        }
        CrossCoreWaitFlag(FINISH_MM_QCQR_SPLIT_N);
        // DequantSplitN
        while (colOffsetVec + colQc <= colOffsetCube) {   // 循环singleNumHeadSize次
            DequantQcQrSplitN(DequantQcQrSplitNParams{mmQnPreDequantOffset, mmQnPreDequantResOffset,
                                inputOffset, outputOffset, srcStride, dstStride});
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
            colOffsetVec += (colQc + colQr);
            inputOffset += (colQc + colQr);
            outputOffset += colQc;
        }
        // RopeQrSplitN
        while ((colOffsetRope + colQc + colQr) <= colOffsetCube) {
            RopeQrSplitN(RopeQrSplitNParams{ropeQrOffset, ropeQrResOffset,
                        inputOffsetRope, deqScaleOffset, outputOffsetRope, ropeStride, ropeDstStride});
            colOffsetRope += (colQc + colQr);
            inputOffsetRope += (colQc + colQr);
            deqScaleOffset += (colQc + colQr);
            outputOffsetRope += colQr;
        }
    }
}

// 用于算力切分dequant切N场景
template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset,
                                                                            int64_t mmQnPreDequantResOffset,
                                                                            int64_t qcQrScaleOffset)
{
#if MLA_PROLOG_V3_SFA1
    if (blockIdx_ >= QC_CORE_NUM * 2) {
#else
    if (blockIdx_ >= QC_CORE_NUM * cvRatio_) {
#endif
        return;
    }
#if MLA_PROLOG_V3_SFA1
    uint32_t subBlockIdx_ = blockIdx_ & 1;
#else
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
#endif
    uint32_t oriCol = (baseParams_->headSizeQc) / QC_CORE_NUM;
#if MLA_PROLOG_V3_SFA1
    uint32_t curCol = (baseParams_->headSizeQc) / (QC_CORE_NUM * 2);
#else
    uint32_t curCol = (baseParams_->headSizeQc) / (QC_CORE_NUM * cvRatio_);
#endif
    uint32_t srcStride = baseParams_->headSizeQc - curCol;
    uint32_t dstStride = baseParams_->headSizeQc - curCol;
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    Rectangle rectangleParams {
        (uint32_t)mmQcQrParam_.m,          // row
        (uint32_t)curCol,                  // col
        (uint32_t)baseParams_->headSizeQc  // columnStride
    };
    DequantSplitNQc(mmQcQrResDequantGm_[mmQnPreDequantResOffset], mmQcQrResGm_[mmQnPreDequantOffset],
                deqScaleQcQrW_[qcQrScaleOffset], dequantTool_.deQuantScaleCqLocal_, shareTmpUb,
                rectangleParams, oriCol, dstStride, subBlockIdx_);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::DequantQc(int64_t mmQnPreDequantOffset,
                                                                      int64_t mmQnPreDequantResOffset,
                                                                      int64_t curVecToken,
                                                                      int64_t curBlockTokenOffset)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }

    Rectangle rectangleParams {
        (uint32_t)baseParams_->stepNumHeadDequant,                       // row
        (uint32_t)baseParams_->dimHeadSizeQc,                            // col
        (uint32_t)baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc  // columnStride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        DequantPerTokenQc(mmQcQrResDequantGm_[mmQnPreDequantResOffset],
            mmQcQrResGm_[mmQnPreDequantOffset], deqScaleQcQrW_,
            dequantTool_.deQuantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM],
            shareTmpUb, rectangleParams, baseParams_->numHeadSize);
        mmQnPreDequantOffset += baseParams_->headSizeQc + baseParams_->headSizeQr;
        mmQnPreDequantResOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitN<MLAPT>::DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset,
                                                int64_t curStepBatchSize, int64_t numHeadOffset, int64_t mmQnLoops)
{
    // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
    // >> 1 是将curStepBatchSize分到每个vec核上；
#if MLA_PROLOG_V3_SFA1
    int64_t curStepBatchSizeVec = (curStepBatchSize + (blockIdx_ & 1u)) >> 1;
    if (blockIdx_ >= baseParams_->mm4BlockNum * 2 || curStepBatchSizeVec == 0) {
#else
    int64_t curStepBatchSizeVec = (curStepBatchSize + (blockIdx_ % cvRatio_)) / cvRatio_;
    if (blockIdx_ >= baseParams_->mm4BlockNum * cvRatio_ || curStepBatchSizeVec == 0) {
#endif
        return;
    }

    // 等待前面的Qr部分完成
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY = EVENT_ID0;
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;

#if MLA_PROLOG_V3_SFA1
    int64_t blockBatchOffset = (blockIdx_ & 1u) * (curStepBatchSize >> 1);
#else
    int64_t blockBatchOffset = (blockIdx_ % cvRatio_) * (curStepBatchSize / cvRatio_);
#endif
    int64_t totalSizeCkv = static_cast<int64_t>(baseParams_->numHeadSize) *
        static_cast<int64_t>(baseParams_->headSizeCkv);
    // 由于两个vec核分curStepBatchSize，各处理curStepBatchSize/2，blockIdx_ & 1 表示是否为第二个vec核
    int64_t dynamicQuantQueryOffset = blockBatchOffset * totalSizeCkv +
        numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t dynamicQuantQueryResOffset = batchOffset * totalSizeCkv + blockBatchOffset * totalSizeCkv +
        numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t scaleQueryNopeOffset = batchOffset * static_cast<int64_t>(baseParams_->numHeadSize) +
        blockBatchOffset * static_cast<int64_t>(baseParams_->numHeadSize) + numHeadOffset;
    int64_t queryOutStride = totalSizeCkv;
    int64_t qrOutputStride = baseParams_->numHeadSize * baseParams_->dimHeadRope;
    int64_t qrPostProcessResOffset = batchOffset * static_cast<int64_t>(baseParams_->headSizeQr) +
        numHeadOffset * static_cast<int64_t>(baseParams_->dimHeadRope) +
        blockBatchOffset * static_cast<int64_t>(baseParams_->headSizeQr);


    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();

    float quantScaleCkv = quantScaleCkvGm_.GetValue(0);

    // Dynamic Quant
    SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
    SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);

    // Rope Post Process
    SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);

    // per-head循环
    for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
        CrossCoreWaitFlag(FINISH_MM_QN_SPLIT_N);
        DynamicQuantQnWithMulQr(dequantScaleQNopeGm_[scaleQueryNopeOffset],
            queryOutGm_[dynamicQuantQueryResOffset], qrOutGm_[qrPostProcessResOffset],
            mmQnResGm_[dynamicQuantQueryOffset], shareTmpUb,  curStepBatchSizeVec, baseParams_->headSizeCkv,
            baseParams_->numHeadSize, queryOutStride,
            // Rope Post Process
            qrOutGm_[qrPostProcessResOffset],
            quantScaleCkv, baseParams_->dimHeadRope, qrOutputStride);

        dynamicQuantQueryOffset += baseParams_->headSizeCkv;
        scaleQueryNopeOffset += 1;
        dynamicQuantQueryResOffset += baseParams_->headSizeCkv;
        qrPostProcessResOffset += baseParams_->dimHeadRope;
    }
    // Rope Post Process
    WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    // Dynamic Quant
    WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
    WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
}

} // namespace MlaPrologV3

#endif // MLA_PROLOG_V3_VEC_S1_CUB_S2_H
