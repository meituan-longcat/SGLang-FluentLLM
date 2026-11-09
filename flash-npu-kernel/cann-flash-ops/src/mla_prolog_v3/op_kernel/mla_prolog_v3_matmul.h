/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mla_prolog_v3_matmul.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_MATMUL_H
#define MLA_PROLOG_V3_MATMUL_H

#include "mla_prolog_v3_comm.h"

namespace MlaPrologV3 {

constexpr uint8_t UNIT_FLAG_DISABLE = 0;  // 0: disable: 不配置unitFlag
constexpr uint8_t UNIT_FLAG_CHECK = 0b10; // 2: enable: 将mmadParams.unitFlag设置为 0b10
constexpr uint8_t UNIT_FLAG_SET = 0b11;   // 3: enable: 在k的最后一轮循环，会将mmadParams.unitFlag设置为 0b11


/**
 * @brief Struct to encapsulate all local tensor objects for matrix multiplication
 * @tparam T Data type for A, B, and L0 tensors
 * @tparam O_L0C Data type for C L0 tensor
 */
template <typename T, typename O_L0C>
struct MmLocalTensors {
    LocalTensor<T> aL1Tensor;
    LocalTensor<T> bL1Tensor;
    LocalTensor<T> aL0Tensor;
    LocalTensor<T> bL0Tensor;
    LocalTensor<O_L0C> cL0Tensor;

    /**
     * @brief Initialize all local tensors with buffer addresses from bufParam
     * @param bufParam Buffer parameter object containing all buffer addresses
     */
    __aicore__ inline void Init(const MMBufParams &bufParam)
    {
        aL1Tensor.SetAddr(bufParam.aL1BufAddr);
        bL1Tensor.SetAddr(bufParam.bL1BufAddr);
        aL0Tensor.SetAddr(bufParam.aL0BufAddr);
        bL0Tensor.SetAddr(bufParam.bL0BufAddr);
        cL0Tensor.SetAddr(bufParam.cL0BufAddr);
    }
};

template <typename SrcT>
__aicore__ inline constexpr uint32_t GetC0Num()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return 8;
    } else if (sizeof(SrcT) == sizeof(int8_t)) {
        return 32;
    }
    return 16;
}

template <typename T>
__aicore__ inline void CopyNDGmToL1(LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDivT(srcN, BLOCK_CUBE_SIZE) * BLOCK_CUBE_SIZE; // 对齐到16 单位 block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void CopyNZGmToL1(LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcNstride)
{
    DataCopyParams param;
    param.blockCount = CeilDivT(srcD, GetC0Num<T>());
    param.blockLen = srcN;                 // 单位为32B srcN*16/16
    param.srcStride = (srcNstride - srcN); // 单位为32B (srcNstride - srcN)*16/16
    param.dstStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, param);
}

template <typename T, bool preload = false>
__aicore__ inline void LoadL1A(const GlobalTensor<T> &tensorAGm, const uint32_t mInput, const uint32_t kL1StepSize,
                               const uint32_t kInput, MMBufParams &bufParam)
{
    uint32_t bufIter = bufParam.aL1BufIter;
    if constexpr (preload) {
        bufIter++;
    }
    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam.aL1BufAddr);
    uint32_t aL1Offset = L1_A_SIZE / sizeof(T);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufIter & 1u));
    LocalTensor<T> aL1 = aL1Tensor[(bufIter & 1u) * aL1Offset];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kL1StepSize, kInput);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufIter & 1u));
}

template <typename T, DataFormat loadFormat = DataFormat::NZ, bool preload = false>
__aicore__ inline void LoadL1B(const GlobalTensor<T> &tensorBGm, const uint32_t nL1Size, const uint32_t kL1StepSize,
                               const uint32_t kInput, MMBufParams &bufParam)
{
    uint32_t bufIter = bufParam.bL1BufIter;
    if constexpr (preload) {
        bufIter++;
    }
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufIter & 1u));
    LocalTensor<T> bL1 = bL1Tensor[(bufIter & 1u) * bL1Offset];
    if constexpr (loadFormat == DataFormat::NZ) {
        CopyNZGmToL1(bL1, tensorBGm, kL1StepSize, nL1Size, kInput);
    } else {
        CopyNDGmToL1(bL1, tensorBGm, kInput, nL1Size, nL1Size);
    }
    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufIter & 1u));
}

/**
 * @brief 使用3D Pro模式将数据从L1加载到L0，可选择是否进行转置操作
 * @tparam T 张量的数据类型
 * @tparam enTranspose 是否启用转置操作（用于B矩阵）
 * @param l0Tensor L0缓冲区中的目标张量
 * @param l1Tensor L1缓冲区中的源张量
 * @param mSize 行维度（对应A矩阵的m，B矩阵的k）
 * @param kSize 列维度（对应A矩阵的k，B矩阵的n）
 * @param l1StepSize L1缓冲区中的步长大小，用于B矩阵转置
 */
template <typename T, bool enTranspose = false>
__aicore__ inline void LoadDataL0K3DPro(const LocalTensor<T> &l0Tensor, const LocalTensor<T> &l1Tensor,
                                        const uint32_t mSize, const uint32_t kSize, const uint32_t l1StepSize = 0)
{
    constexpr uint32_t FMATRIX_HEIGHT_UNIT = 16;
    constexpr int EXTCONFIG_SHIFT_0 = 48;
    constexpr int EXTCONFIG_SHIFT_1 = 32;
    constexpr int EXTCONFIG_SHIFT_MSIZE = 16;

    uint8_t padList[4] = {0, 0, 0, 0};
    LocalTensor<T> srcTensor = l1Tensor;
    LoadData3DParamsV2Pro loadData3DV2;

    if constexpr (enTranspose) {
        // LoadDataBL0K3DPro behavior: mSize=kSize, mSize=nSize, l1StepSize=kL1Size
        AscendC::SetFmatrix(l1StepSize / FMATRIX_HEIGHT_UNIT, FMATRIX_HEIGHT_UNIT, padList, FmatrixMode::FMATRIX_LEFT);
        loadData3DV2.enTranspose = 1;
    } else {
        // LoadDataAL0K3DPro behavior: mSize=mSize, mSize=kSize
        AscendC::SetFmatrix(mSize / FMATRIX_HEIGHT_UNIT, FMATRIX_HEIGHT_UNIT, padList, FmatrixMode::FMATRIX_LEFT);
    }
    loadData3DV2.channelSize = kSize;
    loadData3DV2.extConfig = ((uint64_t)0 << EXTCONFIG_SHIFT_0) | ((uint64_t)0 << EXTCONFIG_SHIFT_1) |
                             ((uint64_t)mSize << EXTCONFIG_SHIFT_MSIZE) | (uint64_t)kSize;

    LoadData<T>(l0Tensor, srcTensor, loadData3DV2);
}

template <typename T>
__aicore__ inline void LoadDataBL0KQuant(const LocalTensor<T> &bL0Tensor, const LocalTensor<T> &bL1Tensor,
                                         const uint32_t kSize, const uint32_t nSize, const uint32_t kL1Size)
{
    constexpr uint32_t LOAD_WIDTH = 32;
    constexpr uint32_t DATA_BYTES_PER_REPEAT = 32;
    constexpr uint32_t STRIDE_HEIGHT = 32;

    uint32_t kloops = kSize / (LOAD_WIDTH / sizeof(T));
    LocalTensor<T> srcTensor = bL1Tensor;
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2dTransposeParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = nSize / (DATA_BYTES_PER_REPEAT / sizeof(T));
        loadData2DParams.srcStride = kL1Size / STRIDE_HEIGHT;
        loadData2DParams.dstGap = 1;
        loadData2DParams.dstFracGap = 0;
        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = srcTensor[i * STRIDE_HEIGHT * (DATA_BYTES_PER_REPEAT / sizeof(T))];
        // LoadData 不支持int8的转置，需要用 LoadDataWithTranspose
        LoadDataWithTranspose(bL0Tensor[i * STRIDE_HEIGHT * nSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename T, typename O>
__aicore__ inline void MatmulL0(MMBufParams &bufParam, const LocalTensor<T> &aL1, const LocalTensor<T> &bL1,
                                const LocalTensor<T> &aL0Tensor, const LocalTensor<T> &bL0Tensor,
                                const LocalTensor<O> &cL0Tensor, const MmadParams &mmadParams,
                                const uint32_t kL1StepSize)
{
    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    LocalTensor<T> aL0 = aL0Tensor[(bufParam.aL0BufIter & 1u) * (L0A_PP_SIZE / sizeof(T))];
    LoadDataL0K3DPro<T, false>(aL0, aL1, mmadParams.m, mmadParams.k);
    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    LocalTensor<T> bL0 = bL0Tensor[(bufParam.bL0BufIter & 1u) * (L0B_PP_SIZE / sizeof(T))];
    if constexpr (std::is_same<T, int8_t>::value) {
        LoadDataBL0KQuant(bL0, bL1, mmadParams.k, mmadParams.n, kL1StepSize);
    } else {
        LoadDataL0K3DPro<T, true>(bL0, bL1, mmadParams.k, mmadParams.n, kL1StepSize);
    }

    SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    Mmad(cL0Tensor, aL0, bL0, mmadParams);
    PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    bufParam.bL0BufIter++;
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    bufParam.aL0BufIter++;
}

template <typename T, typename O, typename O_L0C, bool enUnitFlag = false>
__aicore__ inline void GetTensorC(const GlobalTensor<O> &tensorCGm, const LocalTensor<O_L0C> &cL0, const uint32_t mSize,
                                  const uint32_t nSize, const uint32_t srcStride, const uint32_t dstStride,
                                  MMBufParams &bufParam)
{
    FixpipeParamsV220 fixParams;
    fixParams.nSize = nSize;         // 实现切片大小
    fixParams.mSize = mSize;         // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
    fixParams.srcStride = srcStride; // ((fixParams.mSize + 15) / 16) * 16
    fixParams.dstStride = dstStride;
    fixParams.ndNum = 1;
    if constexpr (enUnitFlag) {
        fixParams.unitFlag = UNIT_FLAG_SET;
    } else {
        fixParams.unitFlag = UNIT_FLAG_DISABLE;
    }
    if constexpr (std::is_same<O_L0C, float>::value) {
        fixParams.quantPre = QuantMode_t::F322BF16;
    }
    if constexpr (!enUnitFlag) {
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
    }
    Fixpipe(tensorCGm, cL0, fixParams);
}

/**
 * @brief 将AB矩阵从GM加载到L1缓存以进行矩阵乘法运算
 * @tparam T 张量的数据类型
 * @tparam hasL1ALoaded A矩阵是否已经加载到L1中
 * @param tensorAGm GM中的A矩阵
 * @param tensorBGm GM中的B矩阵
 * @param kL1 当前K L1循环迭代次数
 * @param kL1Loops K L1循环的总次数
 * @param para 包含维度和步长信息的矩阵乘法参数
 * @param nL1Offset L1缓冲区中的N维度偏移
 * @param nL1Size L1缓冲区中的N维度大小
 * @param kOffesetUnit K维度寻址的偏移单位
 * @param bufParam L1缓冲区管理的参数对象
 */
template <typename T, bool hasL1ALoaded>
__aicore__ inline void LoadL1AB(const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm, uint32_t kL1,
                                uint32_t kL1Loops, const MMParams &para, uint32_t nL1Offset, uint32_t nL1Size,
                                uint32_t kOffesetUnit, MMBufParams &bufParam)
{
    if (kL1 == 0) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A(tensorAGm[kL1 * para.kL1StepSize], para.m, para.kL1StepSize, para.orgKa, bufParam);
        }
        LoadL1B(tensorBGm[para.orgKa * nL1Offset + kL1 * kOffesetUnit], nL1Size, para.kL1StepSize, para.k, bufParam);
    }

    uint32_t kL1LoadStepSize = para.kL1StepSize;
    if (kL1 + 2 == kL1Loops) {
        kL1LoadStepSize = para.kL1TailStepSize;
    }

    if (kL1 + 1 < kL1Loops) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A<T, true>(tensorAGm[(kL1 + 1) * para.kL1StepSize], para.m, kL1LoadStepSize, para.k, bufParam);
        }
        LoadL1B<T, DataFormat::NZ, true>(tensorBGm[para.k * nL1Offset + (kL1 + 1) * kOffesetUnit], nL1Size,
            kL1LoadStepSize, para.k, bufParam);
    }
}

/**
 * @brief B矩阵数据加载到L1缓冲区的辅助函数
 * @tparam T B张量的数据类型
 * @tparam isContinuousCopy 是否使用连续复制模式
 * @param bL1 B矩阵L1张量
 * @param tensorBGm B矩阵GM张量
 * @param kL1StepSize L1中K维度的步长大小
 * @param subNL1SplitSize L1中N维度的分割大小
 * @param para MMParams参数
 * @param bufParam 缓冲区管理参数
 * @param nL1 当前N L1迭代
 * @param kL1 当前K L1迭代
 * @param kOffesetUnit K维度偏移单位
 */
template <typename T, bool isContinuousCopy>
__aicore__ inline void LoadL1BGroupCompute(LocalTensor<T> &bL1, const GlobalTensor<T> &tensorBGm,
                                           uint32_t subNL1SplitSize, const MMParams &para, int64_t nL1, uint32_t kL1,
                                           uint32_t kOffesetUnit)
{
    int64_t tensorBGmOffset = para.k * nL1 * para.baseN + kL1 * kOffesetUnit;
    auto tensorBGmForL1 = tensorBGm[tensorBGmOffset];
    if constexpr (isContinuousCopy) {
        CopyNZGmToL1(bL1, tensorBGmForL1, para.kL1StepSize, subNL1SplitSize, para.k);
    } else {
        // 每次搬运两块K*64拼接成K*128
        CopyNZGmToL1(bL1, tensorBGmForL1, para.kL1StepSize, subNL1SplitSize >> 1, para.k);
        LocalTensor<T> b2L1 = bL1[(para.kL1StepSize * subNL1SplitSize) >> 1];
        auto tensorB2GmForL1 = tensorBGmForL1[para.k * DIM_HEAD_SIZE_QCQR];
        CopyNZGmToL1(b2L1, tensorB2GmForL1, para.kL1StepSize, subNL1SplitSize >> 1, para.k);
    }
}

/**
 * @brief 执行矩阵乘法的K L0循环计算
 * @tparam T A和B矩阵的数据类型
 * @tparam O_L0C L0C缓存中的数据类型
 * @tparam enUnitFlag 是否启用unitFlag功能
 * @param cL0 L0缓存中的C矩阵张量
 * @param aL1 L1缓存中的A矩阵张量
 * @param bL1 L1缓存中的B矩阵张量
 * @param localTensors 矩阵乘法所需的本地张量对象
 * @param bufParam L1缓存管理的缓冲区参数对象
 * @param kL1 当前K L1循环迭代次数
 * @param kL1Loops K L1循环的总次数
 * @param stepK K L0步数
 * @param aOffsetUnit A矩阵L0寻址的偏移单位
 * @param bOffsetUnit B矩阵L0寻址的偏移单位
 * @param mmadParams 矩阵乘法参数
 * @param aOffset A矩阵计算的起始偏移
 * @param bOffset B矩阵计算的起始偏移
 */
template <typename T, typename O_L0C, bool enUnitFlag>
__aicore__ inline void MatmulL1(const LocalTensor<O_L0C> &cL0, const LocalTensor<T> &aL1, const LocalTensor<T> &bL1,
                                const MmLocalTensors<T, O_L0C> &localTensors, MMBufParams &bufParam,
                                const MMParams &para, uint32_t kL1,
                                uint32_t kL1Loops, uint32_t aOffsetUnit, uint32_t bOffsetUnit,
                                MmadParams &mmadParams, int64_t &aOffset, int64_t &bOffset)
{
    uint32_t kL1StepSize = para.kL1StepSize;
    uint32_t stepK = para.stepK;
    if (kL1 == kL1Loops - 1) {
        kL1StepSize = para.kL1TailStepSize;
        stepK = para.tailStepK;
    }

    for (int64_t kL0Loops = 0; kL0Loops < stepK; kL0Loops++) {
        mmadParams.cmatrixInitVal = ((kL1 == 0) && (kL0Loops == 0));
        if constexpr (enUnitFlag) {
            mmadParams.unitFlag = (kL1 == kL1Loops - 1) && (kL0Loops == stepK - 1) ? UNIT_FLAG_SET : UNIT_FLAG_CHECK;
        }
        MatmulL0(bufParam, aL1[aOffset], bL1[bOffset], localTensors.aL0Tensor,
                localTensors.bL0Tensor, cL0, mmadParams, kL1StepSize);
        aOffset += aOffsetUnit;
        bOffset += bOffsetUnit;
    }
}

/**
 * @brief MatmulSplitK 通过切分K轴，进行L1的数据管理和矩阵乘运算; 用于mmCq, mmCKvKr和mmQcQr。
 * @param tensorAGm A矩阵在GM的位置
 * @param tensorBGm B矩阵在GM的位置
 * @param tensorCGm C矩阵在GM的位置
 * @param para 表示matmul形状信息的结构体参数
 * @param bufParam 管理L1 buffer地址和同步计数的结构体参数
 * @param nL1Offset 本次计算中，L1 buffer内B矩阵在n轴上的偏移
 * @param nL1Size 本次计算中，L1 buffer内B矩阵在n轴上的计算量
 */
template <typename T, typename O, bool hasL1ALoaded = false, bool enUnitFlag = false>
__aicore__ inline void MatmulSplitK(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm,
                                    const GlobalTensor<T> &tensorBGm, const MMParams &para, MMBufParams &bufParam,
                                    const uint32_t nL1Offset, const uint32_t nL1Size)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

    // 全局L1管理
    uint32_t kOffesetUnit = para.kL1StepSize * GetC0Num<T>();
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);
    constexpr uint32_t aL1Offset = L1_A_SIZE / sizeof(T);
    constexpr uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    uint32_t aOffsetUnit = mSize * para.baseK;
    uint32_t bOffsetUnit = GetC0Num<T>() * para.baseK;

    MmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    LocalTensor<T> aL1;
    LocalTensor<T> bL1;

    if constexpr (hasL1ALoaded) {
        aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * aL1Offset];
    }

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
    LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

    MmadParams mmadParams = MmadParams(mSize, nL1Size, para.baseK, UNIT_FLAG_DISABLE, false, true);

    int64_t aOffset = 0;
    int64_t bOffset = 0;

    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        // Load data from global memory to L1 cache
        LoadL1AB<T, hasL1ALoaded>(tensorAGm, tensorBGm, kL1, kL1Loops, para, nL1Offset, nL1Size, kOffesetUnit,
                                  bufParam);

        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
        bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * bL1Offset];
        if constexpr (!hasL1ALoaded) {
            WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
            aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * aL1Offset];
            aOffset = 0;
        }
        bOffset = 0;

        // Perform core K L0 loops computation
        MatmulL1<T, O_L0C, enUnitFlag>(cL0, aL1, bL1, localTensors, bufParam, para, kL1, kL1Loops, aOffsetUnit,
                                       bOffsetUnit, mmadParams, aOffset, bOffset);

        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
        bufParam.bL1BufIter++;
        if constexpr (!hasL1ALoaded) {
            SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
            bufParam.aL1BufIter++;
        }
    }
    GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[nL1Offset], cL0, para.m, nL1Size, mSize, para.orgKc, bufParam);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
    bufParam.cL0BufIter++;
}


/**
 * @brief MatmulFullLoad A，B矩阵能够全载L1时的矩阵运算，用于mmQn;
 * @param tensorAGm A矩阵在GM的位置
 * @param tensorBGm B矩阵在GM的位置
 * @param tensorCGm C矩阵在GM的位置
 * @param para 表示matmul形状信息的结构体参数
 * @param bufParam 管理L1 buffer地址和同步计数的结构体参数
 */
template <typename T, typename O, bool hasL1BLoaded = false, bool enUnitFlag = false>
__aicore__ inline void MatmulFullLoad(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm,
                                      const GlobalTensor<T> &tensorBGm, const MMParams &para, MMBufParams &bufParam)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    constexpr uint32_t aL1Offset = L1_A_SIZE / sizeof(T);
    constexpr uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    MmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    LoadL1A(tensorAGm, para.m, para.k, para.orgKa, bufParam);
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    LocalTensor<T> aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * aL1Offset];

    if constexpr (!hasL1BLoaded) {
        LoadL1B<T, DataFormat::ND, false>(tensorBGm, para.n, para.k, para.k, bufParam);
        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
    }
    LocalTensor<T> bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * bL1Offset];

    uint32_t nSplitSize = 128;
    uint32_t nSplitSizeAct = nSplitSize;
    uint32_t nloops = CeilDivT(para.n, nSplitSize);

    MmadParams mmadParams = MmadParams(mSize, nSplitSizeAct, para.k, UNIT_FLAG_DISABLE, false, true);

    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            nSplitSizeAct = para.n - (nloops - 1) * nSplitSize;
            mmadParams.n = nSplitSizeAct;
        }
        // Perform matrix multiplication computation for current N-dimension split
        if constexpr (enUnitFlag) {
            mmadParams.unitFlag = UNIT_FLAG_SET;
        }
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];
        MatmulL0(bufParam, aL1, bL1[para.k * nSplitSize * n], localTensors.aL0Tensor, localTensors.bL0Tensor, cL0,
                 mmadParams, para.kL1StepSize);
        GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[n * nSplitSize], cL0, para.m, nSplitSizeAct, mSize, para.orgKc,
                                            bufParam);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        bufParam.cL0BufIter++;
    }

    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
    bufParam.bL1BufIter++;
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    bufParam.aL1BufIter++;
}

/**
 * @brief MatmulGroupComputeAFullLoad
 * A矩阵能够全载，通过切分K轴，进行L1的数据管理和矩阵乘运算，仅适用于算力分组场景下的mmQc和mmQr;
 * @param tensorAGm A矩阵在GM的位置
 * @param tensorBGm B矩阵在GM的位置
 * @param tensorCGm C矩阵在GM的位置
 * @param para 表示matmul形状信息的结构体参数
 * @param bufParam 管理L1 buffer地址和同步计数的结构体参数
 */
template <typename T, typename O, bool isContinuousCopy = true, bool enUnitFlag = false>
__aicore__ inline void MatmulGroupComputeAFullLoad(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm,
                                                   const GlobalTensor<T> &tensorBGm, const MMParams &para,
                                                   MMBufParams &bufParam)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;
    // 全局L1管理
    uint32_t kOffesetUnit = para.kL1StepSize * GetC0Num<T>();
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);
    uint32_t nL1loops = CeilDivT(para.n, para.baseN);
    uint32_t aOffsetUnit = mSize * para.baseK;
    uint32_t bOffsetUnit = GetC0Num<T>() * para.baseK;

    MmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    LocalTensor<T> aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, para.m, para.k, para.k);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));

    uint32_t subNL1SplitSize = para.baseN;

    MmadParams mmadParams = MmadParams(mSize, subNL1SplitSize, para.baseK, UNIT_FLAG_DISABLE, false, true);

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = para.n - (nL1loops - 1) * para.baseN;
            mmadParams.n = subNL1SplitSize;
        }

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

        int64_t aOffset = 0;
        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
            LocalTensor<T> bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * (L1_B_SIZE / sizeof(T))];
            LoadL1BGroupCompute<T, isContinuousCopy>(bL1, tensorBGm, subNL1SplitSize, para, nL1, kL1,
                                                     kOffesetUnit);
            SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));

            int64_t bOffset = 0;
            // Perform core K L0 loops computation using shared function
            MatmulL1<T, O_L0C, enUnitFlag>(cL0, aL1, bL1, localTensors, bufParam, para, kL1, kL1Loops,
                                           aOffsetUnit, bOffsetUnit, mmadParams, aOffset, bOffset);
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
            bufParam.bL1BufIter++;
        }
        GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[nL1 * para.baseN], cL0, para.m, subNL1SplitSize, mSize,
                                            para.orgKc, bufParam);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        bufParam.cL0BufIter++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    bufParam.aL1BufIter++;
}
} // namespace MlaProlog


#endif