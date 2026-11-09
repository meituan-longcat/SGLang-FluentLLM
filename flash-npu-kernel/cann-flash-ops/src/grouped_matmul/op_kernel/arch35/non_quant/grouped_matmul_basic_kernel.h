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
* \file grouped_matmul_kernel.h
* \brief
*/

#ifndef NON_QUANT_GROUPED_MATMUL_BASIC_KERNEL_ACT
#define NON_QUANT_GROUPED_MATMUL_BASIC_KERNEL_ACT

#include "cgmct/kernel/kernel_grouped_matmul.h"
#include "cgmct/block/block_scheduler_grouped_matmul_aswt.h"
#include "../grouped_matmul_tiling_data_apt.h"
#include "../../grouped_matmul_utils.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;
using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;

namespace GROUPED_MATMUL {

constexpr uint64_t DIM_NUM = 2UL;
constexpr uint64_t BUF_SIZE = 3UL;

template<typename layoutA, typename layoutB>
__aicore__ inline void GmmNoQuantAswt(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, gmmNoQuantParam, gmmBaseParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, mmTilingData, mmTilingData_, tiling);
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    // 定义矩阵的类型和布局
    using AType = DTYPE_X;
    using BType = DTYPE_X;
    using CType = DTYPE_Y;
    using BiasType = DTYPE_BIAS;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajor;
    using LayoutBias = layout::RowMajor;
    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtScheduler;
    // 定义MMAD类型
    using BlockMmad = Block::BlockGroupedMatmulBuilder<
            AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias,
            L1TileShape, L0TileShape, BlockScheduler, MatmulMultiBlockBias<>>;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;
    // 定义Kernel类型
    using GroupedMatmulKernel = Kernel::KernelGroupedMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GroupedMatmulKernel::Params;
    using GMMTiling = typename GroupedMatmulKernel::GMMTiling;
    GMMTiling gmmParams {gmmBaseParams_.groupNum, gmmBaseParams_.groupType, gmmBaseParams_.groupListType,
        gmmBaseParams_.singleX, gmmBaseParams_.singleWeight, gmmBaseParams_.singleY, gmmBaseParams_.hasBias,
        gmmBaseParams_.mTailCnt, gmmBaseParams_.nTailCnt, gmmBaseParams_.weightNoL2Cache};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {
        // template shape, gmm shape can not get now
        {1, 1, 1, 1},
        // mmad args
        {x, weight, y, bias, groupList},
        // epilogue args
        {},
        // gmm tiling data
        gmmParams
    };
    GroupedMatmulKernel op;
    op(params);
}

__aicore__ inline int32_t GetSplitValue(uint32_t groupIdx, int32_t &preOffset,
                                        const int32_t groupType, const uint32_t groupListType,
                                        const GlobalTensor<int64_t> &groupListGm) {
    int32_t splitValue = 0;
    if (likely(groupType != -1)) {  // -1: no  need to split
        if (groupListType == 0) { // 0: cumsum 1: count
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}

__aicore__ inline void GetTensorShape(uint32_t groupIdx, GM_ADDR tensorPtr, uint32_t *shape)
{
    AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void *>(tensorPtr));
    AscendC::TensorDesc<int32_t> desc;
    uint64_t buf[BUF_SIZE] = {0UL};
    desc.SetShapeAddr(buf);
    listTensorDesc.GetDesc(desc, groupIdx);
    uint64_t dim = desc.GetDim();
    for (uint64_t index = 0, count = 0; index < dim; index++) {
        if (dim - index > DIM_NUM) {
            continue;
        }
        shape[count++] = desc.GetShape(index);
    }
}

template <typename T>
__aicore__ inline void EmptyTensor(GM_ADDR x, GM_ADDR weight, GM_ADDR groupListPtr, GM_ADDR y, GM_ADDR tiling) {
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, gmmNoQuantParam, gmmBaseParams, tiling);
    // grouptype can only be 2, else return.
    if (groupListPtr == nullptr || gmmBaseParams.groupType != 2) {
        return;
    }

    GlobalTensor<T> yGm;
    GlobalTensor<int64_t> groupListGm;
    yGm.SetGlobalBuffer(GetTensorAddr<T>(0, y));
    if (groupListPtr != nullptr) {
        groupListGm.SetGlobalBuffer((__gm__ int64_t*)groupListPtr);
    }
    uint64_t yBaseOffset = 0;
    int32_t preOffset = 0;
    uint64_t coreIdx = GetBlockIdx();
    uint64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;
    }

    for (uint32_t groupIdx = 0; groupIdx < gmmBaseParams.groupNum; ++groupIdx) {
        int32_t splitValue = GetSplitValue(groupIdx, preOffset, gmmBaseParams.groupType,
            gmmBaseParams.groupListType, groupListGm);
        uint32_t xShape[DIM_NUM] = {0UL};
        uint32_t wShape[DIM_NUM] = {0UL};
        GetTensorShape(gmmBaseParams.singleX == 0 ? groupIdx : 0, x, xShape);
        GetTensorShape(gmmBaseParams.singleWeight == 0 ? groupIdx : 0, weight, wShape);
        uint32_t m = xShape[DIM_NUM - 1];
        uint32_t k = splitValue;
        uint32_t n = wShape[DIM_NUM - 1];
        if (k == 0) {
            uint32_t singleM = Ceil(m, gmmBaseParams.coreNum);
            singleM = AlignUp<HALF_UB_BLOCK_UNIT_SIZE>(singleM);
            uint32_t cursingleM = singleM;
            if (coreIdx * singleM >= m) {
                yBaseOffset += static_cast<uint64_t>(m) * n;
                continue;
            } else if (m - singleM * coreIdx < singleM) {
                cursingleM = m - singleM * coreIdx;
            }
            InitOutput<T>(yGm[yBaseOffset + coreIdx * singleM * n], static_cast<uint64_t>(cursingleM) * n, 0);
        }
        yBaseOffset += static_cast<uint64_t>(m) * n;
    }
}

}
#endif