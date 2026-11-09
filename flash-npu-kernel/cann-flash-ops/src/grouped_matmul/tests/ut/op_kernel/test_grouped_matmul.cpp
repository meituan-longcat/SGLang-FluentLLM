/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "tikicpulib.h"
// #include "grouped_matmul_tiling_def.h"
#include "grouped_matmul_tiling.h"
#include "data_utils.h"
// #define __forceinline__ __global__
#include "../../../op_kernel/grouped_matmul.cpp"

using namespace std;

#define PARAM_LIST_DEF GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,\
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,\
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,\
                                                     GM_ADDR workspace, GM_ADDR tiling

#define PARAM_LIST x,  weight,  bias,  scale, offset,  antiquantScale,  antiquantOffset, groupList, perTokenScale, y, workspace, tiling


class grouped_matmul_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "grouped_matmul_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grouped_matmul_test TearDown\n" << std::endl;
    }
};

//非量化fp16场景
TEST_F(grouped_matmul_test, test_case_fp16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return grouped_matmul<GMM_TPL_FLOAT16, // D_T_A
                                                GMM_TPL_FLOAT16, // D_T_B
                                                GMM_TPL_FLOAT16, // D_T_Y
                                                0, // TRANS_A
                                                0, // TRANS_B
                                                GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                                0, // IS_STATIC_TILING_API
                                                GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                                GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                                GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
                                                false //IS_ENABLE_FIXED_AXIS
                                                >(PARAM_LIST);};
    size_t xSize = M * K * sizeof(half);
    size_t weightSize = E * K * N * sizeof(half);
    size_t groupedListSize = E * sizeof(int64_t);
    size_t ySize = M * N * sizeof(half);
    size_t tilingSize = sizeof(GMMTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* groupedList = (uint8_t*)AscendC::GmAlloc(groupedListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t NumBlocks = 40;

    GMMTilingData* tilingData = reinterpret_cast<GMMTilingData*>(tiling);
    tilingData->gmmBaseParams.groupNum = E;
    tilingData->gmmBaseParams.coreNum = NumBlocks;
    tilingData->gmmBaseParams.groupType = 0;

    tilingData->gmmArray.mList[0] = 345;
    tilingData->gmmArray.kList[0] = -1;
    tilingData->gmmArray.nList[0] = 567;

    tilingData->mmTilingData.usedCoreNum = NumBlocks;
    tilingData->mmTilingData.M = 345;
    tilingData->mmTilingData.N = 567;
    tilingData->mmTilingData.Ka = 1280;
    tilingData->mmTilingData.Kb = 1280;
    tilingData->mmTilingData.singleCoreM = 345;
    tilingData->mmTilingData.singleCoreN = 256;
    tilingData->mmTilingData.singleCoreK = 1280;
    tilingData->mmTilingData.baseM = 128;
    tilingData->mmTilingData.baseN = 256;
    tilingData->mmTilingData.baseK = 64;
    tilingData->mmTilingData.depthA1 = 8;
    tilingData->mmTilingData.depthB1 = 8;
    tilingData->mmTilingData.stepM = 1;
    tilingData->mmTilingData.stepN = 1;
    tilingData->mmTilingData.stepKa = 4;
    tilingData->mmTilingData.stepKb = 4;
    tilingData->mmTilingData.isBias = 0;
    tilingData->mmTilingData.transLength = 0;
    tilingData->mmTilingData.iterateOrder = 0;
    tilingData->mmTilingData.dbL0A = true;
    tilingData->mmTilingData.dbL0B = true;
    tilingData->mmTilingData.dbL0C = false;

    ICPU_RUN_KF(func, NumBlocks, x, weight, groupedList, y, y,y,y,y,y,y,  workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(groupedList);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// TEST_F(grouped_matmul_test, test_case_fp16)
// {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t xSize = 1280 * 345 * sizeof(half);
//     size_t weightSize = 1280 * 567 * sizeof(half);
//     size_t groupedListSize = 1 * 2 * sizeof(int64_t);
//     size_t ySize = 345 * 2 * 567 * sizeof(float);
//     size_t tilingSize = sizeof(GMMTilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
//     uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
//     uint8_t* groupedList = (uint8_t*)AscendC::GmAlloc(groupedListSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
//     uint32_t NumBlocks = 40;

//     GMMTilingData* tilingData = reinterpret_cast<GMMTilingData*>(tiling);
//     tilingData->gmmBaseParams.groupNum = 2;
//     tilingData->gmmBaseParams.coreNum = NumBlocks;
//     tilingData->gmmBaseParams.groupType = 2;

//     tilingData->gmmArray.mList[0] = 345;
//     tilingData->gmmArray.kList[0] = -1;
//     tilingData->gmmArray.nList[0] = 567;

//     tilingData->mmTilingData.usedCoreNum = 1;
//     tilingData->mmTilingData.M = 345;
//     tilingData->mmTilingData.N = 567;
//     tilingData->mmTilingData.Ka = 1280;
//     tilingData->mmTilingData.Kb = 1280;
//     tilingData->mmTilingData.singleCoreM = 345;
//     tilingData->mmTilingData.singleCoreN = 256;
//     tilingData->mmTilingData.singleCoreK = 1280;
//     tilingData->mmTilingData.baseM = 128;
//     tilingData->mmTilingData.baseN = 256;
//     tilingData->mmTilingData.baseK = 64;
//     tilingData->mmTilingData.depthA1 = 8;
//     tilingData->mmTilingData.depthB1 = 8;
//     tilingData->mmTilingData.stepM = 1;
//     tilingData->mmTilingData.stepN = 1;
//     tilingData->mmTilingData.stepKa = 4;
//     tilingData->mmTilingData.stepKb = 4;
//     tilingData->mmTilingData.isBias = 0;
//     tilingData->mmTilingData.transLength = 0;
//     tilingData->mmTilingData.iterateOrder = 0;
//     tilingData->mmTilingData.dbL0A = true;
//     tilingData->mmTilingData.dbL0B = true;
//     tilingData->mmTilingData.dbL0C = false;


//     ICPU_SET_TILING_KEY(0);
//     ICPU_RUN_KF(grouped_matmul, NumBlocks, x, weight, groupedList, y, y, workspace, tiling);

//     AscendC::GmFree(x);
//     AscendC::GmFree(weight);
//     AscendC::GmFree(groupedList);
//     AscendC::GmFree(y);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }
