/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "moe_distribute_dispatch_v2_tiling_def.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2.cpp"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"

class MoeDistributeDispatchV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchV2Test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchV2Test TearDown\n" << std::endl;
    }
};

TEST_F(MoeDistributeDispatchV2Test, MoeDistributeDispatchV2Test1000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tilingData->moeDistributeDispatchV2Info.epWorldSize = 8;
    tilingData->moeDistributeDispatchV2Info.tpWorldSize = 2;
    tilingData->moeDistributeDispatchV2Info.epRankId = 0;
    tilingData->moeDistributeDispatchV2Info.tpRankId = 0;
    tilingData->moeDistributeDispatchV2Info.expertShardType = 0;
    tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum = 1;
    tilingData->moeDistributeDispatchV2Info.moeExpertNum = 7;
    tilingData->moeDistributeDispatchV2Info.quantMode = 0;
    tilingData->moeDistributeDispatchV2Info.globalBs = 64;
    tilingData->moeDistributeDispatchV2Info.bs = 8;
    tilingData->moeDistributeDispatchV2Info.k = 7;
    tilingData->moeDistributeDispatchV2Info.h = 7168;
    tilingData->moeDistributeDispatchV2Info.aivNum = 48;
    tilingData->moeDistributeDispatchV2Info.isTokenMask= false;
    tilingData->moeDistributeDispatchV2Info.totalUbSize = 196352;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchV2Warrper = [] (
        GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
        GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
        GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
        GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_dispatch_v2<false, TILINGKEY_NO_QUANT, false, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE, TILINGKEY_TPL_A3>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
                expandScalesOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeDispatchV2Warrper, 40, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, 
                expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, expandScalesOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
    AscendC::GmFree((void*)elasticInfo);
    AscendC::GmFree((void*)assistInfoOut);
    AscendC::GmFree((void*)expandScalesOut);
    AscendC::GmFree((void*)xActiveMask);
    AscendC::GmFree((void*)expertScales);
}

//MoeDistributeDispatchA2 test do dispatch unquant kernel
TEST_F(MoeDistributeDispatchV2Test, MoeDistributeDispatchV2Test2000001000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tilingData->moeDistributeDispatchV2Info.epWorldSize = 8;
    tilingData->moeDistributeDispatchV2Info.tpWorldSize = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.epRankId = 0;
    tilingData->moeDistributeDispatchV2Info.tpRankId = 0;   //针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.expertShardType = 0;
    tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.moeExpertNum = 8;
    tilingData->moeDistributeDispatchV2Info.quantMode = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.globalBs = 64;
    tilingData->moeDistributeDispatchV2Info.bs = 8;
    tilingData->moeDistributeDispatchV2Info.k = 7;
    tilingData->moeDistributeDispatchV2Info.h = 7168;
    tilingData->moeDistributeDispatchV2Info.aivNum = 40;//??
    tilingData->moeDistributeDispatchV2Info.isTokenMask= false;
    tilingData->moeDistributeDispatchV2Info.totalUbSize = 196352;//??

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchV2Warrper = [] (
        GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
        GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
        GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
        GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_dispatch_v2<false, TILINGKEY_NO_QUANT, false, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE, TILINGKEY_TPL_A3>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
                expandScalesOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeDispatchV2Warrper, 40, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, 
                expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, expandScalesOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
    AscendC::GmFree((void*)elasticInfo);
    AscendC::GmFree((void*)assistInfoOut);
    AscendC::GmFree((void*)expandScalesOut);
    AscendC::GmFree((void*)xActiveMask);
    AscendC::GmFree((void*)expertScales);
}

//MoeDistributeDispatchA2 test do dispatch int8 quant kernel
TEST_F(MoeDistributeDispatchV2Test, MoeDistributeDispatchV2Test2000001002)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tilingData->moeDistributeDispatchV2Info.epWorldSize = 8;
    tilingData->moeDistributeDispatchV2Info.tpWorldSize = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.epRankId = 0;
    tilingData->moeDistributeDispatchV2Info.tpRankId = 0;   //针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.expertShardType = 0;
    tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.moeExpertNum = 8;
    tilingData->moeDistributeDispatchV2Info.quantMode = 2;
    tilingData->moeDistributeDispatchV2Info.globalBs = 64;
    tilingData->moeDistributeDispatchV2Info.bs = 8;
    tilingData->moeDistributeDispatchV2Info.k = 7;
    tilingData->moeDistributeDispatchV2Info.h = 7168;
    tilingData->moeDistributeDispatchV2Info.aivNum = 40;
    tilingData->moeDistributeDispatchV2Info.isTokenMask= false;
    tilingData->moeDistributeDispatchV2Info.totalUbSize = 196352;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchV2Warrper = [] (
        GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
        GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
        GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
        GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_dispatch_v2<false, TILINGKEY_PERTOKEN_QUANT, false, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE, TILINGKEY_TPL_A3>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
                expandScalesOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeDispatchV2Warrper, 40, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, 
                expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, expandScalesOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
    AscendC::GmFree((void*)elasticInfo);
    AscendC::GmFree((void*)assistInfoOut);
    AscendC::GmFree((void*)expandScalesOut);
    AscendC::GmFree((void*)xActiveMask);
    AscendC::GmFree((void*)expertScales);
}

//MoeDistributeDispatchA2 test do dispatch int8 quant kernel with smooth scale
TEST_F(MoeDistributeDispatchV2Test, MoeDistributeDispatchV2Test2000001012)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeDistributeDispatchV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeDistributeDispatchV2TilingData *tilingData = reinterpret_cast<MoeDistributeDispatchV2TilingData*>(tiling);
    tilingData->moeDistributeDispatchV2Info.epWorldSize = 8;
    tilingData->moeDistributeDispatchV2Info.tpWorldSize = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.epRankId = 0;
    tilingData->moeDistributeDispatchV2Info.tpRankId = 0;   //针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.expertShardType = 0;
    tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum = 0;//针对A2传递默认值 0
    tilingData->moeDistributeDispatchV2Info.moeExpertNum = 8;
    tilingData->moeDistributeDispatchV2Info.quantMode = 1;//
    tilingData->moeDistributeDispatchV2Info.globalBs = 64;
    tilingData->moeDistributeDispatchV2Info.bs = 8;
    tilingData->moeDistributeDispatchV2Info.k = 7;
    tilingData->moeDistributeDispatchV2Info.h = 7168;
    tilingData->moeDistributeDispatchV2Info.aivNum = 40;//??
    tilingData->moeDistributeDispatchV2Info.isTokenMask= false;
    tilingData->moeDistributeDispatchV2Info.totalUbSize = 192 * 1024;//??

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertIds = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandXOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dynamicScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandIdxOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertTokenNumsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *epSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *tpSendCountsOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *elasticInfo = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *assistInfoOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expandScalesOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    auto moeDistributeDispatchV2Warrper = [] (
        GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
        GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
        GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
        GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_distribute_dispatch_v2<false, TILINGKEY_PERTOKEN_QUANT, true, TILINGKEY_NO_FULLMESH, TILINGKEY_TPL_MTE, TILINGKEY_TPL_A3>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
                expandScalesOut, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeDistributeDispatchV2Warrper, 40, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, 
                expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, expandScalesOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)expandXOut);
    AscendC::GmFree((void*)dynamicScalesOut);
    AscendC::GmFree((void*)expandIdxOut);
    AscendC::GmFree((void*)expertTokenNumsOut);
    AscendC::GmFree((void*)epSendCountsOut);
    AscendC::GmFree((void*)tpSendCountsOut);
    AscendC::GmFree((void*)elasticInfo);
    AscendC::GmFree((void*)assistInfoOut);
    AscendC::GmFree((void*)expandScalesOut);
    AscendC::GmFree((void*)xActiveMask);
    AscendC::GmFree((void*)expertScales);
}
