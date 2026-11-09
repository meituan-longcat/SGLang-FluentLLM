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
* \file test_aclnn_moe_distribute_dispatch_v2.cpp
* \brief
*/

#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
#include "aclnnop/aclnn_moe_distribute_combine_v2.h"


#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream dispatchV2Stream;
    aclrtStream combineV2Stream;
    aclrtContext context;
};

const uint32_t MACHINE_NUM = 1;
const char* rank_table_file = std::getenv("RANK_TABLE_FILE");
const char* first_rank_id = std::getenv("FIRST_RANK_ID");
const char* env_dev_num = std::getenv("ENV_DEV_NUM");

const uint32_t EP_WORLD_SIZE = (!first_rank_id) ? 2 : 16;
const uint32_t TP_WORLD_SIZE = (!first_rank_id) ? 1 : 0;
const uint32_t DEV_NUM = (!first_rank_id) ? EP_WORLD_SIZE * TP_WORLD_SIZE : EP_WORLD_SIZE;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, 
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
    );
    return 0;
}

int launchOneThreadDispatchV2AndCombineV2(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    if (!rank_table_file && !first_rank_id) {
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
    }
    LOG_PRINT(
        "[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
        args.rankId, hcomEpName, hcomTpName, args.dispatchV2Stream, args.combineV2Stream, args.context
    );

    // 设置场景
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 1;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    if (!rank_table_file && !first_rank_id) {
        sharedExpertNum = 1;
        sharedExpertRankNum = 1;
    } 
    int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
    int64_t quantMode = 0;
    int64_t globalBS = BS * EP_WORLD_SIZE;
    int64_t expertTokenNumsType = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        A = globalBS / sharedExpertRankNum;
    } else { 
        // Moe专家卡
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        A = globalBS * (localExpertNum < K ? localExpertNum : K);
    }
    std::string commAlg = "";

    /* 根据当前场景，构造device侧输入输出变量*/
    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandXDeviceAddr = nullptr;
    void *dynamicScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    void *tpRecvCountsDeviceAddr = nullptr;
    void *expandScalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *expertTokenNums = nullptr;
    aclTensor *epRecvCounts = nullptr;
    aclTensor *tpRecvCounts = nullptr;
    aclTensor *expandScales = nullptr;
    
    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> expandXShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A, H};
    std::vector<int64_t> dynamicScalesShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A};
    std::vector<int64_t> expandIdxShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * localExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1};
    std::vector<int64_t> expandScalesShape{A};

    long long xShapeSize = GetShapeSize(xShape);
    long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
    long long scalesShapeSize = GetShapeSize(scalesShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

    // 构造host侧变量
    std::vector<op::fp16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        // 每个token发给moe专家{0, 1, ... k - 1}
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> scalesHostData(scalesShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);  
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); 
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    /* 声明算子执行必需变量 */
    uint64_t dispatchV2WorkspaceSize = 0;
    aclOpExecutor *dispatchV2Executor = nullptr;
    void *dispatchV2WorkspaceAddr = nullptr;

    uint64_t combineV2WorkspaceSize = 0;
    aclOpExecutor *combineV2Executor = nullptr;
    void *combineV2WorkspaceAddr = nullptr;   

    /* 依次执行dispatchV2及combineV2算子 */
    // 调用dispatchV2算子第一阶段接口
    ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, 
        (quantMode > 0 ? scales : nullptr), nullptr, 
        expertScales, 
        hcomEpName, EP_WORLD_SIZE, args.epRankId,
        moeExpertNum, hcomTpName, TP_WORLD_SIZE,
        args.tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBS,
        expertTokenNumsType, commAlg.c_str(),
        expandX, dynamicScales,
        expandIdx, expertTokenNums,
        epRecvCounts, tpRecvCounts,
        expandScales, &dispatchV2WorkspaceSize,
        &dispatchV2Executor
    );
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
    );
    // 根据dispatchV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (dispatchV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用dispatchV2算子第二阶段接口
    ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor, args.dispatchV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2 failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
    );

    // 调用combineV2算子第一阶段接口
    ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType,
        commAlg.c_str(), x, &combineV2WorkspaceSize, &combineV2Executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
    );
    // 根据combineV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (combineV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用combineV2算子第二阶段接口
    ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2 failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
    CHECK_RET(
        ret == ACL_SUCCESS, 
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
    );

    LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributecombineV2 execute successfully.\n", args.rankId);

    // 释放device资源
    if (dispatchV2WorkspaceSize > 0) {
        aclrtFree(dispatchV2WorkspaceAddr);
    }
    if (combineV2WorkspaceSize > 0) {
        aclrtFree(combineV2WorkspaceAddr);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (scales != nullptr) {                
        aclDestroyTensor(scales);
    }
    if (expertScales != nullptr) {
        aclDestroyTensor(expertScales);
    }
    if (expandX != nullptr) {
        aclDestroyTensor(expandX);
    }
    if (dynamicScales != nullptr) {  
        aclDestroyTensor(dynamicScales);
    }
    if (expandIdx != nullptr) {
        aclDestroyTensor(expandIdx);
    }
    if (expertTokenNums != nullptr) {     
        aclDestroyTensor(expertTokenNums);
    }
    if (epRecvCounts != nullptr) {
        aclDestroyTensor(epRecvCounts);
    }
    if (tpRecvCounts != nullptr) {
        aclDestroyTensor(tpRecvCounts);
    }   
    if (expandScales != nullptr) {         
        aclDestroyTensor(expandScales);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (dynamicScalesDeviceAddr != nullptr) {
        aclrtFree(dynamicScalesDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (expertTokenNumsDeviceAddr != nullptr) {
        aclrtFree(expertTokenNumsDeviceAddr);
    }
    if (epRecvCountsDeviceAddr != nullptr) {
        aclrtFree(epRecvCountsDeviceAddr);
    }
    if (expandScalesDeviceAddr != nullptr) {
        aclrtFree(expandScalesDeviceAddr);
    }
    if (tpRecvCountsDeviceAddr != nullptr) {
        aclrtFree(tpRecvCountsDeviceAddr);
    }

    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    aclrtDestroyStream(args.dispatchV2Stream);
    aclrtDestroyStream(args.combineV2Stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    
    return 0;
}

int run_example_on_A2(int rankId, const char* RANK_TABLE_FILE, const char* FIRST_RANK_ID)
{
    Args args;
    aclrtStream dispatchV2Stream;
    aclrtStream combineV2Stream;
    aclrtContext context;

    int ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret));
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d", ret));
    ret = aclrtCreateStream(&dispatchV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));
    ret = aclrtCreateStream(&combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));

    int first_rank_id = std::stoi(std::string(FIRST_RANK_ID));
    HcclComm hcclComm = nullptr;
    int rank_id = rankId + first_rank_id;
    ret = HcclCommInitClusterInfo(RANK_TABLE_FILE, rank_id, &hcclComm);
    if (ret != HCCL_SUCCESS) {
        std::cout << "[ERROR] HCCL CommInitClusterInfo failed. ret = " << ret << std::endl;
        return ret;
    }
    std::cout << "[INFO] HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << DEV_NUM
            << ", hcclComm:" << hcclComm << std::endl;

    args.rankId = rankId;
    args.epRankId = rankId;
    args.tpRankId = 0;
    args.hcclEpComm = hcclComm;
    args.dispatchV2Stream = dispatchV2Stream;
    args.combineV2Stream = combineV2Stream;
    args.context = context;

    launchOneThreadDispatchV2AndCombineV2(args);
    return 0;
}

int run_example_on_A3A5()
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream dispatchV2Stream[DEV_NUM];
    aclrtStream combineV2Stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&combineV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
        }
    }
    // 初始化ep通信域，ep = 8 {0,2,4,6,8,10,12,14} {1,3,5,7,9,11,13,15}.
    HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
    for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d\n", tpId, ret); return ret
        );
    }

    int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
        }
    }
    // 初始化tp通信域，tp = 2 {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,13} {14,15}.
    HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d\n", epId, ret); return ret
        );
    }

    Args args[DEV_NUM];
    // 各线程调用各卡执行算子
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        uint32_t epRankId = rankId / TP_WORLD_SIZE;
        uint32_t tpRankId = rankId % TP_WORLD_SIZE;

        args[rankId].rankId = rankId;
        args[rankId].epRankId = epRankId;
        args[rankId].tpRankId = tpRankId;
        args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
        args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
        args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
        args[rankId].combineV2Stream = combineV2Stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return 0;
}


int main(int argc, char *argv[])
{
    const char* env_var_name = "RANK_TABLE_FILE and FIRST_RANK_ID";
    if (!env_dev_num) {
        LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
        LOG_PRINT("[WARNING] For details related to ENV_DEV_NUM, see aclnnMoeDistributeDispatchV2.md.\n");
        return 0;
    }
    int actual_env_dev_num = std::stoi(std::string(env_dev_num));
    if (actual_env_dev_num < DEV_NUM) {
        LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
        return 0;
    }
    if (!rank_table_file && !first_rank_id) {
        LOG_PRINT("[INFO] %s are not identified and example on <Atlas A3> will be executed!\n", env_var_name);
        int ret = run_example_on_A3A5();
    }
    else if (rank_table_file && !first_rank_id) {
        LOG_PRINT("[INFO] %s are not identified and example on <Atlas A5> will be executed!\n", env_var_name);
        int ret = run_example_on_A3A5();
    }
    else if (rank_table_file && first_rank_id) {
        LOG_PRINT("[INFO] %s are identified and example on <Atlas A2> will be executed!\n", env_var_name);
        uint32_t single_machine_dev_num = EP_WORLD_SIZE / MACHINE_NUM;
        std::vector<std::unique_ptr<std::thread>> threads(single_machine_dev_num);
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
        for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
            threads[rankId] = std::make_unique<std::thread>([rankId,&ret]()
            {
                int ret = run_example_on_A2(rankId, rank_table_file, first_rank_id);
            });
        }
        for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
            threads[rankId]->join();
        }
        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
    }
    else {
        LOG_PRINT("[WARNING] Please check whether %s are set correctly.\n", env_var_name);
    }

    return 0;
}