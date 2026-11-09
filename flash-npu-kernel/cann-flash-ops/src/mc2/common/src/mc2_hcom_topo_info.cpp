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
 * \file mc2_hcom_topo_info.cc
 * \brief
 */

#include <cstdlib>
#include <string>
#include <dlfcn.h>
#include "log/log.h"
#include "mc2_hcom_topo_info.h"
#include "ops_legacy/op_tiling/hcom_topo_info.h"
#include "tiling/mc2_tiling_utils.h"
#ifndef BUILD_OPEN_PROJECT
#include "hcom/hcom_topo_info.h"
#endif

namespace Mc2Hcom {
const std::string HCOM_GET_COMM_FUNC_NAME = "HcomGetCommHandleByGroup";
#ifdef BUILD_OPEN_PROJECT
const std::string HCOM_GET_RANK_SIZE_EX = "HcomGetRankSizeEx";
const std::string HCOM_GET_L0_TOPO_TYPE_EX = "HcomGetL0TopoTypeEx";
#else
const std::string COMM_GET_NET_LAYERS_NAME = "CommGetNetLayers";
const std::string COMM_GET_TOPO_TYPE_NAME = "CommGetInstTopoTypeByNetLayer";
const std::string COMM_GET_SIZE_NAME = "CommGetInstSizeByNetLayer";
#endif
const std::string COMM_GET_CCL_BUFFER_SIZE_NAME = "CommGetCCLBufSizeCfg";

static const string GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE("", "Ascend home path doesn't exist.");
        return nullptr;
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl.so";
#else
    return nullptr;
#endif
    std::string fullPath = ascendPath + hcclPathPostfix;
    OP_LOGI("", "Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OP_LOGE("", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

MC2HcomTopology::MC2HcomTopology(const char *libPath)
{
    handle_ = dlopen(libPath, RTLD_NOW);
    if (handle_ == nullptr) {
        OP_LOGE("", "Load lib hccl failed.");
        return;
    }

    getCommHandle_ = GetHcclLibFunc<FuncGetHandle>(handle_, HCOM_GET_COMM_FUNC_NAME);
#ifdef BUILD_OPEN_PROJECT
    getRankSizeEx_ = GetHcclLibFunc<FuncGetRankSizeEx>(handle_, HCOM_GET_RANK_SIZE_EX);
    getL0TopoTypeEx_ = GetHcclLibFunc<FuncGetL0TopoTypeEx>(handle_, HCOM_GET_L0_TOPO_TYPE_EX);
#else
    getNetLayers_ = GetHcclLibFunc<FuncGetNetLayers>(handle_, COMM_GET_NET_LAYERS_NAME);
    getTopoType_ = GetHcclLibFunc<FuncGetTopoType>(handle_, COMM_GET_TOPO_TYPE_NAME);
    getInstSize_ = GetHcclLibFunc<FuncGetInstSize>(handle_, COMM_GET_SIZE_NAME);
#endif
    getCclBufferSize_ = GetHcclLibFunc<FuncGetCclBufferSize>(handle_, COMM_GET_CCL_BUFFER_SIZE_NAME);

#ifdef BUILD_OPEN_PROJECT
    if (getCommHandle_ == nullptr || getRankSizeEx_ == nullptr || getL0TopoTypeEx_ == nullptr ||
        getCclBufferSize_ == nullptr) {
        OP_LOGE("", "Lib load new topo functions failed.");
        getCommHandle_ = nullptr;
        getCclBufferSize_ = nullptr;
        getRankSizeEx_ = nullptr;
        getL0TopoTypeEx_ = nullptr;
        return;
    }
#else
    if (getCommHandle_ == nullptr || getNetLayers_ == nullptr || getTopoType_ == nullptr || getInstSize_ == nullptr ||
        getCclBufferSize_ == nullptr) {
        OP_LOGE("", "Lib load new topo functions failed.");
        getCommHandle_ = nullptr;
        getNetLayers_ = nullptr;
        getTopoType_ = nullptr;
        getInstSize_ = nullptr;
        getCclBufferSize_ = nullptr;
        return;
    }
#endif

    OP_LOGI("", "Init MC2HcomTopoLogy Success.");
}

MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    static const char *libPath = GetLibPath().c_str();
    static MC2HcomTopology loader(libPath);
    return loader;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const
{
    if (getCommHandle_ == nullptr) {
        OP_LOGE("", "Failed to get comm handle, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCommHandle_(group, commHandle));
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const
{
    if (getRankSizeEx_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getRankSizeEx_(group, ranksize, flag));
}

HcclResult MC2HcomTopology::CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const
{
    if (getL0TopoTypeEx_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getL0TopoTypeEx_(group, topoType, flag));
}
#else
HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t **netLayer, uint32_t *netLayerNum) const
{
    if (getNetLayers_ == nullptr) {
        OP_LOGE("", "Failed to get net layers, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getNetLayers_(comm, netLayer, netLayerNum));
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType) const
{
    if (getTopoType_ == nullptr) {
        OP_LOGE("", "Failed to get topo type, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getTopoType_(comm, netLayer, topoType));
}

HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum) const
{
    if (getInstSize_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getInstSize_(comm, netLayer, rankNum));
}
#endif

HcclResult MC2HcomTopology::CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const
{
    if (getCclBufferSize_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCclBufferSize_(comm, cclBufferSize));
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm)
{
    if (group == nullptr || cclBufferSize == nullptr || hcclComm == nullptr) {
        OP_LOGE("", "Group or rank num or hcclComm is nullptr.");
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetCommHandleByGroup(group, hcclComm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGI("", "get nullptr comm handle.");
        *hcclComm = nullptr;
        return HCCL_SUCCESS;
    }
    ret = GetInstance().CallCommGetCCLBufSizeCfg(*hcclComm, cclBufferSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get buffer size.");
        return ret;
    }
    OP_LOGI("", "cclBufferSize is %lu", *cclBufferSize);
    return HCCL_SUCCESS;
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t *cclBufferSize)
{
    if (ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize) != ge::GRAPH_SUCCESS) {
        OP_LOGD("", "Get winSize from GetGroupLocalWindowSize=%lu", *cclBufferSize);
        *cclBufferSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
        return HCCL_SUCCESS;
    }
    OP_LOGD("", "Get winSize from GetMaxWindowSize=%lu", *cclBufferSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    if (group == nullptr || rankNum == nullptr) {
        OP_LOGE("", "Group or rank num is nullptr.");
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetRankSizeEx(group, rankNum, COMM_IS_NOT_SET_DEVICE);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get rank size.");
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    if (group == nullptr || topoType == nullptr) {
        OP_LOGE("", "Group or topo type is nullptr.");
        return HCCL_E_PTR;
    }
    CommTopo topoRet;
    HcclResult ret = GetInstance().CallHcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get topo type.");
        return ret;
    }
    *topoType = static_cast<uint32_t>(topoRet);
    return HCCL_SUCCESS;
}
#else
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t* cclBufferSize)
{
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE("", "cclBufferSize not set.");
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    int64_t rankSize = *rankNum;
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupRankSize(group, rankSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return HCCL_E_NOT_FOUND;
    }
    *rankNum = static_cast<uint32_t>(rankSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    ge::HcomTopoInfo::TopoInfo topoInfo;
    if (!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(group, topoInfo)) {
        OP_LOGW("", "GroupTopoInfo not set.");
        *topoType = COMM_ALG_DEFAULT;
        return HCCL_E_NOT_FOUND;
    }
    *topoType = topoInfo.topo_level_descs[static_cast<int32_t>(ge::HcomTopoInfo::TopoLevel::L0)].comm_sets;
    OP_LOGD("", "comm_sets from TopoInfo is %u, COMM_MESH is %u", *topoType, COMM_MESH);
    return HCCL_SUCCESS;
}
#endif
}  // namespace Mc2Hcom
