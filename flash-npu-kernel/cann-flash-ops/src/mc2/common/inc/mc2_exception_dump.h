/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_WIN_DUMP_H
#define MC2_WIN_DUMP_H

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl_dump.h"
#include "kernel/moe_distribute_comm_ctx.h"
#include "mc2_log.h"
#include "tiling/mc2_tiling_utils.h"
#include "mc2_gen_task_ops_utils.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>

using ops::Mc2GenTaskOpsUtils;
using ops::NPUARCH_A5;

namespace Mc2Exception {

const std::string OP_NAME = "Mc2Exception";
const uint32_t WIN_SIZE = 1024U * 1024U;
const uint32_t MS_WIDTH = 3U;
const uint32_t MS_PER_S = 1000U;
const mode_t FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

using HcclOpParam = HcclCombinOpParam;
inline std::string GetTimestampWithMilliseconds()
{
    auto now = std::chrono::system_clock::now();
    time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_utc;
    gmtime_r(&now_time_t, &tm_utc);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % MS_PER_S;

    std::stringstream ss;
    ss << std::put_time(&tm_utc, "%Y%m%d%H%M%S") << std::setfill('0') << std::setw(MS_WIDTH) << ms.count();
    return ss.str();
}

inline std::string GenDumpFileName(aclrtExceptionInfo *args, const char *op)
{
    std::stringstream ss;

    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(args);
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(args);
    std::string ts = GetTimestampWithMilliseconds();

    ss << "mc2_exception_info_" << op << "." << streamId << "." << taskId << "." << ts;

    return ss.str();
}

inline bool IsStrEmpty(std::string str)
{
    // return if the string is null or empty or spaces
    return (str.empty() || std::all_of(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); }));
}

inline int DumpToFile(std::string dir, std::string name, uint32_t id, void *buf)
{
    // check if dir and buf valid
    if (IsStrEmpty(dir) || IsStrEmpty(name)) {
        OP_LOGE(OP_NAME, "Dump path or buf is null.");
        return -1;
    }

    std::string rankPath = dir + "/" + std::to_string(id) + "/";
    std::string path = rankPath + name;
    OP_LOGE(OP_NAME, "Start to dump file. The dump path is %s", path);

    // Open file
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, FILE_MODE);
    if (fd < 0) {
        int openErrno = errno;
        OP_LOGE(OP_NAME, "Failed to open a dump file. errno=%d(%s)", openErrno, strerror(openErrno));
        return -1;
    }

    // Write to file
    ssize_t ret = write(fd, buf, WIN_SIZE);
    if (ret < 0) {
        int writeErrno = errno;
        OP_LOGE(OP_NAME, "Failed to write a dump file. errno=%d(%s)", writeErrno, strerror(writeErrno));
        close(fd);
        return -1;
    }

    // Close file
    close(fd);
    OP_LOGE(OP_NAME, "Dump to file %s done.", path.c_str());
    return 0;
}

inline int ProcessArgs(uint64_t argsAddr, std::vector<uint8_t> &winBuf)
{
    // Get hccl context from its addr
    std::vector<uint8_t> hcclArgs(sizeof(HcclOpParam), 0);
    auto ret = aclrtMemcpy(hcclArgs.data(), sizeof(HcclOpParam), (void *)argsAddr, sizeof(HcclOpParam),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtMemcpy HcclOpParam from device to host failed. ret = %d", ret);
        return -1;
    }
    HcclOpParam* winContext = reinterpret_cast<HcclOpParam *>(hcclArgs.data());
    if (!(winContext != nullptr)) {
        OP_LOGE(OP_NAME, "Cast to winContext failed. HcclOpParam is null.");
        return -1;
    }
    OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u, rankDim=%u", winContext->rankId, winContext->rankDim);

    void* winAddr = reinterpret_cast<void *>(winContext->windowsIn[winContext->rankId]);
    if (!(winAddr != nullptr)) {
        OP_LOGE(OP_NAME, "Get win addr failed.");
        return -1;
    }

    // Get windowsIn of each rank from hccl context
    ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

inline void Mc2ExceptionImpl(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    const char* socName = aclrtGetSocName();
    if(std::strstr(socName, "Ascend950") == nullptr) {
        OP_LOGE(OP_NAME, "The soc version is %s, skip dump process", socName);
        return;
    }
    OP_LOGD(OP_NAME, "Start to handle mc2 exception and dump win info.");

    // Get addr of hccl context from ExceptionInfo
    void* devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    auto ret = aclrtGetArgsFromExceptionInfo(args, &devArgsPtr, &devArgsLen);
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed. ret=%d", ret);
        return;
    }
    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtGetDeviceIdFromExceptionInfo failed. ret=%d", ret);
        return;
    }
    OP_LOGD(OP_NAME, "Get context from args. deviceId=%u, devArgsAddr=%p, devArgsLen=%u", deviceId, devArgsPtr,
            devArgsLen);

    uint64_t argsAddr = 0;
    ret = aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return;
    }

    // Get win content
    std::vector<uint8_t> winContent(WIN_SIZE, 0);
    if (ProcessArgs(argsAddr, winContent) != 0) {
        OP_LOGE(OP_NAME, "Failed to get win content.");
        return;
    }

    // Write to bin file
    if (DumpToFile(std::string(acldumpGetPath(acldumpType::AIC_ERR_BRIEF_DUMP)), GenDumpFileName(args, op), deviceId,
                   winContent.data()) != 0) {
        OP_LOGE(OP_NAME, "Failed to get win content.");
    }
}
} // namespace Mc2Exception

#endif