/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file runtime2_util.cc
 * \brief util functions for tiling runtime2
 */
#include "runtime2_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"

namespace optiling {
bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
  *workspace_size = workspace;
  return true;
}

int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end) {
  int64_t size = 1;
  for (size_t i = begin; i < end; i++) {
    size *= shape[i];
  }
  return size;
}

int64_t CeilAlign(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

int64_t GetRemainder(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = u_value % d_value;

  return res_value;
}

bool GetTilingCoreNum(const gert::TilingParseContext* context, uint32_t& core_num) {
  auto platform_info = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, platform_info, false);

  core_num = platform_info->GetCoreNum();
  OP_LOGD(context->GetNodeName(), "get tiling core num is %u", core_num);
  return true;
}

static bool IsRegBaseSocVersion(platform_ascendc::SocVersion version)
{
    // ASCEND910D or ASCEND910_95 is 4, use ASCEND910_95 when AscendC supported
    const static std::set<platform_ascendc::SocVersion> regbaseSocVersions = {
        static_cast<platform_ascendc::SocVersion>(4)};

    return regbaseSocVersions.find(version) != regbaseSocVersions.end();
}

bool IsRegbaseSocVersion(const gert::TilingParseContext& context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegBaseSocVersion(socVersion);
}

bool IsRegbaseSocVersion(const gert::TilingContext& context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegBaseSocVersion(socVersion);
}

}  // namespace optiling
