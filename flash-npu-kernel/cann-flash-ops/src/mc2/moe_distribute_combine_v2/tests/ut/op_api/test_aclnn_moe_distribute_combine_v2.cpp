/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_moe_distribute_combine_v2.h"

#include <array>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"


using namespace op;
using namespace std;

namespace MoeDistributeCombineV2 {
class L2MoeDistributeCombineV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineV2Test TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeCombineV2Test, TestMoeDistributeCombineFirstApi)
{
  TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expandIdx = TensorDesc({32*8}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc tpSendCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
  TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
  TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc sharedExpertX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t epWorldSize = 288;
  int64_t tpWorldSize = 2;
  int64_t epRankId = 0;
  int64_t tpRankId = 0;
  int64_t expertShardType = 0;
  int64_t sharedExpertNum = 1;
  int64_t sharedExpertRankNum = 32;
  int64_t moeExpertNum = 256;
  int64_t globalBs = 0;
  int64_t outDtype = 0;
  int64_t commQuantMode = 0;
  int64_t groupListType = 0;

  TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMoeDistributeCombineV2, INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts,
                      xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX,
                      "test_moe_distribute_combine_ep", epWorldSize, epRankId, moeExpertNum,
                      "test_moe_distribute_combine_tp", tpWorldSize, tpRankId,
                      expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, "test"),
                                        OUTPUT(x));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}
} // MoeDistributeCombineV2