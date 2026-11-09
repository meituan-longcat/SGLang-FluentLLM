/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_init_routing_v3.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_init_routing_v3_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_init_routing_v3_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_init_routing_v3_test TearDown" << endl;
    }
};


TEST_F(l2_moe_init_routing_v3_test, Ascend910B2_moe_init_routing_v3)
{
    auto x = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expertIdx = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto expandedXOut = TensorDesc({20, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t activeNum = 20;
    int64_t expertCapacity = -1;
    int64_t expertNum = 32;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = -1;
    int64_t rowIdxType = 1;

    auto ut = OP_API_UT(aclnnMoeInitRoutingV3, INPUT(x, expertIdx, nullptr, nullptr, activeNum, expertCapacity, expertNum, dropPadMode,
                                                      expertTokensNumType, expertTokensNumFlag, quantMode, activeExpertRange, rowIdxType),
                        OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}