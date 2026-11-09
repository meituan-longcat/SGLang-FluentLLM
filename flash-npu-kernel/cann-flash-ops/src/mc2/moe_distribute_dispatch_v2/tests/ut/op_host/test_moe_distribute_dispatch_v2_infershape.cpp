/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "mc2_infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#define private public
#include "platform/platform_info.h"

namespace MoeDistributeDispatchV2 {
class MoeDistributeDispatchV2Infershape : public testing::Test {
};

// infer shape with bias, success
TEST_F(MoeDistributeDispatchV2Infershape, inferShape0) 
{
    gert::StorageShape expandXShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeDistributeDispatchV2",
        {
            {expandXShape, ge::DT_INT32, ge::FORMAT_ND},
            {expertIdsShape, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expertOutputShape = {{576, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MoeDistributeDispatchV2Infershape, inferDtype0)
{
    ge::DataType expandXType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(2, 6)
        .NodeAttrs({
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        })
        .InputDataTypes({&expandXType, &expertIdsType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}
}