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

namespace MoeDistributeCombineV2 {
class MoeDistributeCombineV2Infershape : public testing::Test{
};

// infer shape with bias, success
TEST_F(MoeDistributeCombineV2Infershape, InferShape0)
{
    gert::StorageShape expandXShape = {{576, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape expandIdxShape = {{16384}, {}};
    gert::StorageShape epSendCountsShape = {{288}, {}};
    gert::StorageShape tpSendCountsShape = {{2}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeDistributeCombineV2",
        {
            {expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
            {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {epSendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
            {tpSendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
            {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expertOutputShape = {{32, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}


TEST_F(MoeDistributeCombineV2Infershape, InferDtype0)
{
    ge::DataType expandXType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType expandIdxType = ge::DT_INT32;
    ge::DataType epSendCountsType = ge::DT_INT32;
    ge::DataType tpSendCountsType = ge::DT_INT32;
    ge::DataType expertScalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(6, 1)
                    .InputDataTypes({&expandXType, &expertIdsType, &expandIdxType,
                                    &epSendCountsType, &tpSendCountsType, &expertScalesType})
                    .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeCombineV2")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}
}