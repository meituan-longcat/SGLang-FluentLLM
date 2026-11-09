/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

using namespace std;

namespace MoeDistributeCombineV2UT {
class MoeDistributeCombineV2Arch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeDistributeCombineV2Arch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeDistributeCombineV2Arch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineV2Arch32TilingTest, SharedExpertX0)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    // check ge::GRAPH_FAILED
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, SharedExpertX1)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    // check ge::GRAPH_SUCCESS
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 32UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, SharedExpertXThreeDims)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 7168}, {2, 4, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    // check ge::GRAPH_SUCCESS
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 32UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, Test0)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 32UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, Test1)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, Test2)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, Test3)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, EpWorldSize384)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(384)},
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
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, EpWorldSize72)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
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
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, XActivateMask2dims)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, ElasticInfo)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, Moepp)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{148}, {148}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6, 7168}, {6, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, CopyExpertWithoutOriX)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{148}, {148}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6, 7168}, {6, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, ConstExpertWithoutOriX)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{148}, {148}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6, 7168}, {6, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

//===============================================A2====================================================
TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgEmpty)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgEmptyWithEnv)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgFullmesh)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("fullmesh")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgFullmeshWithEnv)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("fullmesh")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgHierarchy)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("hierarchy")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 8UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgError)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("error")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgEmptyWithEnvCommint8)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineV2Arch32TilingTest, A2CommalgHierarchyCommint8)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("hierarchy")},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

}  // moe_distribute_combine_v2_ut