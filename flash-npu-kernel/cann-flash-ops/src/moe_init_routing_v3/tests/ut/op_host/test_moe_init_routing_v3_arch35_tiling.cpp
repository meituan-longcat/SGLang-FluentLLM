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
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace{
// 固定属性
constexpr int64_t EXPERT_NUM = 256;
// 可选quant_mode
constexpr int64_t QUANT_MODE_UNQUANT = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
// 可选row_idx_type
constexpr int64_t ROW_IDX_TYPE_GATHER = 0;
constexpr int64_t ROW_IDX_TYPE_SCATTER = 1;
} //namespace

class MoeInitRoutingV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling TearDown" << std::endl;
    }
};

 static std::string A5SocInfo =
	     "{\n"
	     "  \"hardware_info\": {\n"
	     "    \"BT_SIZE\": 0,\n"
	     "    \"load3d_constraints\": \"1\",\n"
	     "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
	     "    \"Intrinsic_data_move_l12ub\": true,\n"
	     "    \"Intrinsic_data_move_l0c2ub\": true,\n"
	     "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
	     "    \"UB_SIZE\": 262144,\n"
	     "    \"L2_SIZE\": 134217728,\n"
	     "    \"L1_SIZE\": 524288,\n"
	     "    \"L0A_SIZE\": 65536,\n"
	     "    \"L0B_SIZE\": 65536,\n"
	     "    \"L0C_SIZE\": 262144,\n"
	     "    \"CORE_NUM\": 32,\n"
	     "    \"socVersion\": \"Ascend910_95\"\n"
	     "  }\n"
	     "}";

void RunSuccessTestcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity, int64_t dropPadMode, int64_t expertTokensNumType,
                          bool expertTokensNumFlag, int64_t quantMode, int64_t isInputScale, ge::DataType xDataType,
                          std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, ge::graphStatus result,
                          int64_t expectTilingKey, std::string expectTilingData, std::vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertCapacity)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunFailureTestcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity, int64_t dropPadMode, int64_t expertTokensNumType,
                          bool expertTokensNumFlag, int64_t quantMode, int64_t isInputScale, ge::DataType xDataType,
                          std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, ge::graphStatus result,
                          int64_t expectTilingKey, std::string expectTilingData, std::vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 196608, platform_ascendc::SocVersion::ASCEND950};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertCapacity)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + scale not None   1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_01)
{
    std::string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_UNQUANT, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_GATHER,
                         ge::GRAPH_SUCCESS, 1000000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + drop pad + scale None 1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_02)
{
    std::string expectTilingData = "40 1 83 27 180 192 12 -1 1 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_UNQUANT, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_SCATTER,
                         ge::GRAPH_SUCCESS, 1001000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + scale None 1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_03)
{
    std::string expectTilingData = "40 160 96 1450 180 192 12 -1 0 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_UNQUANT, 0, ge::DT_INT8, {180, 192},
                         ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 1100000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + drop pad + scale None 1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_04)
{
    std::string expectTilingData = "40 160 96 1450 180 192 12 -1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_UNQUANT, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}


// 单核 + dynamci quant + scale not None   1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_05)
{
    std::string expectTilingData = "40 1 83 27 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16793892};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + scale None 1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_06)
{
    std::string expectTilingData = "40 1 83 27 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16793892};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + drop mode + scale not None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_07)
{
    std::string expectTilingData = "40 8 60 32 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 1 256 1 256 256 256 1 256 256 6144 0 1024 37 7 4 1 7 7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 0 ";
    std::vector<size_t> expectWorkspaces = {16796976};
    RunSuccessTestcase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_SCATTER,
                         ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + drop mode + scale None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_08)
{
    std::string expectTilingData = "40 8 60 32 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 1 256 1 256 256 256 1 256 256 6144 0 1024 37 7 4 1 7 7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 0 ";
    std::vector<size_t> expectWorkspaces = {16796976};
    RunSuccessTestcase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_SCATTER,
                         ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale not None  11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_09)
{
    std::string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale None 11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_10)
{
    std::string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale not None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_11)
{
    std::string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100},
                         ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_12)
{
    std::string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100},
                         ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + bfloat16  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_13)
{
    std::string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100},
                         ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + float16 11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_14)
{
    std::string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100},
                         ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + drop mode + scale not None   1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_15)
{
    std::string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 ";
    std::vector<size_t> expectWorkspaces = {5329576};
    RunFailureTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_SCATTER, ge::GRAPH_FAILED, 1020000, expectTilingData, expectWorkspaces);
}