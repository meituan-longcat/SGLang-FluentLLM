/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file test_grouped_weight_quant_arch35_tiling.cpp
 * \brief Unit tests for GroupedWeightQuantBatchMatmulTiling
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_grouped_matmul_utils.h"

using namespace std;
using namespace ge;

class GroupedWeightQuantBatchMatmulTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedWeightQuantBatchMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedWeightQuantBatchMatmulTiling TearDown" << std::endl;
    }
};

// A16W4 ND场景 - BF16输入, INT4权重, 单单单模式, Split M
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_single_single_single_splitm)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 53485568;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - FP16输入, INT4权重, 单单单模式, Split M
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_fp16_nd_single_single_single_splitm)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // x
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // antiquantScale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},            // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 53485568;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - 带antiquantOffset
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_with_offset)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantOffset (存在)
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 53485600;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - INT8输入, INT4权重, NZ格式
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_int8_int4)
{
    size_t M = 256;
    size_t K = 512;
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    // NZ格式: (N1, K1, K0, N0)，其中N0=16, K0=32 (对于int4)
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 128, N}, {E, K / 128, N}}, ge::DT_UINT64, ge::FORMAT_ND}, // antiquantScale (groupsize=128)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                          // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 37756929;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// 多多多模式测试 - A16W4 ND
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_multi_multi_multi)
{
    size_t M = 128;
    size_t K = 512;
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND}, // x
            {{{K, N}, {K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // offset
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},         // antiquantOffset
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 53485568;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - 转置weight
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_transb)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND}, // weight (transposed)
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 36773888;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - groupsize=192
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_groupsize_192)
{
    size_t M = 256;
    size_t K = 384; // 192 * 2
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 192, N}, {E, K / 192, N}}, ge::DT_UINT64, ge::FORMAT_ND}, // antiquantScale (groupsize=192)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                          // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 38805505;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - No Split模式
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_no_split)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND}, // x
            {{{K, N}, {K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // offset
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},         // antiquantOffset
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // perTokenScale
        },
        {// output info
         {{{E, M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 53485568;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - FP16输出
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_fp16_output)
{
    size_t M = 256;
    size_t K = 512;
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 128, N}, {E, K / 128, N}}, ge::DT_UINT64, ge::FORMAT_ND},               // antiquantScale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                        // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 37756929;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - 大N值测试（触发尾块重切分）
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_large_n)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 8192; // 大N值
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, // 转置触发尾块重切分
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 36773888;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - groupsize=256
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_groupsize_256)
{
    size_t M = 256;
    size_t K = 512; // 256 * 2
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND}, // antiquantScale (groupsize=256)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                          // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 37756929;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}
