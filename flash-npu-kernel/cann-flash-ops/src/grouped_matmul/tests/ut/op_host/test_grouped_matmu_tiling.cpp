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
 * \file test_grouped_matmul.cpp
 * \brief
 */

#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/grouped_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_grouped_matmul_utils.h"

using namespace std;
using namespace ge;

class GroupedMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulTiling TearDown" << std::endl;
    }
};

namespace {
size_t FindUniqueSerializedInt32(const std::vector<uint8_t> &buffer, int32_t value)
{
    size_t result = buffer.size();
    for (size_t offset = 0; offset + sizeof(value) <= buffer.size(); offset += alignof(int32_t)) {
        int32_t candidate = 0;
        std::memcpy(&candidate, buffer.data() + offset, sizeof(candidate));
        if (candidate != value) {
            continue;
        }
        if (result != buffer.size()) {
            return buffer.size();
        }
        result = offset;
    }
    return result;
}

int32_t ReadSerializedInt32(const uint8_t *buffer, size_t offset)
{
    int32_t value = 0;
    std::memcpy(&value, buffer + offset, sizeof(value));
    return value;
}

gert::TilingContextPara MakeA4W4RoutedDecodeTilingContext(
    size_t k, size_t n, optiling::GMMCompileInfo *compileInfo)
{
    constexpr size_t M = 128;
    constexpr size_t E = 7;
    return gert::TilingContextPara(
        "GroupedMatmul",
        {
            {{{M, k}, {M, k}}, ge::DT_INT4, ge::FORMAT_ND},
            {{{E, k, n}, {E, k / 64, n / 16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{E, 1, n}, {E, 1, n}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{M, n}, {M, n}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        },
        compileInfo);
}

void ExpectA4W4BaseTile(size_t k, size_t n, uint32_t baseM, uint32_t baseN, uint32_t baseK)
{
    optiling::GMMCompileInfo compileInfo = {
        24,
        48,
        196608,
        524288,
        196608,
        131072,
        65536,
        65536,
        platform_ascendc::SocVersion::ASCEND910B,
        NpuArch::DAV_2201,
    };
    auto context = MakeA4W4RoutedDecodeTilingContext(k, n, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));

    // TilingDef's raw-buffer constructor does not populate the getter cache in
    // the UT runtime. Locate the three fields from a serialized layout object,
    // then inspect the actual raw tiling data produced by ExecuteTiling.
    constexpr int32_t BASE_M_SENTINEL = 0x13572468;
    constexpr int32_t BASE_N_SENTINEL = 0x24681357;
    constexpr int32_t BASE_K_SENTINEL = 0x31415926;
    optiling::GMMTilingData layout;
    layout.mmTilingData.set_baseM(BASE_M_SENTINEL);
    layout.mmTilingData.set_baseN(BASE_N_SENTINEL);
    layout.mmTilingData.set_baseK(BASE_K_SENTINEL);
    std::vector<uint8_t> serializedLayout(layout.GetDataSize());
    layout.SaveToBuffer(serializedLayout.data(), serializedLayout.size());

    const size_t baseMOffset = FindUniqueSerializedInt32(serializedLayout, BASE_M_SENTINEL);
    const size_t baseNOffset = FindUniqueSerializedInt32(serializedLayout, BASE_N_SENTINEL);
    const size_t baseKOffset = FindUniqueSerializedInt32(serializedLayout, BASE_K_SENTINEL);
    ASSERT_LE(baseMOffset + sizeof(int32_t), tilingInfo.tilingDataSize);
    ASSERT_LE(baseNOffset + sizeof(int32_t), tilingInfo.tilingDataSize);
    ASSERT_LE(baseKOffset + sizeof(int32_t), tilingInfo.tilingDataSize);

    EXPECT_EQ(ReadSerializedInt32(tilingInfo.tilingData.get(), baseMOffset), static_cast<int32_t>(baseM));
    EXPECT_EQ(ReadSerializedInt32(tilingInfo.tilingData.get(), baseNOffset), static_cast<int32_t>(baseN));
    EXPECT_EQ(ReadSerializedInt32(tilingInfo.tilingData.get(), baseKOffset), static_cast<int32_t>(baseK));
}
}  // namespace

TEST_F(GroupedMatmulTiling, test_tiling_a4w4_routed_decode_gmm1_uses_base_n_256_without_tuning_config)
{
    ExpectA4W4BaseTile(8192, 4096, 128, 256, 256);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4_routed_decode_gmm2_uses_base_n_256_without_tuning_config)
{
    ExpectA4W4BaseTile(2048, 8192, 128, 256, 256);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4_non_routed_shape_keeps_base_n_512_without_tuning_config)
{
    ExpectA4W4BaseTile(2048, 4096, 64, 512, 128);
}

// ANTIQUANT_A8W4
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_autotiling_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096, 1})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "1 0 ";
    std::vector<size_t> expectWorkspaces = {67108864}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,250);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_autotiling_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},  //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096, 1})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "1 0 ";
    std::vector<size_t> expectWorkspaces = {67108864}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,250);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_pergroup_antiqunt_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {121634816}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_pergroup_antiqunt_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},  //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {121634816}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_perchannel_antiqunt_1aic1aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33562624}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_perchannel_antiqunt_1aic1aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33562624}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_msd_vec_1aic2aiv)
{
    size_t M = 1024;
    size_t K = 512;
    size_t N = 32768;
    size_t E = 256;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_vec_1aic2aiv)
{
    size_t M = 1024;
    size_t K = 512;
    size_t N = 32768;
    size_t E = 256;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_msd_api_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_msd_api_1aic1aiv)
{
    size_t M = 128;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_api_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_api_1aic1aiv)
{
    size_t M = 128;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_api_withoffset_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_FLOAT, ge::FORMAT_ND},          //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// ANTIQUANT_A16W4
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_perchannel_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_perchannel_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_pergroup_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_pergroup_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_perchannel_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_perchannel_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_pergroup_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_pergroup_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_perchannel_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_perchannel_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_pergroup_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_pergroup_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}


// ANTIQUANT_A16W8
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_antiquant_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_antiquant_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_msd_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23134208}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_msd_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23134208}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_antiquant_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_antiquant_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_msd_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {27328512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_msd_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {27328512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_antiquant_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_antiquant_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// QUANT_A8W8O16
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 8192;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic1aiv_static)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic1aiv_static)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic1aiv_sparse)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic1aiv_sparse)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}



// QUANT_A8W8O16定轴算法
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_fixed_axis)
{
    size_t M = 345;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256, 0, -1})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        true //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {26669056}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}


// QUANT_A4W4
TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_notrans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 580;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4obf16_notrans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 580;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_trans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 640;//must align to 64
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4obf16_trans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 640;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_trans_dynamic_tiling_1aic2aiv)
{
    size_t M = 8000;
    size_t K = 1024;
    size_t N = 2048;//must align to 64
    size_t E = 80;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4obf16_trans_dynamic_tiling_1aic2aiv)
{
    size_t M = 8000;
    size_t K = 1024;
    size_t N = 2048;
    size_t E = 80;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// QUANT_A4W4优化
TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_optimize)
{
    size_t M = 256;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// QUANT_A8W8O8
TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_notrans_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_transw_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_notrans_static_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_transw_static_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}


TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_notrans_sparse_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_transw_sparse_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}




// QUANT_A8W8O32
TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_notrans_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_transw_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_notrans_1aic_sparse)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_transw_1aic_sparse)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// 非量化场景用例
TEST_F(GroupedMatmulTiling, test_tiling_fp16_notrans_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_FLOAT16, // D_T_A
        GMM_TPL_FLOAT16, // D_T_B
        GMM_TPL_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_bf16_notrans_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_BF16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_BF16, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp32_notrans_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT, // D_T_A
        DT_FLOAT, // D_T_B
        DT_FLOAT, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp16_transw_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_FLOAT16, // D_T_A
        GMM_TPL_FLOAT16, // D_T_B
        GMM_TPL_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_bf16_transw_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_BF16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_BF16, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp32_transw_1aic)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT, // D_T_A
        DT_FLOAT, // D_T_B
        DT_FLOAT, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp16_transx_1aic1aiv)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_FLOAT16, // D_T_A
        GMM_TPL_FLOAT16, // D_T_B
        GMM_TPL_FLOAT16, // D_T_Y
        1, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_bf16_transx_1aic1aiv)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_BF16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_BF16, // D_T_B
        DT_BF16, // D_T_Y
        1, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp32_transx_1aic1aiv)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT, // D_T_A
        DT_FLOAT, // D_T_B
        DT_FLOAT, // D_T_Y
        1, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_weightnz_notrans)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_weightnz_wtrans)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K/32, N/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 17L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_weightnz_pertensor)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_weightnz_illegal_dtype)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_weightnz_illegal_wshape)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

//其余ut（历史出过问题的补充ut）

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_INT8, // D_T_A
        GMM_TPL_INT8, // D_T_B
        GMM_TPL_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {18350080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_illegal_group_num_size)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 1205;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_weightnz_tc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_weightnz_kc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_tc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_kc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_illegal_tt_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{E}, {E}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_illegal_acttype_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_illegal_xdtype_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_tc_case_n_equal_1)
{
    size_t M = 190;
    size_t K = 15;
    size_t N = 1;
    size_t E = 1;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_kerneltype_1_corenum_check_fail_nonzero)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        50,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_kerneltype_1_corenum_check_fail_zero)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        0,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_3510)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 1L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_3510)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 0L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_weightnz_notrans)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 0L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_3510_bias_not_int32_error)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_3510_bias_not_int32_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_3510_scale_not_perchannel_error)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, 1}, {E, 1}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_weightnz_error)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w4_950_aic_aiv_invalid_ratio)
{
    size_t M = 512;
    size_t K = 1024;
    size_t N = 1024;
    size_t E = 2;
    size_t quantGroupNum = K / 256;

    optiling::GMMCompileInfo compileInfo = {
        24,                                      // aicNum
        24,                                      // aivNum
        196608,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        131072,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                     // x
            {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},         // weight
            {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // bias
            {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                                   // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                            // offset
            {{{E, quantGroupNum, N}, {E, quantGroupNum, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // antiquantScale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                          // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                          // groupList
            {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
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
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256})},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
