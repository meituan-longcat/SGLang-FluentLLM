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

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define private public
#include "platform/platform_info.h"

class GroupedMatmulInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulProto TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_0)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
        {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
        {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_group_size)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                //x
        {{{E, N, K}, {E, N, K}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},    //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                         //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N, K/16}, {E, N, K/16}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},            //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_pertokenscale_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                //x
        {{{E, N, K}, {E, N, K}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},    //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                         //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N, K/32}, {E, N, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{M, 1, K/32}, {M, 1, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_group_type)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND},   //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},              //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},            //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},              //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},              //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_antiquant_scale_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, 1, N}, {E, 1, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_mx_antiquant_scale_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //x
        {{{E, K, N/8}, {E, K, N/8}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},      //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                  //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_a16w4_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},  //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

// ==================== Multi/Multi/Multi Scenario Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_invalid_x_dim_num)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmul",
        {
            // input info - x has 7 dims (invalid)
            {{{M, K, 1, 1, 1, 1, 1}, {M, K, 1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x[0] 7 dims
            {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},                                  // weight[0]
            {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                     // bias[0]
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // scale
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // offset
            {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                     // antiquantScale[0]
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                     // antiquantOffset
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                                       // groupList
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // perTokenScale
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== S8S4 with invalid groupsize ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_s8s4_invalid_groupsize)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 640; // Not divisible by 128/192/256/512
    size_t N = 512;
    size_t E = 2;
    size_t groupSize = 160; // Invalid group size
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmul",
        {
            // input info - S8S4 with invalid groupsize
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                  // x
            {{{E, N, K / 8}, {E, N, K / 8}}, ge::DT_INT32, ge::FORMAT_ND},                   // weight
            {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // bias
            {{{E, N, K / groupSize}, {E, N, K / groupSize}}, ge::DT_FLOAT, ge::FORMAT_ND},   // scale
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // offset
            {{{E, N, K / groupSize}, {E, N, K / groupSize}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // antiquantScale
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // antiquantOffset
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                                       // groupList
            {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // perTokenScale
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
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
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== A16W4 INT32 weight format ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_a16w4_int32_weight)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 512;
    size_t N = 512;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmul",
        {
            // input info - A16W4 with INT32 packed weight
            {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // x
            {{{E, N, K / 8}, {E, N, K / 8}}, ge::DT_INT32, ge::FORMAT_ND}, // weight (4bit in int32)
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // scale
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // offset
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // antiquantScale
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // antiquantOffset
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                     // groupList
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // perTokenScale
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
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
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
