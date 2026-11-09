/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "moe_init_routing_v3_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_init_routing_v3(uint8_t *x, uint8_t *expertIdx, uint8_t *scale,
                                                          uint8_t *offset, uint8_t *expandedX, uint8_t *expandedRowIdx,
                                                          uint8_t *expertTokensCountOrCumsum, uint8_t *expandedScale,
                                                          uint8_t *workspace, uint8_t *tiling);

class moe_init_routing_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "moe_init_routing_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_init_routing_v3_test TearDown\n" << endl;
    }
};

// UNQUANTIZED_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_8)
{
    size_t num_rows = 78;
    size_t cols = 2880;
    size_t k = 4;
    size_t expert_num = 946;
    uint64_t tilingKey = 2100000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 78 2880 4 946 float32 -1 141 0 946 0 1 1 [873,889] 1 1"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case7");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, nullptr, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py float32 1 0");
    free(path_);
}

TEST_F(moe_init_routing_v3_test, test_case_9)
{
    size_t num_rows = 97;
    size_t cols = 2880;
    size_t k = 4;
    size_t expert_num = 734;
    uint64_t tilingKey = 2100000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 97 2880 4 734 float32 -1 734 0 313 0 1 1 [0,734] 1 1"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case8");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py float32 1 1");
    free(path_);
}

TEST_F(moe_init_routing_v3_test, test_case_10)
{
    size_t num_rows = 94;
    size_t cols = 2880;
    size_t k = 4;
    size_t expert_num = 958;
    uint64_t tilingKey = 2100000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 94 2880 4 958 float32 -1 199 0 958 0 0 1 [554,650] 0 1"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case9");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py float32 0 1");
    free(path_);
}

// to do
// MOE_INIT_ROUTING_V3_SORTONECORE_GATHER_DROP
TEST_F(moe_init_routing_v3_test, test_case_11)
{
    size_t num_rows = 89;
    size_t cols = 2880;
    size_t k = 8;
    size_t expert_num = 256;
    uint64_t tilingKey = 1000100;
    uint32_t blockDim = 48;
    size_t expertCapacity = 13;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = expert_num * expertCapacity * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = expert_num * expertCapacity * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024 + blockDim * 2); // droppad是否影响
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 89 2880 8 256 float32 -1 712 13 256 1 1 1 [0,256] 0 1");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case10");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, nullptr, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER_DROP
TEST_F(moe_init_routing_v3_test, test_case_13)
{
    size_t num_rows = 55;
    size_t cols = 2880;
    size_t k = 8;
    size_t expert_num = 256;
    uint64_t tilingKey = 1020100;
    uint32_t blockDim = 48;
    size_t expertCapacity = 8;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = expert_num * cols * sizeof(float);
    size_t expandedX_FileSize = expert_num * expertCapacity * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = expert_num * expertCapacity * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024 + blockDim * 2);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 55 2880 8 256 float32 1 440 8 256 1 1 1 [0,256] 0 2"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case12");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// MOE_INIT_ROUTING_V3_SORTONECORE_QUANT_GATHER_DROP
TEST_F(moe_init_routing_v3_test, test_case_15)
{
    size_t num_rows = 76;
    size_t cols = 2880;
    size_t k = 8;
    size_t expert_num = 256;
    uint64_t tilingKey = 1010100;
    uint32_t blockDim = 48;
    size_t expertCapacity = 9;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t offset_FileSize = sizeof(float);
    size_t expandedX_FileSize = expert_num * expertCapacity * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * expertCapacity * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024 + blockDim * 2);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *offset = (uint8_t *)AscendC::GmAlloc(offset_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 76 2880 8 256 float32 0 608 9 256 1 1 1 [0,256] 0 1"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case14");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/offset.bin", offset_FileSize, offset, offset_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, offset, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)offset);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// STATIC_QUANT_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_31)
{
    size_t num_rows = 479;
    size_t cols = 2880;
    size_t k = 5;
    size_t expert_num = 38;
    uint64_t tilingKey = 2200000;
    uint32_t blockDim = 48;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = sizeof(float);
    size_t offset_FileSize = sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *offset = (uint8_t *)AscendC::GmAlloc(offset_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 479 2880 5 38 float32 0 1662 0 38 0 0 0 [10,15] 0 1");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case30");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/offset.bin", offset_FileSize, offset, offset_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, offset, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

// DYNAMIC_QUANT_SCATTER_1H_SCALE_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_39)
{
    size_t num_rows = 61;
    size_t cols = 3403;
    size_t k = 5;
    size_t expert_num = 256;
    uint64_t tilingKey = 2311000;
    uint32_t blockDim = 40;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = cols * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 61 3403 5 256 float32 1 238 0 256 0 1 1 [170,171] 1 1");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case38");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py int8 1 1");
    free(path_);
}

// DYNAMIC_QUANT_SCATTER_EH_SCALE_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_40)
{
    size_t num_rows = 69;
    size_t cols = 3541;
    size_t k = 8;
    size_t expert_num = 256;
    uint64_t tilingKey = 2312000;
    uint32_t blockDim = 40;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = expert_num * cols * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 69 3541 8 256 float32 1 445 0 256 0 1 1 [12,15] 0 2");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case39");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py int8 1 1");
    free(path_);
}

// DYNAMIC_QUANT_SCATTER_NO_SCALE_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_41)
{
    size_t num_rows = 28;
    size_t cols = 773;
    size_t k = 4;
    size_t expert_num = 256;
    uint64_t tilingKey = 2310000;
    uint32_t blockDim = 40;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 28 773 4 256 float32 1 72 0 256 0 1 0 [26,44] 0 0");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case40");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, nullptr, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py int8 0 0");
    free(path_);
}


// DYNAMIC_QUANT_GATHER_NO_SCALE_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_42)
{
    size_t num_rows = 28;
    size_t cols = 773;
    size_t k = 4;
    size_t expert_num = 256;
    uint64_t tilingKey = 2300000;
    uint32_t blockDim = 40;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 28 773 4 256 float32 1 72 0 256 0 1 0 [0,256] 0 0");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case41");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, nullptr, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py int8 0 0");
    free(path_);
}

// DYNAMIC_QUANT_GATHER_EH_SCALE_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_43)
{
    size_t num_rows = 69;
    size_t cols = 3541;
    size_t k = 8;
    size_t expert_num = 256;
    uint64_t tilingKey = 2302000;
    uint32_t blockDim = 40;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = expert_num * cols * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 69 3541 8 256 float32 1 445 0 256 0 1 1 [0,256] 0 2");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case42");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py int8 1 1");
    free(path_);
}

// DYNAMIC_QUANT_SCATTER_1H_SCALE_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_44)
{
    size_t num_rows = 61;
    size_t cols = 3403;
    size_t k = 5;
    size_t expert_num = 256;
    uint64_t tilingKey = 2301000;
    uint32_t blockDim = 40;
    size_t expertCapacity = 0;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = cols * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(int8_t);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 61 3403 5 256 float32 1 238 0 256 0 1 1 [0,256] 1 1");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case43");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);

    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py int8 1 1");
    free(path_);
}
