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
 * \file grouped_matmul_utils.h
 * \brief GroupedMatmul 测试工具.
 */

#ifndef UTEST_GROUPED_MATMUL_UTILS_H
#define UTEST_GROUPED_MATMUL_UTILS_H

#include <cstdint>

namespace gmmTestUtils {
#define GMM_TPL_INVALID 0xFFFFFFFF
#define GMM_TPL_FLOAT 0
#define GMM_TPL_FLOAT16 1
#define GMM_TPL_INT8 2
#define GMM_TPL_INT32 3
#define GMM_TPL_BF16 27
#define GMM_TPL_INT4 29

#define GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM 0
#define GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT 1
#define GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM 2

#define GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE 0
#define GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT 1
#define GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT 2
#define GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT 3
#define GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT 4
#define GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING 5

#define GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE 0
#define GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD 1
#define GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT 2

#define GROUPED_MATMUL_CUBE_ONLY 0
#define GROUPED_MATMUL_AIV_AIC_RATIO_1 1
#define GROUPED_MATMUL_AIV_AIC_RATIO_2 2
    inline uint64_t GMMEncodeTilingKey(int xDtype, int weightDtype, int yDtype, int transX, int transWeight,
                                int groupListType, int isStaticTilingApi, int a8w4KernelTemplate,
                                int a16w8KernelTemplate, int aivAicRatio, bool isEnableFixedAxis) {
        uint64_t value = 0;
        int shift = 0;

        value |= (uint64_t)(xDtype & 0xFF) << shift;
        shift += 8;

        value |= (uint64_t)(weightDtype & 0xFF) << shift;
        shift += 8;

        value |= (uint64_t)(yDtype & 0xFF) << shift;
        shift += 8;

        value |= (uint64_t)(transX & 0x1) << shift;
        shift += 1;

        value |= (uint64_t)(transWeight & 0x1) << shift;
        shift += 1;

        value |= (uint64_t)(groupListType & 0x3) << shift;
        shift += 2;

        value |= (uint64_t)(isStaticTilingApi & 0x1) << shift;
        shift += 1;

        value |= (uint64_t)(a8w4KernelTemplate & 0xF) << shift;
        shift += 4;

        value |= (uint64_t)(a16w8KernelTemplate & 0x3) << shift;
        shift += 2;

        value |= (uint64_t)(aivAicRatio & 0x3) << shift;
        shift += 2;

        value |= (uint64_t)(isEnableFixedAxis & 0x1) << shift;
        shift += 1;

        return value;
    }


}
#endif // UTEST_GROUPED_MATMUL_UTILS_H