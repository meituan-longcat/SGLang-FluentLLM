# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
message(STATUS "=== Debug: CURRENT_SOURCE_DIRS =${CURRENT_SOURCE_DIRS} ")
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的tiling文件路径
    set(moe_init_routing_v3_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_init_routing_v3_tiling_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_init_routing_v3_tiling_base.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_init_routing_v3_tiling.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："Ascend950PR_9599;AscendB1"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(moe_init_routing_v3 "ascend910B1" "-DDTYPE_X=float" "${moe_init_routing_v3_tiling_files}")
endif()