# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        moe_distribute_dispatch_v2_def.cpp
    )
    add_modules_sources(
        OP_API_INDEPENDENT ON 
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api 
        OP_MC2_ENABLE ON 
        OPTYPE moe_distribute_dispatch_v2 ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(moe_distribute_dispatch_v2_depends mc2/common mc2/3rd PARENT_SCOPE)

    # --cce-auto-sync=off：指定CCE编译器是否自动执行线程间或模块间的同步操作
    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend910_93")
        add_ops_compile_options(
            OP_NAME MoeDistributeDispatchV2
            OPTIONS --cce-auto-sync=off
        )
    elseif("${CONDITION_UNIT}" STREQUAL "ascend910b")
        add_ops_compile_options(
            OP_NAME MoeDistributeDispatchV2
            OPTIONS --cce-auto-sync=off
        )
    elseif("${CONDITION_UNIT}" STREQUAL "ascend950")  # 考虑多版本兼容dump功能
        add_definitions(-DMC2_EXCEPTION_HANDLER)
    endif()
else() # 回黄host
    add_mc2_modules_sources(OPTYPE moe_distribute_dispatch_v2 ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE) 
endif()
