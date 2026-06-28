# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# ascendc_ops is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
# ============================================================================

# 供 Tiling 链接的 Utils
#[[
#]]
add_library(ops_utils_tiling INTERFACE)
target_include_directories(ops_utils_tiling INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        # 头文件自包含, 包含 ops/utils/include 内头文件可引用的头文件路径
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/experiment/slog
)
target_compile_definitions(ops_utils_tiling INTERFACE
        OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"  # 定义日志输出时 SUB_MODE 名称
)
target_link_libraries(ops_utils_tiling INTERFACE
        c_sec
        alog error_manager  # 日志
        exe_graph           # Context
)

# 供 Proto 链接的 Utils
#[[
#]]
add_library(ops_utils_proto INTERFACE)
target_include_directories(ops_utils_proto INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        # 头文件自包含, 包含 ops/utils/include 内头文件可引用的头文件路径
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/experiment/slog
)
target_compile_definitions(ops_utils_proto INTERFACE
        OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
)
target_link_libraries(ops_utils_proto INTERFACE
        c_sec
        alog error_manager  # 日志
        exe_graph           # Context
)
