# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0.

add_ops_compile_options(
        OP_NAME MoeExpertIdMap
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/moe_expert_id_map_def.cpp
)

target_sources(optiling PRIVATE
        op_host/moe_expert_id_map_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            op_host/moe_expert_id_map_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/moe_expert_id_map_proto.cpp
)
