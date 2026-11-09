# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_ops_compile_options(
        OP_NAME SelectedKvMetadata
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/selected_kv_metadata_def.cpp
)

target_sources(optiling PRIVATE
        op_host/selected_kv_metadata_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            op_host/selected_kv_metadata_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/selected_kv_metadata_proto.cpp
)
