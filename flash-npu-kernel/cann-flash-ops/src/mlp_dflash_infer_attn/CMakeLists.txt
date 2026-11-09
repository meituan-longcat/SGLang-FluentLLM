# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

add_ops_compile_options(
        OP_NAME MlpDflashInferAttn
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        op_host/mlp_dflash_infer_attn_def.cpp
)

target_sources(optiling PRIVATE
        op_host/mlp_dflash_infer_attn_tiling.cpp
)

if (BUILD_OPEN_PROJECT)
    add_device_tiling_library(
            TARGET mlp_dflash_infer_attn_opmaster
            OPTION SHARED
            SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mlp_dflash_infer_attn_tiling.cpp
            INCLUDE_DIRS ${OPS_ADV_DIR}/src/utils/inc
    )
endif ()

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            op_host/mlp_dflash_infer_attn_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/mlp_dflash_infer_attn_proto.cpp
)

# The generated dynamic-op wrapper compiles the kernel from
# flash_ops_impl/ascendc/mlp_dflash_infer_attn.  The repository-wide install
# rule only copies top-level *.cpp/*.h files there, while this kernel also
# needs a top-level *.hpp and the attn_infra header tree.  Keep the regular
# op_kernel copy for offline compilation and additionally provide the include
# layout required by torchair/GE runtime compilation.
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fia_kernel_common.hpp
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/mlp_dflash_infer_attn
)

install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/attn_infra
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/mlp_dflash_infer_attn
)
