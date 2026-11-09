add_ops_compile_options(
    OP_NAME SparseFlashAttentionDecode
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true)

target_sources(op_host_aclnnInner PRIVATE op_host/sparse_flash_attention_decode_def.cpp)
target_sources(opsproto PRIVATE op_host/sparse_flash_attention_decode_proto.cpp)
target_sources(optiling PRIVATE op_host/sparse_flash_attention_decode_tiling.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

if (BUILD_OPEN_PROJECT)
    add_device_tiling_library(
        TARGET sparse_flash_attention_decode_opmaster
        OPTION SHARED
        SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/sparse_flash_attention_decode_tiling.cpp)
endif()

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/sparse_flash_attention_decode_tiling.cpp)
endif()

target_sources(opapi PRIVATE
    op_host/op_api/aclnn_sparse_flash_attention_decode.cpp)
target_include_directories(opapi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api)
install(FILES op_host/op_api/aclnn_sparse_flash_attention_decode.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR})
