add_ops_compile_options(
    OP_NAME SparseMlaAttentionPrefill
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true)

target_sources(op_host_aclnnInner PRIVATE op_host/sparse_mla_attention_prefill_def.cpp)
target_sources(opsproto PRIVATE op_host/sparse_mla_attention_prefill_proto.cpp)
target_sources(optiling PRIVATE op_host/sparse_mla_attention_prefill_tiling.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

if (BUILD_OPEN_PROJECT)
    add_device_tiling_library(
        TARGET sparse_mla_attention_prefill_opmaster
        OPTION SHARED
        SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/sparse_mla_attention_prefill_tiling.cpp)
endif()

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/sparse_mla_attention_prefill_tiling.cpp)
endif()

target_sources(opapi PRIVATE
    op_host/op_api/aclnn_sparse_mla_attention_prefill.cpp)
target_include_directories(opapi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api)
install(FILES op_host/op_api/aclnn_sparse_mla_attention_prefill.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR})
