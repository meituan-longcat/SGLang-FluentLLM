add_ops_compile_options(
    OP_NAME SelectLocalTopkIndices
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-aicore-hoist-movemask=false)

target_sources(op_host_aclnn PRIVATE
    op_host/select_local_topk_indices_def.cpp)
target_sources(optiling PRIVATE
    op_host/select_local_topk_indices_tiling.cpp)
target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
target_sources(opsproto PRIVATE
    op_host/select_local_topk_indices_proto.cpp)
