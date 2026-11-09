add_ops_compile_options(
    OP_NAME KvCacheCopyRowsBurst
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
)
target_sources(op_host_aclnn PRIVATE op_host/kv_cache_copy_rows_burst_def.cpp)
target_sources(optiling PRIVATE op_host/kv_cache_copy_rows_burst_tiling.cpp)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE op_host/kv_cache_copy_rows_burst_tiling.cpp)
endif ()
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
target_sources(opsproto PRIVATE op_host/kv_cache_copy_rows_burst_proto.cpp)
