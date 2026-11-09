# MC2 构建适配层：将 ops-transformer 的 MC2 算子构建体系接入 cann-flash-ops
# 基于 ops-transformer-9.0.0/cmake/obj_func.cmake 简化移植

# 定义 BUILD_OPEN_PROJECT 宏，让 mc2 源码走 custom 分支（跳过 task.pb.h 等 built-in 依赖）
add_definitions(-DBUILD_OPEN_PROJECT)

# 变量定义（映射 9.0.0 体系）
set(PKG_NAME transformer)
set(OPHOST_NAME ophost_${PKG_NAME})
set(GRAPH_PLUGIN_NAME graph_plugin_${PKG_NAME})
set(COMMON_NAME common_${PKG_NAME})
set(OPS_TRANSFORMER_DIR ${CMAKE_SOURCE_DIR}/src)
set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)

# Include 路径集合
set(MC2_INC
  ${OPS_TRANSFORMER_DIR}/mc2/common
  ${OPS_TRANSFORMER_DIR}/mc2/common/inc
  ${OPS_TRANSFORMER_DIR}/mc2/common/new_mc2_mm
  ${OPS_TRANSFORMER_DIR}/mc2/3rd
  ${OPS_TRANSFORMER_DIR}/mc2
  ${CMAKE_SOURCE_DIR}/src/utils/inc
  ${CMAKE_SOURCE_DIR}/src/common/include
)

set(CANN_INC
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include/external
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog
  ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/external
  ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
  ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/exe_graph
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/runtime
  ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
)

set(OP_TILING_INCLUDE ${MC2_INC} ${CANN_INC})
set(OP_PROTO_INCLUDE ${MC2_INC} ${CANN_INC} ${OPS_TRANSFORMER_DIR})
set(OPAPI_INCLUDE ${MC2_INC} ${CANN_INC} ${OPS_TRANSFORMER_DIR})
set(INFER_OBJ_INCLUDE ${OP_PROTO_INCLUDE})

# 创建缺失的 interface library stub
if(NOT TARGET intf_pub_cxx17)
  add_library(intf_pub_cxx17 INTERFACE)
  target_link_libraries(intf_pub_cxx17 INTERFACE intf_pub)
  target_compile_options(intf_pub_cxx17 INTERFACE -std=c++17)
endif()

if(NOT TARGET intf_pub_cxx14)
  add_library(intf_pub_cxx14 INTERFACE)
  target_link_libraries(intf_pub_cxx14 INTERFACE intf_pub)
  target_compile_options(intf_pub_cxx14 INTERFACE -std=c++14)
endif()

if(NOT TARGET json)
  add_library(json INTERFACE IMPORTED)
  set_target_properties(json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/third_party/json/include
  )
endif()

if(NOT TARGET ops_transformer_utils_tiling_headers)
  add_library(ops_transformer_utils_tiling_headers INTERFACE)
  target_include_directories(ops_transformer_utils_tiling_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/src/utils/inc
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
  )
  target_compile_definitions(ops_transformer_utils_tiling_headers INTERFACE
    OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
    OPS_UTILS_LOG_PACKAGE_TYPE="[Custom]"
  )
endif()

if(NOT TARGET ops_transformer_utils_proto_headers)
  add_library(ops_transformer_utils_proto_headers INTERFACE)
  target_include_directories(ops_transformer_utils_proto_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/src/utils/inc
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
  )
  target_compile_definitions(ops_transformer_utils_proto_headers INTERFACE
    OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
    OPS_UTILS_LOG_PACKAGE_TYPE="[Custom]"
  )
endif()

if(NOT TARGET rt2_registry_static)
  add_library(rt2_registry_static INTERFACE)
endif()

if(NOT TARGET alog_headers)
  add_library(alog_headers INTERFACE)
endif()

if(NOT TARGET dlog_headers)
  add_library(dlog_headers INTERFACE)
endif()

if(NOT TARGET ops_base_util_objs)
  add_library(ops_base_util_objs INTERFACE)
endif()

if(NOT TARGET ops_base_infer_objs)
  add_library(ops_base_infer_objs INTERFACE)
endif()

# 创建 INTERFACE library（9.0.0 variables.cmake 定义）
add_library(${OPHOST_NAME}_opdef_aclnn_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE)
add_library(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE)
add_library(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE)

# Macro/Function 定义（从 9.0.0 obj_func.cmake 简化移植）

function(add_opapi_modules)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj)
    add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
      ${OPAPI_INCLUDE}
      ${CANN_INC}
    )
    target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE
      _GLIBCXX_USE_CXX11_ABI=0
      BUILD_OPEN_PROJECT_API=1
      BUILD_OPEN_PROJECT
    )
    target_compile_options(${OPHOST_NAME}_opapi_obj PRIVATE
      -Dgoogle=ascend_private
      -DACLNN_LOG_FMT_CHECK
    )
    target_link_libraries(${OPHOST_NAME}_opapi_obj PRIVATE intf_pub)
  endif()
endfunction()

function(add_infer_modules)
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    add_library(${OPHOST_NAME}_infer_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_infer_obj PRIVATE ${INFER_OBJ_INCLUDE})
    target_compile_definitions(${OPHOST_NAME}_infer_obj PRIVATE
      LOG_CPP
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
      BUILD_OPEN_PROJECT
    )
    target_compile_options(${OPHOST_NAME}_infer_obj PRIVATE
      -Dgoogle=ascend_private
      -fvisibility=hidden
    )
    target_link_libraries(${OPHOST_NAME}_infer_obj PRIVATE intf_pub ops_utils_proto_headers)
  endif()
endfunction()

function(add_tiling_modules)
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    add_library(${OPHOST_NAME}_tiling_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${OP_TILING_INCLUDE})
    target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE
      LOG_CPP
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
      BUILD_OPEN_PROJECT
    )
    target_compile_options(${OPHOST_NAME}_tiling_obj PRIVATE
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    target_link_libraries(${OPHOST_NAME}_tiling_obj PRIVATE intf_pub ops_utils_tiling_headers json)
  endif()
endfunction()

function(add_graph_plugin_modules)
  if(NOT TARGET ${GRAPH_PLUGIN_NAME}_obj)
    add_library(${GRAPH_PLUGIN_NAME}_obj OBJECT)
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_definitions(${GRAPH_PLUGIN_NAME}_obj PRIVATE
      OPS_UTILS_LOG_SUB_MOD_NAME="GRAPH_PLUGIN"
      LOG_CPP
      BUILD_OPEN_PROJECT
    )
    target_compile_options(${GRAPH_PLUGIN_NAME}_obj PRIVATE
      -Dgoogle=ascend_private
      -fvisibility=hidden
    )
    target_link_libraries(${GRAPH_PLUGIN_NAME}_obj PRIVATE intf_pub)
  endif()
endfunction()

function(add_opmaster_ct_gentask_modules)
  if(NOT TARGET ${OPHOST_NAME}_opmaster_ct_gentask_obj)
    add_library(${OPHOST_NAME}_opmaster_ct_gentask_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE ${OP_TILING_INCLUDE})
    target_compile_definitions(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE
      LOG_CPP
      BUILD_OPEN_PROJECT
    )
    target_compile_options(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    target_link_libraries(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE intf_pub)
  endif()
endfunction()

# add_modules_sources macro（从 9.0.0 obj_func.cmake:17-159 移植）
macro(add_modules_sources)
  set(oneValueArgs OP_API_INDEPENDENT OP_API_DIR OP_MC2_ENABLE)
  set(multiValueArgs OPTYPE ACLNNTYPE ACLNN_EXTRA_VERSION)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  if(NOT DEFINED MODULE_OP_MC2_ENABLE)
    set(MODULE_OP_MC2_ENABLE OFF)
  endif()

  if(NOT DEFINED MODULE_OP_API_INDEPENDENT)
    set(MODULE_OP_API_INDEPENDENT OFF)
  endif()
  if(MODULE_OP_API_INDEPENDENT)
    if(NOT DEFINED MODULE_OP_API_DIR OR NOT EXISTS "${MODULE_OP_API_DIR}")
      message(FATAL_ERROR "OP_API_INDEPENDENT=ON时，必须传递有效的OP_API_DIR路径")
    endif()
    set(OP_API_SRC_DIR "${MODULE_OP_API_DIR}")
  else()
    set(OP_API_SRC_DIR "${SOURCE_DIR}/op_api")
  endif()

  if(MODULE_OP_MC2_ENABLE)
    set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
    set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)
  endif()

  list(LENGTH MODULE_OPTYPE OpTypeLen)
  list(LENGTH MODULE_ACLNN_EXTRA_VERSION AclnnExtraVersionLen)
  if((AclnnExtraVersionLen GREATER 1) AND (OpTypeLen GREATER 1))
    message(FATAL_ERROR "There should be only 1 optype if there are more than 1 aclnn extra versions!")
  endif()

  file(GLOB OPAPI_SRCS ${OP_API_SRC_DIR}/*.cpp)
  if(OPAPI_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  else()
    if(NOT TARGET ${OPHOST_NAME}_opapi_obj)
      add_opapi_modules()
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
      target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
    endif()
  endif()

  if(NOT MODULE_OP_MC2_ENABLE)
    file(GLOB OPAPI_HEADERS ${OP_API_SRC_DIR}/aclnn_*.h)
    if(OPAPI_HEADERS)
      target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
    endif()
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  add_infer_modules()
  set(PROTO_STUB_FILE ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp)

  if(NOT EXISTS ${PROTO_STUB_FILE})
    file(WRITE ${PROTO_STUB_FILE} "// Auto-generated stub file\n")
  endif()

  set_source_files_properties(
    ${PROTO_STUB_FILE}
    PROPERTIES GENERATED TRUE
  )

  if(OPINFER_SRCS)
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  else()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${PROTO_STUB_FILE})
  endif()

  file(GLOB_RECURSE SUB_OPTILING_SRC ${SOURCE_DIR}/op_tiling/*.cpp)
  file(GLOB OPTILING_SRCS
    ${SOURCE_DIR}/*fallback*.cpp
    ${SOURCE_DIR}/*_tiling*.cpp
    ${SOURCE_DIR}/op_tiling/arch35/*.cpp
    ${SOURCE_DIR}/../op_graph/fallback_*.cpp
    ${SOURCE_DIR}/../graph_plugin/fallback_*.cpp)
  if(OPTILING_SRCS OR SUB_OPTILING_SRC)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS} ${SUB_OPTILING_SRC})
  endif()

  if(MODULE_OP_MC2_ENABLE)
    file(GLOB GENTASK_SRCS
      ${SOURCE_DIR}/../op_graph/*_gen_task*.cpp
    )
    if(GENTASK_SRCS)
      add_opmaster_ct_gentask_modules()
      target_sources(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE ${GENTASK_SRCS})
    endif()
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if(${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)

        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
        if(AclnnExtraVersionLen GREATER 0)
          # concat_op_names stub (moe 不使用 ACLNN_EXTRA_VERSION)
        endif()
      elseif(${AclnnType} STREQUAL "no_need_aclnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
        "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
        "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
        "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

# add_graph_plugin_sources macro（从 9.0.0 obj_func.cmake:511-539 移植）
macro(add_graph_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(DEFINED ASCEND_OP_NAME
    AND NOT "${ASCEND_OP_NAME}" STREQUAL ""
    AND NOT "${ASCEND_OP_NAME}" STREQUAL "all"
    AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL"
  )
    if(NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
      return()
    endif()
  endif()

  file(GLOB GRAPH_PLUGIN_SRCS
    ${SOURCE_DIR}/*_graph_plugin*.cpp
  )
  if(GRAPH_PLUGIN_SRCS)
    add_graph_plugin_modules()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${GRAPH_PLUGIN_SRCS})
  endif()

  file(GLOB GRAPH_PLUGIN_PROTO_HEADERS ${SOURCE_DIR}/*_proto*.h)
  if(GRAPH_PLUGIN_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${GRAPH_PLUGIN_PROTO_HEADERS})
  endif()
endmacro()

# 将 MC2 OBJECT library link 到 cann-flash-ops 主 target
function(mc2_link_to_targets)
  if(TARGET ${OPHOST_NAME}_tiling_obj)
    target_sources(optiling PRIVATE $<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>)
  endif()
  if(TARGET ${OPHOST_NAME}_opmaster_ct_gentask_obj)
    target_sources(optiling PRIVATE $<TARGET_OBJECTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>)
  endif()
  if(TARGET ${OPHOST_NAME}_opapi_obj)
    target_sources(opapi PRIVATE $<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>)
  endif()
  if(TARGET ${OPHOST_NAME}_infer_obj)
    target_sources(opsproto PRIVATE $<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>)
  endif()
endfunction()
