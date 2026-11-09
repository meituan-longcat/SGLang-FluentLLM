# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
 set(CONDITION_UNIT_COMMON ${ASCEND_COMPUTE_UNIT})
 if(BUILD_OPEN_PROJECT)
 file(GLOB OPTILING_SRCS 
   src/context_transfer.cpp
   src/hccl_formulaic_tiling.cpp
   src/hccl_performance.cpp
   src/hccl_performance_arch35.cpp
   src/matmul_all_reduce_gen_task_ops_utils.cpp
   src/matmul_formulaic_tiling.cpp
   src/matmul_performance.cpp
   src/mc2_fit_based_balance_tiling.cpp
   src/mc2_common_infershape.cpp
   src/mc2_copy_quant_matmul_params.cpp
   src/mc2_gen_task_ops_utils.cpp
   src/mc2_hcom_topo_info.cpp
   src/mc2_log.cpp
   src/mc2_matmul_tiling_cfg.cpp
   src/mc2_moe_gen_task_ops_utils.cpp
   src/mc2_moe_utils.cpp
   src/mc2_tiling_utils.cpp
   src/moe_tiling_base.cpp
   src/one_calc_two_comm_tiling.cpp
   )
 if("${CONDITION_UNIT_COMMON}" STREQUAL "ascend950")
   list(APPEND OPTILING_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mc2_gen_task_ops_utils_arch35.cpp)
 else()
   list(APPEND OPTILING_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mc2_gen_task_ops_utils_arch35_stub.cpp)
 endif()
 if(UT_TEST_ALL OR OP_HOST_UT)
   list(FILTER OPTILING_SRCS EXCLUDE REGEX "src/mc2_hcom_topo_info\\.cpp")
   file(GLOB MC2_TILING_UT_STUB ${UT_COMMON_INC}/mc2_hcom_topology_mocker.cpp ${UT_COMMON_INC}/op_cache_tiling.cpp)
   add_library(mc2_tiling_ut_stub SHARED ${MC2_TILING_UT_STUB})
   target_include_directories(mc2_tiling_ut_stub PRIVATE
     ${OPS_TRANSFORMER_DIR}/mc2/common/inc
     ${OP_TILING_INCLUDE}
   )
  endif()
  if(OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
    if(UT_TEST_ALL OR OP_HOST_UT)
     target_link_directories(${OPHOST_NAME}_tiling_obj PRIVATE mc2_tiling_ut_stub)
    endif()
  endif()
  if(NOT ENABLE_TEST)
   file(GLOB OPPROTO_SRCS)
   if(OPPROTO_SRCS)
     add_infer_modules()
     target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPPROTO_SRCS})
   endif()
   file(GLOB OPAPI_SRC)
   if(OPAPI_SRC)
     add_opapi_modules()
     target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRC})
   endif()
 endif()
endif()

file(GLOB NEW_MC2_MM_SRC new_mc2_mm/*.cpp)
# MC2_COMPILE used to avoid tefusion_llt 
# when blue use SUB_MC2_COMPILE when rty use MC2_COMPILE
if(MC2_COMPILE OR SUB_MC2_COMPILE OR UT_TEST OR OP_HOST_UT)
 add_tiling_modules()
 target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${NEW_MC2_MM_SRC})
endif()

file(GLOB OPAPI_SRC
 src/mc2_aclnn_util.cpp
)
if(OPAPI_SRC)
 add_opapi_modules()
 target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRC})
endif()