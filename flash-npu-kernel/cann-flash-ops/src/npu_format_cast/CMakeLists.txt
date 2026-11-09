# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").

target_sources(opapi PRIVATE
        op_host/op_api/aclnn_npu_format_cast.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api
        ${CMAKE_SOURCE_DIR}/src/common/include
        ${CMAKE_SOURCE_DIR}/src/common/include/external
)

# NpuFormatCast is a host-only ACLNN wrapper around the math L0 TransData APIs.
target_link_libraries(opapi PRIVATE opapi_math)

install(FILES op_host/op_api/aclnn_npu_format_cast.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
