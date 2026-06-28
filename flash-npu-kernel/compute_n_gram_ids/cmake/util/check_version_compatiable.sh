#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

ASCEND_CANN_PACKAGE_PATH=$1
PACKAGE_NAME=$2
CANN_VERSION_INFO_FILE=${ASCEND_CANN_PACKAGE_PATH}/${PACKAGE_NAME}/version.info

function main()
{
    if [ ! -f "${CANN_VERSION_INFO_FILE}" ];then
        echo "Error: ${CANN_VERSION_INFO_FILE} does not exist, please check whether the cann package is installed."
        exit 1
    fi

    cann_version=$(grep -w "Version"  ${CANN_VERSION_INFO_FILE} | cut -d"=" -f2)
    
    _cann_version=$(echo ${cann_version} | cut -d'.' -f1-4)

    echo "==============cann_version: ${cann_version}, _cann_version: ${_cann_version}==============="

    COMPATIBLE_VERSION="7.5.0.1"

    if [[ $(echo -e "$_cann_version\n$COMPATIBLE_VERSION" | sort -V | tail -n1) == "$_cann_version" ]]; then
        echo "=====COMPATIBLE_VERSION===="
        exit 2
    else
        echo "=====NOT COMPATIBLE_VERSION===="
        exit 3
    fi

    echo "${cann_version}"
}

main



