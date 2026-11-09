#!/bin/bash
set -euo pipefail
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

BASE_DIR=$(pwd)

# remove historical compilation results
rm -rf build
rm -f dist/flash_ops-*.whl

# use ninja to build system and parallel compilation
export USE_NINJA=1

# set parallel jobs for compilation
if [ -z "${MAX_JOBS:-}" ]; then
    export MAX_JOBS=`nproc`
    echo "Using $MAX_JOBS parallel jobs for compilation"
fi

# compile wheel package using incremental compilation
python3 setup.py build_ext && python3 setup.py bdist_wheel

# install wheel package
cd ${BASE_DIR}/dist
#pip3 install *.whl -I
cd -
