# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import glob
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = glob.glob(os.path.join(BASE_DIR, "custom_ops/csrc", "*.cpp"), recursive=True)

exts = []
ext = NpuExtension(
    name="cp_core.custom_ops_lib",
    sources=source_files,
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)
exts.append(ext)

setup(
    name="cp_core",
    version='1.0',
    keywords='cp_core',
    ext_modules=exts,
    package_dir={'cp_core': 'custom_ops'},
    package_data={
        'cp_core': ['*.py', '*.so'],
        'cp_core.converter': ['*.py', '*.so'],
    },
    packages=['cp_core', 'cp_core.converter'],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
