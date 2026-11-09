# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import glob
import os

import torch_npu  # noqa: F401 - registers device/operator implementations
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension  # noqa: F401 - registers device/operator implementations

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv("USE_NINJA") == "1"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

MINIMAL_OPS = os.getenv("FLASH_OPS_MINIMAL") == "1"
if MINIMAL_OPS:
    source_files = [
        os.path.join(BASE_DIR, "custom_ops/csrc", source_name)
        for source_name in (
            "ops_common.cpp",
            "ops_def_registration.cpp",
            "flat_quant.cpp",
            "mlp_dflash_infer_attn.cpp",
            "mlp_compressed_lightning_indexer.cpp",
            "npu_mla_prolog_v3.cpp",
            "npu_kv_quant_sparse_flash_attention_lse.cpp",
            "mlp_lightning_indexer.cpp",
            "mlp_sparse_lightning_indexer.cpp",
            "mlp_split_swiglu.cpp",
            "npu_moe_distribute_dispatch_v2.cpp",
            "npu_moe_distribute_combine_v2.cpp",
            "npu_causal_conv1d.cpp",
            "npu_compute_n_gram_ids.cpp",
            "npu_gather_element.cpp",
            "npu_select_local_topk_indices.cpp",
            "npu_router_input_slice_cast.cpp",
            "npu_kvp_attention_merge.cpp",
            "npu_packed_kv_dequant.cpp",
            "npu_selected_kv_prepare.cpp",
            "npu_packed_kv_prefill_prepare.cpp",
            "npu_moe_expert_id_map.cpp",
            "npu_moe_gating_top_k_expert_map.cpp",
            "npu_kv_cache_copy_rows.cpp",
            "npu_lightning_indexer.cpp",
            "npu_sparse_flash_attention.cpp",
            "npu_sparse_flash_attention_decode.cpp",
            "npu_sparse_mla_attention_prefill.cpp",
        )
    ]
else:
    source_files = glob.glob(os.path.join(BASE_DIR, "custom_ops/csrc", "*.cpp"), recursive=True)

exts = []
ext = NpuExtension(
    name="flash_ops.custom_ops_lib",
    sources=source_files,
    extra_compile_args=[
        "-I" + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ]
    + (["-DFLASH_OPS_MINIMAL"] if MINIMAL_OPS else []),
)
exts.append(ext)

# Both ABIs ship in the default wheel; original custom operators stay intact.
sfa1_sources = glob.glob(os.path.join(BASE_DIR, "fluent_sfa1_ops/csrc", "*.cpp"))
sfa1_sources.append(os.path.join(BASE_DIR, "custom_ops/csrc/ops_common.cpp"))
exts.append(NpuExtension(
    name="fluent_sfa1_ops._C",
    sources=sfa1_sources,
    include_dirs=[os.path.join(BASE_DIR, "custom_ops/csrc")],
    extra_compile_args=[
        "-I" + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        "-fvisibility=hidden", "-fvisibility-inlines-hidden",
        "-DFLUENT_PRIVATE_OFFLOAD_PROVIDER",
    ],
))

setup(
    name="flash_ops",
    version="1.0",
    keywords="flash_ops",
    ext_modules=exts,
    package_dir={"flash_ops": "custom_ops", "fluent_sfa1_ops": "fluent_sfa1_ops"},
    package_data={
        "flash_ops": ["*.py", "*.so"],
        "flash_ops.converter": ["*.py", "*.so"],
        "flash_ops.offload": ["*.py"],
        "fluent_sfa1_ops": ["*.py", "*.so"],
        "fluent_sfa1_ops.converter": ["*.py", "*.so"],
    },
    packages=["flash_ops", "flash_ops.offload", "flash_ops.converter", "fluent_sfa1_ops", "fluent_sfa1_ops.converter"],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
