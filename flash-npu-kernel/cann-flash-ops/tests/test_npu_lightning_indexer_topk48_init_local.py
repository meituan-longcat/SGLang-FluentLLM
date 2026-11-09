# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 小规模单测: topk=48, q_len=4, kv_len=128, init=16, local=16
# layout: query=TND, key=PA_BSND (与 nsa_indexer.py 线上调用形态一致)
#
# 验证点:
#   1. causal(sparse_mode=3)下, 第s行有效key数 = kv_len - q_len + s + 1 = 125+s
#   2. init/local 强制入选: 前16个key 与 每行最后16个有效key 必出现在topk结果中
#      (kernel 内将其 score 抬为 SECOND_MAX_LOGITS)
#   3. PA 路径: block_size 可配, block_table 乱序, 校验跨 block 边界取数正确
#   4. NPU 输出与 CPU golden 逐行做"集合"比较 —— 被提升的32个entry分值完全相同,
#      并列名次在 NPU Sort/MrgSort 与 torch.sort 间不保证顺序一致, 故不比严格顺序
#
# 入口优先级 (可用 LI_OP_ENTRY 环境变量强制指定):
#   1. aclnn      : ctypes 直调 vendor 包 libcust_opapi.so 的 aclnnLightningIndexer
#                   —— 即 "cann-flash-ops build.sh -n lightning_indexer" 新编译安装的算子,
#                   无需 torch 侧绑定. 用 LI_VENDOR_OPAPI 可覆盖 .so 路径
#   2. custom_li  : flash_ops wheel 的 torch.ops.custom.npu_lightning_indexer
#   3. builtin_li : torch_npu 内置 npu_lightning_indexer (需 schema 带 init_tensor)
#
# 运行: source set_env.sh && python3 test_npu_lightning_indexer_topk48_init_local.py

import ctypes
import os

import torch
import torch_npu
import numpy as np

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

# kernel 中 init/local 提升到的分值 (MAX_LOGITS/2), 任何真实 score 不可能达到
SECOND_MAX_LOGITS = np.float32(0x7F7FFFFF).view(np.float32) / 2.0

INT64_MAX = 9223372036854775807

DEFAULT_VENDOR_OPAPI = "/usr/local/Ascend/cann-9.0.0/opp/vendors/flash_ops/op_api/lib/libcust_opapi.so"

# acl 常量 (acl_base.h)
ACL_FORMAT_ND = 2
ACL_DTYPE = {torch.float32: 0, torch.float16: 1, torch.int32: 3, torch.int64: 9, torch.bfloat16: 27}


class AclnnLightningIndexer:
    """ctypes 直调 vendor 包 aclnnLightningIndexer (绕开 torch_npu eager 绑定, 直接测新编算子)。"""

    def __init__(self, opapi_path):
        self.lib = ctypes.CDLL(opapi_path, mode=ctypes.RTLD_GLOBAL)
        # aclCreateTensor/aclDestroyTensor 在 libnnopbase.so
        cann_lib64 = os.path.join(os.environ.get("ASCEND_HOME_PATH",
                                                 "/usr/local/Ascend/ascend-toolkit/latest"), "lib64")
        self.nnop = ctypes.CDLL(os.path.join(cann_lib64, "libnnopbase.so"), mode=ctypes.RTLD_GLOBAL)
        self.nnop.aclCreateTensor.restype = ctypes.c_void_p
        self.nnop.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,   # dims, ndim, dtype
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,                   # stride, offset, format
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p]  # storageDims, num, data
        self.nnop.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.lib.aclnnLightningIndexerGetWorkspaceSize.restype = ctypes.c_int
        self.lib.aclnnLightningIndexer.restype = ctypes.c_int

    def _tensor(self, t):
        if t is None:
            return None  # 可选输入传空指针
        assert t.is_contiguous(), "tensor must be contiguous"
        shape = (ctypes.c_int64 * t.dim())(*t.shape)
        handle = self.nnop.aclCreateTensor(shape, t.dim(), ACL_DTYPE[t.dtype], None, 0,
                                           ACL_FORMAT_ND, shape, t.dim(),
                                           ctypes.c_void_p(t.data_ptr()))
        assert handle, f"aclCreateTensor failed for shape {tuple(t.shape)}"
        return handle

    def __call__(self, query, key, weights, actual_seq_q, actual_seq_k, block_table,
                 init_tensor, local_tensor, layout_query, layout_key,
                 sparse_count, sparse_mode, return_value):
        keep = []  # 防 GC
        handles = [self._tensor(t) for t in
                   (query, key, weights, actual_seq_q, actual_seq_k, block_table, init_tensor, local_tensor)]
        out_idx = torch.empty((*query.shape[:-2], 1, sparse_count), dtype=torch.int32, device=query.device)
        # Match the production binding: ACLNN requires an output descriptor,
        # including a zero-length tensor when scores are not requested.
        out_val = (torch.empty_like(out_idx, dtype=query.dtype) if return_value
                   else torch.empty(0, dtype=query.dtype, device=query.device))
        h_idx = self._tensor(out_idx)
        h_val = self._tensor(out_val)

        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p(0)
        ret = self.lib.aclnnLightningIndexerGetWorkspaceSize(
            *[ctypes.c_void_p(h) if h else None for h in handles],
            ctypes.c_char_p(layout_query.encode()), ctypes.c_char_p(layout_key.encode()),
            ctypes.c_int64(sparse_count), ctypes.c_int64(sparse_mode),
            ctypes.c_int64(INT64_MAX), ctypes.c_int64(INT64_MAX),
            ctypes.c_bool(return_value),
            ctypes.c_void_p(h_idx), ctypes.c_void_p(h_val),
            ctypes.byref(ws_size), ctypes.byref(executor))
        assert ret == 0, f"aclnnLightningIndexerGetWorkspaceSize failed: {ret}"

        workspace = torch.empty(ws_size.value, dtype=torch.uint8, device=query.device) if ws_size.value else None
        keep.append(workspace)
        stream = torch.npu.current_stream().npu_stream
        ret = self.lib.aclnnLightningIndexer(
            ctypes.c_void_p(workspace.data_ptr()) if workspace is not None else None,
            ctypes.c_uint64(ws_size.value), executor, ctypes.c_void_p(stream))
        assert ret == 0, f"aclnnLightningIndexer failed: {ret}"
        torch.npu.synchronize()

        for h in handles + [h_idx, h_val]:
            if h:
                self.nnop.aclDestroyTensor(ctypes.c_void_p(h))
        if not return_value:
            out_val = torch.empty(0, dtype=query.dtype)
        return out_idx, out_val


def pick_entry():
    forced = os.environ.get("LI_OP_ENTRY")
    vendor = os.environ.get("LI_VENDOR_OPAPI", DEFAULT_VENDOR_OPAPI)
    candidates = [forced] if forced else ["aclnn", "custom_li", "builtin_li"]
    errors = []
    for e in candidates:
        try:
            if e == "aclnn":
                if os.path.exists(vendor):
                    return "aclnn", AclnnLightningIndexer(vendor)
                errors.append(f"aclnn: vendor opapi not found: {vendor}")
            elif e == "custom_li":
                import flash_ops  # noqa: F401
                return "custom_li", torch.ops.custom.npu_lightning_indexer
            elif e == "builtin_li":
                schema = str(torch.ops.npu.npu_lightning_indexer._schemas.get('', ''))
                if 'init_tensor' in schema:
                    return "builtin_li", torch_npu.npu_lightning_indexer
                errors.append("builtin_li: torch_npu schema has no init_tensor")
            else:
                errors.append(f"unknown LI_OP_ENTRY: {e}")
        except ModuleNotFoundError:
            errors.append("custom_li: flash_ops not installed")
    raise RuntimeError("no usable lightning_indexer entry:\n  " + "\n  ".join(errors))


def depage_key(key, block_table, act_s2):
    """PA_BSND key [BlockNum, BlockSize, 1, D] + block_table -> [act_s2, D]"""
    block_num, block_size, n2, d = key.shape
    assert n2 == 1, f"n2 must be 1, got {n2}"
    blocks = key[block_table.long()]            # [need_blocks, BlockSize, 1, D]
    return blocks.reshape(-1, d)[:act_s2]


def golden_lightning_indexer(query, key_paged, weights, block_table,
                             actual_s1, actual_s2, init_num, local_num,
                             sparse_count, sparse_mode=3):
    """CPU golden, B=1, query=TND, key=PA_BSND, 复刻 kernel 语义。

    query:     [T, N1, D]   (T = q_len)
    key_paged: [BlockNum, BlockSize, 1, D]
    weights:   [T, N1]
    return:    indices [T, 1, sparse_count] int32, values 同形状 fp32
    """
    t, n1, d = query.shape
    q = query.transpose(0, 1).float()             # [N1, T, D]
    k = depage_key(key_paged, block_table[0], actual_s2).t().float()  # [D, S2]
    w = weights.t().float().unsqueeze(-1)         # [N1, T, 1]

    score = torch.sum(torch.maximum(q @ k, torch.tensor(0.0)) * w, dim=0)  # [T, S2]

    out_idx = torch.full((t, sparse_count), -1, dtype=torch.int32)
    out_val = torch.full((t, sparse_count), float('-inf'), dtype=torch.float32)
    for s in range(t):
        valid_len = actual_s2 - actual_s1 + s + 1 if sparse_mode == 3 else actual_s2
        row = score[s].clone()
        row[valid_len:] = float('-inf')
        # init/local 提升: 前init个 + 最后local个有效key 抬为极大值, 保证入选
        if init_num + local_num < valid_len:
            row[:init_num] = SECOND_MAX_LOGITS
            if local_num > 0:
                row[valid_len - local_num:valid_len] = SECOND_MAX_LOGITS
        val, idx = torch.sort(row, descending=True, stable=True)
        out_idx[s] = idx[:sparse_count].to(torch.int32)
        out_val[s] = val[:sparse_count]
    return out_idx.reshape(t, 1, sparse_count), out_val.reshape(t, 1, sparse_count)


def test_golden_init_local_gate_uses_local_num():
    """锁定 init != local 时的门控，避免 golden 退回 init + init。"""
    query = torch.zeros(1, 1, 2, dtype=torch.float32)
    key = torch.zeros(1, 32, 1, 2, dtype=torch.float32)
    weights = torch.ones(1, 1, dtype=torch.float32)
    block_table = torch.tensor([[0]], dtype=torch.int32)

    _, values = golden_lightning_indexer(
        query, key, weights, block_table,
        actual_s1=1, actual_s2=20, init_num=16, local_num=0,
        sparse_count=20, sparse_mode=0)

    # 16 + 0 < 20，必须进入强选；旧门控 16 + 16 < 20 会错误跳过。
    assert int((values[0, 0] > 1e30).sum()) == 16


def test_topk48_q4_kv128_init16_local16_tnd_pa():
    b = 1
    q_len, kv_len = 4, 128
    n1, n2, d = 64, 1, 128
    init_num, local_num = 16, 16
    sparse_count = 48
    sparse_mode = 3
    dtype = torch.bfloat16

    entry, li_op = pick_entry()

    # PA 配置: block_size 可配(16的倍数, <=1024), block_num/block_table 由它自动推导,
    # block_table 取乱序排列以覆盖跨 block 边界取数;
    # 注意 block_size >= kv_len 时只有 1 个物理块, 乱序退化为 [[0]], 不覆盖跨块路径
    block_size = 16
    assert block_size % 16 == 0 and 0 < block_size <= 1024, \
        f"block_size must be a multiple of 16 in (0, 1024], got {block_size}"
    block_num = (kv_len + block_size - 1) // block_size
    block_perm = np.random.RandomState(0).permutation(block_num).tolist()
    block_table = torch.tensor([block_perm], dtype=torch.int32)
    assert block_table.shape == (b, block_num), (
        f"block_table shape mismatch: got {tuple(block_table.shape)}, "
        f"expected (b={b}, block_num={block_num}) "
        f"(block_num = ceil(kv_len={kv_len} / block_size={block_size})). "
        f"If you changed block_size, block_table must be regenerated to length block_num.")

    np.random.seed(42)
    # TND: query [T, N1, D], weights [T, N1]
    query = torch.tensor(np.random.uniform(-1, 1, (q_len, n1, d))).to(dtype)
    weights = torch.tensor(np.random.uniform(0, 1, (q_len, n1))).to(dtype)
    # PA_BSND: key [BlockNum, BlockSize, N2, D]
    key = torch.tensor(np.random.uniform(-1, 1, (block_num, block_size, n2, d))).to(dtype)
    init_tensor = torch.tensor([init_num], dtype=torch.int32)
    # 算子自 2026-08-07 起只接受 per-token local_tensor [T]: 逐行传 local_num(与旧 [B] 定宽同语义)
    local_tensor = torch.tensor([local_num] * q_len, dtype=torch.int32)

    # ---------- CPU golden ----------
    golden_idx, golden_val = golden_lightning_indexer(
        query, key, weights, block_table, q_len, kv_len,
        init_num, local_num, sparse_count, sparse_mode)

    # ---------- NPU ----------
    # 严格复刻 fluentllm nsa_indexer.py npu_indexer_normal 的输入形态:
    # actual_seq_q 为无前导0的前缀和(B,), actual_seq_k 为普通长度(B,)
    actual_seq_q = torch.tensor([q_len], dtype=torch.int32)
    actual_seq_k = torch.tensor([kv_len], dtype=torch.int32)
    if entry == "aclnn":
        npu_idx, npu_val = li_op(
            query.npu(), key.npu(), weights.npu(),
            actual_seq_q.npu(), actual_seq_k.npu(), block_table.npu(),
            init_tensor.npu(), local_tensor.npu(),
            "TND", "PA_BSND", sparse_count, sparse_mode, True)
        entry_name = f"aclnnLightningIndexer (vendor: {os.environ.get('LI_VENDOR_OPAPI', DEFAULT_VENDOR_OPAPI)})"
    else:
        npu_idx, npu_val = li_op(
            query.npu(), key.npu(), weights.npu(),
            actual_seq_lengths_query=actual_seq_q.npu(),
            actual_seq_lengths_key=actual_seq_k.npu(),
            block_table=block_table.npu(),                  # (B, max_blocks)
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            init_tensor=init_tensor.npu(),
            local_tensor=local_tensor.npu(),
            sparse_mode=sparse_mode,
        )
        entry_name = ("torch.ops.custom.npu_lightning_indexer (flash_ops)" if entry == "custom_li"
                      else "torch_npu.npu_lightning_indexer (built-in)")
    npu_idx = npu_idx.cpu()
    npu_val = npu_val.cpu()

    # ---------- 打印 topk_indices ----------
    print("=" * 70)
    print(f"entry: {entry_name}")
    print(f"config: topk={sparse_count}, q_len={q_len}, kv_len={kv_len}, "
          f"init={init_num}, local={local_num}, n1={n1}, "
          f"layout=TND/PA_BSND, block_size={block_size}, dtype={dtype}")
    print(f"block_table: {block_table.tolist()}")
    for s in range(q_len):
        valid_len = kv_len - q_len + s + 1
        print(f"--- row s1={s} (valid kv = {valid_len}) ---")
        print("golden indices:", golden_idx[s, 0].tolist())
        print("npu    indices:", npu_idx[s, 0].tolist())
        if npu_val.numel() > 0:
            print("npu    values :", [round(v, 4) for v in npu_val[s, 0].float().tolist()])
    print("=" * 70)

    # ---------- 校验 ----------
    for s in range(q_len):
        valid_len = kv_len - q_len + s + 1
        npu_row = npu_idx[s, 0].tolist()
        golden_row = golden_idx[s, 0].tolist()

        # 1) topk=48 <= 最短有效行125, 不应出现 -1
        assert -1 not in npu_row, f"row {s}: unexpected -1 in npu indices: {npu_row}"

        # 2) init/local 强制入选: 前init个 + 最后local个有效key 必须都在结果里
        forced = set(range(init_num)) | set(range(valid_len - local_num, valid_len))
        missing = forced - set(npu_row)
        assert not missing, f"row {s}: forced init/local indices missing: {sorted(missing)}"

        # 3) 与 golden 逐行集合比较(并列名次顺序不比)
        assert set(npu_row) == set(golden_row), \
            f"row {s}: index set mismatch\n npu={npu_row}\n golden={golden_row}"

        # 4) 所有索引都在有效范围内
        assert all(0 <= i < valid_len for i in npu_row), \
            f"row {s}: index out of valid range [0, {valid_len}): {npu_row}"

    print(f"PASS: topk48 q4 kv128 init16 local16 (TND + PA_BSND)\nentry={entry_name}")


if __name__ == "__main__":
    test_topk48_q4_kv128_init16_local16_tnd_pa()
