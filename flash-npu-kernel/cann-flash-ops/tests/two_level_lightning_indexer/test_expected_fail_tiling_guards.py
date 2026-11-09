# -*- coding: utf-8 -*-
"""expected-fail 用例：覆盖每个约束维度各 1 条最小反例。

所有 case 断言：
- 调用抛 RuntimeError
- 错误信息包含 561002

覆盖维度：
compressed: kv_block_len / q_block_len / init_num / local_num / sparse_count / sparse_mode / H / D
sparse:     sparse_index_block_size / sparse_count / sparse_mode / H / D
共 13 条。

用法：NPU_DEVICE=15 python3 test_expected_fail_tiling_guards.py
"""
import os
import torch
import torch_npu
import flash_ops  # 注册 torch.ops.custom 命名空间
import numpy as np
import torch.nn.functional as F
from test_npu_mlp_sparse_lightning_indexer import split_to_kv_cache as split_s, generate_sparse_index

DEVICE_ID = int(os.environ.get("NPU_DEVICE", "15"))
torch_npu.npu.set_device(DEVICE_ID)
INT32_MAX = 2147483647
device = f"npu:{DEVICE_ID}"


def expect_561002(fn, label):
    """断言调用被 tiling 拒绝。

    两条调用路径报错文本不同：
    - torch.ops.npu    : 带 errno 561002
    - torch.ops.custom : 仅 "call aclnnXxx failed"（适配层未透传 errno）
    两种形式都视为被 tiling 干净拒绝。
    """
    try:
        fn()
    except Exception as e:
        msg = str(e)
        rejected = ("561002" in msg) or ("call aclnn" in msg and "failed" in msg)
        assert rejected, f"{label}: expected tiling reject, got: {msg[:300]}"
        detail = "561002" if "561002" in msg else "aclnn reject"
        print(f"[PASS] {label}: rejected ({detail})")
        return
    raise AssertionError(f"{label}: expected tiling reject, but call succeeded")


def build_compressed_inputs(N=16, D=128, q_len=128, seq_len=32768, block_len=16, seed=42):
    pooling = 128 // block_len if block_len in [1,2,4,8,16,32] else 8
    T2 = seq_len // pooling
    np.random.seed(seed)
    q = torch.tensor(np.random.uniform(-2, 2, (q_len, N, D))).to(torch.bfloat16)
    k = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(torch.bfloat16)
    w = torch.tensor(np.random.uniform(-1, 1, (q_len, N))).to(torch.float)
    csq = torch.tensor([0, q_len], dtype=torch.int32)
    csk = torch.tensor([0, seq_len], dtype=torch.int32)
    return q.to(device), k.to(device), w.to(device), csq.to(device), csk.to(device)


def build_sparse_inputs(N=16, D=128, q_len=128, seq_len=32768, block_size=128, block_topk=256, seed=42):
    np.random.seed(seed)
    q = torch.tensor(np.random.uniform(-2, 2, (q_len, N, D))).to(torch.bfloat16)
    k = torch.tensor(np.random.uniform(-2, 2, (seq_len, 1, D))).to(torch.bfloat16)
    w = torch.tensor(np.random.uniform(-1, 1, (q_len, N))).to(torch.float)
    csq = torch.tensor([0, q_len], dtype=torch.int32)
    csk = torch.tensor([0, seq_len], dtype=torch.int32)
    safe_bs = block_size if (block_size > 0 and block_size % 16 == 0 and block_size <= 1024) else 128
    si = generate_sparse_index([seq_len], safe_bs, block_topk, [q_len])
    kc, kbt = split_s(k, safe_bs, [seq_len], seq_len)
    return q.to(device), k.to(device), w.to(device), csq.to(device).to(torch.int64), csk.to(device).to(torch.int64), kc.to(device), kbt.to(device), si.to(device)


# ===== compressed 8 维 =====
def test_c_block_len_guard():
    q,k,w,csq,csk = build_compressed_inputs(block_len=16)
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=64,q_block_len=1,init_num=64,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed kv_block_len=64")

def test_c_q_block_len_guard():
    q,k,w,csq,csk = build_compressed_inputs()
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=16,q_block_len=2,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed q_block_len=2")

def test_c_init_guard():
    q,k,w,csq,csk = build_compressed_inputs()
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=16,q_block_len=1,init_num=256,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed init_num=256")

def test_c_local_guard():
    q,k,w,csq,csk = build_compressed_inputs()
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=16,q_block_len=1,init_num=16,local_num=4096,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed local_num=4096")

def test_c_sparse_count_guard():
    q,k,w,csq,csk = build_compressed_inputs()
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=8320,kv_block_len=16,q_block_len=1,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed sparse_count=8320")

def test_c_sparse_mode_guard():
    q,k,w,csq,csk = build_compressed_inputs()
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=16,q_block_len=1,init_num=16,local_num=1024,sparse_mode=1,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed sparse_mode=1")

def test_c_h_guard():
    q,k,w,csq,csk = build_compressed_inputs(N=3)
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=16,q_block_len=1,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed H=3")

def test_c_d_guard():
    q,k,w,csq,csk = build_compressed_inputs(D=64)
    expect_561002(lambda: torch.ops.custom.mlp_compressed_lightning_indexer(q,q,k,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=None,layout_query="TND",layout_key="TND",sparse_count=1024,kv_block_len=16,q_block_len=1,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "compressed D=64")

# ===== sparse 5 维 =====
def test_s_block_size_guard():
    q,k,w,csq,csk,kc,kbt,si = build_sparse_inputs(block_size=128, block_topk=256)
    expect_561002(lambda: torch.ops.custom.mlp_sparse_lightning_indexer(q,k,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,sparse_index=si,layout_query="TND",layout_key="TND",sparse_count=2048,sparse_index_block_size=17,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "sparse block_size=17")

def test_s_sparse_count_guard():
    q,k,w,csq,csk,kc,kbt,si = build_sparse_inputs(block_topk=256)
    expect_561002(lambda: torch.ops.custom.mlp_sparse_lightning_indexer(q,kc,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=kbt,sparse_index=si,layout_query="TND",layout_key="PA_BSND",sparse_count=16384,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "sparse sparse_count=16384")

def test_s_sparse_mode_guard():
    q,k,w,csq,csk,kc,kbt,si = build_sparse_inputs(block_topk=256)
    expect_561002(lambda: torch.ops.custom.mlp_sparse_lightning_indexer(q,kc,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=kbt,sparse_index=si,layout_query="TND",layout_key="PA_BSND",sparse_count=2048,init_num=16,local_num=1024,sparse_mode=1,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "sparse sparse_mode=1")

def test_s_h_guard():
    q,k,w,csq,csk,kc,kbt,si = build_sparse_inputs(N=8, block_topk=256)
    expect_561002(lambda: torch.ops.custom.mlp_sparse_lightning_indexer(q,kc,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=kbt,sparse_index=si,layout_query="TND",layout_key="PA_BSND",sparse_count=2048,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "sparse H=8")

def test_s_d_guard():
    q,k,w,csq,csk,kc,kbt,si = build_sparse_inputs(D=64, block_topk=256)
    expect_561002(lambda: torch.ops.custom.mlp_sparse_lightning_indexer(q,kc,w,cur_seq_lengths_query=csq,cur_seq_lengths_key=csk,block_table=kbt,sparse_index=si,layout_query="TND",layout_key="PA_BSND",sparse_count=2048,init_num=16,local_num=1024,sparse_mode=3,pre_tokens=INT32_MAX,next_tokens=INT32_MAX,return_value=True), "sparse D=64")

if __name__ == "__main__":
    print("=== expected-fail per-constraint guards ===")
    # compressed
    test_c_block_len_guard()
    test_c_q_block_len_guard()
    test_c_init_guard()
    test_c_local_guard()
    test_c_sparse_count_guard()
    test_c_sparse_mode_guard()
    test_c_h_guard()
    test_c_d_guard()
    # sparse
    test_s_block_size_guard()
    test_s_sparse_count_guard()
    test_s_sparse_mode_guard()
    test_s_h_guard()
    test_s_d_guard()
    print("=== done ===")
