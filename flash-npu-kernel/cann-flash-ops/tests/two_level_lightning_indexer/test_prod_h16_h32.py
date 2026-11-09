# -*- coding: utf-8 -*-
"""生产配置测试: 只 H∈{16,32}(3B/pro), 其余全生产值, mode3, q2048.
2 case: 每条是完整级联 (level1 compressed -> level2 sparse) + golden 比对."""
import os, sys, io, contextlib
import torch, torch_npu, numpy as np
import torch.nn.functional as F
import flash_ops  # 注册 torch.ops.custom 命名空间
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_mlp_sparse_lightning_indexer import golden_lightning_indexer_with_sparse_index, split_to_kv_cache as split_s, compare_indices_with_value_tolerance
DEVICE_ID = int(os.environ.get("NPU_DEVICE", "15")); torch_npu.npu.set_device(DEVICE_ID)
INT32_MAX = 2147483647; device = "npu:%d" % DEVICE_ID
Q_LEN, SEQ_LEN, D = 2048, 32768, 128
def test_cascade(N):
    bl, ci, cl, csc, mode = 16, 16, 1024, 1024, 3
    sbs, ssc, sinit, slocal = 128, 2048, 16, 1024
    pooling = 128 // bl; T2 = SEQ_LEN // pooling
    np.random.seed(42)
    q = torch.tensor(np.random.uniform(-2, 2, (Q_LEN, N, D))).to(torch.bfloat16)
    k = torch.tensor(np.random.uniform(-2, 2, (SEQ_LEN, 1, D))).to(torch.bfloat16)
    w = torch.tensor(np.random.uniform(-1, 1, (Q_LEN, N))).to(torch.float)
    k_avg = F.pad(k.squeeze(1), [0, 0, 0, (pooling - SEQ_LEN % pooling) % pooling], value=0).reshape(-1, pooling, D).mean(1).unsqueeze(1).contiguous()
    csq = torch.tensor([0, Q_LEN], dtype=torch.int32); csk = torch.tensor([0, SEQ_LEN], dtype=torch.int32)
    l1_out, _ = torch.ops.custom.mlp_compressed_lightning_indexer(
        q.to(device), q.to(device), k_avg.to(device), k_avg.to(device), w.to(device),
        cur_seq_lengths_query=csq.to(device), cur_seq_lengths_key=csk.to(device), block_table=None,
        layout_query="TND", layout_key="TND", sparse_count=csc, kv_block_len=bl, q_block_len=1,
        init_num=ci, local_num=cl, sparse_mode=mode, pre_tokens=INT32_MAX, next_tokens=INT32_MAX, return_value=False)
    si = l1_out.reshape(Q_LEN, csc).cpu().to(torch.int32)
    kc, kbt = split_s(k, sbs, [SEQ_LEN], SEQ_LEN)
    l2_out, l2_val = torch.ops.custom.mlp_sparse_lightning_indexer(
        q.to(device), kc.to(device), w.to(device),
        cur_seq_lengths_query=csq.to(device).to(torch.int64), cur_seq_lengths_key=csk.to(device).to(torch.int64),
        block_table=kbt.to(device), sparse_index=si.to(device), layout_query="TND", layout_key="PA_BSND",
        sparse_count=ssc, init_num=sinit, local_num=slocal, sparse_mode=mode,
        pre_tokens=INT32_MAX, next_tokens=INT32_MAX, return_value=True)
    with contextlib.redirect_stdout(io.StringIO()):
        cpu_l2, cpu_l2v = golden_lightning_indexer_with_sparse_index(
            q, k, w, csq, csk, si, sbs, ssc, sparse_mode=0, INIT_NUM=sinit, LOCAL_NUM=slocal)
    co = cpu_l2.reshape(-1, ssc).cpu(); no = l2_out.reshape(-1, ssc).cpu()
    cv = cpu_l2v.reshape(-1, ssc).cpu().to(torch.bfloat16).to(torch.float32)
    nv = l2_val.reshape(-1, ssc).cpu().to(torch.float32)
    ok, _ = compare_indices_with_value_tolerance(co, no, cv, nv, ssc, rtol=5e-3, verbose=False)
    return ok
if __name__ == "__main__":
    print("=== 生产级联测试 (H16/H32, mode3, q2048) ===")
    for N in [16, 32]:
        label = "3B" if N == 16 else "pro"
        ok = test_cascade(N)
        print(f"  cascade {label}(H={N}): {'PASS' if ok else 'FAIL'}")
    print("=== done ===")
