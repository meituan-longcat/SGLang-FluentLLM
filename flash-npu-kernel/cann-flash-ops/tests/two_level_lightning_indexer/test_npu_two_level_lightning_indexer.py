"""
两级稀疏索引联合测试（MlpCompressedLightningIndexer + MlpSparseLightningIndexer）。

端到端串联两个算子，验证 two-level k-avg 稀疏索引流程：
  Level 1 (MlpCompressedLightningIndexer):
      以 k_avg（每 LEVEL1_SIZE 个 token 平均为一个块摘要）粗筛，
      从全部 page 中选出 level1_topk 个候选 block 索引。
  Level 2 (MlpSparseLightningIndexer):
      将 Level 1 选出的候选 block 索引作为 sparse_index 输入，
      在候选 block 内细筛出 level2_topk 个 token 索引。

每一级均与 CPU golden（einsum 手算 + topk）做对称差集（symmetric difference）比对。

运行：python3 test_npu_two_level_lightning_indexer.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
import flash_ops  # 注册 torch.ops.custom 命名空间

DEVICE_ID = 12
torch_npu.npu.set_device(int(DEVICE_ID))
torch.set_printoptions(precision=6, sci_mode=False)

PAGE_SIZE = 128
MAX_SEQ_LEN = 1024 * 1024
MAX_PAGE_NUM = MAX_SEQ_LEN // PAGE_SIZE
LEVEL1_SIZE = 16
KV_BLOCK_LEN = PAGE_SIZE // LEVEL1_SIZE

def ceil_div(a:int, b:int):
    return (a+b-1) // b

class ModelConfig:
    init_tokens = 16
    local_tokens = 1024
    init_pages = ceil_div(init_tokens, PAGE_SIZE) # 1
    local_pages = local_tokens // PAGE_SIZE + 1 # 9
    level1_topk = 256
    level2_topk = 2048

def print_symmetric_difference(a, b):
    a = a.cpu().flatten()
    b = b.cpu().flatten()
    """对称差集：在a或b中但不同时在两者中的元素"""
    # 找出在a但不在b的元素
    a_unique = a[~torch.isin(a, b)]
    # 找出在b但不在a的元素
    b_unique = b[~torch.isin(b, a)]
    if a_unique.numel() != 0 or b_unique.numel() != 0:
        print("check fail")
        print("在a但不在b的元素: ", a_unique, a_unique.numel())
        print("在b但不在a的元素: ", b_unique, b_unique.numel())
        return False
    else:
        print("check pass")
        return True

def get_k_avg_pad_k(
    all_k: torch.Tensor, # [l, d]
):
    l, d = all_k.shape
    assert PAGE_SIZE % LEVEL1_SIZE == 0
    padded_k = F.pad(
        all_k,
        [0, 0, 0, (PAGE_SIZE - l % PAGE_SIZE) % PAGE_SIZE],
        value=0
    )

    # [l/LEVEL1_SIZE, d]
    k_avg = padded_k.reshape(-1, LEVEL1_SIZE, d).mean(dim=1)
    return k_avg, padded_k

def sort_tensor_rows(tensor):
    # 1. 确定用于替换的“极大值”
    max_val = torch.iinfo(tensor.dtype).max # 整数用该类型的最大值

    # 2. 将所有 -1（即小于0的数）替换为极大值
    # 使用 tensor < 0 比 tensor == -1 更安全
    temp_tensor = torch.where(tensor < 0, max_val, tensor)
    # 3. 按行（dim=1）进行升序排序
    sorted_temp, _ = torch.sort(temp_tensor, dim=-1)

    # 4. 把排在最后面的极大值，重新替换回 -1
    result = torch.where(sorted_temp == max_val, -1, sorted_temp)

    return result

def each_level2(
    q: torch.Tensor, # [s, h, d]
    all_k: torch.Tensor, # [l, d]
    w: torch.Tensor, # [s, h]
    seqlens_k: torch.Tensor, # [s]
):
    s, h, d = q.shape
    l, dk = all_k.shape
    assert d == dk
    remainder = l % PAGE_SIZE
    # [l/16, d]
    k_avg, pad_k = get_k_avg_pad_k(all_k)

    level2_idxs = torch.full([s, ModelConfig.level2_topk], -1, dtype=torch.int64)
    #####
    q_npu = q.to("npu:%s" % DEVICE_ID)

    k_avg_npu = k_avg.to("npu:%s" % DEVICE_ID)
    k_avg_npu = k_avg_npu.unsqueeze(1)

    weight_npu = w.to("npu:%s" % DEVICE_ID)

    cur_seq_lengths_query = torch.tensor([0, s]).to(torch.int64)
    cur_seq_lengths_key = torch.tensor([0, l]).to(torch.int64) #原始seq_k
    cur_seq_lengths_query = cur_seq_lengths_query.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_key = cur_seq_lengths_key.to("npu:%s" % DEVICE_ID)
    print("cur_seq_lengths_query ", cur_seq_lengths_query)
    print("cur_seq_lengths_key ", cur_seq_lengths_key)

    npu_out, npu_valuesOut = torch.ops.custom.mlp_compressed_lightning_indexer(
            q_npu, q_npu, k_avg_npu, k_avg_npu, weight_npu, cur_seq_lengths_query=cur_seq_lengths_query,
                cur_seq_lengths_key=cur_seq_lengths_key, block_table=None, layout_query="TND",
                layout_key="TND",
                sparse_count=ModelConfig.level1_topk,
                kv_block_len=KV_BLOCK_LEN, q_block_len=1,
                init_num=ModelConfig.init_pages * KV_BLOCK_LEN,
                local_num=ModelConfig.local_pages * KV_BLOCK_LEN,
                sparse_mode=3, pre_tokens=2147483647,
                next_tokens=2147483647, return_value=True)
    pad_k_npu = pad_k.to("npu:%s" % DEVICE_ID)
    pad_k_npu = pad_k_npu.unsqueeze(1)
    q_npu = q.to("npu:%s" % DEVICE_ID)
    print("first level 1 npu_valuesOut shape is ", npu_valuesOut.shape)
    # mask = npu_out >= 0
    # npu_out1 = sort_tensor_rows(npu_out)
    npu_out1 = npu_out

    npu_out1 = npu_out1.squeeze(1)
    print("pad_k_npu shape is ", pad_k_npu.shape)
    l2, n2, d = pad_k_npu.shape
    all_k_npu = all_k.to("npu:%s" % DEVICE_ID)
    all_k_npu = all_k_npu.unsqueeze(1)
    print("all_k_npu shape is ", all_k_npu.shape)
    cur_seq_lengths_key_org = torch.tensor([0, l]).to(torch.int64)
    cur_seq_lengths_key_org = cur_seq_lengths_key_org.to("npu:%s" % DEVICE_ID)
    npu_out2, npu_valuesOut2 = torch.ops.custom.mlp_sparse_lightning_indexer(
        q_npu, all_k_npu, weight_npu,
        cur_seq_lengths_query=cur_seq_lengths_query,
        cur_seq_lengths_key=cur_seq_lengths_key_org,
        sparse_index=npu_out1,
        layout_query="TND",
        layout_key="TND",
        sparse_count=ModelConfig.level2_topk,
        sparse_index_block_size=PAGE_SIZE,
        init_num=ModelConfig.init_tokens,
        local_num=ModelConfig.local_tokens,
        sparse_mode=3,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        return_value=True,
    )
    for i in range(s):
        qi = q[i]
        # [h]
        wi = w[i]
        seqlen = seqlens_k[i].item()
        print("i is ", i, " seqlen ", seqlen)
        page_num = ceil_div(seqlen, PAGE_SIZE)
        # [h, l/16]
        qxk = torch.einsum("hd,ld->hl", qi.float(), k_avg.float())
        qxk = F.relu(qxk)
        scores = (wi.unsqueeze(1) * qxk).sum(0)
        scores = scores.reshape(-1, PAGE_SIZE // LEVEL1_SIZE).sum(1)
        tmp_value = PAGE_SIZE // LEVEL1_SIZE
        scores = scores / tmp_value # 平均数，对比输出value
        scores = scores[:page_num]
        scores[:ModelConfig.init_pages] = float(88888) # 必选
        scores[-ModelConfig.local_pages:] = float(88888) # 必选
        value, idxs = torch.topk(scores, k=min(page_num, ModelConfig.level1_topk), dim=0)
        level1_idxs, _ = idxs.sort(dim=0)
        # value, idxs = torch.sort(scores, dim=0, descending=True, stable=True)
        # k=min(page_num, ModelConfig.level1_topk)
        # idxs = idxs[:k + 1]


        tmp_out = npu_out1[i, :]
        result = tmp_out[tmp_out != -1]
        ret = print_symmetric_difference(idxs, result)
        if ret == False:
            print("npu level 1 last value is ", npu_valuesOut[i, ..., -1])
            print("golden level 1 last value is ", value[-1])
        pad_k = pad_k.reshape(-1, PAGE_SIZE, d)
        # [level1topk, PAGE_SIZE, d] -> [l', d]
        ki = pad_k[level1_idxs].reshape(-1, d)
        qxk = torch.einsum("hd,ld->hl", qi.float(), ki.float())
        qxk = qxk[:,:qxk.shape[1] - (-seqlen % PAGE_SIZE)]
        scores = (wi.unsqueeze(1) * F.relu(qxk)).sum(0)
        scores[:ModelConfig.init_tokens] = float(88888)
        scores[-ModelConfig.local_tokens:] = float(88888)
        my_values, idxs = torch.topk(scores, k=min(seqlen, ModelConfig.level2_topk), dim=0)
        # this is token idxs in 256 pages, then transform them to seq level token idxs
        seq_token_idxs = level1_idxs[idxs // PAGE_SIZE] * PAGE_SIZE + idxs % PAGE_SIZE
        k=min(seqlen, ModelConfig.level2_topk)
        seq_token_idxs = seq_token_idxs[:k]
        ret = print_symmetric_difference(seq_token_idxs, npu_out2[i, :])
        if ret == False:
            print("npu level 2 last value is ", npu_valuesOut2[i, ..., -1])
            print("golden level 2 last value is ", my_values[-1])
        level2_idxs[i, :seq_token_idxs.shape[0]] = seq_token_idxs
    return level2_idxs


def main():
    torch.manual_seed(42)

    s = 128
    h = 32
    d = 128
    l = 32768*2+888
    seqlens_k = torch.arange(l - s + 1, l+1)
    dtype = torch.bfloat16

    q = torch.randn(s, h, d, dtype=dtype)
    all_k = torch.randn(l, d, dtype=dtype)
    w = torch.randn(s, h, dtype=torch.float)

    ref_level2_idxs = each_level2(q, all_k, w, seqlens_k)
    print(ref_level2_idxs)
    ref_level2_idxs = torch.where(ref_level2_idxs==-1, 999999, ref_level2_idxs)
    idxs, _ = torch.sort(ref_level2_idxs, dim=1)
    idxs = torch.where(idxs == 999999, -1, idxs)

if __name__ == "__main__":
    main()
