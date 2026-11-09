# DCP rank0 local_cnt 少算当前 token 的算子级探针 (对应 tasks/dsa_dcp_local_cnt_audit.md)
#
# 复刻 DCP rank0 输入形态:
#   - 有效序列 = hist 个历史 token + q_len 个当前 token(拼在段尾)
#   - 调用方(sglang build_req_level_owner_stats)传 local_tensor = 窗口内仅历史 token 数
#     = num_local - 1 ("+1" 只给 1 个当前 token 留位, verify 时 q_len>1 就少算 j+1)
#   - q 全零 => 所有真实 score = 0, 被强选的 entry value = SECOND_MAX_LOGITS ≈ 1.7e38 成为哨兵
#
# 判定: 每行 value>1e30 的索引集合 = 实际强选集合。
#   调用方意图: 历史窗口 {hist+1-num_local .. hist-1} + 当前 token 全部有保障
#   若算子强选 = {valid-cnt .. valid-1} (含段尾当前 token),
#   则历史窗口最老的 j+1 个 {hist+1-num_local .. hist-num_local+j} 不在强选集 => 漏算实锤
#
# 运行: source set_env.sh && python3 test_npu_lightning_indexer_dcp_local_probe.py

import torch
import torch_npu
import numpy as np

from test_npu_lightning_indexer_topk48_init_local import AclnnLightningIndexer, DEFAULT_VENDOR_OPAPI

torch_npu.npu.set_device(0)

HIST = 100          # 历史 token 数
NUM_LOCAL = 4       # 设计意图: 末尾 4 个历史 token 必须有强选保障
SPARSE_COUNT = 16   # < kv_len, 保证不是全覆盖
N1, D = 64, 128
BOOST_THR = 1e30    # 强选哨兵阈值


def run_case(li, q_len, local_cnt):
    """q_len 个当前 token 拼在 hist 之后, 每行都传 local_cnt(per-request 标量的逐行复制),
    返回每行强选集合。注: 算子自 2026-08-07 起只接受 per-token [T]; 逐行同值 == 旧 [B] 标量语义。"""
    kv_len = HIST + q_len
    query = torch.zeros(q_len, N1, D, dtype=torch.bfloat16)     # q=0 => 真实score=0
    key = torch.randn(kv_len, 1, D).bfloat16()                  # TND key, 内容不影响(score恒0)
    weights = torch.ones(q_len, N1, dtype=torch.bfloat16)
    asq = torch.tensor([q_len], dtype=torch.int32)              # TND 前缀和(B=1)
    ask = torch.tensor([kv_len], dtype=torch.int32)             # TND 前缀和(B=1)
    init_t = torch.tensor([0], dtype=torch.int32)               # 关闭 init 强选
    local_t = torch.tensor([local_cnt] * q_len, dtype=torch.int32)  # per-token [T], 逐行同值

    idx, val = li(query.npu(), key.npu(), weights.npu(),
                  asq.npu(), ask.npu(), None,
                  init_t.npu(), local_t.npu(),
                  "TND", "TND", SPARSE_COUNT, 3, True)
    idx, val = idx.cpu(), val.cpu().float()
    forced = []
    for j in range(q_len):
        row_forced = sorted(idx[j, 0][val[j, 0] > BOOST_THR].tolist())
        forced.append(set(row_forced))
    return forced


def run_case_pertoken(li, hists, q_lens, local_counts_per_req):
    """per-token local_tensor 模式: local_tensor[t] = 该行(请求内第 j 行)的强选窗口宽度。

    hists: 各请求历史 token 数; q_lens: 各请求当前 token 数;
    local_counts_per_req: [req][j] = 该行的 cnt。返回 [req][j] = 强选索引集合(请求内偏移)。
    """
    kv_lens = [h + q for h, q in zip(hists, q_lens)]
    total_q, total_kv = sum(q_lens), sum(kv_lens)
    query = torch.zeros(total_q, N1, D, dtype=torch.bfloat16)
    key = torch.randn(total_kv, 1, D).bfloat16()
    weights = torch.ones(total_q, N1, dtype=torch.bfloat16)
    asq = torch.tensor(np.cumsum(q_lens), dtype=torch.int32)    # TND 前缀和
    ask = torch.tensor(np.cumsum(kv_lens), dtype=torch.int32)
    init_t = torch.tensor([0] * len(hists), dtype=torch.int32)  # 关闭 init 强选
    local_t = torch.tensor([c for req in local_counts_per_req for c in req], dtype=torch.int32)
    assert local_t.shape == (total_q,)

    idx, val = li(query.npu(), key.npu(), weights.npu(),
                  asq.npu(), ask.npu(), None,
                  init_t.npu(), local_t.npu(),
                  "TND", "TND", SPARSE_COUNT, 3, True)
    idx, val = idx.cpu(), val.cpu().float()
    forced_per_req = []
    t = 0
    for r, q_len in enumerate(q_lens):
        rows = []
        for j in range(q_len):
            rows.append(set(sorted(idx[t, 0][val[t, 0] > BOOST_THR].tolist())))
            t += 1
        forced_per_req.append(rows)
    return forced_per_req


def test_pertoken(li):
    """路线 B 修复验证: per-token local_tensor, 逐行断言强选集 = 意图集(无多无少)"""
    print("=" * 72)
    print("[per-token 模式] local_tensor[t] = w_j + (j+1), 逐行精确断言")
    # --- B=1 verify: 意图窗口 = 历史 {hist+1-num_local..hist-1} + 当前 {hist..hist+j} ---
    q_len = 4
    intended_hist = set(range(HIST + 1 - NUM_LOCAL, HIST))
    w_j = NUM_LOCAL - 1                                    # 窗口内归本 rank 的历史数(探针内恒定)
    counts = [w_j + (j + 1) for j in range(q_len)]         # 路线 B 调用方应传值
    forced = run_case_pertoken(li, [HIST], [q_len], [counts])[0]
    for j in range(q_len):
        valid_len = HIST + j + 1
        intended = intended_hist | set(range(HIST, HIST + j + 1))
        expect = set(range(valid_len - counts[j], valid_len))
        assert intended == expect, "探针自洽性失败: 意图集 != 末尾cnt个"
        assert forced[j] == intended, \
            f"per-token 行 j={j} 强选集 != 意图集: 多{sorted(forced[j] - intended)} 少{sorted(intended - forced[j])}"
        print(f"  row j={j} (cnt={counts[j]}): 强选={sorted(forced[j])} == 意图集 ✓")
    # --- B=1 verify: 逐行 cnt 非均匀, 验证逐行独立取值 ---
    counts = [2, 5, 3, 7]
    forced = run_case_pertoken(li, [HIST], [q_len], [counts])[0]
    for j in range(q_len):
        valid_len = HIST + j + 1
        expect = set(range(valid_len - counts[j], valid_len))
        assert forced[j] == expect, \
            f"非均匀 cnt 行 j={j}: 强选={sorted(forced[j])} != 预测末尾{counts[j]}个={sorted(expect)}"
    print(f"  非均匀 cnt={counts}: 4 行强选集与逐行预测全部精确吻合 ✓")
    # --- B=2: 验证 q 前缀和索引(第二请求 qPrefix != 0) ---
    hists, q_lens = [50, 70], [2, 3]
    counts_req = [[3, 6], [4, 2, 5]]
    forced_req = run_case_pertoken(li, hists, q_lens, counts_req)
    for r, q_len_r in enumerate(q_lens):
        for j in range(q_len_r):
            valid_len = hists[r] + j + 1
            cnt = counts_req[r][j]
            expect = set(range(valid_len - cnt, valid_len))
            assert forced_req[r][j] == expect, \
                f"B=2 请求{r} 行 j={j}: 强选={sorted(forced_req[r][j])} != 预测={sorted(expect)}"
    print(f"  B=2 (q_lens={q_lens}, 请求1 qPrefix=2): 5 行强选集全部精确吻合 ✓")
    print("PER-TOKEN PROBE PASS")


def run_case_init(li, hist, q_len, num_init, local_cnts, sparse_count=SPARSE_COUNT):
    """带 init_tensor 的探针: local_cnts 为逐行 cnt 列表(len==q_len), 返回每行强选集合。
    sparse_count 须 >= 预期强选数, 否则强选 entry 被 topk 输出截断而观测不全。"""
    kv_len = hist + q_len
    query = torch.zeros(q_len, N1, D, dtype=torch.bfloat16)
    key = torch.randn(kv_len, 1, D).bfloat16()
    weights = torch.ones(q_len, N1, dtype=torch.bfloat16)
    asq = torch.tensor([q_len], dtype=torch.int32)
    ask = torch.tensor([kv_len], dtype=torch.int32)
    init_t = torch.tensor([num_init], dtype=torch.int32)
    local_t = torch.tensor(local_cnts, dtype=torch.int32)
    assert local_t.shape == (q_len,)
    idx, val = li(query.npu(), key.npu(), weights.npu(),
                  asq.npu(), ask.npu(), None,
                  init_t.npu(), local_t.npu(),
                  "TND", "TND", sparse_count, 3, True)
    idx, val = idx.cpu(), val.cpu().float()
    return [set(sorted(idx[j, 0][val[j, 0] > BOOST_THR].tolist())) for j in range(q_len)]


def test_init_gate(li):
    """Returned scores must retain forced priority even when local windows cover the row."""
    print("=" * 72)
    print("[init 门控] return_value=True: 全覆盖时仍保留强选分数")
    # Each causal row has 2..17 valid tokens; promote its valid init prefix.
    forced = run_case_init(li, 1, 16, 16, [0] * 16)
    for j in range(16):
        expect = set(range(min(16, j + 2)))
        assert forced[j] == expect, f"行 j={j}: 强选={sorted(forced[j])} != {sorted(expect)}"
    # --- 用例2: local<init 判别区间, hist=40, q=4, init=32, local=0 ---
    # seq=41..44 ∈ (32, 64]: 旧门控(2*32=64>=seq)全跳过 => 强选为空(FAIL); 新门控每行强选 [0,32)
    # sparse_count=40: >= 32 个强选 entry 可完整观测, 且 < seq 保持 topk 非全覆盖
    forced = run_case_init(li, 40, 4, 32, [0] * 4, sparse_count=40)
    for j in range(4):
        assert forced[j] == set(range(32)), \
            f"用例2 行 j={j}: 强选={sorted(forced[j])} != 前32个 (旧门控此处会整行漏强选)"
    print("  用例2 (hist=40,q=4,init=32,local=0): 4 行全部强选前32个 ✓ (旧门控此处为空调过)")
    # Overlapping init/local windows cover all 21 valid tokens. The output
    # budget covers the full forced set so no tie truncation obscures it.
    forced = run_case_init(li, 20, 1, 16, [8], sparse_count=24)
    assert forced[0] == set(range(21)), f"全覆盖仍须提分: {sorted(forced[0])}"
    # --- 用例4: 双强选回归, init=16, local=4, hist=100, q=1, seq=101 > 20 ---
    # sparse_count=24: >= 20 个强选 entry 可完整观测
    forced = run_case_init(li, 100, 1, 16, [4], sparse_count=24)
    expect = set(range(16)) | {97, 98, 99, 100}
    assert forced[0] == expect, f"用例4: 强选={sorted(forced[0])} != {sorted(expect)}"
    print("  用例4 (seq=101 > init16+local4): 前16 + 末4 双强选精确 ✓")
    print("INIT-GATE PROBE PASS")


def test_reject_legacy_B_shape(li):
    """[B] 形状(B != T)必须被 tiling 显式拒绝(只支持 per-token [T])"""
    q_len = 4
    kv_len = HIST + q_len
    query = torch.zeros(q_len, N1, D, dtype=torch.bfloat16)
    key = torch.randn(kv_len, 1, D).bfloat16()
    weights = torch.ones(q_len, N1, dtype=torch.bfloat16)
    asq = torch.tensor([q_len], dtype=torch.int32)
    ask = torch.tensor([kv_len], dtype=torch.int32)
    init_t = torch.tensor([0], dtype=torch.int32)
    local_t = torch.tensor([NUM_LOCAL - 1], dtype=torch.int32)  # [B=1], T=4, 应被拒绝
    try:
        li(query.npu(), key.npu(), weights.npu(),
           asq.npu(), ask.npu(), None,
           init_t.npu(), local_t.npu(),
           "TND", "TND", SPARSE_COUNT, 3, True)
    except BaseException as e:
        print(f"  [B] 形状被显式拒绝 ✓ ({type(e).__name__})")
        print("REJECT-[B] PASS")
        return
    raise AssertionError("[B] 形状(B != T)未被拒绝, [T]-only 约束失效!")


def main():
    li = AclnnLightningIndexer(DEFAULT_VENDOR_OPAPI)
    # 调用方实际传的值: 窗口 [hist+1-num_local, hist) 内的历史 token 数 = num_local-1
    caller_cnt = NUM_LOCAL - 1

    for q_len, tag in [(1, "decode"), (4, "target_verify")]:
        kv_len = HIST + q_len
        print("=" * 72)
        print(f"[{tag}] hist={HIST}, q_len={q_len}, kv_len={kv_len}, "
              f"num_local={NUM_LOCAL}, 调用方传 local_tensor={caller_cnt}")
        intended_hist = set(range(HIST + 1 - NUM_LOCAL, HIST))   # 意图保障的历史窗口
        forced = run_case(li, q_len, caller_cnt)
        all_ok = True
        for j in range(q_len):
            valid_len = HIST + j + 1
            expect_forced = set(range(valid_len - caller_cnt, valid_len))  # 算子语义: 末尾cnt个(含段尾)
            missed = intended_hist - forced[j]
            current_covered = set(range(HIST, HIST + j + 1)) <= forced[j]
            match = forced[j] == expect_forced
            all_ok &= match
            print(f"  row j={j} (valid={valid_len}): 强选={sorted(forced[j])}")
            print(f"    算子语义预测={sorted(expect_forced)} -> {'吻合' if match else '不吻合!'}")
            print(f"    当前 token {{{HIST}..{HIST + j}}} 被强选: {current_covered}; "
                  f"意图历史窗口 {sorted(intended_hist)} 中失去保障: {sorted(missed)} "
                  f"(漏 {len(missed)} 个, 预测 j+1={j + 1})")
        # 对照组: 传"含段尾"的修正计数, 验证修复方案
        # decode: +1 即精确; verify: +q_len 使每行都是意图窗口超集
        fix_cnt = caller_cnt + q_len
        fixed = run_case(li, q_len, fix_cnt)
        for j in range(q_len):
            intended = intended_hist | set(range(HIST, HIST + j + 1))
            covered = intended <= fixed[j]
            print(f"  [对照] local_tensor={fix_cnt}(+q_len): row j={j} 强选={sorted(fixed[j])}, "
                  f"意图窗口(历史+当前)全覆盖: {covered}")
            assert covered, f"修复方案在行 j={j} 仍未全覆盖!"
        assert all_ok, "算子强选集合与'含段尾'语义预测不符!"
    print("=" * 72)
    print("结论: 算子按'末尾 local_tensor 个(含段尾当前 token)'强选, 与预测完全一致;")
    print("      在调用方只传历史计数的前提下, 历史窗口最老 j+1 个 token 确定失去强选保障。")
    print("PROBE PASS")
    # 路线 B: per-token local_tensor 逐行精确验证
    test_pertoken(li)
    # init 门控修复: init+local < seq 判别(放在 [B] 拒绝用例前)
    test_init_gate(li)
    # [T]-only 约束: [B] 形状必须被拒绝(放最后, 避免失败后影响其他用例)
    test_reject_legacy_B_shape(li)


if __name__ == "__main__":
    main()
