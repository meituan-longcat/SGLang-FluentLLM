import argparse
import time
import torch
import torch_npu
import flash_ops


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse Flash Attention 单次跑（看算子日志）")
    parser.add_argument("--operator", choices=("sfad", "sfa"), default="sfad",
                        help="测试decode新算子或原sparse flash attention算子")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--s1", type=int, default=4, help="seq_q per batch")
    parser.add_argument("--n1", type=int, default=64, help="num_query_heads")
    parser.add_argument("--dn", type=int, default=512, help="head_dim (nope)")
    parser.add_argument("--dr", type=int, default=64, help="head_dim (rope)")
    parser.add_argument("--block_size", type=int, default=128, help="page attention block size")
    parser.add_argument("--req_max_block_num", type=int, default=2050, help="block_table.shape[-1]")
    parser.add_argument("--total_block_num", type=int, default=1581, help="实际分配的block数")
    parser.add_argument("--s2_act", type=int, default=64 * 1024, help="每请求实际KV长度")
    parser.add_argument("--sparse_token_count", type=int, default=2048, help="sparse_indices 第三维大小")
    parser.add_argument("--sparse_block_size", type=int, default=1)
    parser.add_argument("--scale_value", type=float, default=1.0)
    parser.add_argument("--attention_mode", type=int, choices=(2,), default=2)
    parser.add_argument("--sparse_prep_mode", type=int, choices=(0, 1), default=0,
                        help="0=full scan, 1=valid_chunks bounded scan")
    parser.add_argument("--sparse_mode", type=int, choices=(0, 3), default=3)
    parser.add_argument("--kvp", type=int, default=0,
                        help="0或1=全部有效; N>1=mask后1/N为有效其余填-1")
    parser.add_argument("--all_neg1", action="store_true", help="sparse_indices全部填-1")
    parser.add_argument("--random_valid_counts", action="store_true",
                        help="每token随机保留0到2048个有效KV，-1保持在后缀")
    parser.add_argument("--structured_indices", action="store_true",
                        help="使用降序topk：最后1024+中间随机+最前16")
    parser.add_argument(
        "--structured_block_runs", action="store_true",
        help="使用多个降序128-token连续块和随机后缀")
    parser.add_argument("--query_type", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--reference", action="store_true", help="与纯PyTorch参考实现比较")
    parser.add_argument("--warmup", type=int, default=0, help="计时前预热次数")
    parser.add_argument("--iters", type=int, default=1, help="计时迭代次数")
    parser.add_argument("--benchmark_only", action="store_true", help="仅输出计时结果")
    parser.add_argument("--timed_all_valid", action="store_true",
                        help="预热后把indices恢复为全部有效，再开始计时")
    parser.add_argument("--check_each", action="store_true",
                        help="逐次检查完整output/LSE与预热基线一致，用于长稳测试")
    parser.add_argument("--repeat_output_atol", type=float, default=0.0,
                        help="逐次output与预热基线的最大允许绝对误差")
    parser.add_argument("--repeat_lse_atol", type=float, default=0.0,
                        help="逐次LSE与预热基线的最大允许绝对误差")
    return parser.parse_args()


args = parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)

query_type = torch.bfloat16 if args.query_type == "bfloat16" else torch.float16
return_softmax_lse = True

b = args.b
s1 = args.s1
n1 = args.n1
dn = args.dn
dr = args.dr
block_size = args.block_size
req_max_block_num = args.req_max_block_num
total_block_num = args.total_block_num
s2_act = args.s2_act
sparse_token_count = args.sparse_token_count
sparse_block_size = args.sparse_block_size
scale_value = args.scale_value
attention_mode = args.attention_mode
sparse_prep_mode = args.sparse_prep_mode
sparse_mode = args.sparse_mode
kvp = args.kvp

total_q = b * s1
query = torch.rand(total_q, n1, dn, dtype=query_type)
query_rope = torch.rand(total_q, n1, dr, dtype=query_type)

key = torch.rand(total_block_num, block_size, 1, dn, dtype=query_type)
value = key
key_rope = torch.rand(total_block_num, block_size, 1, dr, dtype=query_type)

def make_sparse_indices():
    if args.structured_block_runs:
        if sparse_token_count < 1040 or s2_act < 2048:
            raise ValueError("structured_block_runs要求至少1040个topk和2048个KV")
        run_blocks = 8
        runs = []
        for i in range(run_blocks):
            block = s2_act // block_size - 1 - i * 2
            end = (block + 1) * block_size
            runs.append(torch.arange(
                end - 1, end - block_size - 1, -1, dtype=torch.int32))
        run_tokens = torch.cat(runs)
        prefix = torch.arange(15, -1, -1, dtype=torch.int32)
        random_count = sparse_token_count - run_tokens.numel() - prefix.numel()
        middle_end = int(run_tokens[-1].min().item())
        rows = []
        for _ in range(total_q):
            middle = (torch.randperm(middle_end - 16, dtype=torch.int32)
                      [:random_count] + 16)
            middle = torch.sort(middle, descending=True).values
            rows.append(torch.cat((run_tokens, middle, prefix)))
        return torch.stack(rows)

    if not args.structured_indices:
        return torch.stack([
            torch.randperm(s2_act, dtype=torch.int32)[:sparse_token_count]
            for _ in range(total_q)
        ])

    tail = min(1024, sparse_token_count)
    prefix = min(16, sparse_token_count - tail)
    random_count = sparse_token_count - tail - prefix
    middle_begin = prefix
    middle_end = s2_act - tail
    if random_count > middle_end - middle_begin:
        raise ValueError("structured_indices的中间候选token不足")
    rows = []
    for _ in range(total_q):
        middle = (torch.randperm(middle_end - middle_begin,
                                 dtype=torch.int32)[:random_count] +
                  middle_begin)
        selected = torch.cat((
            torch.arange(s2_act - tail, s2_act, dtype=torch.int32),
            middle,
            torch.arange(prefix, dtype=torch.int32),
        ))
        rows.append(torch.sort(selected, descending=True).values)
    return torch.stack(rows)


sparse_indices = make_sparse_indices().reshape(
    total_q, 1, sparse_token_count)
full_sparse_indices = sparse_indices.clone()

if kvp and kvp > 1:
    sparse_indices[:, :, sparse_token_count // kvp:] = -1

if args.all_neg1:
    sparse_indices.fill_(-1)

if args.random_valid_counts:
    random_valid_counts = torch.randint(
        0, sparse_token_count + 1, (total_q,), dtype=torch.int32)
    for token, valid in enumerate(random_valid_counts.tolist()):
        sparse_indices[token, :, valid:] = -1

valid_counts_cpu = (sparse_indices != -1).sum(dim=2).squeeze(1).to(
    torch.int32)
valid_chunks_cpu = (valid_counts_cpu + block_size - 1) // block_size

block_table = torch.full((b, req_max_block_num), -1, dtype=torch.int32)
blocks_per_req = s2_act // block_size
for i in range(b):
    block_table[i, :blocks_per_req] = torch.randperm(total_block_num, dtype=torch.int32)[:blocks_per_req]

act_seq_q = torch.arange(1, b + 1, dtype=torch.int32) * s1
act_seq_kv = torch.full((b,), s2_act, dtype=torch.int32)

cpu_inputs = (query, key, value, sparse_indices, query_rope, key_rope,
              block_table, act_seq_q, act_seq_kv)
query = query.npu()
key = key.npu()
value = value.npu()
sparse_indices = sparse_indices.npu()
query_rope = query_rope.npu()
key_rope = key_rope.npu()
block_table = block_table.npu()
act_seq_q = act_seq_q.npu()
act_seq_kv = act_seq_kv.npu()
valid_chunks = valid_chunks_cpu.npu()

print("=" * 60)
print("配置:")
print(f"  b={b}, s1={s1}, n1={n1}, dn={dn}, dr={dr}, seed={args.seed}")
print(f"  s2_act={s2_act}, sparse_token_count={sparse_token_count}, kvp={kvp}")
print(f"  block_size={block_size}, total_block_num={total_block_num}, req_max_block_num={req_max_block_num}")
print(f"  sparse_block_size={sparse_block_size}, sparse_mode={sparse_mode}, "
      f"attention_mode={attention_mode}, sparse_prep_mode={sparse_prep_mode}")
print(f"query shape (TND):     {tuple(query.shape)}")
print(f"key/value shape (PA):  {tuple(key.shape)}")
print(f"sparse_indices shape:  {tuple(sparse_indices.shape)}")
print(f"valid_chunks:          {valid_chunks_cpu.tolist()}")
print(f"block_table shape:     {tuple(block_table.shape)}")
print("=" * 60)

# 直接单次调用算子，不编图，方便看算子日志
torch.npu.set_compile_mode(jit_compile=False)

operator_name = ("npu_sparse_flash_attention_decode"
                 if args.operator == "sfad"
                 else "npu_sparse_flash_attention")
print(f"\n[run] 单次调用 {operator_name} ...")
def run_op():
    operator = (torch_npu.npu_sparse_flash_attention_decode
                if args.operator == "sfad"
                else torch_npu.npu_sparse_flash_attention)
    kwargs = ({"valid_chunks": valid_chunks,
               "sparse_prep_mode": sparse_prep_mode}
              if args.operator == "sfad" else {})
    return operator(
        query, key, value, sparse_indices, scale_value,
        block_table=block_table,
        actual_seq_lengths_query=act_seq_q,
        actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
        **kwargs,
    )


for _ in range(args.warmup):
    warmup_output = run_op()
    torch_npu.npu.synchronize()
if args.timed_all_valid:
    sparse_indices.copy_(full_sparse_indices.npu())
    valid_chunks.fill_(sparse_token_count // block_size)
    torch_npu.npu.synchronize()

repeat_baseline_out = None
repeat_baseline_lse = None
if args.check_each:
    baseline_out, baseline_max, baseline_sum = run_op()
    torch_npu.npu.synchronize()
    repeat_baseline_out = baseline_out.float().clone()
    repeat_baseline_lse = (baseline_max.float() + torch.log(baseline_sum.float())).clone()

start = time.time()
device_time = 0.0
device_latencies = []
max_repeat_output_diff = 0.0
max_repeat_lse_diff = 0.0
repeat_output_mismatch_count = 0
repeat_lse_mismatch_count = 0
first_repeat_output_mismatch = None
first_repeat_lse_mismatch = None
repeat_output_samples = []
for iteration in range(args.iters):
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)
    start_event.record()
    attention_out, max_o, sum_o = run_op()
    end_event.record()
    torch_npu.npu.synchronize()
    latency = start_event.elapsed_time(end_event)
    device_time += latency
    device_latencies.append(latency)
    if args.check_each:
        current_out = attention_out.float()
        current_lse = max_o.float() + torch.log(sum_o.float())
        output_finite = bool(torch.isfinite(current_out).all().item())
        lse_finite = bool(torch.isfinite(current_lse).all().item())
        output_delta = (current_out - repeat_baseline_out).abs()
        lse_delta = (current_lse - repeat_baseline_lse).abs()
        output_diff = output_delta.max().item()
        lse_diff = lse_delta.max().item()
        if output_diff != 0.0:
            repeat_output_mismatch_count += 1
            flat = int(output_delta.argmax().item())
            col = flat % dn
            flat //= dn
            head = flat % n1
            token = flat // n1
            mismatch_elements = int((output_delta != 0).sum().item())
            baseline_value = repeat_baseline_out[token, head, col].item()
            current_value = current_out[token, head, col].item()
            sample = (iteration, token, head, col, output_diff,
                      mismatch_elements, baseline_value, current_value)
            if len(repeat_output_samples) < 20:
                repeat_output_samples.append(sample)
            if first_repeat_output_mismatch is None:
                first_repeat_output_mismatch = (
                    iteration, token, head, col, output_diff,
                    mismatch_elements, baseline_value, current_value)
        if lse_diff != 0.0:
            repeat_lse_mismatch_count += 1
            if first_repeat_lse_mismatch is None:
                flat = int(lse_delta.argmax().item())
                first_repeat_lse_mismatch = (
                    iteration, flat // n1, flat % n1, lse_diff)
        max_repeat_output_diff = max(max_repeat_output_diff, output_diff)
        max_repeat_lse_diff = max(max_repeat_lse_diff, lse_diff)
        if (not output_finite or not lse_finite or
                output_diff > args.repeat_output_atol or
                lse_diff > args.repeat_lse_atol):
            raise RuntimeError(
                f"iteration {iteration} stability check failed: "
                f"output_finite={output_finite}, lse_finite={lse_finite}, "
                f"output_diff={output_diff}, lse_diff={lse_diff}")
end = time.time()


def percentile(values, ratio):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]

print("=" * 60)
print(f"平均耗时: {(end - start) * 1000 / args.iters:.3f} ms")
print(f"NPU平均耗时: {device_time / args.iters:.3f} ms")
print("NPU耗时分布(ms): "
      f"min={min(device_latencies):.3f}, "
      f"p50={percentile(device_latencies, 0.50):.3f}, "
      f"p95={percentile(device_latencies, 0.95):.3f}, "
      f"p99={percentile(device_latencies, 0.99):.3f}, "
      f"max={max(device_latencies):.3f}")
if args.check_each:
    print(f"逐次稳定性: {args.iters}/{args.iters} 通过, "
          f"output最大重复差={max_repeat_output_diff}, "
          f"LSE最大重复差={max_repeat_lse_diff}")
    print(f"重复差异次数: output={repeat_output_mismatch_count}, "
          f"LSE={repeat_lse_mismatch_count}")
    if first_repeat_output_mismatch is not None:
        print("首次output差异(iteration, token, head, col, diff, "
              "elements, baseline, current): "
              f"{first_repeat_output_mismatch}")
        print("前20次output差异样本:", repeat_output_samples)
    if first_repeat_lse_mismatch is not None:
        print("首次LSE差异(iteration, token, head, diff): "
              f"{first_repeat_lse_mismatch}")
if args.benchmark_only:
    raise SystemExit(0)
softmax_lse = max_o + torch.log(sum_o)
print(f"输出 shape: {tuple(attention_out.shape)}")
print(f"输出 dtype: {attention_out.dtype}")
if args.operator == "sfad" and attention_out.dtype != torch.float32:
    raise AssertionError(f"attention_out dtype应为float32，实际为{attention_out.dtype}")
out_f = attention_out.float()
print(f"nan 个数: {torch.isnan(out_f).sum().item()} / {out_f.numel()}")
print(f"inf 个数: {torch.isinf(out_f).sum().item()}")
print(f"abs max: {out_f.abs().max().item()}, mean: {out_f.mean().item()}")
print(f"全零元素个数: {(out_f == 0).sum().item()} / {out_f.numel()}")
print(f"前8个元素: {out_f.flatten()[:8].tolist()}")
if softmax_lse is not None:
    lse_f = softmax_lse.float()
    print(f"lse shape: {tuple(softmax_lse.shape)}")
    print(f"lse nan 个数: {torch.isnan(lse_f).sum().item()} / {lse_f.numel()}")
    print(f"lse inf 个数: {torch.isinf(lse_f).sum().item()}")
    print(f"lse abs max: {lse_f.abs().max().item()}, mean: {lse_f.mean().item()}")
    print(f"lse 前8个元素: {lse_f.flatten()[:8].tolist()}")
if args.reference:
    q, k, v, indices, qr, kr, table, seq_q, seq_kv = cpu_inputs
    ref = torch.zeros_like(q, dtype=torch.float32)
    ref_lse = torch.full((total_q, n1), -torch.inf)
    for t in range(total_q):
        batch = int(torch.searchsorted(seq_q, torch.tensor(t), right=True))
        q_begin = 0 if batch == 0 else int(seq_q[batch - 1])
        kv_len = int(seq_kv[batch])
        threshold = kv_len
        if sparse_mode == 3:
            threshold = max(
                0, kv_len - (int(seq_q[batch]) - q_begin) + t - q_begin + 1)
        pos = indices[t, 0]
        pos = pos[(pos >= 0) & (pos < threshold)].long()
        if pos.numel() == 0:
            continue
        block = table[batch, pos // block_size].long()
        row = pos % block_size
        selected_k = k[block, row, 0].float()
        selected_v = v[block, row, 0].float()
        selected_kr = kr[block, row, 0].float()
        logits = (q[t].float() @ selected_k.T + qr[t].float() @ selected_kr.T) * scale_value
        ref[t] = torch.softmax(logits, dim=-1) @ selected_v
        ref_lse[t] = torch.logsumexp(logits, dim=-1)
    out_diff = (out_f.cpu() - ref).abs()
    actual_lse = lse_f.cpu().reshape(total_q, n1)
    same_inf = torch.isinf(actual_lse) & torch.isinf(ref_lse) & (actual_lse == ref_lse)
    lse_diff = torch.where(same_inf, 0.0, (actual_lse - ref_lse).abs())
    print(f"reference output max/mean diff: {out_diff.max().item()} / {out_diff.mean().item()}")
    max_pos = torch.unravel_index(out_diff.argmax(), out_diff.shape)
    print(f"reference output max diff position: {tuple(int(x) for x in max_pos)}")
    print("reference output diff by 128 columns:",
          [out_diff[..., i:i + 128].max().item() for i in range(0, dn, 128)])
    print(f"reference lse max/mean diff: {lse_diff.max().item()} / {lse_diff.mean().item()}")
print("=" * 60)
