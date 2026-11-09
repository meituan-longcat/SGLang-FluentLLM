import argparse
import time
import torch
import torch.nn as nn
import torch_npu
import flash_ops
import torchair


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse Flash Attention 性能测试")
    # 算子参数
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
    parser.add_argument("--sparse_mode", type=int, choices=(0, 3), default=3,
                        help="sparse attention mode (0 or 3)")
    # kvp: 0 或 1 表示全部有效; N>1 表示仅前 1/N 有效，其余位置保持原 shape 并填 -1
    parser.add_argument("--kvp", type=int, default=0,
                        help="0或1=全部有效; N>1=仅前1/N有效，其余位置填-1")
    parser.add_argument("--even_tiling", action="store_true",
                        help="性能对照：传入均匀cost，保持indices随机分布")
    parser.add_argument("--random_valid_counts", action="store_true",
                        help="每token随机保留0到2048个有效KV，-1保持在后缀")
    parser.add_argument("--structured_indices", action="store_true",
                        help="使用降序topk：最后1024+中间随机+最前16")
    parser.add_argument(
        "--structured_block_runs", action="store_true",
        help="使用多个降序128-token连续块和随机后缀")
    # 性能测试参数
    parser.add_argument("--graph_layers", type=int, default=100, help="编入单张图的算子层数")
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--test_iters", type=int, default=10)
    parser.add_argument("--query_type", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=None, help="随机种子，设置后可复现输入数据")
    parser.add_argument("--timeline", type=str, default=None,
                        help="抓取 NPU timeline 的输出路径（如 ./profile），设置后在正式计时前额外跑一次并导出 chrome trace")
    return parser.parse_args()


args = parse_args()

# 固定随机种子，保证输入数据可复现
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)

if args.even_tiling and args.sparse_prep_mode != 0:
    raise ValueError("--even_tiling is only valid with --sparse_prep_mode=0")

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
GRAPH_LAYERS = args.graph_layers

# === 构建 query (TND 格式) ===
# TND: shape [sum(seq_q), n1, dn]，seq_q 各 batch 相同，T = b * s1
total_q = b * s1
query = torch.rand(total_q, n1, dn, dtype=query_type)
query_rope = torch.rand(total_q, n1, dr, dtype=query_type)

# === 构建 key/value (PA_BSND 格式) ===
# shape: [total_block_num, block_size, 1, dn] = [1581, 128, 1, 512]
key = torch.rand(total_block_num, block_size, 1, dn, dtype=query_type)
value = key
key_rope = torch.rand(total_block_num, block_size, 1, dr, dtype=query_type)

# === kvp 截断: 保持 sparse_indices shape，尾部填 -1 ===
# shape 不变，用来验证算子是否停止发射 -1 后面的计算块

# === 构建 sparse_indices ===
# shape: [total_q, 1, sparse_token_count]
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

if kvp and kvp > 1:
    sparse_indices[:, :, sparse_token_count // kvp:] = -1

if args.random_valid_counts:
    random_valid_counts = torch.randint(
        0, sparse_token_count + 1, (total_q,), dtype=torch.int32)
    edge_counts = (0, 1, block_size - 1, block_size,
                   block_size + 1, sparse_token_count - 1,
                   sparse_token_count)
    for token, valid in enumerate(edge_counts[:total_q]):
        random_valid_counts[token] = max(0, min(valid, sparse_token_count))
    for token, valid in enumerate(random_valid_counts.tolist()):
        sparse_indices[token, :, valid:] = -1

valid_counts = (sparse_indices != -1).sum(dim=2).squeeze(1).to(torch.int32)
valid_chunks = (valid_counts + block_size - 1) // block_size
if args.even_tiling:
    valid_chunks.fill_(1)

# === 构建 block_table ===
# shape: [b, req_max_block_num]，全填-1表示未分配
block_table = torch.full((b, req_max_block_num), -1, dtype=torch.int32)
blocks_per_req = s2_act // block_size
for i in range(b):
    block_table[i, :blocks_per_req] = torch.randperm(total_block_num, dtype=torch.int32)[:blocks_per_req]

# TND query uses cumulative endpoints; PA_BSND KV uses per-batch lengths.
act_seq_q = torch.arange(1, b + 1, dtype=torch.int32) * s1
act_seq_kv = torch.full((b,), s2_act, dtype=torch.int32)

# === 搬到 NPU ===
query = query.npu()
key = key.npu()
value = value.npu()
sparse_indices = sparse_indices.npu()
query_rope = query_rope.npu()
key_rope = key_rope.npu()
block_table = block_table.npu()
act_seq_q = act_seq_q.npu()
act_seq_kv = act_seq_kv.npu()
valid_chunks = valid_chunks.npu()

from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
config.experimental_config.tiling_schedule_optimize = True
npu_backend = torchair.get_npu_backend(compiler_config=config)


# === 将 N 层算子调用编入同一张图，整图只下发一次，摊薄 launch 开销 ===
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, query, key, value, sparse_indices, scale_value,
                block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
                query_rope, key_rope, valid_chunks, sparse_block_size,
                layout_query, layout_kv, sparse_mode, attention_mode,
                return_softmax_lse, sparse_prep_mode):
        output = None
        for i in range(GRAPH_LAYERS):
            out, a, b = torch_npu.npu_sparse_flash_attention_decode(
                query, key, value, sparse_indices, scale_value,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                query_rope=query_rope, key_rope=key_rope,
                valid_chunks=valid_chunks,
                sparse_block_size=sparse_block_size,
                layout_query=layout_query, layout_kv=layout_kv,
                sparse_mode=sparse_mode,
                attention_mode=attention_mode,
                return_softmax_lse=return_softmax_lse,
                sparse_prep_mode=sparse_prep_mode)
            if i == 0:
                output = out
            else:
                output += out
        return output


mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True, dynamic=False)

# === 性能测试 ===
warmup_iters = args.warmup_iters
test_iters = args.test_iters

# warmup
print(f"[warmup] {warmup_iters} iters (x{GRAPH_LAYERS} layers/iter)...")
for _ in range(warmup_iters):
    _ = mod(
        query, key, value, sparse_indices, scale_value,
        block_table=block_table,
        actual_seq_lengths_query=act_seq_q,
        actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        valid_chunks=valid_chunks,
        sparse_block_size=sparse_block_size,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
        sparse_prep_mode=sparse_prep_mode
    )
torch_npu.npu.synchronize()

# 正式计时
print(f"[bench] {test_iters} iters (x{GRAPH_LAYERS} layers/iter)...")

# 可选: 抓取 timeline（单独一次 forward，不计时）
if args.timeline:
    print(f"[timeline] 导出到 {args.timeline} ...")

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU,
            ],
        experimental_config=experimental_config,
        record_shapes=True,  # Whether to record operator shapes
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.timeline),
    ) as prof:
        _ = mod(
            query, key, value, sparse_indices, scale_value,
            block_table=block_table,
            actual_seq_lengths_query=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            valid_chunks=valid_chunks,
            sparse_block_size=sparse_block_size,
            layout_query='TND',
            layout_kv='PA_BSND',
            sparse_mode=sparse_mode,
            attention_mode=attention_mode,
            return_softmax_lse=return_softmax_lse,
            sparse_prep_mode=sparse_prep_mode
        )
        prof.step()
    torch_npu.npu.synchronize()
    print(f"[timeline] 完成，可用 chrome://tracing 打开 {args.timeline} 下的 json")

start = time.time()
for _ in range(test_iters):
    attention_out = mod(
        query, key, value, sparse_indices, scale_value,
        block_table=block_table,
        actual_seq_lengths_query=act_seq_q,
        actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        valid_chunks=valid_chunks,
        sparse_block_size=sparse_block_size,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
        sparse_prep_mode=sparse_prep_mode
    )
torch_npu.npu.synchronize()
end = time.time()

# === 结果统计 ===
total_layers = test_iters * GRAPH_LAYERS
total_time_s = end - start
avg_time_ms = total_time_s / total_layers * 1000

# 计算量 FLOPs: kvp 场景下实际有效 token 数为 sparse_token_count / kvp
# 截断后 sparse_token_count 已是有效 token 数
valid_token_count = sparse_token_count // kvp if kvp > 0 else sparse_token_count
flops_per_req = 2 * s1 * valid_token_count * (dn + dr) * 2 * n1  # QK + AV
total_flops = flops_per_req * b
tflops = total_flops / (avg_time_ms / 1000) / 1e12

print("=" * 60)
print(f"配置: b={b}, s1={s1}, n1={n1}, dn={dn}, dr={dr}, seed={args.seed}")
print(f"      s2_act={s2_act}, sparse_token_count={sparse_token_count}, kvp={kvp}, valid_token_count={valid_token_count}")
print(f"      block_size={block_size}, total_block_num={total_block_num}, req_max_block_num={req_max_block_num}")
print(f"query shape (TND):     {tuple(query.shape)}")
print(f"key/value shape (PA):  {tuple(key.shape)}")
print(f"sparse_indices shape:  {tuple(sparse_indices.shape)}")
print(f"valid_chunks:          {valid_chunks.cpu().tolist()}")
print(f"block_table shape:     {tuple(block_table.shape)}")
print("-" * 60)
print(f"总耗时:   {total_time_s*1000:.2f} ms ({test_iters} iters x {GRAPH_LAYERS} layers = {total_layers} ops)")
print(f"平均耗时: {avg_time_ms:.3f} ms / iter")
print(f"计算量:   {total_flops/1e9:.2f} GFLOP")
print(f"算力:     {tflops:.2f} TFLOPS")
print(f"输出 shape: {tuple(attention_out.shape)}")
print("=" * 60)
