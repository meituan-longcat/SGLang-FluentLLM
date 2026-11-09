import torch
from tabulate import tabulate

from sglang.srt.layers.attention.chunker import ChunkFlashAttn3, ChunkFlashInferAttn

# Simulation environment configuration
global_server_args_dict = {"chunker_backend": "flashinfer"}
global_workspace_buffer = None

# --- Make sure ChunkFlashAttn3 and ChunkFlashInferAttn classes are loaded in your code ---


def benchmark_forward_only(backend, batch_size, q_len_val, kv_len_val, num_qo_heads, num_kv_heads, head_dim=128):
    device = "cuda"
    dtype = torch.bfloat16
    global_server_args_dict["chunker_backend"] = backend

    # 1. Prepare data
    q_lens_list = [q_len_val] * batch_size
    kv_lens_list = [kv_len_val] * batch_size
    q_lens = torch.tensor(q_lens_list, device=device, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, device=device, dtype=torch.int32)

    q = torch.randn(batch_size * q_len_val, num_qo_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size * kv_len_val, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size * kv_len_val, num_kv_heads, head_dim, device=device, dtype=dtype)
    scaling = 1.0 / (head_dim**0.5)

    if backend == "flashinfer":
        attn = ChunkFlashInferAttn(num_qo_heads, num_kv_heads, head_dim, head_dim, dtype, device, True, None)
    else:
        attn = ChunkFlashAttn3(num_qo_heads, num_kv_heads, head_dim, head_dim, dtype, device, True, None)

    attn.plan(q_lens, kv_lens, q_lens_list, kv_lens_list)

    for _ in range(20):
        attn.forward_extend(q, k, v, scaling, None)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    num_iters = 200
    start_event.record()
    for _ in range(num_iters):
        attn.forward_extend(q, k, v, scaling, None)
    end_event.record()

    torch.cuda.synchronize()
    avg_forward_ms = start_event.elapsed_time(end_event) / num_iters

    # Calculate TFLOPS (approximate)
    # Formula: 2 * batch * q_len * kv_len * n_heads * head_dim / time
    flops = 2 * batch_size * q_len_val * kv_len_val * num_qo_heads * head_dim
    tflops = (flops / 1e12) / (avg_forward_ms / 1e3)

    return avg_forward_ms, tflops


def run_benchmark():
    # Test configuration: (batch_size, q_len, kv_len, n_qo, n_kv)
    test_cases = [
        (1, 512, 512, 4, 4),
        (1, 1024, 1024, 4, 4),
        (1, 2048, 2048, 4, 4),
        (1, 4096, 4096, 4, 4),
        (1, 8192, 8192, 4, 4),
        (1, 16384, 16384, 4, 4),
        (1, 512, 8192, 4, 4),
        (1, 1024, 8192, 4, 4),
        (1, 2048, 8192, 4, 4),
        (1, 3072, 8192, 4, 4),
        (1, 4096, 8192, 4, 4),
        (1, 6144, 8192, 4, 4),
        (2, 2048, 8192, 4, 4),
        (4, 2048, 8192, 4, 4),
    ]

    results = []
    print(f"Running Benchmark on {torch.cuda.get_device_name(0)}...\n")

    for bs, ql, kvl, nqh, nkh in test_cases:
        # FA3 execution
        fa3_time, fa3_tflops = benchmark_forward_only("fa3", bs, ql, kvl, nqh, nkh)
        # FlashInfer execution
        fi_time, fi_tflops = benchmark_forward_only("flashinfer", bs, ql, kvl, nqh, nkh)

        speedup = fa3_time / fi_time if fi_time > 0 else 0

        results.append(
            [f"{bs}x{ql}x{kvl}", f"{nqh}/{nkh}", f"{fa3_time:.3f}", f"{fi_time:.3f}", f"{speedup:.2f}x", f"{fa3_tflops:.1f}", f"{fi_tflops:.1f}"]
        )

    headers = ["Config (BxQxKV)", "Heads(Q/K)", "FA3 (ms)", "FlashInfer (ms)", "FI Speedup", "FA3 TFLOPS", "FI TFLOPS"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run_benchmark()
