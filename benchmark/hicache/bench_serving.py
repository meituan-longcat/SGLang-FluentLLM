# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

"""
Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate-range 1,2,4,8,16,32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --multi
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
import numpy as np
import requests

# Add current directory to sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import MsgContent, SampleOutput, get_dataset
from sglang.bench_serving import get_tokenizer, remove_prefix, set_ulimit
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

global args


@dataclass
class RequestFuncInput:
    prompts: list[tuple[MsgContent, int, int]]
    api_url: str
    model: str
    lora_name: str
    extra_request_body: dict[str, Any]

    # For multiturn chat, store the context
    prev_messages: list = field(default_factory=list)
    finished_prompts: int = 0


@dataclass
class RequestFuncOutput:
    generated_text: list[str] = field(default_factory=list)
    prompt_len: list[int] = field(default_factory=list)
    output_len: list[int] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    ttft: list[float] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    success: bool = False
    error: str = ""


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    queue: asyncio.Queue,
    tokenizer: PreTrainedTokenizerBase,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "temperature": 0.0,
            "best_of": 1,
            "stream": not args.disable_stream,
            "stream_options": {"include_usage": True},
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()

        prompt_idx = request_func_input.finished_prompts
        messages = request_func_input.prev_messages
        prompt, input_len, max_tokens = request_func_input.prompts[prompt_idx]
        prompt_len = sum(
            prompt[1] + prompt[2]  # input_len + output_len
            for prompt in request_func_input.prompts[:prompt_idx]
        )
        prompt_len += input_len

        # Messages
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        payload["messages"] = messages
        payload["max_tokens"] = max_tokens

        # output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    actual_prompt_len = prompt_len - 1
                    actual_output_len = 0
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)
                            timestamp = time.perf_counter()
                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["usage"] is not None and len(data["usage"]) > 0:
                                actual_prompt_len = data["usage"]["prompt_tokens"]
                                actual_output_len = data["usage"]["completion_tokens"]
                                continue
                            delta = data["choices"][0]["delta"]

                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft.append(ttft)

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += delta["content"]
                            most_recent_timestamp = timestamp

                    output.prompt_len.append(actual_prompt_len)  # truncate <s>
                    output.output_len.append(actual_output_len)
                    output.generated_text.append(generated_text)
                    output.success = True
                    output.latency.append(latency)

                    # Prepare for the new request
                    request_func_input.prompts[prompt_idx] = (
                        prompt,
                        input_len,
                        actual_output_len,  # changes from max_tokens to output_len
                    )
                    prompt_idx += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": generated_text,
                        }
                    )

                    # Move the new request to the end of the queue
                    if prompt_idx < len(request_func_input.prompts):
                        request_func_input.finished_prompts = prompt_idx
                        request_func_input.prev_messages = messages
                        await queue.put(request_func_input)
                else:
                    output.error = response.reason or f"HTTP {response.status}"
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or f"HTTP {response.status}"
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_openai_completions,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


async def get_requests(
    input_requests_queue: asyncio.Queue,
    request_rate: float,
    num_actual_requests: int,
) -> AsyncGenerator[RequestFuncInput, None]:
    for _ in range(num_actual_requests):
        try:
            request = await asyncio.wait_for(
                input_requests_queue.get(), timeout=300
            )  # Wait for 5 minutes then abort
        except Exception as e:
            print(f"exception: {e}")
            break

        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> tuple[BenchmarkMetrics, list[int]]:
    output_lens: list[int] = []
    retokenized_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    e2e_latencies: list[float] = []
    output_success = 0
    for i in range(len(outputs)):
        if outputs[i].success:
            output_success += 1
            assert len(outputs[i].generated_text) == len(outputs[i].latency)
            assert len(outputs[i].generated_text) == len(outputs[i].ttft)
            for j in range(len(outputs[i].generated_text)):
                output_len = outputs[i].output_len[j]
                output_lens.append(output_len)
                retokenized_output_len = len(
                    tokenizer.encode(
                        outputs[i].generated_text[j], add_special_tokens=False
                    )
                )
                retokenized_output_lens.append(retokenized_output_len)
                total_input += outputs[i].prompt_len[j]
                if output_len > 1:
                    tpots.append(
                        (outputs[i].latency[j] - outputs[i].ttft[j]) / (output_len - 1)
                    )

                completed += 1
            itls += outputs[i].itl
            ttfts += outputs[i].ttft
            e2e_latencies += outputs[i].latency

        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p90_itl_ms=np.percentile(itls or 0, 90) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: SampleOutput,
    request_rate: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
    lora_name: str,
    extra_request_body: dict[str, Any],
    profile: bool,
    enable_shared_prefix: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, queue, tokenizer, pbar):
        if semaphore is None:
            return await request_func(
                request_func_input=request_func_input,
                queue=queue,
                tokenizer=tokenizer,
                pbar=pbar,
            )
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input,
                queue=queue,
                tokenizer=tokenizer,
                pbar=pbar,
            )

    num_actual_requests = sum(len(r) for r in input_requests)
    print(f"Num of shared prefixes or conversations: {len(input_requests)}")
    print(f"Num of total requests: {num_actual_requests}")

    # flatten the requests for shared prefix
    if enable_shared_prefix:
        input_requests = [[r] for requests in input_requests for r in requests]
    inputs_requests_queue = asyncio.Queue(maxsize=len(input_requests))
    print("Starting initial single prompt test run...")
    # NOTE: Just use the first request of the first conversation for warmup
    test_input = RequestFuncInput(
        model=model_id,
        prompts=input_requests[0][:1],
        api_url=api_url,
        lora_name=lora_name,
        extra_request_body=extra_request_body,
    )
    test_output = await request_func(
        request_func_input=test_input, queue=inputs_requests_queue, tokenizer=tokenizer
    )
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    # Check the states
    assert inputs_requests_queue.empty()

    # Flush cache
    if "sglang" in backend:
        requests.post(base_url + "/flush_cache")

    time.sleep(1.0)

    # Start profiler
    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    for request in input_requests:
        request_func_input = RequestFuncInput(
            model=model_id,
            prompts=request,
            api_url=api_url,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        inputs_requests_queue.put_nowait(request_func_input)
    if (
        not args.enable_multiturn
        and not args.enable_shared_prefix
        and not args.dataset_name == "generated-shared-prefix"
    ):
        pass  # assert len(input_requests) == num_actual_requests  # disabled for sharegpt multiturn conversations

    pbar = None if disable_tqdm else tqdm(total=num_actual_requests)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_requests(
        inputs_requests_queue, request_rate, num_actual_requests
    ):
        tasks.append(
            asyncio.create_task(
                limited_request_func(
                    request_func_input=request,
                    queue=inputs_requests_queue,
                    tokenizer=tokenizer,
                    pbar=pbar,
                )
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    # Stop profiler
    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    # Compute metrics and print results
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max request concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", metrics.p90_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P90 ITL (ms):", metrics.p90_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # Arguments
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "fixed_output_len": args.fixed_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            # Results
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "concurrency": metrics.concurrency,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "fixed_output_len": args.fixed_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "duration": benchmark_duration,
            "completed": metrics.completed,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "random":
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"
            )

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p90_itl_ms": metrics.p90_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }
    return result


def flush_cache(base_url: str) -> bool:
    """Flush server cache before running benchmark.

    Args:
        base_url: Base URL of the server (e.g., http://127.0.0.1:8192)

    Returns:
        True if flush was successful, False otherwise
    """
    try:
        print(f"Flushing cache at {base_url}/flush_cache ...")
        response = requests.post(f"{base_url}/flush_cache", timeout=30)
        if response.status_code == 200:
            print("Cache flushed successfully")
            time.sleep(2)  # Wait for flush to complete
            return True
        else:
            print(f"Failed to flush cache: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to flush cache: {e}")
        return False


def run_single_benchmark(args_: argparse.Namespace):
    """Run a single benchmark with given arguments."""
    global args
    args = args_

    # Set default value for max_concurrency if not present
    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Set url
    if args.port is None:
        args.port = {
            "sglang": 30000,
            "lmdeploy": 23333,
            "vllm": 8000,
        }.get(args.backend, 30000)

    model_url = (
        f"{args.base_url}/v1/models"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/models"
    )

    if args.backend in ["sglang", "vllm", "lmdeploy"]:
        api_url = (
            f"{args.base_url}/v1/chat/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/chat/completions"
        )
    base_url = (
        f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url
    )

    # Flush cache if requested
    if args.flush_cache:
        if not flush_cache(base_url):
            print("Warning: Failed to flush cache, continuing anyway...")

    # Get model name
    if args.model is None:
        if args.backend == "truss":
            print(
                "Please provide a model with `--model` when using truss backend. e.g. --model meta-llama/Llama-3.1-8B-Instruct"
            )
            sys.exit(1)
        try:
            response = requests.get(model_url)
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    # Dataset compatibility check
    if args.enable_multiturn:
        # TODO: Support multiturn for random
        if args.dataset_name not in ["sharegpt", "ultrachat", "loogle", "nextqa"]:
            print(
                "Multiturn conversation is only supported for sharegpt, ultrachat, loogle, and nextqa datasets."
            )
            sys.exit(1)

    if args.enable_shared_prefix:
        if args.dataset_name not in ["loogle", "nextqa"]:
            print("Shared prefix is only supported for loogle and nextqa datasets.")
            sys.exit(1)

    print(f"{args}\n")

    # Read dataset
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id)

    input_requests = get_dataset(args, tokenizer)

    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            lora_name=args.lora_name,
            extra_request_body=extra_request_body,
            profile=args.profile,
            enable_shared_prefix=args.enable_shared_prefix,
        )
    )


def run_benchmark(args_: argparse.Namespace):
    """Run benchmark with multiple request rates."""
    global args
    args = args_

    # Determine request rates to test
    if args.disable_auto_run:
        print("Running with specified request rate...")
        request_rates = [args.request_rate]
    else:
        print("Auto-running with different request rates...")
        request_rates = [16, 8, 4, 2, 1]

    all_results = []
    for rate in request_rates:
        args.request_rate = rate
        print(f"\n{'='*60}")
        print(f"Testing request rate: {rate} req/s")
        print(f"{'='*60}")
        
        result = run_single_benchmark(args)
        if result:
            all_results.append({"request_rate": rate, **result})
        
        # Wait between tests to let system stabilize
        if rate != request_rates[-1]:
            print("\nWaiting 5 seconds before next test...")
            time.sleep(5)
    
    # Print summary of all results
    print_summary(all_results)
    
    return all_results


def print_summary(all_results):
    """Print summary of all benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for result in all_results:
        rate = result["request_rate"]
        print(f"\nRequest Rate: {rate} req/s")
        print(f"  Completed Requests: {result.get('completed', 'N/A')}")
        
        # Handle numeric formatting safely
        req_throughput = result.get('request_throughput', 'N/A')
        if req_throughput != 'N/A':
            print(f"  Request Throughput: {req_throughput:.2f} req/s")
        else:
            print(f"  Request Throughput: {req_throughput} req/s")
            
        total_throughput = result.get('total_throughput', 'N/A')
        if total_throughput != 'N/A':
            print(f"  Total Throughput: {total_throughput:.2f} tok/s")
        else:
            print(f"  Total Throughput: {total_throughput} tok/s")
            
        mean_ttft = result.get('mean_ttft_ms', 'N/A')
        if mean_ttft != 'N/A':
            print(f"  Mean TTFT: {mean_ttft:.2f} ms")
        else:
            print(f"  Mean TTFT: {mean_ttft} ms")
            
        p99_ttft = result.get('p99_ttft_ms', 'N/A')
        if p99_ttft != 'N/A':
            print(f"  P99 TTFT: {p99_ttft:.2f} ms")
        else:
            print(f"  P99 TTFT: {p99_ttft} ms")
            
        mean_latency = result.get('mean_e2e_latency_ms', 'N/A')
        if mean_latency != 'N/A':
            print(f"  Mean Latency: {mean_latency:.2f} ms")
        else:
            print(f"  Mean Latency: {mean_latency} ms")
            
        p99_latency = result.get('p99_e2e_latency_ms', 'N/A')
        if p99_latency != 'N/A':
            print(f"  P99 Latency: {p99_latency:.2f} ms")
        else:
            print(f"  P99 Latency: {p99_latency} ms")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Request Rate':<12} {'Throughput':<12} {'TTFT (mean)':<12} {'Latency (mean)':<12} {'Cache Hit':<12}")
    print(f"{' (req/s)':<12} {' (req/s)':<12} {' (ms)':<12} {' (ms)':<12} {' Rate':<12}")
    print("-"*80)
    
    for result in all_results:
        rate = result["request_rate"]
        throughput = result.get('request_throughput', 0)
        ttft = result.get('mean_ttft_ms', 0)
        latency = result.get('mean_e2e_latency_ms', 0)
        cache_hit = result.get('cache_hit_rate', 0) * 100 if 'cache_hit_rate' in result else 0
        
        print(f"{rate:<12.1f} {throughput:<12.2f} {ttft:<12.2f} {latency:<12.2f} {cache_hit:<12.2f}%")
    
    print("="*80)


def load_config(args=None):
    """Load configuration from either CONFIG_PATH environment variable or command line arguments."""
    config_path = os.getenv("CONFIG_PATH")
    
    if config_path:
        # Load from config file
        with open(config_path, "r") as f:
            config = json.load(f)
    elif args:
        # Load from command line arguments
        config = {
            "num_rounds": args.num_rounds,
            "num_clients": args.num_clients,
            "round_ratios": [int(x) for x in args.round_ratios.split(",")],
            "mean_new_tokens_per_round": [int(x) for x in args.mean_new_tokens_per_round.split(",")],
            "mean_return_tokens_per_round": [int(x) for x in args.mean_return_tokens_per_round.split(",")],
            "mean_inter_round_interval": [int(x) for x in args.mean_inter_round_interval.split(",")],
        }
    else:
        raise ValueError("Either set CONFIG_PATH environment variable or provide configuration via command line arguments.")

    required_keys = [
        "num_rounds",
        "num_clients",
        "round_ratios",
        "mean_new_tokens_per_round",
        "mean_return_tokens_per_round",
        "mean_inter_round_interval",
    ]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")

    num_rounds = config["num_rounds"]
    assert len(config["round_ratios"]) == num_rounds
    assert len(config["mean_new_tokens_per_round"]) == num_rounds
    assert len(config["mean_return_tokens_per_round"]) == num_rounds
    assert len(config["mean_inter_round_interval"]) == num_rounds

    print("Configuration:")
    print(f"  num_rounds: {config['num_rounds']}")
    print(f"  num_clients: {config['num_clients']}")
    print(f"  round_ratios: {config['round_ratios']}")
    print(f"  mean_new_tokens_per_round: {config['mean_new_tokens_per_round']}")
    print(f"  mean_return_tokens_per_round: {config['mean_return_tokens_per_round']}")
    print(f"  mean_inter_round_interval: {config['mean_inter_round_interval']}")

    return config


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default=os.environ.get("SERVER_HOST", "0.0.0.0"), help="Default host is 0.0.0.0. Can be set via SERVER_HOST env var."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SERVER_PORT", "30000")) if os.environ.get("SERVER_PORT") else None,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines. Can be set via SERVER_PORT env var.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "random",
            "generated-shared-prefix",
            "ultrachat",
            "loogle",
            "nextqa",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--fixed-output-len",
        type=int,
        default=1,
        help="Output length for each request. Overrides the output length from the dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1,
        type=int,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use request rate range rather than single value.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step. Default is 2,34,2. It also supports a list of request rates, requiring the parameters to not equal three.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--enable-multiturn",
        action="store_true",
        help="Enable multiturn chat for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "sharegpt, ultrachat, loogle, nextqa",
    )
    parser.add_argument(
        "--enable-shared-prefix",
        action="store_true",
        help="Enable shared prefix for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "loogle, nextqa",
    )

    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable shuffling datasets. This is useful to generate stable output "
        "in benchmarking",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help="The name of LoRA adapter",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    # videos specific
    parser.add_argument(
        "--max-frames",
        type=int,
        default=sys.maxsize,
        help="The maximum number of frames to extract from each video. "
        "This option is specific to the nextqa dataset (video benchmark). ",
    )

    # Fixed dataset arguments for deterministic testing
    fixed_group = parser.add_argument_group("fixed dataset arguments")
    fixed_group.add_argument(
        "--use-fixed-data",
        action="store_true",
        default=True,
        help="Use fixed, deterministic dataset subsets for reproducible benchmarks. Default: True",
    )
    fixed_group.add_argument(
        "--use-random-data",
        action="store_true",
        help="Use legacy random data sampling mode (overrides --use-fixed-data)",
    )
    fixed_group.add_argument(
        "--data-scenario",
        type=str,
        choices=['serving', 'multiturn', 'mix', 'long_context'],
        default='serving',
        help="Data scenario for fixed dataset selection. "
        "serving: short conversations for general tests, "
        "multiturn: medium conversations for multi-turn scenarios, "
        "mix: balanced mix of conversation types, "
        "long_context: long conversations for cache reuse testing. Default: serving",
    )
    fixed_group.add_argument(
        "--fixed-data-cache-dir",
        type=str,
        default=None,
        help="Directory to cache fixed datasets. Default: ~/.cache/sglang/fixed_datasets",
    )

    # Cache management arguments
    cache_group = parser.add_argument_group("cache management arguments")
    cache_group.add_argument(
        "--flush-cache",
        action="store_true",
        default=False,
        help="Flush server cache before running benchmark",
    )
    
    # HiCache configuration arguments
    hicache_group = parser.add_argument_group("HiCache configuration arguments")
    hicache_group.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of rounds for multi-turn conversations (default: 10)",
    )
    hicache_group.add_argument(
        "--num-clients",
        type=int,
        default=60,
        help="Maximum number of concurrent client requests (default: 60)",
    )
    hicache_group.add_argument(
        "--round-ratios",
        type=str,
        default="50,25,15,15,10,10,9,8,7,6",
        help="Distribution of requests across rounds as comma-separated list (default: 50,25,15,15,10,10,9,8,7,6)",
    )
    hicache_group.add_argument(
        "--mean-new-tokens-per-round",
        type=str,
        default="1000,400,350,300,280,260,240,220,210,200",
        help="Mean new tokens per round as comma-separated list (default: 1000,400,350,300,280,260,240,220,210,200)",
    )
    hicache_group.add_argument(
        "--mean-return-tokens-per-round",
        type=str,
        default="100,100,100,100,100,100,100,100,100,100",
        help="Mean return tokens per round as comma-separated list (default: 100,100,100,100,100,100,100,100,100,100)",
    )
    hicache_group.add_argument(
        "--mean-inter-round-interval",
        type=str,
        default="30,30,30,30,30,30,30,30,30,30",
        help="Mean interval between rounds as comma-separated list (default: 30,30,30,30,30,30,30,30,30,30)",
    )

    args = parser.parse_args()

    # Handle backward compatibility for random data mode
    if args.use_random_data:
        args.use_fixed_data = False
        print("Using legacy random data mode (--use-random-data specified)")
    else:
        print(f"Using fixed data mode with scenario: {args.data_scenario}")

    if args.enable_multiturn and args.enable_shared_prefix:
        parser.error(
            "--enable-multiturn and --enable-shared-prefix cannot be set at the same time."
        )

    run_benchmark(args)