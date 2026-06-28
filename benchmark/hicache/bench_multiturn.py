import argparse
import asyncio
import json
import queue
import random
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

# Add current directory to sys.path for local imports
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)
from data_processing import sample_fixed_sharegpt_requests

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=256,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=128,
        help="Maximum number of parallel requests",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=1024,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=1000,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson",
        choices=["poisson", "uniform"],
        help="Distribution type for request intervals (poisson or uniform)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Average number of requests per second",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("SERVER_HOST", "localhost"),
        help="Server hostname or IP (default: localhost). Can be set via SERVER_HOST env var.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SERVER_PORT", "30000")) if os.environ.get("SERVER_PORT") else 30000,
        help="Server port (default: 30000). Can be set via SERVER_PORT env var.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="local dataset to sample tokens from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="File to log performance metrics",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--disable-random-sample",
        action="store_true",
        help="If set, disable random sampling of requests from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--enable-round-barrier",
        action="store_true",
        help="If set, only send i-th turn requests after all (i-1)-th turn requests finished.",
    )
    parser.add_argument(
        "--sub-question-input-length",
        type=int,
        default=0,
        help="Length of the sub question input for each request, if set 0 use request_length",
    )
    parser.add_argument(
        "--ready-queue-policy",
        type=str,
        default="random",
        help="Policy for popping requests from the ready queue (random or fifo)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag of a certain run in the log file",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="String of LoRA path. Currently we only support benchmarking on a single LoRA adaptor.",
    )
    parser.add_argument(
        "--round-interval",
        type=float,
        default=1,
        help="Interval between rounds in seconds. Default: 0.0 (no interval)",
    )

    # Fixed dataset arguments for deterministic testing
    parser.add_argument(
        "--use-fixed-data",
        action="store_true",
        default=True,
        help="Use fixed, deterministic dataset subsets for reproducible benchmarks. Default: True",
    )
    parser.add_argument(
        "--use-random-data",
        action="store_true",
        help="Use legacy random data sampling mode (overrides --use-fixed-data)",
    )
    parser.add_argument(
        "--data-scenario",
        type=str,
        choices=['serving', 'multiturn', 'mix', 'long_context'],
        default='multiturn',
        help="Data scenario for fixed dataset selection. Default: multiturn",
    )
    parser.add_argument(
        "--fixed-data-cache-dir",
        type=str,
        default=None,
        help="Directory to cache fixed datasets. Default: ~/.cache/sglang/fixed_datasets",
    )

    # Cache management arguments
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        default=False,
        help="Flush server cache before running benchmark",
    )

    return parser.parse_args()


async def async_request_sglang_generate(
    payload,
    url,
    pbar: Optional[tqdm] = None,
):
    """
    Sends a non-streaming request to the server. Returns complete response.
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        st = time.perf_counter()
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                latency = time.perf_counter() - st
                if response.status == 200:
                    # Parse JSON response
                    response_data = await response.json()
                    
                    # Check if this is an error response
                    if "error" in response_data:
                        output.error = response_data.get("error", "Unknown error")
                        output.success = False
                    else:
                        # Extract data from response
                        generated_text = response_data.get("text", "")
                        meta_info = response_data.get("meta_info", {})
                        prompt_tokens = meta_info.get("prompt_tokens", 0)
                        cached_tokens = meta_info.get("cached_tokens", 0)
                        completion_tokens = meta_info.get("completion_tokens", 0)
                        
                        # Set output fields
                        output.success = True
                        output.generated_text = generated_text
                        output.latency = latency
                        output.prompt_len = prompt_tokens
                        output.cached_tokens = cached_tokens
                        output.output_len = completion_tokens
                        
                        # For non-streaming, we don't have ttft and itl metrics
                        # Set ttft to latency (time to first and only token)
                        output.ttft = latency
                        
                else:
                    output.error = response.reason or f"HTTP {response.status}"
                    output.success = False
        except Exception as e:
            output.success = False
            error_msg = str(e) if e else "Unknown exception"
            output.error = error_msg
            print(f"Request failed: {error_msg}")

    if pbar:
        pbar.update(1)
    return output


def gen_payload(prompt, output_len, lora_path=""):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,  # 改为非流式
        "lora_path": lora_path,
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    return payload


def log_to_jsonl_file(data, file_path="performance_metrics.jsonl", tag=""):
    """Append the data with a timestamp and tag to the specified JSONL file."""
    timestamped_data = {"timestamp": datetime.now().isoformat(), "tag": tag, **data}
    try:
        with open(file_path, "a") as file:
            file.write(
                json.dumps(timestamped_data) + "\n"
            )  # Write as a single line in JSONL format
    except IOError as e:
        print(f"Error writing to JSONL file: {e}")


class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self.lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                # todo, varying thinking time of clients
                raise ValueError(f"{self.policy} not implemented")


class WorkloadGenerator:
    def __init__(self, args):
        # Construct the base URL for requests
        self.url = f"http://{args.host}:{args.port}/generate"

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.sent_requests = 0
        self.completed_requests = 0

        # Handle fixed vs random data mode
        if getattr(args, 'use_random_data', False):
            args.use_fixed_data = False
            print("Using legacy random data mode for multiturn benchmark")

        if getattr(args, 'use_fixed_data', True):
            # Use fixed dataset for deterministic results
            print(f"Using fixed data mode with scenario: {getattr(args, 'data_scenario', 'multiturn')}")

            # Get fixed dataset for main inputs
            fixed_dataset = sample_fixed_sharegpt_requests(
                scenario=getattr(args, 'data_scenario', 'multiturn'),
                num_requests=args.num_clients,
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                cache_dir=getattr(args, 'fixed_data_cache_dir', None),
                fixed_output_len=args.output_length,
            )

            # Convert to the format expected by this benchmark
            self.candidate_inputs = []
            for conversation in fixed_dataset:
                if conversation:  # conversation is a list of (prompt, input_len, output_len) tuples
                    # Use the first turn as the candidate input
                    self.candidate_inputs.append(conversation[0][0])  # conversation[0][0] is the prompt

            # Ensure we have enough inputs
            while len(self.candidate_inputs) < args.num_clients:
                self.candidate_inputs.extend(self.candidate_inputs[:args.num_clients - len(self.candidate_inputs)])
            self.candidate_inputs = self.candidate_inputs[:args.num_clients]

            # Get sub-question inputs from the same fixed dataset
            sub_question_dataset = sample_fixed_sharegpt_requests(
                scenario=getattr(args, 'data_scenario', 'multiturn'),
                num_requests=args.num_clients * max(args.num_rounds - 1, 1),
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                cache_dir=getattr(args, 'fixed_data_cache_dir', None),
                fixed_output_len=args.output_length,
            )

            # Convert sub-question inputs
            self.sub_question_inputs = []
            for conversation in sub_question_dataset:
                if conversation:
                    # Use subsequent turns or repeat first turn for sub-questions
                    if len(conversation) > 1:
                        self.sub_question_inputs.append(conversation[1][0])  # Second turn
                    else:
                        self.sub_question_inputs.append(conversation[0][0])  # First turn

        else:
            # Legacy random sampling mode
            print("Using legacy random data sampling")
            self.candidate_inputs = sample_random_requests(
                input_len=args.request_length,
                output_len=args.output_length,
                num_prompts=args.num_clients,
                range_ratio=1.0,
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                random_sample=not args.disable_random_sample,
            )
            # sample_random_requests returns tuples (prompt, input_len, output_len)
            # Extract the prompt from each tuple
            self.candidate_inputs = [i[0] for i in self.candidate_inputs]

            if args.sub_question_input_length != 0:
                sub_question_input_length = args.sub_question_input_length
            else:
                sub_question_input_length = args.request_length

            self.sub_question_inputs = sample_random_requests(
                input_len=sub_question_input_length,
                output_len=args.output_length,
                num_prompts=args.num_clients * max(args.num_rounds - 1, 1),
                range_ratio=1.0,
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                random_sample=not args.disable_random_sample,
            )
            # Convert to the same format as candidate_inputs
            # sample_random_requests returns tuples (prompt, input_len, output_len)
            # Extract the prompt from each tuple
            self.sub_question_inputs = [i[0] for i in self.sub_question_inputs]

        init_requests = [
            (
                i,
                gen_payload(
                    self.candidate_inputs[i], args.output_length, args.lora_path
                ),
            )
            for i in range(args.num_clients)
        ]
        self.client_records = {
            i: {"round": 0, "history": init_requests[i][1]["text"]}
            for i in range(args.num_clients)
        }
        self.ready_queue = ReadyQueue(
            init_requests=init_requests, policy=args.ready_queue_policy
        )
        self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=args.num_clients * args.num_rounds)
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }
        self.enable_round_barrier = args.enable_round_barrier
        if self.enable_round_barrier:
            # Add round-specific metrics while preserving the original structure
            for i in range(args.num_rounds):
                self.performance_metrics[f"round_{i}"] = {
                    "ttft": [],
                    "latency": [],
                    "prompt_len": [],
                    "cached_tokens": [],
                    "generated_len": [],
                }
        self.num_clients = args.num_clients

        self.num_rounds = args.num_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length
        self.round_interval = args.round_interval
        self.args = args  # Store args for later use

    async def handle_request(self, item):
        try:
            client_id, payload = item
            response = await async_request_sglang_generate(payload, self.url, self.pbar)
            if self.pbar.n == self.pbar.total:
                self.finished_time = time.perf_counter()
            self.response_queue.put((client_id, response))
        except Exception as e:
            print(f"Request failed: {e}")

    def request_sender(self):
        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.05)
                    continue

                if self.pbar.n == self.pbar.total:
                    break

                # Calculate Poisson-distributed wait time
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / self.request_rate if self.request_rate > 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                await asyncio.sleep(sleep_time)  # Wait before sending the next request

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        next_round_reqs = []
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    # Get error message, handle empty string
                    error_msg = str(response.error) if response.error else "Unknown error"
                    
                    # Check if the error is about input length exceeding context length
                    if "longer than the model's context length" in error_msg:
                        print(f"Warning: Skipping client {client_id} due to input length exceeding context length: {error_msg}")
                        # Mark this client as completed to avoid further rounds
                        self.client_records[client_id]["round"] = self.num_rounds
                        self.completed_requests += 1
                        continue
                    else:
                        raise ValueError(f"Request failed with error: {error_msg}")
                # Extract values from response
                generated_text = response.generated_text
                ttft_value = response.ttft
                latency_value = response.latency
                prompt_len_value = response.prompt_len
                cached_tokens_value = response.cached_tokens
                output_len_value = response.output_len
                
                self.client_records[client_id]["history"] += generated_text
                current_round = self.client_records[client_id]["round"]
                self.client_records[client_id]["round"] += 1
                self.performance_metrics["ttft"].append(ttft_value)
                self.performance_metrics["latency"].append(latency_value)
                self.performance_metrics["prompt_len"].append(prompt_len_value)
                self.performance_metrics["cached_tokens"].append(cached_tokens_value)
                self.performance_metrics["generated_len"].append(output_len_value)
                if self.enable_round_barrier:
                    self.performance_metrics[f"round_{current_round}"]["ttft"].append(
                        ttft_value
                    )
                    self.performance_metrics[f"round_{current_round}"][
                        "latency"
                    ].append(latency_value)
                    self.performance_metrics[f"round_{current_round}"][
                        "prompt_len"
                    ].append(prompt_len_value)
                    self.performance_metrics[f"round_{current_round}"][
                        "cached_tokens"
                    ].append(cached_tokens_value)
                    self.performance_metrics[f"round_{current_round}"][
                        "generated_len"
                    ].append(output_len_value)
                self.completed_requests += 1

                if self.client_records[client_id]["round"] < self.num_rounds:
                    # Add interval between rounds if specified
                    if self.round_interval > 0:
                        time.sleep(self.round_interval)
                    
                    # append new request to client's history
                    self.client_records[client_id][
                        "history"
                    ] += self.sub_question_inputs.pop()
                    new_req = (
                        client_id,
                        gen_payload(
                            self.client_records[client_id]["history"],
                            self.output_length,
                            self.args.lora_path,
                        ),
                    )
                    if self.enable_round_barrier:
                        next_round_reqs.append(new_req)
                        if len(next_round_reqs) == self.num_clients:
                            for req in next_round_reqs:
                                self.ready_queue.append(req)
                            next_round_reqs = []
                    else:
                        self.ready_queue.append(new_req)
            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break
            except ValueError as e:
                print(f"Error processing response for client {client_id}: {e}")
                continue

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()
        self.pbar.close()

        duration = self.finished_time - self.start_time
        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "request_rate": self.request_rate,
                "average_prompt_len": (
                    sum(self.performance_metrics["prompt_len"])
                    / len(self.performance_metrics["prompt_len"])
                    if self.performance_metrics["prompt_len"]
                    else 0.0
                ),
                "average_output_len": (
                    sum(self.performance_metrics["generated_len"])
                    / len(self.performance_metrics["generated_len"])
                    if self.performance_metrics["generated_len"]
                    else 0.0
                ),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": sorted(self.performance_metrics["ttft"])[
                    int(0.9 * len(self.performance_metrics["ttft"]))
                ],
                "median_ttft": sorted(self.performance_metrics["ttft"])[
                    len(self.performance_metrics["ttft"]) // 2
                ],
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": sorted(self.performance_metrics["latency"])[
                    int(0.9 * len(self.performance_metrics["latency"]))
                ],
                "median_latency": sorted(self.performance_metrics["latency"])[
                    len(self.performance_metrics["latency"]) // 2
                ],
                "input_token_throughput": sum(self.performance_metrics["prompt_len"])
                / duration,
                "output_token_throughput": sum(
                    self.performance_metrics["generated_len"]
                )
                / duration,
                "throughput": self.pbar.total / duration,
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
            },
        }
        if self.enable_round_barrier:
            performance_data["round"] = {}
            for round_num in range(args.num_rounds):
                round_key = f"round_{round_num}"
                round_metrics = self.performance_metrics[round_key]
                performance_data["round"][round_key] = {
                    "average_ttft": (
                        sum(round_metrics["ttft"]) / len(round_metrics["ttft"])
                        if round_metrics["ttft"]
                        else 0
                    ),
                    "cache_hit_rate": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "request_count": len(round_metrics["ttft"]),
                }
        print("All requests completed")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {performance_data['summary']['total_requests']} at {performance_data['summary']['request_rate']} requests per second"
        )
        print(
            f"  Average Prompt Length: {performance_data['summary']['average_prompt_len']:.2f} tokens"
        )
        print(
            f"  Average Output Length: {performance_data['summary']['average_output_len']:.2f} tokens"
        )
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(
            f"  Input token throughput: {performance_data['summary']['input_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Output token throughput: {performance_data['summary']['output_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Request Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        if self.enable_round_barrier:
            # Print round-basedsummary
            print("Per-round metrics:")
            if "round" in performance_data:
                for round_num in range(self.num_rounds):
                    round_key = f"round_{round_num}"
                    if round_key in performance_data["round"]:
                        round_data = performance_data["round"][round_key]
                        avg_ttft = round_data["average_ttft"]
                        cache_hit_rate = round_data["cache_hit_rate"]
                        request_count = round_data["request_count"]
                        print(
                            f"  Round {round_num}: Average TTFT = {avg_ttft:.2f}s, "
                            f"Cache Hit Rate = {cache_hit_rate:.6f} "
                            f"({request_count} requests)"
                        )
                    else:
                        print(f"  Round {round_num}: No requests completed")

        return performance_data


if __name__ == "__main__":
    args = parse_args()
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.disable_auto_run:
        print("Running with specified request rate...")
        request_rates = [args.request_rate]
    else:
        print("Auto-running with different request rates...")
        request_rates = [16, 8, 4, 2, 1]

    for rate in request_rates:
        args.request_rate = rate
        if args.flush_cache:
            try:
                requests.post(flush_cache_url, timeout=30)
                print("Cache flushed successfully")
                time.sleep(2)
            except Exception as e:
                print(f"Warning: Failed to flush cache: {e}")
        performance_data = WorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, tag=args.tag)