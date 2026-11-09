import json
import os
import queue
import sys
import threading
import time

import requests

# Add current directory to sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_multiturn import (
    ReadyQueue,
    WorkloadGenerator,
    gen_payload,
    log_to_jsonl_file,
    parse_args,
)
from data_processing import sample_fixed_sharegpt_requests
from sglang.bench_serving import get_tokenizer
from tqdm.asyncio import tqdm


class ContextWorkloadGenerator(WorkloadGenerator):
    def __init__(self, args):
        # Construct the base URL for requests
        self.baseurl = f"http://{args.host}:{args.port}/"
        self.url = self.baseurl + "generate"
        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.sent_requests = 0
        self.completed_requests = 0

        # Handle fixed vs original dataset mode
        use_fixed_data = getattr(args, 'use_fixed_data', True)
        if getattr(args, 'use_random_data', False):
            use_fixed_data = False

        if use_fixed_data:
            print(f"Using fixed data mode with scenario: {getattr(args, 'data_scenario', 'long_context')}")
            # Use fixed dataset for deterministic long context results
            fixed_dataset = sample_fixed_sharegpt_requests(
                scenario=getattr(args, 'data_scenario', 'long_context'),
                num_requests=args.num_clients,
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                cache_dir=getattr(args, 'fixed_data_cache_dir', None),
                fixed_output_len=getattr(args, 'output_length', 100),  # Default output length
            )

            init_requests = []
            for i, conversation in enumerate(fixed_dataset):
                if i >= args.num_clients:
                    break
                if conversation:
                    # Use the conversation as a long context prompt
                    # Combine multiple turns to create a longer context
                    combined_prompt = ""
                    total_output_len = 0
                    for turn_prompt, prompt_len, output_len in conversation:
                        combined_prompt += turn_prompt + " "
                        total_output_len += output_len

                    # Use the combined prompt as the long context
                    init_requests.append(
                        (
                            i,
                            gen_payload(
                                combined_prompt.strip(),
                                total_output_len // len(conversation) if conversation else 100,
                                getattr(args, 'lora_path', ''),
                            ),
                        )
                    )

            num_requests = len(init_requests)
        else:
            print("Using legacy JSON dataset format for long context benchmark")
            # Original JSON dataset mode
            try:
                self.dataset = json.load(open(args.dataset_path))
                num_requests = min(args.num_clients, len(self.dataset["queries"]))

                init_requests = []
                for i in range(num_requests):
                    context_id = self.dataset["queries"][i]["context"]
                    init_requests.append(
                        (
                            i,
                            gen_payload(
                                self.dataset["contexts"][context_id]
                                + self.dataset["queries"][i]["question"],
                                len(
                                    self.tokenizer(
                                        self.dataset["queries"][i]["reference_answer"]
                                    )["input_ids"]
                                ),
                                getattr(args, 'lora_path', ''),
                            ),
                        )
                    )
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load JSON dataset ({e}). Falling back to fixed ShareGPT data.")
                # Fallback to fixed data if JSON loading fails
                fixed_dataset = sample_fixed_sharegpt_requests(
                    scenario='long_context',
                    num_requests=args.num_clients,
                    tokenizer=self.tokenizer,
                    dataset_path=args.dataset_path,
                    cache_dir=getattr(args, 'fixed_data_cache_dir', None),
                    fixed_output_len=getattr(args, 'output_length', 100),
                )

                init_requests = []
                for i, conversation in enumerate(fixed_dataset):
                    if i >= args.num_clients:
                        break
                    if conversation:
                        combined_prompt = " ".join([turn[0] for turn in conversation])
                        init_requests.append(
                            (
                                i,
                                gen_payload(
                                    combined_prompt,
                                    100,  # Default output length
                                    getattr(args, 'lora_path', ''),
                                ),
                            )
                        )
                num_requests = len(init_requests)

        self.ready_queue = ReadyQueue(init_requests=init_requests)
        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=num_requests)
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "itl": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }

        self.max_parallel = args.max_parallel
        self.logfile = args.log_file
        self.enable_round_barrier = False

    def response_handler(self):
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
                        self.completed_requests += 1
                        continue
                    else:
                        raise ValueError(f"Request failed with error: {error_msg}")
                # Extract values (they may be wrapped in lists)
                ttft_value = response.ttft[0] if isinstance(response.ttft, list) else response.ttft
                latency_value = response.latency[0] if isinstance(response.latency, list) else response.latency
                prompt_len_value = response.prompt_len[0] if isinstance(response.prompt_len, list) else response.prompt_len
                cached_tokens_value = response.cached_tokens[0] if isinstance(response.cached_tokens, list) else response.cached_tokens
                output_len_value = response.output_len[0] if isinstance(response.output_len, list) else response.output_len
                
                self.performance_metrics["ttft"].append(ttft_value)
                self.performance_metrics["itl"].extend(response.itl if isinstance(response.itl, list) else [response.itl])
                self.performance_metrics["latency"].append(latency_value)
                self.performance_metrics["prompt_len"].append(prompt_len_value)
                self.performance_metrics["cached_tokens"].append(cached_tokens_value)
                self.performance_metrics["generated_len"].append(output_len_value)
                self.completed_requests += 1

            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()

        self.pbar.close()

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
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
                "throughput": self.completed_requests
                / (self.finished_time - self.start_time),
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
            },
        }
        print("All requests completed")
        print("Performance metrics summary:")
        print(f"  Total requests: {performance_data['summary']['total_requests']}")
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(f"  Average latency: {performance_data['summary']['average_latency']:.2f}")
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(f"  Throughput: {performance_data['summary']['throughput']:.2f} requests per second")
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        # Note: user_generator is not used in long context benchmark
        # These stats are only relevant for multi-round benchmarks like mix.py
        return performance_data


if __name__ == "__main__":
    args = parse_args()
    args.num_rounds = 1
    args.max_parallel = 24
    base_url = f"http://{args.host}:{args.port}"

    # Flush cache if requested
    if args.flush_cache:
        try:
            flush_cache_url = f"{base_url}/flush_cache"
            print(f"Flushing cache at {flush_cache_url} ...")
            response = requests.post(flush_cache_url, timeout=30)
            if response.status_code == 200:
                print("Cache flushed successfully")
                time.sleep(2)
            else:
                print(f"Warning: Failed to flush cache: HTTP {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to flush cache: {e}")

    # 运行一次基准测试，请求率由外部脚本控制
    performance_data = ContextWorkloadGenerator(args).run()
    log_to_jsonl_file(performance_data, args.log_file, args.tag)