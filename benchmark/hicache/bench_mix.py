import argparse
import asyncio
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from functools import wraps

import aiohttp

# Add current directory to sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import sample_fixed_sharegpt_requests
from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)

# Set up logger
logger = logging.getLogger(__name__)

# Set up JSONL file for debug logging
debug_log_file = None
# Create a lock for thread-safe debug log writing
debug_log_lock = threading.Lock()


def write_debug_log(data):
    global debug_log_file

    """Write debug information to a JSONL file"""
    if debug_log_file is None:
        return

    # Acquire lock for thread-safe writing
    with debug_log_lock:
        # Write as JSONL (JSON Line format)
        debug_log_file.write(json.dumps(data) + "\n")
        debug_log_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/models/Qwen3-0.6B",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data/models/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="local dataset to sample tokens from",
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
        "--duration",
        type=int,
        default=600,
        help="Duration to run the benchmark in seconds (default: 600 seconds)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=16.0,
        help="Target request rate in requests per second (default: 16.0)",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info"],
        help="Set the logging level (default: info)",
    )
    parser.add_argument(
        "--debug-log-file",
        type=str,
        default="debug.log.jsonl",
        help="File to write debug logs in JSONL format",
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
        default='mix',
        help="Data scenario for fixed dataset selection. Default: mix",
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

    # Configuration arguments (alternative to CONFIG_PATH)
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of rounds for multi-turn conversations (default: 10)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=60,
        help="Maximum number of concurrent client requests (default: 60)",
    )
    parser.add_argument(
        "--round-ratios",
        type=str,
        default="50,25,15,15,10,10,9,8,7,6",
        help="Distribution of requests across rounds as comma-separated list (default: 50,25,15,15,10,10,9,8,7,6)",
    )
    parser.add_argument(
        "--mean-new-tokens-per-round",
        type=str,
        default="1000,400,350,300,280,260,240,220,210,200",
        help="Mean new tokens per round as comma-separated list (default: 1000,400,350,300,280,260,240,220,210,200)",
    )
    parser.add_argument(
        "--mean-return-tokens-per-round",
        type=str,
        default="100,100,100,100,100,100,100,100,100,100",
        help="Mean return tokens per round as comma-separated list (default: 100,100,100,100,100,100,100,100,100,100)",
    )
    parser.add_argument(
        "--mean-inter-round-interval",
        type=str,
        default="30,30,30,30,30,30,30,30,30,30",
        help="Mean interval between rounds as comma-separated list (default: 30,30,30,30,30,30,30,30,30,30)",
    )

    return parser.parse_args()


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


@dataclass
class UserData:
    user_id: int
    current_round: int
    total_rounds: int
    prompt: str
    return_tokens: int
    start: int


def synchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.lock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


class UserGenerator:
    def __init__(self, config, model_path, dataset_path, use_fixed_data=True, data_scenario='mix', fixed_data_cache_dir=None):
        self.tokenizer_path = model_path
        self.tokenizer = get_tokenizer(self.tokenizer_path)
        self.dataset_path = dataset_path
        self.use_fixed_data = use_fixed_data
        self.data_scenario = data_scenario
        self.fixed_data_cache_dir = fixed_data_cache_dir

        self.user_id = 0
        self.lock = threading.Lock()

        self.num_rounds = config["num_rounds"]

        self.cumulative_ratios = [
            sum(config["round_ratios"][: i + 1])
            for i in range(len(config["round_ratios"]))
        ]
        self.mean_new_tokens_per_round = config["mean_new_tokens_per_round"]
        self.mean_return_tokens_per_round = config["mean_return_tokens_per_round"]
        self.mean_inter_round_interval = config["mean_inter_round_interval"]

        self.sigma = 100
        self.range_ratio = 0.8
        assert self.range_ratio <= 1

        if self.use_fixed_data:
            print(f"Using fixed data mode with scenario: {self.data_scenario}")
            # Use fixed dataset for deterministic results
            self.candidate_inputs = []
            for i in range(self.num_rounds):
                # Get fixed dataset for each round
                fixed_dataset = sample_fixed_sharegpt_requests(
                    scenario=self.data_scenario,
                    num_requests=config["num_clients"],
                    tokenizer=self.tokenizer,
                    dataset_path=self.dataset_path,
                    cache_dir=self.fixed_data_cache_dir,
                    fixed_output_len=int(self.mean_return_tokens_per_round[i] * (2 - self.range_ratio)),
                )

                # Convert to the format expected by this benchmark
                round_inputs = []
                for conversation in fixed_dataset:
                    if conversation:
                        # Create a mock object with the expected attributes
                        class MockRequest:
                            def __init__(self, prompt, prompt_len, output_len):
                                self.prompt = prompt
                                self.prompt_len = prompt_len
                                self.output_len = output_len

                        # Use the first turn of each conversation
                        prompt, prompt_len, output_len = conversation[0]
                        round_inputs.append(MockRequest(prompt, prompt_len, output_len))

                self.candidate_inputs.append(round_inputs)
        else:
            print("Using legacy random data sampling for mix benchmark")
            # Legacy random sampling mode
            self.candidate_inputs = [
                [
                    r
                    for r in sample_random_requests(
                        input_len=(
                            self.mean_new_tokens_per_round[i] * (2 - self.range_ratio)
                        ),
                        output_len=(
                            self.mean_return_tokens_per_round[i] * (2 - self.range_ratio)
                        ),
                        num_prompts=config["num_clients"],
                        range_ratio=self.range_ratio / (2 - self.range_ratio),
                        tokenizer=self.tokenizer,
                        dataset_path=self.dataset_path,
                        random_sample=False,
                    )
                ]
                for i in range(self.num_rounds)
            ]

        self.multiturn_queue = []

        self.user_stats = [0 for _ in range(self.num_rounds)]
        self.input_stats = [[0, 0] for _ in range(self.num_rounds)]
        self.output_stats = [[0, 0] for _ in range(self.num_rounds)]

    def gen(self):
        user_id = self.user_id
        self.user_id += 1

        rand_ratio = random.randint(0, self.cumulative_ratios[-1])
        i = len(self.cumulative_ratios)
        for idx, cumulative_ratio in enumerate(self.cumulative_ratios):
            if rand_ratio >= cumulative_ratio:
                continue
            else:
                i = idx + 1
                break
        total_rounds = i
        current_round = 0

        candidate_input = random.sample(self.candidate_inputs[current_round], 1)[0]
        self.input_stats[0][0] += candidate_input.prompt_len
        self.input_stats[0][1] += 1
        prompt = f"{user_id} " + candidate_input.prompt
        return_tokens = int(
            random.gauss(self.mean_return_tokens_per_round[current_round], self.sigma)
        )
        if return_tokens <= 0:
            return_tokens = self.mean_return_tokens_per_round[current_round]
        start = 0

        user_data = UserData(
            user_id, current_round, total_rounds, prompt, return_tokens, start
        )

        self.user_stats[total_rounds - 1] += 1

        return user_data

    @synchronized()
    def push(self, user_data, generated_text, len_itl):
        self.output_stats[user_data.current_round][0] += len_itl + 1
        self.output_stats[user_data.current_round][1] += 1
        user_data.current_round += 1
        if user_data.current_round >= user_data.total_rounds:
            return

        candidate_input = random.sample(
            self.candidate_inputs[user_data.current_round], 1
        )[0]
        self.input_stats[user_data.current_round][0] += candidate_input.prompt_len
        self.input_stats[user_data.current_round][1] += 1
        user_data.prompt += generated_text + candidate_input.prompt
        user_data.return_tokens = int(
            random.gauss(
                self.mean_return_tokens_per_round[user_data.current_round], self.sigma
            )
        )
        if user_data.return_tokens <= 0:
            user_data.return_tokens = self.mean_return_tokens_per_round[
                user_data.current_round
            ]
        interval = random.gauss(
            self.mean_inter_round_interval[user_data.current_round], self.sigma
        )
        if interval <= 0:
            interval = self.mean_inter_round_interval[user_data.current_round]
        user_data.start = time.perf_counter() + interval

        if len(self.multiturn_queue) == 0:
            self.multiturn_queue.append(user_data)
        else:
            i = len(self.multiturn_queue)
            for idx, d in enumerate(self.multiturn_queue):
                if user_data.start < d.start:
                    i = idx
                    break
            self.multiturn_queue.insert(idx, user_data)

    @synchronized()
    def pop(self):
        if (
            len(self.multiturn_queue)
            and time.perf_counter() > self.multiturn_queue[0].start
        ):
            return self.multiturn_queue.pop(0)
        return self.gen()


def gen_payload(prompt, output_len):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "lora_path": "",
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    return payload


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


async def async_request_sglang_generate(
    user_data,
    url,
    atomic_counter,
):
    """
    Sends a streaming request to the server. Gathers text token-by-token.
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()
        payload = gen_payload(user_data.prompt, user_data.return_tokens)
        write_debug_log({"timestamp": st, "user_data": user_data.__dict__})

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
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

                            if data.get("text"):
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    prompt_tokens = (data.get("meta_info") or {}).get(
                                        "prompt_tokens", 0
                                    )
                                    cached_tokens = (data.get("meta_info") or {}).get(
                                        "cached_tokens", 0
                                    )

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text = data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                else:
                    output.error = response.reason or f"HTTP {response.status}"
                    output.success = False
        except Exception as e:
            output.success = False
            error_msg = str(e) if e else "Unknown exception"
            output.error = error_msg
            print(f"Request failed: {error_msg}")

    atomic_counter.increment(1)
    return output


class AtomicCounter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self.lock = threading.Lock()

    @synchronized()
    def increment(self, amount=1):
        self._value += amount

    @synchronized()
    def get(self):
        return self._value


class WorkloadGenerator:
    def __init__(self, args):
        config = load_config(args)

        # Handle fixed vs random data mode
        use_fixed_data = getattr(args, 'use_fixed_data', True)
        if getattr(args, 'use_random_data', False):
            use_fixed_data = False
            print("Using legacy random data mode for mix benchmark")

        user_generator = UserGenerator(
            config,
            args.model_path,
            args.dataset_path,
            use_fixed_data=use_fixed_data,
            data_scenario=getattr(args, 'data_scenario', 'mix'),
            fixed_data_cache_dir=getattr(args, 'fixed_data_cache_dir', None),
        )

        self.url = f"http://{args.host}:{args.port}/generate"

        self.tokenizer = user_generator.tokenizer
        self.start_time = None
        self.finished_time = None
        self.duration = args.duration
        self.request_rate = getattr(args, 'request_rate', 16.0)
        self.done = False

        self.sent_requests = 0
        self.completed_requests = 0

        self.user_generator = user_generator
        self.response_queue = queue.Queue()
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
        }
        self.max_parallel = config["num_clients"]
        
        # Calculate target interval between requests based on request rate
        if self.request_rate > 0:
            self.target_interval = 1.0 / self.request_rate
        else:
            self.target_interval = 0
        self.last_request_time = None

        self.atomic_counter = AtomicCounter()

    async def handle_request(self, user_data):
        try:
            response = await async_request_sglang_generate(
                user_data, self.url, self.atomic_counter
            )
            self.response_queue.put((user_data, response))
        except Exception as e:
            print(f"Request failed: {e}")
            self.completed_requests += 1

    def request_sender(self):
        async def request_loop():
            while True:
                current_time = time.perf_counter()
                
                # Check if we should send a new request based on request rate
                if self.last_request_time is None or \
                   (current_time - self.last_request_time >= self.target_interval):
                    
                    if self.sent_requests - self.completed_requests < self.max_parallel:
                        new_request = self.user_generator.pop()
                        if new_request:
                            asyncio.create_task(self.handle_request(new_request))
                            self.sent_requests += 1
                            self.last_request_time = current_time
                    else:
                        await asyncio.sleep(0.01)
                        continue
                else:
                    # Wait until it's time to send the next request
                    wait_time = self.target_interval - (current_time - self.last_request_time)
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time, 0.01))
                    continue

                if time.perf_counter() - self.start_time > self.duration:
                    self.done = True
                    break

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        while True:
            try:
                user_data, response = self.response_queue.get(timeout=10)
                logger.info(
                    f"{((time.perf_counter()-self.start_time)/self.duration*100):.2f}%"
                )
                if not response.success:
                    # Get error message, handle empty string
                    error_msg = str(response.error) if response.error else "Unknown error"
                    
                    # Check if the error is about input length exceeding context length
                    if "longer than the model's context length" in error_msg:
                        print(f"Warning: Skipping request due to input length exceeding context length: {error_msg}")
                        self.completed_requests += 1
                        continue
                    else:
                        raise ValueError(f"Request failed with error: {error_msg}")

                self.user_generator.push(
                    user_data, response.generated_text, len(response.itl)
                )
                # Extract values (they may be wrapped in lists)
                ttft_value = response.ttft[0] if isinstance(response.ttft, list) else response.ttft
                latency_value = response.latency[0] if isinstance(response.latency, list) else response.latency
                prompt_len_value = response.prompt_len[0] if isinstance(response.prompt_len, list) else response.prompt_len
                cached_tokens_value = response.cached_tokens[0] if isinstance(response.cached_tokens, list) else response.cached_tokens
                
                self.performance_metrics["ttft"].append(ttft_value)
                self.performance_metrics["latency"].append(latency_value)
                self.performance_metrics["prompt_len"].append(prompt_len_value)
                self.performance_metrics["cached_tokens"].append(cached_tokens_value)
                self.completed_requests += 1
                self.finished_time = time.perf_counter()

            except queue.Empty:
                if self.done:
                    break
            except ValueError as e:
                print(f"Error processing response for client {user_data}: {e}")
                continue

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()

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
                "throughput": self.atomic_counter.get()
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
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(
            f"  Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        user_stats = self.user_generator.user_stats
        input_stats = self.user_generator.input_stats
        output_stats = self.user_generator.output_stats
        print(f"round_ratios: {user_stats}")
        print(
            f"mean_new_tokens_per_round: {[int(a/b) if b > 0 else 0 for a, b in input_stats]}"
        )
        print(
            f"mean_return_tokens_per_round: {[int(a/b) if b > 0 else 0 for a, b in output_stats]}"
        )
        return performance_data


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


def main():
    global debug_log_file

    args = parse_args()
    if args.log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
        logger.info("use log_level debug")
        # Initialize debug log file
        debug_log_file = open(args.debug_log_file, "w")
    else:
        logging.basicConfig(level=logging.INFO)
        logger.info("use log_level info")

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
        
        # Flush cache if requested
        base_url = f"http://{args.host}:{args.port}"
        if args.flush_cache:
            if not flush_cache(base_url):
                print("Warning: Failed to flush cache, continuing anyway...")

        performance_data = WorkloadGenerator(args).run()
        if performance_data:
            all_results.append({"request_rate": rate, **performance_data})
        
        # Wait between tests to let system stabilize
        if rate != request_rates[-1]:
            print("\nWaiting 5 seconds before next test...")
            time.sleep(5)
    
    # Print summary of all results
    print_summary(all_results)
    
    # Close debug log file if it was opened
    if debug_log_file:
        debug_log_file.close()
    
    return all_results


def print_summary(all_results):
    """Print summary of all benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for result in all_results:
        rate = result["request_rate"]
        summary = result.get("summary", {})
        print(f"\nRequest Rate: {rate} req/s")
        print(f"  Total Requests: {summary.get('total_requests', 'N/A')}")
        print(f"  Average TTFT: {summary.get('average_ttft', 'N/A'):.2f} ms")
        print(f"  P90 TTFT: {summary.get('p90_ttft', 'N/A'):.2f} ms")
        print(f"  Average Latency: {summary.get('average_latency', 'N/A'):.2f} ms")
        print(f"  P90 Latency: {summary.get('p90_latency', 'N/A'):.2f} ms")
        print(f"  Throughput: {summary.get('throughput', 'N/A'):.2f} req/s")
        print(f"  Cache Hit Rate: {summary.get('cache_hit_rate', 'N/A'):.2%}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Request Rate':<12} {'Throughput':<12} {'TTFT (mean)':<12} {'Latency (mean)':<12} {'Cache Hit':<12}")
    print(f"{' (req/s)':<12} {' (req/s)':<12} {' (ms)':<12} {' (ms)':<12} {' Rate':<12}")
    print("-"*80)
    
    for result in all_results:
        rate = result["request_rate"]
        summary = result.get("summary", {})
        throughput = summary.get('throughput', 0)
        ttft = summary.get('average_ttft', 0)
        latency = summary.get('average_latency', 0)
        cache_hit = summary.get('cache_hit_rate', 0) * 100
        
        print(f"{rate:<12.1f} {throughput:<12.2f} {ttft:<12.2f} {latency:<12.2f} {cache_hit:<12.2f}%")
    
    print("="*80)


if __name__ == "__main__":
    main()