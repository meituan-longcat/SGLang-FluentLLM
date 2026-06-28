import dataclasses
import os
import random
import string
import subprocess
import tempfile
import time
import unittest
from functools import cache
from multiprocessing import Process
from types import SimpleNamespace
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import kill_process_tree, popen_launch_pd_server

default_lb_url: str = "http://127.0.0.1:8192"
default_timeout_for_server_launch = 300
default_model_path: str = (
    "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite"
)

np.random.seed(1234)


@cache
def get_user_name():
    return os.getenv("USER", None)


def kill_all_sglang(kill_lb: bool = False):
    kill_lb_pat = f"| grep -v mini_lb" if not kill_lb else ""
    kill_cmds = (
        f"ps aux | grep sglang {kill_lb_pat} | awk '{{print $2}}' | xargs kill -9"
    )
    os.system(kill_cmds)


def seperator(func):
    def wrapper(*args, **kwargs):
        print("=" * 30, flush=True)
        print(f"开始执行函数: {func.__name__}", flush=True)
        result = func(*args, **kwargs)
        print(f"结束执行函数: {func.__name__}", flush=True)
        print("=" * 30 + "\n", flush=True)
        return result

    return wrapper


def assert_output_and_match_length(
    outputs,
    ref_outputs,
    match_length=None,
    ref_match_length=None,
    input_length=None,
):
    assert (
        outputs == ref_outputs
    ), f"The output of pd-disagg server is {outputs!r}, while the output of single server is {ref_outputs!r}"
    if match_length is None or ref_match_length is None:
        return
    if input_length % 64 == 0:
        assert (
            match_length - ref_match_length
        ) == 64, f"The match length of pd-disagg server is {match_length}, while the match length of single server is {ref_match_length}"
    else:
        assert (
            match_length == ref_match_length
        ), f"The match length of pd-disagg server is {match_length}, while the match length of single server is {ref_match_length}"


def construct_prefix_cache_test_chunked_prefill(input_length=20000) -> list[int]:
    input_ids = np.random.randint(
        low=0, high=102400, size=(input_length,), dtype=np.int64
    )
    return input_ids.tolist()


def construct_str_cache_test(input_length=20000) -> str:
    alphabets = string.ascii_letters + string.digits
    text = random.choices(alphabets, k=input_length)
    return "".join(text)


def send_request_and_get_resp(
    url: str, req: any, max_new_tokens: int = 128, input_type: str = "input_ids"
):
    endpoint = f"{url}/generate"
    response = requests.post(
        endpoint,
        json={
            input_type: req,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
        },
        timeout=600,
    )
    if response.status_code != 200:
        error = response.json()
        raise RuntimeError(f"Sync request failed: {error}")
    d = response.json()
    # print(d, flush=True)
    if isinstance(d, list):
        texts, match_lengths = ([], [])
        for item in d:
            texts.append(item["text"])
            match_lengths.append(item["meta_info"]["cached_tokens"])
        return texts, match_lengths
    else:
        text = d["text"]
        matched_length = d["meta_info"]["cached_tokens"]
        return (text, matched_length)


class DisaggregationSimulatedBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kill_all_sglang(kill_lb=True)

        cls.setUpHook()
        cls.error = None

        cls.model = default_model_path
        parsed_url = urlparse(default_lb_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)
        base_port = cls.base_port
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")
        cls.prefill_in, cls.prefill_out = cls.get_prefill_out_file()
        cls.decode_in, cls.decode_out = cls.get_decode_out_file()

        cls.dependency_env = {
            "EPS_HOME": "/workdir/lvyongkang/code/eps",
            "PYTHONPATH": "/workdir/lvyongkang/code/eps/python:/home/FlashMLA/build/lib.linux-x86_64-cpython-39",
            "LD_LIBRARY_PATH": f"/usr/local/lib:/usr/local/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
            "LD_PRELOAD": "/usr/lib64/libcuda.so",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        }

        cls.common_args = cls.get_comm_args()

        lb_command = [
            "python3",
            "-m",
            "sglang.srt.disaggregation.mini_lb",
            "--host",
            "0.0.0.0",
            "--port",
            cls.lb_port,
        ]

        print("Starting load balancer:", " ".join(lb_command), flush=True)
        env = os.environ.copy()
        env.update(cls.dependency_env)
        cls.process_lb = subprocess.Popen(
            " ".join(lb_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env=env,
            text=True,
        )

        cls.wait_server_ready(cls.lb_url + "/health")

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.wait_server_ready(cls.lb_url + "/health_generate")

        time.sleep(5)

    @classmethod
    def get_comm_args(cls):
        return [
            "--enable-flashinfer-mla",
            "--trust-remote-code",
            "--moe-parallel-strategy",
            "ep",
            "--dense-parallel-strategy",
            "rep",
            "--nprocs-per-node",
            "1",
            "--attn-tp-size",
            "1",
            "--dp-size",
            "1",
            "--random-seed",
            "1234",
            "--host",
            "0.0.0.0",
            "--disable-radix-cache",
        ]

    @classmethod
    def get_pd_args(cls) -> list[str]:
        return [
            "--pdlb-url",
            cls.lb_url,
        ]

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--base-gpu-id",
            "0",
            "--port",
            cls.prefill_port,
        ]
        prefill_args += cls.common_args
        prefill_args += cls.get_pd_args()
        prefill_env = os.environ.copy()
        prefill_env.update(cls.dependency_env)
        prefill_env.update(cls.get_test_env())
        prefill_env.update(
            {
                "SGLANG_ENABLE_TORCH_COMPILE": "0",
            }
        )
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=default_timeout_for_server_launch,
            other_args=prefill_args,
            env=prefill_env,
            stdout=cls.prefill_out,
            stderr=cls.prefill_in,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--base-gpu-id",
            "1",
            "--port",
            cls.decode_port,
            "--disable-cuda-graph",
        ]
        decode_args += cls.common_args
        decode_args += cls.get_pd_args()
        decode_env = os.environ.copy()
        decode_env.update(cls.dependency_env)
        decode_env.update(cls.get_test_env())
        decode_env.update(
            {
                "SGLANG_ENABLE_TORCH_COMPILE": "0",
            }
        )
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=default_timeout_for_server_launch,
            other_args=decode_args,
            env=decode_env,
            stdout=cls.decode_out,
            stderr=cls.decode_in,
        )

    @classmethod
    def get_test_env(cls) -> dict[str, str]:
        return {}

    @classmethod
    def wait_server_ready(
        cls, url, timeout=default_timeout_for_server_launch, method="get"
    ):
        start_time = time.perf_counter()
        while True:
            try:
                func = getattr(requests, method)
                response = func(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready", flush=True)
                    return
                else:
                    print(
                        f"Server {url} is not ready, status code: {response.status_code}",
                        flush=True,
                    )
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownHook()
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # wait for 5 seconds
        time.sleep(5)
        if cls.error:
            raise Exception(cls.error)

    @classmethod
    def get_prefill_out_file(cls):
        return [cls.temp_prefill_stdout_file, cls.temp_prefill_stdout_file]

    @classmethod
    def get_decode_out_file(cls):
        return [cls.temp_decode_stdout_file, cls.temp_decode_stdout_file]

    @classmethod
    def setUpHook(cls):
        cls.temp_decode_stdout_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        )
        print("decode_log", cls.temp_decode_stdout_file.name, flush=True)
        cls.temp_prefill_stdout_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        )
        print("prefill_log:", cls.temp_prefill_stdout_file.name, flush=True)
        return

    @classmethod
    def tearDownHook(cls):
        return


class TestDisaggregationSimulatedGSM8k(DisaggregationSimulatedBase):

    def test_gsm8k(self):
        print("=============================", flush=True)
        print(self.base_host, flush=True)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
            temperature=0.0,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["accuracy"], 0.33)


class TestDisaggregationSimulatedRetract(TestDisaggregationSimulatedGSM8k):
    @classmethod
    def setUpHook(cls):
        os.environ["SGLANG_TEST_RETRACT"] = "true"
        super().setUpHook()

    @classmethod
    def get_decode_out_file(cls):
        return [cls.temp_decode_stdout_file, cls.temp_decode_stdout_file]

    @classmethod
    def get_comm_args(cls):
        args = super().get_comm_args()
        args.extend(["--log-level", "debug"])
        return args

    @classmethod
    def get_test_env(cls) -> dict[str, str]:
        return {
            "SGLANG_TEST_RETRACT": "true",
            "FLUENTLLM_LOG_LEVEL": "debug",
        }

    @classmethod
    def tearDownHook(cls):
        with open(cls.temp_decode_stdout_file.name, "r") as f:
            found = False
            for line in f:
                if "[retract_decode] retracting " in line:
                    found = True
                    break
            if found:
                print("Retracting detected.")
                os.remove(cls.temp_decode_stdout_file.name)
            else:
                cls.error = f"Retracting not found {cls.temp_decode_stdout_file.name}"

        os.environ.pop("SGLANG_TEST_RETRACT")
        super().tearDownHook()


@unittest.skip("tmp")
class DisaggregationSimulatedBaseWithSingle(DisaggregationSimulatedBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.single_port = f"{int(cls.base_port) + 300}"
        cls.single_url = f"http://{cls.base_host}:{cls.single_port}"
        cls.single_in_file, cls.single_out_file = cls.get_single_out_file()
        cls.start_single_server()
        cls.wait_server_ready(cls.single_url + "/health")
        cls.error = 0

        return

    @classmethod
    def setUpHook(cls):
        cls.temp_decode_stdout_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        )
        print(cls.temp_decode_stdout_file.name, flush=True)

    def tearDown(self):
        outcome = self._outcome
        errors = outcome.errors
        for test, exc_info in errors:
            if test is self and exc_info is not None:
                print(f"Found exception in test: {exc_info}", flush=True)
                DisaggregationSimulatedBaseWithSingle.error += 1

    @classmethod
    def tearDownHook(cls):
        if cls.error == 0:
            os.remove(cls.temp_decode_stdout_file.name)

    @classmethod
    def get_single_out_file(cls):
        return [subprocess.DEVNULL, subprocess.DEVNULL]
        # return [None, None]

    @classmethod
    def get_decode_out_file(cls):
        return [cls.temp_decode_stdout_file, cls.temp_decode_stdout_file]

    @classmethod
    def start_single_server(cls):
        single_args = [
            "--base-gpu-id",
            "7",
            "--port",
            cls.single_port,
        ]
        single_args += cls.common_args
        single_env = os.environ.copy()
        single_env.update(cls.dependency_env)
        single_env.update(cls.get_test_env())
        single_env.update(
            {
                "SGLANG_ENABLE_TORCH_COMPILE": "0",
            }
        )
        cls.process_single = popen_launch_pd_server(
            cls.model,
            cls.single_url,
            timeout=default_timeout_for_server_launch,
            other_args=single_args,
            env=single_env,
            stdout=cls.single_in_file,
            stderr=cls.single_out_file,
        )

    def prefix_cache_test(cls, url: str, req: list[int]):
        no_prefix_resp = send_request_and_get_resp(url, req)
        prefix_resp = send_request_and_get_resp(url, req)
        return no_prefix_resp, prefix_resp

    def flush_cache(self):
        self.wait_server_ready(
            self.single_url + "/flush_cache", method="post", timeout=3
        )
        self.wait_server_ready(self.lb_url + "/flush_cache", method="post", timeout=3)

    def length_test(self, input_length):
        self.flush_cache()

        req = construct_prefix_cache_test_chunked_prefill(input_length)
        ((no_prefix_output_ids, ml1), (prefix_out_ids, ml2)) = self.prefix_cache_test(
            url=self.single_url,
            req=req,
        )
        ((no_prefix_output_ids_pd, ml1_pd), (prefix_out_ids_pd, ml2_pd)) = (
            self.prefix_cache_test(
                url=self.lb_url,
                req=req,
            )
        )
        assert_output_and_match_length(no_prefix_output_ids_pd, no_prefix_output_ids)
        assert (
            ml1_pd == ml1
        ), f"the prefix match length is not equal, pd {ml1_pd} != single server {ml1}"
        assert_output_and_match_length(
            prefix_out_ids_pd, prefix_out_ids, ml2_pd, ml2, input_length
        )

    @classmethod
    def get_test_env(cls) -> dict[str, str]:
        return {
            "FLUENTLLM_LOG_LEVEL": "debug",
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.process_single:
            try:
                kill_process_tree(cls.process_single.pid)
            except Exception as e:
                print(f"Error killing process {cls.process_single.pid}: {e}")


class TestDisaggregationPrefixRetraction_1(DisaggregationSimulatedBaseWithSingle):

    @classmethod
    def get_test_env(cls) -> dict[str, str]:
        return {
            "SGLANG_CI_SMALL_KV_SIZE": "1408",
            "FLUENTLLM_LOG_LEVEL": "debug",
        }

    def test_retract(self):
        input_length = 64
        reqs = [
            construct_prefix_cache_test_chunked_prefill(input_length) for _ in range(10)
        ]
        (outputs, match_lengths) = send_request_and_get_resp(
            self.lb_url, reqs, max_new_tokens=768
        )
        (ref_outputs, ref_match_lengths) = send_request_and_get_resp(
            self.single_url, reqs, max_new_tokens=768
        )
        assert_output_and_match_length(outputs, ref_outputs)


class TestDisaggregationPrefixRetraction_2(DisaggregationSimulatedBaseWithSingle):

    @classmethod
    def get_test_env(cls) -> dict[str, str]:
        return {
            "SGLANG_CI_SMALL_KV_SIZE": "512",
            "SGLANG_NUM_RESERVED_DECODE_TOKENS": "64",
            "FLUENTLLM_LOG_LEVEL": "debug",
        }

    def test_retract(self):
        input_length = 64
        reqs = [
            construct_prefix_cache_test_chunked_prefill(input_length) for _ in range(2)
        ]
        (outputs, match_lengths) = send_request_and_get_resp(
            self.lb_url, reqs, max_new_tokens=256
        )
        (ref_outputs, ref_match_lengths) = send_request_and_get_resp(
            self.single_url, reqs, max_new_tokens=256
        )
        assert_output_and_match_length(outputs, ref_outputs)


@unittest.skip("tmp")
class TestDisaggregationPrefixRetraction_3(DisaggregationSimulatedBase):

    @classmethod
    def get_test_env(cls) -> dict[str, str]:
        return {
            "SGLANG_CI_SMALL_KV_SIZE": "320",
            "SGLANG_NUM_RESERVED_DECODE_TOKENS": "1",
            "FLUENTLLM_LOG_LEVEL": "debug",
        }

    def test_retract(self):
        input_length = 64
        reqs = [
            construct_prefix_cache_test_chunked_prefill(input_length) for _ in range(10)
        ]
        (outputs, match_lengths) = send_request_and_get_resp(
            default_lb_url, reqs, max_new_tokens=125
        )


if __name__ == "__main__":
    unittest.main()
