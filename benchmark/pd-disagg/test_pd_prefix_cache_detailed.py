#!/usr/bin/env python3
"""
PD分离前缀缓存详细测试代码
在八卡机上测试前缀缓存功能，包括：
1. 完全复用前缀
2. 无复用前缀  
3. 复用部分前缀
4. 混合场景测试
5. 压力测试
使用方法：
python test_pd_prefix_cache_detailed.py --test_case all
python test_pd_prefix_cache_detailed.py --test_case full_reuse
"""

import argparse
import dataclasses
import os
import sys
import random
import string
import subprocess
import tempfile
import time
import unittest
from functools import cache
from multiprocessing import Process
from types import SimpleNamespace
from typing import Callable, List, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import orjson

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import kill_process_tree, popen_launch_pd_server

# 测试配置
DEFAULT_LB_URL: str = "http://0.0.0.0:8192"
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600
DEFAULT_MODEL_PATH: str = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite"

# 设置随机种子确保结果可重现
np.random.seed(1234)
random.seed(1234)

def _get_user_python():
    """获取用户的Python解释器路径"""
    import shutil
    
    # 首先检查当前使用的Python
    current_python = sys.executable
    if current_python and '/usr/local/' not in current_python:
        return current_python
    
    # 如果当前是系统Python，查找虚拟环境中的Python
    home_dir = os.path.expanduser('~')
    possible_python_paths = [
        os.path.join(home_dir, 'miniconda3', 'bin', 'python'),
        os.path.join(home_dir, 'anaconda3', 'bin', 'python'),
        os.path.join(home_dir, '.conda', 'bin', 'python'),
        os.path.join(home_dir, 'venv', 'bin', 'python'),
        os.path.join(home_dir, 'env', 'bin', 'python'),
    ]
    
    # 检查which python3是否指向虚拟环境
    try:
        which_result = subprocess.run(['which', 'python3'], capture_output=True, text=True)
        if which_result.returncode == 0:
            python_path = which_result.stdout.strip()
            if python_path and '/usr/local/' not in python_path and os.path.exists(python_path):
                return python_path
    except Exception:
        pass
    
    # 查找虚拟环境中的Python
    for python_path in possible_python_paths:
        if os.path.exists(python_path):
            return python_path
    
    # 如果都找不到，返回None，让子进程使用shell来激活环境
    return None

# 获取用户Python路径
_get_user_python = _get_user_python()

@cache
def get_user_name():
    return os.getenv("USER", None)

def kill_all_sglang(kill_lb: bool = False):
    """清理所有sglang进程"""
    kill_lb_pat = f"| grep -v mini_lb" if not kill_lb else ""
    kill_cmds = (
        f"ps aux | grep sglang {kill_lb_pat} | awk '{{print $2}}' | xargs kill -9"
    )
    os.system(kill_cmds)

def seperator(func):
    """函数执行分隔符装饰器"""
    def wrapper(*args, **kwargs):
        print("=" * 60, flush=True)
        print(f"🚀 开始执行测试: {func.__name__}", flush=True)
        print("=" * 60, flush=True)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"✅ 测试完成: {func.__name__}, 耗时: {end_time - start_time:.2f}秒", flush=True)
        print("=" * 60 + "\n", flush=True)
        return result
    return wrapper

class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self, tokenizer_path: str = None):
        if tokenizer_path:
            self.tokenizer = get_tokenizer(tokenizer_path)
        else:
            self.tokenizer = None
    
    def generate_prefix_ids(self, length: int) -> List[int]:
        """生成指定长度的前缀token IDs"""
        if self.tokenizer:
            # 生成有意义的文本前缀
            prefix_text = "The quick brown fox jumps over the lazy dog. " * (length // 10 + 1)
            prefix_text = prefix_text[:length * 2]  # 粗略估计token长度
            token_ids = self.tokenizer.encode(prefix_text)[:length]
            # 填充或截断到指定长度
            if len(token_ids) < length:
                token_ids.extend([random.randint(0, 102400) for _ in range(length - len(token_ids))])
            else:
                token_ids = token_ids[:length]
        else:
            token_ids = np.random.randint(low=0, high=102400, size=(length,), dtype=np.int64).tolist()
        return token_ids
    
    def generate_different_suffix(self, base_length: int, suffix_length: int) -> List[List[int]]:
        """生成不同后缀的测试用例"""
        base_prefix = self.generate_prefix_ids(base_length)
        
        suffixes = []
        for i in range(5):  # 生成5个不同的后缀
            suffix = self.generate_prefix_ids(suffix_length)
            full_input = base_prefix + suffix
            suffixes.append({
                'input_ids': full_input,
                'description': f'case_{i}_suffix_len_{suffix_length}',
                'prefix_len': base_length,
                'suffix_len': suffix_length
            })
        
        return suffixes

class PrefixCacheTestCase:
    """前缀缓存测试用例基类"""
    
    def __init__(self):
        self.test_results = []
        self.data_generator = TestDataGenerator()
        
    def send_request_and_get_resp(
        self, 
        url: str, 
        req: any, 
        max_new_tokens: int = 64,
        input_type: str = "input_ids",
        stream: bool = False
    ) -> Tuple[any, Dict]:
        """
        发送请求并获取响应
        
        Returns:
            Tuple[response_text, metadata_dict]
        """
        endpoint = f"{url}/generate"
        json_data = {
            input_type: req,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
            "stream": stream,
        }
        
        try:
            response = requests.post(
                endpoint,
                json=json_data,
                timeout=300,
            )
            if response.status_code != 200:
                try:
                    error = response.json()
                except:
                    error = response.text
                raise RuntimeError(f"请求失败 (status={response.status_code}): {error}")
            
            d = response.json()
            
            if isinstance(d, list):
                texts = [item["text"] for item in d]
                metas = [item["meta_info"] for item in d]
                output_extra_info = [item["output_extra_info"] for item in d]
                return texts, metas, output_extra_info
            else:
                text = d["text"]
                meta_info = d.get("meta_info", {})
                output_extra_info = d.get("output_extra_info", {})
                return text, meta_info, output_extra_info
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            raise

class DetailedPrefixCacheTest(unittest.TestCase):
    """详细的前缀缓存测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试环境初始化"""
        print("🔧 初始化PD分离测试环境...", flush=True)
        cls.base_host = "0.0.0.0"
        cls.base_port = 8192
        cls.lb_port = str(cls.base_port)
        cls.prefill_port = str(cls.base_port + 100)
        cls.decode_port = str(cls.base_port + 200)
        
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        
        print(f"📍 服务地址配置:")
        print(f"   - Load Balancer: {cls.lb_url}")
        print(f"   - Prefill Server: {cls.prefill_url}")
        print(f"   - Decode Server: {cls.decode_url}")
        
        # 日志文件 - 为每个服务创建独立的日志文件
        timestamp = int(time.time())
        
        # Load Balancer 日志
        cls.lb_log_file = f"/tmp/pd_lb_log_{timestamp}.log"
        
        # Prefill Server 日志
        cls.prefill_log_file = f"/tmp/pd_prefill_log_{timestamp}.log"
        
        # Decode Server 日志 
        cls.decode_log_file = f"/tmp/pd_decode_log_{timestamp}.log"
        
        print(f"📝 日志文件:")
        print(f"   - Load Balancer: {cls.lb_log_file}")
        print(f"   - Prefill Server: {cls.prefill_log_file}")
        print(f"   - Decode Server: {cls.decode_log_file}")
        
        # 创建文件句柄用于实时输出
        cls.lb_log_handle = open(cls.lb_log_file, 'w')
        cls.prefill_log_handle = open(cls.prefill_log_file, 'w')
        cls.decode_log_handle = open(cls.decode_log_file, 'w')
        
        # 环境变量配置
        cls.dependency_env = {
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
            "FLUENTLLM_LOG_LEVEL": "debug",
            "CONDA_DEFAULT_ENV": os.environ.get('CONDA_DEFAULT_ENV', 'base'),
            "VIRTUAL_ENV": os.environ.get('VIRTUAL_ENV', ''),
        }
        
        # 通用服务参数
        cls.common_args = [
            "--enable-flashinfer-mla",
            "--trust-remote-code",
            "--moe-parallel-strategy", "ep",
            "--dense-parallel-strategy", "rep",
            "--nprocs-per-node", "1",
            "--attn-tp-size", "1",
            "--dp-size", "1",
            "--random-seed", "1234",
            "--host", "0.0.0.0",
            "--disable-radix-cache",
            "--log-level", "debug",
        ]
        
        # PD分离参数
        cls.pd_args = [
            "--pdlb-url", cls.lb_url,
        ]
        
        cls.error = None
        cls.processes = []
        cls.all_test_results = []  # 全局测试结果收集
        
        # 启动服务
        cls._start_services()
        
        print("✅ 测试环境初始化完成")
        print(f"🐍 当前Python环境: {_get_user_python or 'python3'}")
        time.sleep(3)
    
    @classmethod
    def _start_services(cls):
        """启动所有服务"""
        # 启动负载均衡器
        # cls._start_load_balancer()
        
        # 启动prefill服务
        # cls._start_prefill_server()
        
        # 启动decode服务
        # cls._start_decode_server()
        
        # 等待服务就绪
        # cls._wait_services_ready()

        # 虚拟环境下存在问题，需手动拉起服务， 待后续修正
        pass 
    
    @classmethod
    def _start_load_balancer(cls):
        """启动负载均衡器"""
        # 获取用户Python路径
        user_python = _get_user_python
        
        if user_python:
            # 使用找到的Python路径
            python_cmd = user_python
            print(f"🐍 使用用户Python环境: {user_python}")
            
            # 设置环境
            env = os.environ.copy()
            env.update(cls.dependency_env)
            env["PYTHON"] = user_python
            
            lb_command = [
                python_cmd, "-m", "sglang.srt.disaggregation.mini_lb",
                "--host", "0.0.0.0",
                "--port", cls.lb_port,
            ]
            
            print(f"🚀 启动Load Balancer: {' '.join(lb_command)}")
            
            with open(cls.lb_log_file, 'w') as lb_log:
                cls.process_lb = subprocess.Popen(
                    " ".join(lb_command),
                    stdout=lb_log,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    env=env,
                    text=True,
                )
        else:
            # 没找到虚拟环境Python，使用shell方式来激活环境
            print(f"🔧 未找到虚拟环境Python，使用shell方式启动")
            
            # 构造通过bash激活环境的命令
            bash_command = f"""
                source ~/.bashrc | python3 -m sglang.srt.disaggregation.mini_lb --host 0.0.0.0 --port {cls.lb_port} "$@"
                """
            
            # 创建临时shell脚本
            shell_script = f"/tmp/start_lb_{cls.lb_port}.sh"
            with open(shell_script, 'w') as f:
                f.write(bash_command)
            os.chmod(shell_script, 0o755)
            
            print(f"🚀 通过bash启动Load Balancer: {shell_script}")
            
            env = os.environ.copy()
            env.update(cls.dependency_env)
            
            with open(cls.lb_log_file, 'w') as lb_log:
                cls.process_lb = subprocess.Popen(
                    ['bash', shell_script],
                    stdout=lb_log,
                    stderr=lb_log,
                    env=env,
                    text=True,
                    shell=True,
                )
    
    @classmethod
    def _start_prefill_server(cls):
        """启动prefill服务器"""
        prefill_args = [
            "--disaggregation-mode", "prefill",
            "--base-gpu-id", "0",  # 使用GPU 0-3作为prefill
            "--port", cls.prefill_port,
            "--disable-cuda-graph",
        ]
        prefill_args.extend(cls.common_args)
        prefill_args.extend(cls.pd_args)
        
        prefill_env = os.environ.copy()
        prefill_env.update(cls.dependency_env)
        prefill_env.update({
            "SGLANG_ENABLE_TORCH_COMPILE": "0",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",  # 前4张GPU给prefill
        })
        
        # 获取用户Python路径
        user_python = _get_user_python
        
        if user_python:
            # 使用找到的Python路径
            print(f"🐍 使用用户Python环境: {user_python}")
            prefill_env["PYTHON"] = user_python
            
            print(f"🚀 启动Prefill服务器，GPU: 0,1,2,3")
            with open(cls.prefill_log_file, 'w') as prefill_log:
                cls.process_prefill = popen_launch_pd_server(
                    DEFAULT_MODEL_PATH,
                    cls.prefill_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=prefill_args,
                    env=prefill_env,
                    stdout=prefill_log,
                    stderr=prefill_log,
                )
        else:
            # 没找到虚拟环境Python，使用shell方式来激活环境
            print(f"🔧 未找到虚拟环境Python，使用shell方式启动Prefill")
            
            # 构造通过bash激活环境的命令
            bash_command = f"""
                #!/bin/bash
                source ~/.bashrc && \
                CUDA_VISIBLE_DEVICES=0,1,2,3 SGLANG_ENABLE_TORCH_COMPILE=0 \
                exec python3 -m sglang.srt.launch_server \
                    --model-path {DEFAULT_MODEL_PATH} \
                    --port {cls.prefill_url.replace('http://127.0.0.1:', '')} \
                    --disaggregation-mode prefill \
                    --base-gpu-id 0 \
                    --attn-tp-size 1 \
                    --dp-size 1 \
                    --pdlb-url {cls.lb_url} \
                    --enable-flashinfer-mla \
                    --trust-remote-code \
                    --moe-parallel-strategy ep \
                    --dense-parallel-strategy rep \
                    --nprocs-per-node 1 \
                    --host 0.0.0.0 \
                    --disable-radix-cache \
                    --log-level debug \
                    --disable-cuda-graph \
                    "$@"
                """
            
            # 创建临时shell脚本
            shell_script = f"/tmp/start_prefill_{cls.prefill_port}.sh"
            with open(shell_script, 'w') as f:
                f.write(bash_command)
            os.chmod(shell_script, 0o755)
            
            print(f"🚀 通过bash启动Prefill: {shell_script}")
            
            env = os.environ.copy()
            env.update(cls.dependency_env)
            
            with open(cls.prefill_log_file, 'w') as prefill_log:
                cls.process_prefill = subprocess.Popen(
                    ['bash', shell_script],
                    stdout=prefill_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
    
    @classmethod
    def _start_decode_server(cls):
        """启动decode服务器"""
        decode_args = [
            "--disaggregation-mode", "decode",
            "--base-gpu-id", "4",  # 使用GPU 4-7作为decode
            "--port", cls.decode_port,
            "--disable-cuda-graph",
        ]
        decode_args.extend(cls.common_args)
        decode_args.extend(cls.pd_args)
        
        decode_env = os.environ.copy()
        decode_env.update(cls.dependency_env)
        decode_env.update({
            "SGLANG_ENABLE_TORCH_COMPILE": "0",
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",  # 后4张GPU给decode
        })
        
        # 获取用户Python路径
        user_python = _get_user_python
        
        if user_python:
            # 使用找到的Python路径
            print(f"🐍 使用用户Python环境: {user_python}")
            decode_env["PYTHON"] = user_python
            
            print(f"🚀 启动Decode服务器，GPU: 4,5,6,7")
            with open(cls.decode_log_file, 'w') as decode_log:
                cls.process_decode = popen_launch_pd_server(
                    DEFAULT_MODEL_PATH,
                    cls.decode_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=decode_args,
                    env=decode_env,
                    stdout=decode_log,
                    stderr=decode_log,
                )
        else:
            # 没找到虚拟环境Python，使用shell方式来激活环境
            print(f"🔧 未找到虚拟环境Python，使用shell方式启动Decode")
            
            # 构造通过bash激活环境的命令
            bash_command = f"""
                #!/bin/bash
                source ~/.bashrc && \
                CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_ENABLE_TORCH_COMPILE=0 \
                exec python3 -m sglang.srt.launch_server \
                    --model-path {DEFAULT_MODEL_PATH} \
                    --port {cls.decode_port} \
                    --disaggregation-mode decode \
                    --base-gpu-id 4 \
                    --attn-tp-size 1 \
                    --dp-size 1 \
                    --pdlb-url {cls.lb_url} \
                    --enable-flashinfer-mla \
                    --trust-remote-code \
                    --moe-parallel-strategy ep \
                    --dense-parallel-strategy rep \
                    --nprocs-per-node 1 \
                    --host 0.0.0.0 \
                    --disable-radix-cache \
                    --log-level debug \
                    --disable-cuda-graph \
                    "$@"
                """
            
            # 创建临时shell脚本
            shell_script = f"/tmp/start_decode_{cls.decode_port}.sh"
            with open(shell_script, 'w') as f:
                f.write(bash_command)
            os.chmod(shell_script, 0o755)
            
            print(f"🚀 通过bash启动Decode: {shell_script}")
            
            env = os.environ.copy()
            env.update(cls.dependency_env)
            
            with open(cls.decode_log_file, 'w') as decode_log:
                cls.process_decode = subprocess.Popen(
                    ['bash', shell_script],
                    stdout=decode_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
    
    @classmethod
    def _wait_services_ready(cls, timeout=3000):
        """等待所有服务就绪"""
        print(f"⏳ 等待服务就绪（最长{timeout}秒）...")
        
        health_endpoints = [
            (cls.lb_url,"health", "Load Balancer"),
            (cls.prefill_url, "/get_model_info","Prefill Server"),
            (cls.decode_url, "health","Decode Server"),
        ]
        
        # 添加LB的特殊健康检查端点
        health_endpoints.append((cls.lb_url, "health_generate", "LB Generate"))
        
        for url, endpoint, name in health_endpoints:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{url}/{endpoint}", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ {name} 就绪")
                        break
                    else:
                        print(f"{name} {endpoint}  : response {response}")
                except Exception as e:
                    print(f"{name} {endpoint} error {e}")
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f"❌ {name} 启动超时")

    def setUp(self):
        """每个测试方法执行前的设置"""
        # 清理缓存
        self.flush_cache()
        time.sleep(1)
        
        # 初始化数据生成器
        self.data_generator = TestDataGenerator()
        self.test_results = []
    
    def send_request_and_get_resp(
        self, 
        url: str, 
        req: any, 
        max_new_tokens: int = 64,
        input_type: str = "input_ids",
        stream: bool = False
    ) -> Tuple[any, Dict]:
        """
        发送请求并获取响应
        
        Returns:
            Tuple[response_text, metadata_dict]
        """
        endpoint = f"{url}/generate"
        json_data = {
            input_type: req,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
            "stream": stream,
        }
        
        try:
            response = requests.post(
                endpoint,
                json=json_data,
                timeout=300,
            )
            if response.status_code != 200:
                try:
                    error = response.json()
                except:
                    error = response.text
                raise RuntimeError(f"请求失败 (status={response.status_code}): {error}")
            
            d = response.json()
            if isinstance(d, list):
                texts = [item["text"] for item in d]
                metas = [item["meta_info"] for item in d]
                output_extra_info = [item["output_extra_info"] for item in d]
                return texts, metas, output_extra_info
            else:
                text = d["text"]
                meta_info = d.get("meta_info", {})
                output_extra_info = d.get("output_extra_info", {})
                return text, meta_info, output_extra_info
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            raise
    
    def flush_cache(self):
        """清理缓存"""
        try:
            requests.post(f"{self.lb_url}/flush_cache", timeout=10)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ 清理缓存失败: {e}")
    
    @seperator
    def test_full_prefix_reuse(self):
        """
        测试完整前缀复用场景
        相同前缀的多个请求应该能够复用缓存
        """
        print("🧪 测试场景：完整前缀复用")
        
        # 构造测试数据：相同前缀 + 不同后缀
        prefix_length = 256
        suffix_length = 64
        
        base_prefix = self.data_generator.generate_prefix_ids(prefix_length)
        
        test_cases = []
        for i in range(5):
            suffix = self.data_generator.generate_prefix_ids(suffix_length)
            input_ids = base_prefix + suffix
            test_cases.append({
                'name': f'full_reuse_case_{i}',
                'input_ids': input_ids,
                'prefix_len': prefix_length,
                'suffix_len': suffix_length
            })
        
        print(f"📊 测试用例：{len(test_cases)}个请求，前缀长度: {prefix_length}, 后缀长度: {suffix_length}")
        
        # 发送第一个请求（建立缓存）
        print("📤 发送第一个请求（建立前缀缓存）...")
        text1, meta1, output_extra_info = self.send_request_and_get_resp(self.lb_url, test_cases[0]['input_ids'])
        cached_tokens_1 = output_extra_info.get('decode_prefix_len', 0)
        print(f"   结果: 缓存匹配token数 = {cached_tokens_1}")

        # 发送后续请求（应该复用缓存）
        cache_results = []
        for i in range(1, len(test_cases)):
            print(f"📤 发送第{i+1}个请求...")
            text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, test_cases[i]['input_ids'])
            cached_tokens = output_extra_info.get('decode_prefix_len', 0)
            cache_results.append(cached_tokens)
            print(f"   结果: 缓存匹配token数 = {cached_tokens}")
        
        # 验证缓存复用效果
        expected_cached = prefix_length  # 应该匹配完整前缀
        successful_reuse = sum(1 for cached in cache_results if cached >= expected_cached * 0.9)  # 允许10%误差
        
        print(f"📈 缓存复用分析:")
        print(f"   - 预期缓存匹配: {expected_cached}")
        print(f"   - 实际平均缓存: {np.mean(cache_results):.1f}")
        print(f"   - 成功复用率: {successful_reuse}/{len(cache_results)} = {successful_reuse/len(cache_results)*100:.1f}%")
        
        # 生成测试报告
        test_report = {
            'test_name': 'full_prefix_reuse',
            'prefix_length': prefix_length,
            'suffix_length': suffix_length,
            'num_requests': len(test_cases),
            'expected_cached': expected_cached,
            'avg_cached': float(np.mean(cache_results)),
            'min_cached': int(min(cache_results)),
            'max_cached': int(max(cache_results)),
            'successful_reuse_rate': successful_reuse/len(cache_results),
            'cache_efficiency': float(np.mean(cache_results) / expected_cached)
        }
        self.test_results.append(test_report)
        
        # 断言验证
        self.assertGreater(np.mean(cache_results), expected_cached * 0.8, 
                        "完整前缀复用效果不佳")
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 完整前缀复用测试通过")
    
    @seperator
    def test_no_prefix_reuse(self):
        """
        测试无前缀复用场景
        完全不同的请求应该无缓存命中
        """
        print("🧪 测试场景：无前缀复用")
        
        # 构造完全不同的测试数据
        test_cases = []
        for i in range(5):
            input_length = 320
            input_ids = self.data_generator.generate_prefix_ids(input_length)
            test_cases.append({
                'name': f'no_reuse_case_{i}',
                'input_ids': input_ids,
                'length': input_length
            })
        
        print(f"📊 测试用例：{len(test_cases)}个完全不同的请求，长度: 320")
        
        cache_results = []
        for i, case in enumerate(test_cases):
            print(f"📤 发送第{i+1}个请求...")
            text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, case['input_ids'])
            cached_tokens = output_extra_info.get('decode_prefix_len', 0)
            cache_results.append(cached_tokens)
            print(f"   结果: 缓存匹配token数 = {cached_tokens}")
        
        # 验证无缓存命中
        avg_cached = np.mean(cache_results)
        max_cached = max(cache_results)
        
        print(f"📈 无缓存复用分析:")
        print(f"   - 平均缓存匹配: {avg_cached:.1f}")
        print(f"   - 最大缓存匹配: {max_cached}")
        
        # 断言验证：应该很少有缓存命中
        self.assertLess(avg_cached, 50, "无前缀复用场景下缓存命中过多")
        
        # 生成测试报告
        test_report = {
            'test_name': 'no_prefix_reuse',
            'num_requests': len(test_cases),
            'avg_cached': float(avg_cached),
            'max_cached': int(max_cached),
            'cache_efficiency': avg_cached / 320.0  # 假设请求长度为320
        }
        self.test_results.append(test_report)
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 无前缀复用测试通过")
    
    @seperator
    def test_partial_prefix_reuse(self):
        """
        测试部分前缀复用场景
        不同长度的共享前缀应该能够部分复用
        """
        print("🧪 测试场景：部分前缀复用")
        
        # 构造分层测试数据
        base_prefix_length = 256
        base_prefix = self.data_generator.generate_prefix_ids(base_prefix_length)
        
        test_cases = [
            {
                'name': 'partial_50',
                'input_ids': base_prefix[:base_prefix_length//2] + self.data_generator.generate_prefix_ids(128),
                'expected_prefix': base_prefix_length//2,
            },
            {
                'name': 'partial_75', 
                'input_ids': base_prefix[:base_prefix_length*3//4] + self.data_generator.generate_prefix_ids(96),
                'expected_prefix': base_prefix_length*3//4,
            },
            {
                'name': 'partial_90',
                'input_ids': base_prefix[:base_prefix_length*9//10] + self.data_generator.generate_prefix_ids(64),
                'expected_prefix': base_prefix_length*9//10,
            },
            {
                'name': 'full_100',
                'input_ids': base_prefix + self.data_generator.generate_prefix_ids(64),
                'expected_prefix': base_prefix_length,
            },
        ]
        
        print(f"📊 测试用例：{len(test_cases)}个部分前缀复用请求")
        print(f"   - 基础前缀长度: {base_prefix_length}")
        
        # 首先发送完整前缀请求建立缓存
        print("📤 建立基础前缀缓存...")
        base_case = test_cases[-1]  # 完整前缀的用例
        _, _, _ = self.send_request_and_get_resp(self.lb_url, base_case['input_ids'])
        
        # 发送部分前缀复用请求
        cache_results = []
        for case in test_cases[:-1]:  # 排除完整前缀用例
            print(f"📤 发送{case['name']}请求...")
            text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, case['input_ids'])
            cached_tokens = output_extra_info.get('decode_prefix_len', 0)
            expected = case['expected_prefix']
            reuse_ratio = cached_tokens / expected if expected > 0 else 0
            cache_results.append({
                'name': case['name'],
                'cached': cached_tokens,
                'expected': expected,
                'reuse_ratio': reuse_ratio
            })
            print(f"   结果: 缓存={cached_tokens}, 期望={expected}, 复用率={reuse_ratio:.2%}")
        
        # 验证部分前缀复用效果
        avg_reuse_ratio = np.mean([r['reuse_ratio'] for r in cache_results])
        
        print(f"📈 部分前缀复用分析:")
        for result in cache_results:
            print(f"   - {result['name']}: {result['cached']}/{result['expected']} = {result['reuse_ratio']:.2%}")
        print(f"   - 平均复用率: {avg_reuse_ratio:.2%}")
        
        # 断言验证
        self.assertGreater(avg_reuse_ratio, 0.7, "部分前缀复用效果不佳")
        
        # 生成测试报告
        test_report = {
            'test_name': 'partial_prefix_reuse',
            'base_prefix_length': base_prefix_length,
            'num_requests': len(test_cases),
            'avg_reuse_ratio': float(avg_reuse_ratio),
            'cache_efficiency': float(avg_reuse_ratio),  # 复用率就是缓存效率
            'detailed_results': cache_results
        }
        self.test_results.append(test_report)
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 部分前缀复用测试通过")
    
    @seperator
    def test_mixed_scenario(self):
        """
        测试混合场景
        同时处理不同类型的请求，验证系统稳定性
        """
        print("🧪 测试场景：混合请求场景")
        
        # 构造混合测试数据
        base_prefix_length = 256
        base_prefix = self.data_generator.generate_prefix_ids(base_prefix_length)
        self.send_request_and_get_resp(self.lb_url, base_prefix)
        test_cases = []
        
        # 完全复用请求组
        for i in range(3):
            input_ids = base_prefix + self.data_generator.generate_prefix_ids(50)
            test_cases.append({
                'type': 'full_reuse',
                'name': f'mixed_full_{i}',
                'input_ids': input_ids,
                'expected_cache': base_prefix_length,
            })
        
        # 部分复用请求组
        for i in range(3):
            partial_len = base_prefix_length // 2 + i * 20
            input_ids = base_prefix[:partial_len] + self.data_generator.generate_prefix_ids(100)
            test_cases.append({
                'type': 'partial_reuse',
                'name': f'mixed_partial_{i}',
                'input_ids': input_ids,
                'expected_cache': partial_len,
            })
        
        # 无复用请求组
        for i in range(2):
            input_ids = self.data_generator.generate_prefix_ids(250)
            test_cases.append({
                'type': 'no_reuse',
                'name': f'mixed_none_{i}',
                'input_ids': input_ids,
                'expected_cache': 0,
            })
        
        # 随机打乱测试顺序
        random.shuffle(test_cases)
        
        print(f"📊 测试用例：{len(test_cases)}个混合请求")
        print(f"   - 完全复用: 3个")
        print(f"   - 部分复用: 3个") 
        print(f"   - 无复用: 2个")
        
        # 发送请求
        results_by_type = {'full_reuse': [], 'partial_reuse': [], 'no_reuse': []}
        
        for i, case in enumerate(test_cases):
            print(f"📤 发送{case['name']} ({case['type']})...")
            text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, case['input_ids'])
            cached_tokens = output_extra_info.get('decode_prefix_len', 0)
            
            results_by_type[case['type']].append({
                'cached': cached_tokens,
                'expected': case['expected_cache'],
                'name': case['name']
            })
            
            cache_efficiency = cached_tokens / case['expected_cache'] if case['expected_cache'] > 0 else 0
            print(f"   结果: 缓存={cached_tokens}, 期望={case['expected_cache']}, 效率={cache_efficiency:.2%}")
        
        # 分析结果
        print(f"📈 混合场景分析:")
        for req_type, results in results_by_type.items():
            if results:
                avg_efficiency = np.mean([r['cached'] / r['expected'] if r['expected'] > 0 else 0 for r in results])
                avg_cached = np.mean([r['cached'] for r in results])
                print(f"   - {req_type}: 平均缓存={avg_cached:.1f}, 平均效率={avg_efficiency:.2%}")
        
        # 验证不同类型请求的预期行为
        full_reuse_results = results_by_type['full_reuse']
        if full_reuse_results:
            avg_full_efficiency = np.mean([r['cached'] / r['expected'] for r in full_reuse_results])
            self.assertGreater(avg_full_efficiency, 0.8, "混合场景下完全复用效果不佳")
        
        no_reuse_results = results_by_type['no_reuse']
        if no_reuse_results:
            avg_no_cache = np.mean([r['cached'] for r in no_reuse_results])
            self.assertLess(avg_no_cache, 50, "混合场景下无复用请求缓存命中过多")
        
        # 生成测试报告
        test_report = {
            'test_name': 'mixed_scenario',
            'num_requests': len(test_cases),
            'results_by_type': results_by_type,
            'cache_efficiency': 0.0  # 需要计算整体效率
        }
        
        # 计算整体缓存效率
        all_cached = []
        all_expected = []
        for case in test_cases:
            cached = sum([r['cached'] for r in results_by_type[case['type']] if r['name'] == case['name']])
            expected = case['expected_cache']
            all_cached.append(cached)
            all_expected.append(expected)
        
        if all_expected:
            test_report['cache_efficiency'] = sum(all_cached) / sum(all_expected)
        
        self.test_results.append(test_report)
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 混合场景测试通过")
    
    @seperator
    def test_stress_concurrent_requests(self):
        """
        测试并发请求场景
        验证高并发下缓存机制的稳定性
        """
        print("🧪 测试场景：并发请求压力测试")
        
        import threading
        import queue
        
        # 构造大量相似请求
        base_prefix_length = 128
        base_prefix = self.data_generator.generate_prefix_ids(base_prefix_length)
        
        num_requests = 60
        test_cases = []
        for i in range(num_requests):
            if i % 4 == 0:  # 每4个请求中有1个使用完整前缀
                input_ids = base_prefix + self.data_generator.generate_prefix_ids(60)
                expected_type = 'full_reuse'
            elif i % 4 == 1:  # 部分前缀
                partial_len = base_prefix_length // 2
                input_ids = base_prefix[:partial_len] + self.data_generator.generate_prefix_ids(80)
                expected_type = 'partial_reuse'
            else:  # 无复用
                input_ids = self.data_generator.generate_prefix_ids(192)
                expected_type = 'no_reuse'
            
            test_cases.append({
                'index': i,
                'input_ids': input_ids,
                'expected_type': expected_type,
                'expected_cache': base_prefix_length if 'full' in expected_type else (base_prefix_length // 2 if 'partial' in expected_type else 0)
            })
        
        print(f"📊 并发测试：{num_requests}个请求")
        print(f"   - 完全复用: ~{num_requests//4}个")
        print(f"   - 部分复用: ~{num_requests//4}个")
        print(f"   - 无复用: ~{num_requests//2}个")
        
        # 并发执行
        results = queue.Queue()
        errors = queue.Queue()
        
        def send_request(case):
            try:
                start_time = time.time()
                text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, case['input_ids'])
                end_time = time.time()
                
                cached_tokens = output_extra_info.get('decode_prefix_len', 0)
                results.put({
                    'case': case,
                    'cached': cached_tokens,
                    'latency': end_time - start_time,
                    'success': True
                })
            except Exception as e:
                errors.put({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 启动所有请求线程
        threads = []
        start_time = time.time()
        
        for case in test_cases:
            thread = threading.Thread(target=send_request, args=(case,))
            thread.start()
            threads.append(thread)
            #time.sleep(0.1)  # 错开发送时间避免雪崩
        
        # 等待所有请求完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 收集结果
        successful_results = []
        failed_results = []
        
        while not results.empty():
            successful_results.append(results.get())
        
        while not errors.empty():
            failed_results.append(errors.get())
        
        # 分析结果
        print(f"📈 并发测试结果:")
        print(f"   - 总请求数: {num_requests}")
        print(f"   - 成功请求: {len(successful_results)}")
        print(f"   - 失败请求: {len(failed_results)}")
        print(f"   - 总耗时: {total_time:.2f}秒")
        print(f"   - QPS: {num_requests/total_time:.2f}")
        
        if successful_results:
            latencies = [r['latency'] for r in successful_results]
            cached_tokens = [r['cached'] for r in successful_results]
            
            print(f"   - 平均延迟: {np.mean(latencies):.2f}秒")
            print(f"   - P95延迟: {np.percentile(latencies, 95):.2f}秒")
            print(f"   - 平均缓存: {np.mean(cached_tokens):.1f}")
            
            # 按类型分析
            by_type = {}
            for result in successful_results:
                t = result['case']['expected_type']
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(result['cached'])
            
            for t, caches in by_type.items():
                print(f"   - {t}: 平均缓存={np.mean(caches):.1f}, {caches=}")
        
        # 断言验证
        success_rate = len(successful_results) / num_requests
        self.assertGreater(success_rate, 0.9, "并发请求成功率过低")
        
        if successful_results:
            avg_latency = np.mean([r['latency'] for r in successful_results])
            self.assertLess(avg_latency, 60, "并发请求延迟过高")
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 并发压力测试通过")
    
    @seperator
    def test_long_prefix_cache(self):
        """
        测试长前缀缓存场景
        验证大长度前缀的缓存效果
        """
        print("🧪 测试场景：长前缀缓存")
        
        # 测试不同长度的长前缀
        prefix_lengths = [512, 1024, 2048, 4096]
        suffix_length = 128
        
        for prefix_len in prefix_lengths:
            print(f"📤 测试前缀长度: {prefix_len}")
            
            # 生成测试数据
            base_prefix = self.data_generator.generate_prefix_ids(prefix_len)
            
            # 第一个请求建立缓存
            case1_input = base_prefix + self.data_generator.generate_prefix_ids(suffix_length)
            text1, meta1,output_extra_info = self.send_request_and_get_resp(self.lb_url, case1_input)
            cached1 = output_extra_info.get('decode_prefix_len', 0)
            print(f"   建立缓存: 匹配token数 = {cached1}")
            
            # 第二个请求复用缓存
            case2_input = base_prefix + self.data_generator.generate_prefix_ids(suffix_length + 32)
            text2, meta2, output_extra_info = self.send_request_and_get_resp(self.lb_url, case2_input)
            cached2 = output_extra_info.get('decode_prefix_len', 0)
            reuse_ratio = cached2 / prefix_len if prefix_len > 0 else 0
            print(f"   复用缓存: 匹配token数 = {cached2}, 复用率 = {reuse_ratio:.2%}")
            
            # 验证长前缀复用效果
            self.assertGreater(reuse_ratio, 0.8, f"长度{prefix_len}的前缀复用率过低")
        
# 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)

        print("✅ 长前缀缓存测试通过")
    
    @seperator
    def test_retract_with_env_variable(self):
        """
        测试通过SGLANG_TEST_RETRACT环境变量触发retract机制
        参照bench_pd_prefix_cache.py的TestDisaggregationSimulatedRetract实现
        """
        print("🧪 测试场景：通过环境变量触发Retract机制")
        
        # 设置环境变量触发retract
        old_retract_env = os.environ.get("SGLANG_TEST_RETRACT")
        os.environ["SGLANG_TEST_RETRACT"] = "true"
        
        try:
            # 简单的测试：发送少量请求
            input_length = 64
            reqs = [
                self.data_generator.generate_prefix_ids(input_length) for _ in range(4)
            ]
            
            print(f"📤 发送{len(reqs)}个请求触发retract...")
            self.send_request_and_get_resp(self.lb_url, reqs, max_new_tokens=128)
            # 发送请求
            outputs, match_lengths = [], []
            for i, req in enumerate(reqs):
                print(f"   发送第{i+1}个请求...")
                text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, req, max_new_tokens=128)
                cached_tokens = output_extra_info.get('decode_prefix_len', 0)
                outputs.append(text)
                match_lengths.append(cached_tokens)
                print(f"      缓存匹配: {cached_tokens}")
            
            # 检查decode日志中是否有retract标记
            retract_detected = self._check_retract_in_logs()
            
            # 分析结果
            avg_match = np.mean(match_lengths) if match_lengths else 0
            print(f"📈 Retract测试分析:")
            print(f"   - 平均缓存匹配: {avg_match:.1f}")
            print(f"   - Retract检测: {'✅ 检测到' if retract_detected else '❌ 未检测到'}")
            
            # 记录测试结果
            test_report = {
                'test_name': 'retract_with_env_variable',
                'num_requests': len(reqs),
                'avg_match_length': float(avg_match),
                'retract_detected': retract_detected,
                'cache_efficiency': float(avg_match / input_length) if input_length > 0 else 0
            }
            self.test_results.append(test_report)
            
            if retract_detected:
                print("✅ SGLANG_TEST_RETRACT触发的retract机制工作正常")
            else:
                print("⚠️ 未检测到retract，可能需要更多请求或不同的配置")
        
        finally:
            # 恢复原始环境变量
            if old_retract_env is not None:
                os.environ["SGLANG_TEST_RETRACT"] = old_retract_env
            else:
                os.environ.pop("SGLANG_TEST_RETRACT", None)
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 环境变量触发retract测试完成")
    
    @seperator
    def test_duplicate_requests_reuse(self):
        """
        测试多次发送完全相同请求的缓存复用
        验证相同请求的缓存命中率和性能稳定性
        """
        print("🧪 测试场景：多次重复请求缓存复用")
        
        # 构造测试数据：固定的input_ids
        input_length = 256
        fixed_input_ids = self.data_generator.generate_prefix_ids(input_length)
        
        # 测试不同轮次的重复请求
        test_rounds = 2
        requests_per_round = 5
        
        print(f"📊 重复请求测试配置:")
        print(f"   - 输入长度: {input_length}")
        print(f"   - 测试轮次: {test_rounds}")
        print(f"   - 每轮请求数: {requests_per_round}")
        print(f"   - 总请求数: {test_rounds * requests_per_round}")
        
        all_results = []
        round_summaries = []
        self.send_request_and_get_resp(self.lb_url, fixed_input_ids)
        for round_num in range(test_rounds):
            print(f"\n🔄 第{round_num + 1}轮测试...")
            
            round_results = []
            round_start_time = time.time()
            
            # 发送多轮相同请求
            for req_num in range(requests_per_round):
                print(f"   📤 发送第{req_num + 1}个重复请求...{self.lb_url}")
                
                start_time = time.time()
                text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, fixed_input_ids)
                end_time = time.time()
                print(f"   📈 请求结束，耗时: {end_time - start_time:.3f}秒")
                cached_tokens = output_extra_info.get('decode_prefix_len', 0)
                latency = end_time - start_time
                
                result = {
                    'round': round_num + 1,
                    'request': req_num + 1,
                    'cached_tokens': cached_tokens,
                    'latency': latency,
                    'cache_efficiency': cached_tokens / input_length if input_length > 0 else 0
                }
                
                round_results.append(result)
                all_results.append(result)
                
                print(f"      缓存匹配: {cached_tokens}/{input_length} ({result['cache_efficiency']:.2%})")
                print(f"      响应延迟: {latency:.3f}秒")
            
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            
            # 计算本轮统计
            round_cached = [r['cached_tokens'] for r in round_results]
            round_latencies = [r['latency'] for r in round_results]
            round_efficiencies = [r['cache_efficiency'] for r in round_results]
            
            round_summary = {
                'round': round_num + 1,
                'avg_cached': float(np.mean(round_cached)),
                'min_cached': int(min(round_cached)),
                'max_cached': int(max(round_cached)),
                'avg_latency': float(np.mean(round_latencies)),
                'p95_latency': float(np.percentile(round_latencies, 95)),
                'avg_efficiency': float(np.mean(round_efficiencies)),
                'duration': round_duration,
                'qps': requests_per_round / round_duration
            }
            
            round_summaries.append(round_summary)
            
            print(f"   📈 第{round_num + 1}轮统计:")
            print(f"      平均缓存: {round_summary['avg_cached']:.1f} ({round_summary['avg_efficiency']:.2%})")
            print(f"      平均延迟: {round_summary['avg_latency']:.3f}秒")
            print(f"      P95延迟: {round_summary['p95_latency']:.3f}秒")
            print(f"      QPS: {round_summary['qps']:.2f}")
        
        # 整体分析
        print(f"\n📊 重复请求测试整体分析:")
        
        # 缓存效率分析
        all_cached = [r['cached_tokens'] for r in all_results]
        all_efficiencies = [r['cache_efficiency'] for r in all_results]
        all_latencies = [r['latency'] for r in all_results]
        
        overall_avg_cached = float(np.mean(all_cached))
        overall_min_cached = int(min(all_cached))
        overall_max_cached = int(max(all_cached))
        overall_avg_efficiency = float(np.mean(all_efficiencies))
        overall_avg_latency = float(np.mean(all_latencies))
        overall_p95_latency = float(np.percentile(all_latencies, 95))
        
        print(f"   - 总体缓存效率:")
        print(f"     * 平均缓存匹配: {overall_avg_cached:.1f}/{input_length} ({overall_avg_efficiency:.2%})")
        print(f"     * 最低缓存匹配: {overall_min_cached}/{input_length}")
        print(f"     * 最高缓存匹配: {overall_max_cached}/{input_length}")
        
        print(f"   - 总体性能:")
        print(f"     * 平均延迟: {overall_avg_latency:.3f}秒")
        print(f"     * P95延迟: {overall_p95_latency:.3f}秒")
        print(f"     * 总QPS: {len(all_results) / sum(r['duration'] for r in round_summaries):.2f}")
        
        # 稳定性分析
        cache_stability = np.std(all_cached) / np.mean(all_cached) if np.mean(all_cached) > 0 else 0
        latency_stability = np.std(all_latencies) / np.mean(all_latencies) if np.mean(all_latencies) > 0 else 0
        
        print(f"   - 稳定性分析:")
        print(f"     * 缓存匹配变异系数: {cache_stability:.3f} (越小越稳定)")
        print(f"     * 延迟变异系数: {latency_stability:.3f} (越小越稳定)")
        
        # 轮次间比较
        print(f"   - 轮次间比较:")
        for summary in round_summaries:
            print(f"     * 第{summary['round']}轮: {summary['avg_efficiency']:.2%}效率, {summary['qps']:.2f} QPS")
        
        # 记录测试结果
        test_report = {
            'test_name': 'duplicate_requests_reuse',
            'input_length': input_length,
            'test_rounds': test_rounds,
            'requests_per_round': requests_per_round,
            'total_requests': test_rounds * requests_per_round,
            'overall_avg_cached': overall_avg_cached,
            'overall_min_cached': overall_min_cached,
            'overall_max_cached': overall_max_cached,
            'overall_avg_efficiency': overall_avg_efficiency,
            'overall_avg_latency': overall_avg_latency,
            'overall_p95_latency': overall_p95_latency,
            'cache_stability': cache_stability,
            'latency_stability': latency_stability,
            'round_summaries': round_summaries
        }
        self.test_results.append(test_report)
        
        # 断言验证
        # 1. 缓存效率应该很高（重复请求应该命中缓存）
        self.assertGreater(overall_avg_efficiency, 0.8, "重复请求的缓存命中率过低")
        
        # 2. 缓存匹配应该稳定（变异系数应该较小）
        self.assertLess(cache_stability, 0.2, "缓存匹配稳定性不佳")
        
        # 3. 延迟应该稳定
        self.assertLess(latency_stability, 0.3, "响应延迟稳定性不佳")
        
        # 4. 延迟应该在合理范围内
        self.assertLess(overall_avg_latency, 5.0, "重复请求响应延迟过高")
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 多次重复请求缓存复用测试通过")
    
    @seperator
    def test_evict_with_small_kv(self):
        """
        测试通过SGLANG_CI_SMALL_KV_SIZE限制KV大小触发evict机制
        参照bench_pd_prefix_cache.py的TestDisaggregationPrefixRetraction实现
        """
        print("🧪 测试场景：小KV缓存空间触发Evict机制")
        
        # 设置小KV缓存环境变量
        old_kv_size_env = os.environ.get("SGLANG_CI_SMALL_KV_SIZE")
        old_reserved_env = os.environ.get("SGLANG_NUM_RESERVED_DECODE_TOKENS")
        
        os.environ["SGLANG_CI_SMALL_KV_SIZE"] = "512"
        os.environ["SGLANG_NUM_RESERVED_DECODE_TOKENS"] = "64"
        
        try:
            # 使用与参考文件相同的测试参数
            input_length = 128
            reqs = [
                self.data_generator.generate_prefix_ids(input_length) for _ in range(4)
            ]
            
            print(f"📤 发送{len(reqs)}个请求填满小缓存...")
            for i, req in enumerate(reqs):
                print(f"   发送第{i+1}个请求...")
                text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, req, max_new_tokens=128)
            # 逐个发送请求
            print(f"📤 发送{len(reqs)}个请求出发evict...")
            outputs, match_lengths = [], []
            for i, req in reversed(list(enumerate(reqs))):
                print(f"   发送第{i+1}个请求...")
                text, meta, output_extra_info = self.send_request_and_get_resp(self.lb_url, req, max_new_tokens=128)
                cached_tokens = output_extra_info.get('decode_prefix_len', 0)
                outputs.append(text)
                match_lengths.append(cached_tokens)
                print(f"      缓存匹配: {cached_tokens}/{input_length}")
            
            # 检查decode日志中是否有evict标记
            evict_detected = self._check_evict_in_logs()
            
            # 分析缓存匹配情况
            avg_match = np.mean(match_lengths) if match_lengths else 0
            min_match = min(match_lengths) if match_lengths else 0
            max_match = max(match_lengths) if match_lengths else 0
            
            print(f"📈 Evict测试分析:")
            print(f"   - 平均缓存匹配: {avg_match:.1f}")
            print(f"   - 最小缓存匹配: {min_match}")
            print(f"   - 最大缓存匹配: {max_match}")
            print(f"   - 缓存效率: {avg_match/input_length:.2%}")
            print(f"   - Evict检测: {'✅ 检测到' if evict_detected else '❌ 未检测到'}")
            
            # 记录测试结果
            test_report = {
                'test_name': 'evict_with_small_kv',
                'kv_size': '512',
                'reserved_tokens': '64',
                'num_requests': len(reqs),
                'avg_match_length': float(avg_match),
                'min_match_length': int(min_match),
                'max_match_length': int(max_match),
                'evict_detected': evict_detected,
                'cache_efficiency': float(avg_match / input_length)
            }
            self.test_results.append(test_report)
            
            if evict_detected:
                print("✅ SGLANG_CI_SMALL_KV_SIZE触发的evict机制工作正常")
            else:
                print("⚠️ 未检测到evict，可能需要更多请求或不同的配置")
        
        finally:
            # 恢复原始环境变量
            if old_kv_size_env is not None:
                os.environ["SGLANG_CI_SMALL_KV_SIZE"] = old_kv_size_env
            else:
                os.environ.pop("SGLANG_CI_SMALL_KV_SIZE", None)
            
            if old_reserved_env is not None:
                os.environ["SGLANG_NUM_RESERVED_DECODE_TOKENS"] = old_reserved_env
            else:
                os.environ.pop("SGLANG_NUM_RESERVED_DECODE_TOKENS", None)
        
        # 将测试结果添加到全局容器
        DetailedPrefixCacheTest.all_test_results.extend(self.test_results)
        
        print("✅ 小KV缓存evict测试完成")
    
    def _check_retract_in_logs(self):
        """检查decode日志中是否有retract标记"""
        try:
            # 尝试检查默认的decode日志位置
            decode_log_paths = [
                './de.log',
                getattr(self, 'decode_log_file', None)
            ]
            
            for log_path in decode_log_paths:
                if log_path and os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        content = f.read()
                        if "[retract_decode] retracting" in content:
                            return True
        except Exception as e:
            print(f"   ⚠️ 检查retract日志失败: {e}")
        return False
    
    def _check_evict_in_logs(self):
        """检查decode日志中是否有evict标记"""
        try:
            # 尝试检查默认的decode日志位置
            decode_log_paths = [
                './de.log',
                getattr(self, 'decode_log_file', None)
            ]
            
            for log_path in decode_log_paths:
                if log_path and os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        content = f.read()
                        if "[evict]" in content:
                            return True
        except Exception as e:
            print(f"   ⚠️ 检查evict日志失败: {e}")
        return False

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        print("🧹 清理测试环境...")
        
        if cls.error:
            print(f"❌ 测试过程中发现错误: {cls.error}")
        
        # 清理进程
        processes = [getattr(cls, 'process_lb', None), 
                    getattr(cls, 'process_prefill', None), 
                    getattr(cls, 'process_decode', None)]
        
        for process in processes:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"清理进程失败: {e}")
        
        # 清理日志文件
        log_files = [
            (getattr(cls, 'lb_log_file', None), 'Load Balancer'),
            (getattr(cls, 'prefill_log_file', None), 'Prefill Server'),
            (getattr(cls, 'decode_log_file', None), 'Decode Server')
        ]
        
        for log_file, service_name in log_files:
            if log_file:
                try:
                    print(f"   清理{service_name}日志: {log_file}")
                    os.unlink(log_file)
                except Exception as e:
                    print(f"   清理{service_name}日志文件失败: {e}")
        
        # 关闭文件句柄
        for handle_name in ['lb_log_handle', 'prefill_log_handle', 'decode_log_handle']:
            handle = getattr(cls, handle_name, None)
            if handle and hasattr(handle, 'close'):
                try:
                    handle.close()
                except Exception as e:
                    print(f"   关闭{handle_name}失败: {e}")
        
        print("✅ 测试环境清理完成")
        
        if cls.error:
            raise Exception(cls.error)
    
    @classmethod
    def generate_test_summary(cls):
        """生成测试总结报告"""
        if not hasattr(cls, 'all_test_results'):
            print("📊 无测试结果可供分析")
            return
            
        print("\n" + "="*50)
        print("🎯 PD分离前缀缓存测试结果")
        print("="*50)
        
        all_results = cls.all_test_results
        passed_tests = 0
        total_tests = 0
        
        # 按测试名称分组，统计每个测试用例的通过情况
        test_cases = {}
        for result in all_results:
            test_name = result['test_name']
            if test_name not in test_cases:
                test_cases[test_name] = []
            test_cases[test_name].append(result)
        
        for test_name, results in test_cases.items():
            total_tests += 1
            # 取该测试用例的平均结果进行判断
            avg_result = {}
            for key in ['cache_efficiency', 'successful_reuse_rate', 'avg_cached', 'expected_cached', 'overall_avg_efficiency']:
                if key in results[0]:
                    avg_result[key] = np.mean([r[key] for r in results])
            
            # 根据不同测试用例设置不同的通过条件
            passed = False
            reason = ""
            
            if test_name == 'full_prefix_reuse':
                # 完整前缀复用：缓存效率应该>90%
                efficiency = avg_result.get('cache_efficiency', 0)
                passed = efficiency > 0.9
                reason = f"缓存效率: {efficiency:.1%}"
            
            elif test_name == 'no_prefix_reuse':
                # 无前缀复用：缓存效率应该<30%（表示没有意外复用）
                efficiency = avg_result.get('cache_efficiency', 0)
                passed = efficiency < 0.3
                reason = f"缓存效率: {efficiency:.1%}"
            
            elif test_name == 'partial_prefix_reuse':
                # 部分前缀复用：应该有50%-90%的复用率
                efficiency = avg_result.get('cache_efficiency', 0)
                passed = 0.5 <= efficiency <= 0.95
                reason = f"缓存效率: {efficiency:.1%}"
            
            elif test_name == 'mixed_scenario':
                # 混合场景：整体复用率应该>60%
                efficiency = avg_result.get('cache_efficiency', 0)
                passed = efficiency > 0.6
                reason = f"缓存效率: {efficiency:.1%}"
            
            elif test_name == 'duplicate_requests_reuse':
                # 重复请求复用：复用率应该>95%
                # duplicate_requests_reuse测试使用overall_avg_efficiency字段
                reuse_rate = avg_result.get('successful_reuse_rate', avg_result.get('overall_avg_efficiency', 0))
                passed = reuse_rate > 0.95
                reason = f"复用率: {reuse_rate:.1%}"
            
            elif test_name == 'retract_with_env_variable':
                # 回缩测试：检查日志中是否有retract标记
                retract_detected = any(r.get('retract_detected', False) for r in results)
                passed = retract_detected
                reason = f"Retract检测: {'✅' if retract_detected else '❌'}"
            
            elif test_name == 'evict_with_small_kv':
                # 驱逐测试：检查日志中是否有evict标记
                evict_detected = any(r.get('evict_detected', False) for r in results)
                passed = evict_detected
                reason = f"Evict检测: {'✅' if evict_detected else '❌'}"
            
            else:
                # 其他测试：默认缓存效率>70%
                efficiency = avg_result.get('cache_efficiency', 0)
                passed = efficiency > 0.7
                reason = f"缓存效率: {efficiency:.1%}"
            
            if passed:
                print(f"✅ {test_name} - 通过 ({reason})")
                passed_tests += 1
            else:
                print(f"❌ {test_name} - 失败 ({reason})")
        
        print("="*50)
        print(f"📊 总计: {passed_tests}/{total_tests} 个测试用例通过")
        
        if passed_tests == total_tests:
            print("🎉 所有测试用例通过！")
        else:
            print(f"⚠️ {total_tests - passed_tests} 个测试用例未通过")
        print("="*50)

def tail_logs():
    """实时查看日志文件"""
    import glob
    
    # 查找最新的日志文件
    log_files = {}
    
    # 查找LB日志
    lb_logs = sorted(glob.glob('/tmp/pd_lb_log_*.log'))
    if lb_logs:
        log_files['Load Balancer'] = lb_logs[-1]
    
    # 查找Prefill日志
    prefill_logs = sorted(glob.glob('/tmp/pd_prefill_log_*.log'))
    if prefill_logs:
        log_files['Prefill Server'] = prefill_logs[-1]
    
    # 查找Decode日志
    decode_logs = sorted(glob.glob('/tmp/pd_decode_log_*.log'))
    if decode_logs:
        log_files['Decode Server'] = decode_logs[-1]
    
    if not log_files:
        print("❌ 未找到日志文件")
        return
    
    print("📝 实时日志文件位置:")
    for service, log_file in log_files.items():
        print(f"   - {service}: {log_file}")
    
    print("\n💡 使用以下命令实时查看日志:")
    for service, log_file in log_files.items():
        print(f"   # {service}:")
        print(f"   tail -f {log_file}")

def run_specific_test(test_name: str):
    """运行指定的测试用例"""
    suite = unittest.TestSuite()
    
    test_map = {
        'full_reuse': DetailedPrefixCacheTest('test_full_prefix_reuse'),
        'no_reuse': DetailedPrefixCacheTest('test_no_prefix_reuse'),
        'partial_reuse': DetailedPrefixCacheTest('test_partial_prefix_reuse'),
        'mixed': DetailedPrefixCacheTest('test_mixed_scenario'),
        'duplicate': DetailedPrefixCacheTest('test_duplicate_requests_reuse'),
        'stress': DetailedPrefixCacheTest('test_stress_concurrent_requests'),
        'long_prefix': DetailedPrefixCacheTest('test_long_prefix_cache'),
        'retract': DetailedPrefixCacheTest('test_retract_with_env_variable'),
        'evict': DetailedPrefixCacheTest('test_evict_with_small_kv'),
    }
    
    if test_name == 'all':
        for name, test in test_map.items():
            suite.addTest(test)
    elif test_name in test_map:
        suite.addTest(test_map[test_name])
    else:
        print(f"❌ 未知的测试用例: {test_name}")
        print(f"可用的测试用例: {list(test_map.keys())}")
        return
    
    # 添加结果收集
    class TestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
            
        def addSuccess(self, test):
            super().addSuccess(test)
            # 收集测试结果
            if hasattr(test, '_testMethodName'):
                test_instance = test._testMethodName
                print(f"✅ {test_instance} 测试完成")
    
    runner = unittest.TextTestRunner(verbosity=2, resultclass=TestResult)
    result = runner.run(suite)
    
    # 生成总结报告
    if hasattr(DetailedPrefixCacheTest, 'all_test_results'):
        DetailedPrefixCacheTest.generate_test_summary()
    
    if result.wasSuccessful():
        print("🎉 所有测试通过！")
    else:
        print(f"❌ 测试失败: {len(result.failures)}个失败, {len(result.errors)}个错误")
        for test, traceback in result.failures + result.errors:
            print(f"   失败: {test}")
            print(f"   详情: {traceback}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PD分离前缀缓存详细测试")
    parser.add_argument(
        "--test_case", 
        choices=['all', 'full_reuse', 'no_reuse', 'partial_reuse', 'mixed', 'stress', 'long_prefix', 'retract', 'evict', 'duplicate'],
        default='all',
        help="指定要运行的测试用例，默认运行所有测试"
    )
    parser.add_argument(
        "--tail-logs",
        action='store_true',
        help="查看最新的日志文件位置"
    )
    
    args = parser.parse_args()
    
    # 如果只是查看日志，直接显示并退出
    if args.tail_logs:
        tail_logs()
        sys.exit(0)
    
    print("🔥 PD分离前缀缓存详细测试")
    print(f"🎯 运行测试: {args.test_case}")
    print("=" * 80)
    
    # 在开始测试前先清理所有进程
    # print("🧹 清理现有sglang进程...")
    # kill_all_sglang(kill_lb=True)
    time.sleep(2)
    
    try:
        run_specific_test(args.test_case)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        print("🧹 最终清理...")
        kill_all_sglang(kill_lb=True)