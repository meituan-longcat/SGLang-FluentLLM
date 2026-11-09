#!/usr/bin/env python3
"""
HiCache 最优参数搜索脚本 - 改进版本

改进点：
1. 更好的目录结构管理
2. 更健壮的指标收集
3. 更好的错误处理
4. 更清晰的输出

用法: python search_optimal_params_v2.py [选项]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# 颜色输出
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def log_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def log_debug(msg: str):
    print(f"{Colors.BLUE}[DEBUG]{Colors.NC} {msg}")

class ImprovedParameterSearch:
    """改进的参数搜索类"""
    
    def __init__(self, args):
        self.args = args
        
        # 脚本路径
        self.script_dir = Path(__file__).parent
        self.launch_script = self.script_dir / "launch_hicache_pd.sh"
        self.benchmark_script = self.script_dir / "run_all_benchmarks.sh"
        
        # 数据集路径
        self.dataset_path = self.args.dataset_path
        if not Path(self.dataset_path).exists():
            self.dataset_path = self.script_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"
        
        # 输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_output_dir = Path(args.output_dir) / f"param_search_v2_{timestamp}"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存全局配置
        self.save_global_config()
        
        # 待搜索的参数定义
        self.parameters = {
            "PREFETCH_POLICY": {
                "values": ["best_effort", "timeout"],
                "description": "预取策略",
                "arg_name": "prefetch_policy"
            },
            "MEM_LAYOUT": {
                "values": ["page_first", "layer_first"],
                "description": "内存布局",
                "arg_name": "mem_layout"
            },
            "WRITE_POLICY": {
                "values": ["write_through", "write_back"],
                "description": "写策略",
                "arg_name": "write_policy"
            },
            "PREFETCH_THRESHOLD": {
                "values": ["1", "64", "128", "256"],
                "description": "预取阈值",
                "arg_name": "prefetch_threshold"
            },
            "IO_BACKEND": {
                "values": ["kernel", "direct"],
                "description": "IO后端",
                "arg_name": "io_backend"
            }
        }
        
        # 当前最优参数
        self.current_best_params = {
            "PREFETCH_POLICY": "timeout",
            "MEM_LAYOUT": "page_first",
            "WRITE_POLICY": "write_through",
            "PREFETCH_THRESHOLD": "1",
            "IO_BACKEND": "kernel"
        }
        
        # 所有测试结果
        self.all_results = []
        
        # 测试状态跟踪
        self.test_status = {}
    
    def save_global_config(self):
        """保存全局配置"""
        config_file = self.base_output_dir / "global_config.json"
        config = {
            "timestamp": datetime.now().isoformat(),
            "base_output_dir": str(self.base_output_dir),
            "model": self.args.model,
            "dataset_path": str(self.dataset_path),
            "num_requests": self.args.num_requests,
            "server_host": self.args.server_host,
            "server_port": self.args.server_port,
            "quick_test": self.args.quick_test,
            "timeout": self.args.timeout,
            "git_commit": subprocess.getoutput("git rev-parse HEAD 2>/dev/null || echo 'unknown'"),
            "git_branch": subprocess.getoutput("git branch --show-current 2>/dev/null || echo 'unknown'"),
            "hostname": subprocess.getoutput("hostname"),
            "user": subprocess.getoutput("whoami")
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        log_info(f"全局配置已保存到: {config_file}")
    
    def create_param_directory(self, params: dict[str, str]) -> Path:
        """创建参数目录"""
        # 使用更清晰的目录名
        param_parts = []
        for param_name in ["PREFETCH_POLICY", "MEM_LAYOUT", "WRITE_POLICY", "PREFETCH_THRESHOLD", "IO_BACKEND"]:
            if param_name in params:
                param_parts.append(f"{param_name[:3]}_{params[param_name]}")
        
        param_str = "_".join(param_parts)
        output_dir = self.base_output_dir / param_str
        
        # 检查目录是否已存在且有测试结果
        if output_dir.exists():
            # 检查是否有详细结果文件
            results_file = output_dir / "detailed_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        existing_results = json.load(f)
                    # 如果有有效的测试结果，返回目录但不重新创建
                    if "error" not in existing_results:
                        log_info(f"✓ 参数组合 {param_str} 已存在有效测试结果，跳过重复测试")
                        return output_dir
                except Exception as e:
                    log_warn(f"读取现有结果文件失败: {e}")
            
            # 如果目录存在但没有有效结果，可以继续使用
            log_info(f"⚠ 参数组合 {param_str} 目录已存在，但无有效结果，继续测试")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存参数配置
        config_file = output_dir / "params.json"
        with open(config_file, 'w') as f:
            json.dump({
                "params": params,
                "timestamp": datetime.now().isoformat(),
                "test_id": len(self.all_results) + 1
            }, f, indent=2)
        
        return output_dir
    
    def cleanup(self):
        """清理进程和资源"""
        log_info("清理进程和资源...")
        
        # 杀掉所有相关进程
        processes_to_kill = ["sglang", "mini_lb", "mooncake_master"]
        for proc in processes_to_kill:
            subprocess.run(["pkill", "-f", proc], stderr=subprocess.DEVNULL)
        
        # 等待进程结束
        time.sleep(5)
    
    def launch_service(self, params: dict[str, str]) -> bool:
        """启动 HiCache 服务"""
        log_info("=" * 60)
        log_info("启动 HiCache 服务...")
        log_info("=" * 60)
        
        # 构建启动命令
        cmd = [
            str(self.launch_script),
            self.args.model,
            params.get("PREFETCH_POLICY", "timeout"),
            params.get("MEM_LAYOUT", "page_first"),
            params.get("WRITE_POLICY", "write_through"),
            params.get("PREFETCH_THRESHOLD", "1"),
            params.get("IO_BACKEND", "kernel"),
        ]
        
        log_info(f"启动命令: {' '.join(cmd)}")
        log_info(f"参数配置: {json.dumps(params, indent=2)}")
        
        try:
            # 执行启动脚本
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 5分钟超时
            )
            
            if result.returncode != 0:
                log_error(f"启动服务失败: {result.stderr}")
                return False
            
            log_info(result.stdout)
            
            # 等待服务完全启动
            log_info("等待服务启动完成...")
            time.sleep(30)  # 额外等待
            
            # 健康检查
            return self.check_service_health()
            
        except subprocess.TimeoutExpired:
            log_error("启动服务超时")
            return False
        except Exception as e:
            log_error(f"启动服务异常: {e}")
            return False
    
    def check_service_health(self) -> bool:
        """检查服务健康状态"""
        urls = [
            "http://127.0.0.1:8192/health",  # Load Balancer
            "http://127.0.0.1:8392/health",  # Decode Worker
            "http://127.0.0.1:8292/health",  # Prefill Worker
        ]
        
        max_retries = 30
        retry_interval = 5
        
        healthy_count = 0
        for url in urls:
            for i in range(max_retries):
                try:
                    response = subprocess.run(
                        ["curl", "-s", "-f", url],
                        capture_output=True,
                        timeout=10
                    )
                    if response.returncode == 0:
                        log_info(f"✓ {url} 健康检查通过")
                        healthy_count += 1
                        break
                except:
                    pass
                
                if i < max_retries - 1:
                    time.sleep(retry_interval)
            else:
                log_warn(f"⚠ {url} 健康检查失败")
        
        # 至少需要两个服务健康
        return healthy_count >= 2
    
    def run_benchmarks(self, params: dict[str, str], output_dir: Path) -> dict[str, Any] | None:
        """运行基准测试"""
        log_info("=" * 60)
        log_info("运行基准测试...")
        log_info("=" * 60)
        
        # 构建基准测试命令
        cmd = [
            str(self.benchmark_script),
            "--server-host", self.args.server_host,
            "--server-port", self.args.server_port,
            "--output-dir", str(output_dir),
            "--dataset-path", str(self.dataset_path),
            "--num-requests", str(self.args.num_requests),
        ]
        
        if self.args.quick_test:
            cmd.append("--quick-test")
        
        if self.args.timeout:
            cmd.extend(["--timeout", str(self.args.timeout)])
        
        log_info(f"基准测试命令: {' '.join(cmd)}")
        
        try:
            # 执行基准测试
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.args.timeout * 2 if self.args.timeout else 3600 * 2
            )
            end_time = time.time()
            
            # 保存命令输出
            stdout_file = output_dir / "benchmark_stdout.log"
            stderr_file = output_dir / "benchmark_stderr.log"
            
            with open(stdout_file, 'w') as f:
                f.write(result.stdout)
            with open(stderr_file, 'w') as f:
                f.write(result.stderr)
            
            log_info(f"标准输出已保存到: {stdout_file}")
            if result.stderr:
                log_warn(f"标准错误: {result.stderr[:500]}...")
            
            # 收集结果
            metrics = self.collect_metrics(output_dir)
            metrics["duration"] = end_time - start_time
            metrics["exit_code"] = result.returncode
            metrics["params"] = params.copy()
            
            # 保存详细结果
            results_file = output_dir / "detailed_results.json"
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except subprocess.TimeoutExpired:
            log_error("基准测试超时")
            return {
                "error": "timeout",
                "params": params.copy(),
                "duration": self.args.timeout * 2 if self.args.timeout else "timeout"
            }
        except Exception as e:
            log_error(f"基准测试异常: {e}")
            return {
                "error": str(e),
                "params": params.copy()
            }
    
    def collect_metrics(self, output_dir: Path) -> dict[str, Any]:
        """从输出目录收集指标"""
        metrics = {
            "summary": {},
            "detailed": {},
            "files_found": []
        }
        
        # 查找所有的 metrics 文件
        metrics_files = list(output_dir.rglob("*_metrics.json"))
        metrics["files_found"] = [str(f.relative_to(output_dir)) for f in metrics_files]
        
        # 按测试类型组织指标（不包含mix场景）
        test_types = ["serving", "multiturn", "longcontext"]
        
        for test_type in test_types:
            metrics["detailed"][test_type] = {}
            
            # 查找该测试类型的所有 metrics 文件
            test_metrics_files = [f for f in metrics_files if test_type in f.name]
            
            for metrics_file in test_metrics_files:
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    
                    # 确定指标来源
                    source = "unknown"
                    if "rate_" in str(metrics_file):
                        # 从路径中提取 rate
                        parts = str(metrics_file).split("/")
                        for part in parts:
                            if part.startswith("rate_"):
                                source = part
                                break
                    else:
                        source = "direct"
                    
                    metrics["detailed"][test_type][source] = data
                    
                except Exception as e:
                    log_warn(f"读取 metrics 文件失败 {metrics_file}: {e}")
        
        # 计算汇总指标
        self.calculate_summary_metrics(metrics)
        
        return metrics
    
    def calculate_summary_metrics(self, metrics: dict[str, Any]):
        """计算汇总指标"""
        summary = {
            "total_tests": 0,
            "successful_tests": 0,
            "average_throughput": 0.0,
            "average_ttft": 0.0,
            "best_throughput": 0.0,
            "best_ttft": float('inf'),
            "test_details": {}
        }
        
        throughput_sum = 0.0
        ttft_sum = 0.0
        count = 0
        
        for test_type, test_data in metrics["detailed"].items():
            for source, data in test_data.items():
                if isinstance(data, dict):
                    summary["total_tests"] += 1
                    
                    throughput = data.get("throughput", 0)
                    ttft = data.get("ttft_p50", 0)
                    
                    if throughput > 0 or ttft > 0:
                        summary["successful_tests"] += 1
                        
                        if throughput > 0:
                            throughput_sum += throughput
                            summary["best_throughput"] = max(summary["best_throughput"], throughput)
                        
                        if ttft > 0:
                            ttft_sum += ttft
                            summary["best_ttft"] = min(summary["best_ttft"], ttft)
                        
                        count += 1
                    
                    # 记录测试详情
                    test_key = f"{test_type}_{source}"
                    summary["test_details"][test_key] = {
                        "throughput": throughput,
                        "ttft_p50": ttft,
                        "latency_p50": data.get("latency_p50", 0),
                        "cache_hit_rate": data.get("cache_hit_rate", 0)
                    }
        
        if count > 0:
            summary["average_throughput"] = throughput_sum / count
            summary["average_ttft"] = ttft_sum / count
        
        if summary["best_ttft"] == float('inf'):
            summary["best_ttft"] = 0
        
        metrics["summary"] = summary
    
    def calculate_score(self, metrics: dict[str, Any]) -> float:
        """计算性能分数"""
        if "summary" not in metrics:
            return 0.0
        
        summary = metrics["summary"]
        
        # 使用加权分数
        score = 0.0
        weights = {
            "average_throughput": 0.5,  # 吞吐量权重
            "best_throughput": 0.3,     # 最佳吞吐量权重
            "best_ttft": 0.2,           # 最佳TTFT权重（倒数）
        }
        
        # 吞吐量越高越好
        if summary["average_throughput"] > 0:
            score += weights["average_throughput"] * summary["average_throughput"]
        
        if summary["best_throughput"] > 0:
            score += weights["best_throughput"] * summary["best_throughput"]
        
        # TTFT越低越好（使用倒数）
        if summary["best_ttft"] > 0:
            score += weights["best_ttft"] * (1000 / summary["best_ttft"])  # 转换为分数
        
        # 考虑成功测试的比例
        success_ratio = summary["successful_tests"] / max(summary["total_tests"], 1)
        score *= success_ratio
        
        return score
    
    def search_parameter(self, param_name: str) -> str:
        """搜索单个参数的最优值"""
        param_config = self.parameters[param_name]
        log_info("=" * 60)
        log_info(f"搜索参数: {param_name} ({param_config['description']})")
        log_info("=" * 60)
        
        results = []
        
        for value in param_config["values"]:
            log_info(f"测试 {param_name} = {value}")
            
            # 构建测试参数
            test_params = self.current_best_params.copy()
            test_params[param_name] = value
            
            # 创建输出目录（会检查是否已有有效结果）
            output_dir = self.create_param_directory(test_params)
            
            # 检查是否已有有效结果
            results_file = output_dir / "detailed_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        existing_results = json.load(f)
                    # 如果有有效的测试结果，跳过测试
                    if "error" not in existing_results:
                        # 计算现有结果的分数
                        score = self.calculate_score(existing_results)
                        log_info(f"✓ 使用现有测试结果，分数: {score:.2f}")
                        results.append({
                            "value": value,
                            "metrics": existing_results,
                            "score": score,
                            "output_dir": str(output_dir),
                            "from_cache": True
                        })
                        # 保存到总结果中
                        self.all_results.append({
                            "param_name": param_name,
                            "value": value,
                            "params": test_params.copy(),
                            "metrics": existing_results,
                            "score": score,
                            "timestamp": datetime.now().isoformat(),
                            "output_dir": str(output_dir),
                            "from_cache": True
                        })
                        continue
                except Exception as e:
                    log_warn(f"读取现有结果文件失败: {e}")
            
            # 启动服务
            if not self.launch_service(test_params):
                log_error(f"无法启动服务，跳过 {param_name}={value}")
                results.append({
                    "value": value,
                    "error": "service_launch_failed",
                    "score": -1,
                    "output_dir": str(output_dir)
                })
                continue
            
            # 运行基准测试
            metrics = self.run_benchmarks(test_params, output_dir)
            
            # 计算分数
            if metrics and "error" not in metrics:
                score = self.calculate_score(metrics)
                log_info(f"✓ 测试完成，分数: {score:.2f}")
                results.append({
                    "value": value,
                    "metrics": metrics,
                    "score": score,
                    "output_dir": str(output_dir)
                })
            else:
                log_warn(f"⚠ 测试失败: {metrics.get('error', 'unknown')}")
                results.append({
                    "value": value,
                    "error": metrics.get("error", "unknown"),
                    "score": -1,
                    "output_dir": str(output_dir)
                })
            
            # 保存结果
            self.all_results.append({
                "param_name": param_name,
                "value": value,
                "params": test_params.copy(),
                "metrics": metrics,
                "score": results[-1]["score"],
                "timestamp": datetime.now().isoformat(),
                "output_dir": str(output_dir)
            })
            
            # 清理
            self.cleanup()
            
            # 等待系统稳定
            time.sleep(10)
        
        # 分析结果，选择最优值
        log_info("=" * 60)
        log_info(f"{param_name} 测试结果汇总:")
        log_info("=" * 60)
        
        best_value = None
        best_score = -1
        
        for result in results:
            value = result["value"]
            score = result.get("score", -1)
            status = "✓" if score > 0 else "✗"
            
            log_info(f"{status} {param_name}={value}: 分数={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_value = value
        
        if best_value:
            log_info(f"最优 {param_name} = {best_value} (分数: {best_score:.2f})")
            self.current_best_params[param_name] = best_value
        else:
            log_warn(f"无法确定最优 {param_name}，使用默认值")
            best_value = param_config["values"][0]
        
        return best_value
    
    def run(self) -> dict[str, Any]:
        """运行完整的参数搜索"""
        log_info("=" * 60)
        log_info("HiCache 最优参数搜索 (改进版本)")
        log_info("=" * 60)
        log_info(f"输出目录: {self.base_output_dir}")
        log_info(f"模型: {self.args.model}")
        log_info(f"数据集: {self.dataset_path}")
        log_info(f"请求数: {self.args.num_requests}")
        log_info(f"快速测试模式: {self.args.quick_test}")
        log_info("=" * 60)
        
        # 按顺序搜索每个参数
        search_order = ["PREFETCH_POLICY", "MEM_LAYOUT", "WRITE_POLICY", "PREFETCH_THRESHOLD", "IO_BACKEND"]
        
        for param_name in search_order:
            self.search_parameter(param_name)
        
        # 生成最终报告
        self.generate_final_report()
        
        return {
            "best_params": self.current_best_params,
            "all_results": self.all_results,
            "output_dir": str(self.base_output_dir)
        }
    
    def generate_final_report(self):
        """生成最终报告"""
        report_file = self.base_output_dir / "final_report.json"
        summary_file = self.base_output_dir / "summary.txt"
        
        # 保存完整报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.args.model,
            "dataset_path": str(self.dataset_path),
            "num_requests": self.args.num_requests,
            "best_params": self.current_best_params,
            "all_results": self.all_results,
            "search_summary": {
                "total_tests": len(self.all_results),
                "successful_tests": sum(1 for r in self.all_results if r.get("score", -1) > 0),
                "average_score": sum(r.get("score", 0) for r in self.all_results) / max(len(self.all_results), 1)
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        log_info(f"完整报告已保存到: {report_file}")
        
        # 生成文本摘要
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HiCache 最优参数搜索结果 (改进版本)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {self.args.model}\n")
            f.write(f"数据集: {self.dataset_path}\n")
            f.write(f"请求数: {self.args.num_requests}\n")
            f.write(f"总测试数: {report['search_summary']['total_tests']}\n")
            f.write(f"成功测试数: {report['search_summary']['successful_tests']}\n")
            f.write(f"平均分数: {report['search_summary']['average_score']:.2f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("最优参数组合\n")
            f.write("=" * 60 + "\n")
            f.writelines(f"  {param}: {value}\n" for param, value in self.current_best_params.items())
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("参数搜索详情\n")
            f.write("=" * 60 + "\n")
            
            # 按参数分组显示结果
            param_results = {}
            for result in self.all_results:
                param = result["param_name"]
                if param not in param_results:
                    param_results[param] = []
                param_results[param].append(result)
            
            for param, results in param_results.items():
                f.write(f"\n{param}:\n")
                for r in results:
                    value = r["value"]
                    score = r.get("score", -1)
                    f.write(f"  {value}: 分数={score:.2f}")
                    if score == max(res.get("score", -1) for res in results):
                        f.write(" [最优]")
                    f.write("\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("使用最优参数的启动命令:\n")
            f.write("=" * 60 + "\n")
            f.write(f"{self.launch_script} {self.args.model} ")
            f.write(f"{self.current_best_params['PREFETCH_POLICY']} ")
            f.write(f"{self.current_best_params['MEM_LAYOUT']} ")
            f.write(f"{self.current_best_params['WRITE_POLICY']} ")
            f.write(f"{self.current_best_params['PREFETCH_THRESHOLD']} ")
            f.write(f"{self.current_best_params['IO_BACKEND']}\n")
        
        log_info(f"摘要已保存到: {summary_file}")
        
        # 输出最优参数
        log_info("")
        log_info("=" * 60)
        log_info("最优参数组合:")
        log_info("=" * 60)
        for param, value in self.current_best_params.items():
            log_info(f"  {param}: {value}")
        log_info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="HiCache 最优参数搜索脚本 (改进版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行完整搜索
  python search_optimal_params_v2.py
  
  # 快速测试模式
  python search_optimal_params_v2.py --quick-test
  
  # 指定模型和输出目录
  python search_optimal_params_v2.py --model deepseek --output-dir ./my_results
  
  # 自定义超时时间
  python search_optimal_params_v2.py --timeout 1800
  
  # 自定义数据集和请求数
  python search_optimal_params_v2.py --dataset-path /path/to/dataset.json --num-requests 200
        """
    )
    
    parser.add_argument(
        "--model",
        default="qwen3",
        choices=["qwen3", "deepseek"],
        help="模型名称 (默认: qwen3)"
    )
    
    parser.add_argument(
        "--server-host",
        default="127.0.0.1",
        help="服务器地址 (默认: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--server-port",
        default="8192",
        help="服务器端口 (默认: 8192)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="输出目录 (默认: ./benchmark_results)"
    )
    
    parser.add_argument(
        "--dataset-path",
        default="benchmark/hicache/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="数据集路径 (默认: benchmark/hicache/ShareGPT_V3_unfiltered_cleaned_split.json)"
    )
    
    parser.add_argument(
        "--num-requests",
        default="100",
        help="每个测试的请求数 (默认: 100)"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="快速测试模式（减少请求数和超时时间）"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="单个测试的超时时间（秒） (默认: 7200)"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="测试完成后不清理进程（用于调试）"
    )
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = ImprovedParameterSearch(args)
    
    try:
        # 运行搜索
        result = searcher.run()
        
        log_info("")
        log_info("=" * 60)
        log_info("参数搜索完成！")
        log_info("=" * 60)
        log_info(f"结果目录: {result['output_dir']}")
        log_info(f"最优参数: {result['best_params']}")
        log_info("=" * 60)
        
    except KeyboardInterrupt:
        log_warn("用户中断搜索")
        searcher.cleanup()
        sys.exit(1)
    except Exception as e:
        log_error(f"搜索过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        searcher.cleanup()
        sys.exit(1)
    finally:
        if not args.no_cleanup:
            searcher.cleanup()

if __name__ == "__main__":
    main()