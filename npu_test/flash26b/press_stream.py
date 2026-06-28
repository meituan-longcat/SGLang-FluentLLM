import os
import sys
import aiohttp
import asyncio
import json
import time
import random
import argparse
from datetime import datetime
from urllib.parse import urlparse
from aiolimiter import AsyncLimiter
from dataclasses import dataclass
from typing import List, Optional, Union
from rich import print
from rich.progress import Progress, TaskID, BarColumn, Progress, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from contextlib import asynccontextmanager

@dataclass
class Config:
    url: str
    qps: Optional[float]
    request_num: Optional[int]
    input_len: Optional[int]
    output_len: int
    temperature: float
    dataset: Optional[str]
    input_text: Optional[str]
    ignore_eos: Optional[bool]
    record_file: str

    def __post_init__(self):
        self._validate_config()

    def _validate_config(self):
        if self.input_len and not self.request_num:
            raise ValueError("Must specify request_num when input_len is specified.")
        if self.input_text and self.qps:
            raise ValueError("Cannot specify both input_text and qps.")
        if os.path.isdir(self.record_file):
            mode_scheme = f"async_qps_{self.qps}" if self.qps else "sync"
            url_scheme = urlparse(self.url).scheme
            time_scheme = f"time_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            record_file_name = f"{url_scheme}_{mode_scheme}_{time_scheme}.json"
            self.record_file = os.path.join(self.record_file, record_file_name)

    @property
    def generate_url(self) -> str:
        return f"http://{self.url}/generate"

    @property 
    def rate_limit(self) -> tuple:
        return 1, 1/self.qps

@dataclass
class RequestRecord:
    id: int
    duration: float
    ttft: float
    tpot: float
    input_len: int
    output_len: int
    status: str
    output: str
    accept_rate: float

@dataclass
class ProgressState:
    total: int = 0
    progress: Optional[Progress] = None
    send_task_id: Optional[TaskID] = None
    recv_task_id: Optional[TaskID] = None

    @staticmethod
    def create_progress() -> Progress:
        return Progress(
            TextColumn("[bold blue]{task.fields[flag]}", justify="right"),
            BarColumn(bar_width=60),
            "[progress.percentage]{task.percentage:>3.1f}%", "•",
            MofNCompleteColumn(), "•",
            TimeRemainingColumn(elapsed_when_finished=True)
        )
    def update_send(self) -> None:     
        if self.progress and self.send_task_id is not None:
            self.progress.update(self.send_task_id, advance=1)

    def update_recv(self) -> None:     
        if self.progress and self.recv_task_id is not None:
            self.progress.update(self.recv_task_id, advance=1)

class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.limiter = self._init_rate_limiter(config)
        self.progress_state = None

    def _init_rate_limiter(self, config: Config) -> Optional[AsyncLimiter]:
        if config.qps:
            max_rate, time_period = config.rate_limit
            return AsyncLimiter(max_rate, time_period)
        return None
    
    @asynccontextmanager
    async def _progress_tracker(self, total: int):
        self.progress_state = ProgressState(total=total)
        
        with ProgressState.create_progress() as progress:
            send_task_id = progress.add_task("send", flag='send', start=True)
            self.progress_state.send_task_id = send_task_id
            progress.update(send_task_id, total=total)

            recv_task_id = progress.add_task("recv", flag='recv', start=True)
            self.progress_state.recv_task_id = recv_task_id
            progress.update(recv_task_id, total=total)

            self.progress_state.progress = progress
            yield self.progress_state

    async def send_request(self, session: aiohttp.ClientSession, id: int, input: Union[str, list]) -> RequestRecord:
        try:
            if self.limiter:
                async with self.limiter:
                    if self.progress_state:
                        self.progress_state.update_send()
                    record = await self._do_stream_request(session, id, input, True)
            else:
                print(f"[bold magenta]Input:[/bold magenta] {input}")
                record = await self._do_stream_request(session, id, input, False)
        except Exception as e:
            record = self._handle_error(id, str(e))
        
        if self.limiter and self.progress_state:
            self.progress_state.update_recv()
        return record

    def _prepare_request(self, input: Union[str, list]) -> dict:
        input_key = "text" if isinstance(input, str) else "input_ids"
        return {
            input_key: input,
            "sampling_params": {
                "temperature": self.config.temperature,
                "top_k": 10,
                "max_new_tokens": self.config.output_len,
                "ignore_eos": self.config.ignore_eos,
            },
            "stream": True
        }

    async def _do_stream_request(self, session: aiohttp.ClientSession, id: int, input: Union[str, list], async_mode: bool) -> RequestRecord:
        input_len = 0
        prev_len = 0
        completion_tokens = 0
        ttft = None
        recv_ft_time = None
        output = ""
        error = None
        
        try:
            request = self._prepare_request(input)
            start_time = time.time()
            async with session.post(self.config.generate_url, json=request, timeout=aiohttp.ClientTimeout(total=6000)) as response:
                if async_mode is False:
                    print(f"[bold green]Output: [/bold green]", end="", flush=True)
                async for raw_chunk in response.content:
                    chunk = raw_chunk.decode('utf-8').strip()
                    if not chunk or chunk == "data: [DONE]":
                        continue

                    if chunk.startswith('data:'):
                        chunk_data = json.loads(chunk[5:])
                        if chunk_data.get("error") is not None:
                            error = chunk_data.get("error").get("message")
                            continue
                        
                        current_text = chunk_data.get("text", "")
                        current_length = len(current_text)
                        
                        delta_text = current_text[prev_len:current_length]
                        prev_len = current_length
                        
                        if async_mode is False:
                            print(delta_text, end="", flush=True)

                        meta_info = chunk_data.get("meta_info", {})
                        current_tokens = meta_info.get("completion_tokens", 0)
                        accept_rate = meta_info.get("accept_draft_tokens", 0)

                        if recv_ft_time is None and delta_text:
                            recv_ft_time = time.time()
                            ttft = recv_ft_time - start_time
                            completion_tokens = current_tokens
                            input_len = meta_info.get("prompt_tokens", 0)

                        completion_tokens = max(completion_tokens, current_tokens)
                        output += delta_text
                    else:
                        print(f"[Line {sys._getframe().f_lineno}] [bold red]Request failed: {error}")
                        error = chunk
                if async_mode is False:
                    print("\n", flush=True)

                duration = time.time() - start_time
                if recv_ft_time and completion_tokens > 1:
                    output_duration = time.time() - recv_ft_time
                    tpot = (output_duration * 1000) / (completion_tokens - 1)
                else:
                    tpot = 0

                if async_mode is False and error:
                    print(f"[Line {sys._getframe().f_lineno}] [bold red]Request failed: {error}")

                return RequestRecord(
                    id=id,
                    duration=duration,
                    ttft=ttft * 1000 if ttft else 0,
                    tpot=tpot,
                    input_len=input_len,
                    output_len=completion_tokens,
                    accept_rate=accept_rate,
                    status=error if error else "success",
                    output=output
                )
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"[Line {sys._getframe().f_lineno}] [bold red]Request failed: {str(e)}")
            return RequestRecord(
                id=id,
                duration=duration,
                ttft=0,
                tpot=0,
                input_len=0,
                output_len=0,
                status=f"failed: {str(e)}",
                output=""
            )

    def _handle_error(self, id: int, error: str) -> RequestRecord:
        print(f"[Line {sys._getframe().f_lineno}] [bold red]Request failed: {error}")
        return RequestRecord(id, None, 0, 0, 0, 0, f"failed: {error}", "")

class RecordReporter:
    def __init__(self, record_file: str):
        self.record_file = record_file
        os.makedirs(os.path.dirname(record_file), exist_ok=True)
        with open(self.record_file, "w", encoding="utf-8") as f:
            pass  # Just create or clear the file

    def report_summary(self, records: List[RequestRecord], total_duration: float) -> None:
        import math
        success_records = [r for r in records if r.status == "success"]
        total_records = len(records)
        success_count = len(success_records)

        print("\n=== Summary ===")
        print(f"总请求数: {total_records}")
        print(f"成功数: {success_count}")
        print(f"失败数: {total_records - success_count}")
        print(f"总耗时: {total_duration:.2f} s")

        if success_count > 0:
            def get_percentile(sorted_data: List[float], percent: float) -> float:
                n = len(sorted_data)
                if n == 0:
                    return 0.0
                index = (percent / 100) * (n - 1)
                lower = int(math.floor(index))
                upper = lower + 1
                if upper >= n:
                    return sorted_data[lower]
                weight = index - lower
                return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

            metrics_info = [
                ("TTFT", "ms", [r.ttft for r in success_records]),
                ("TPOT", "ms", [r.tpot for r in success_records]),
                ("Duration", "s", [r.duration for r in success_records]),
                ("Input Length", "", [r.input_len for r in success_records]),
                ("Output Length", "", [r.output_len for r in success_records]),
                ("Accept Length", "", [r.accept_rate for r in success_records]),
            ]

            table_rows = []
            for name, unit, data in metrics_info:
                sorted_data = sorted(filter(None, data))
                avg = sum(sorted_data) / success_count
                tp50 = get_percentile(sorted_data, 50)
                tp90 = get_percentile(sorted_data, 90)
                tp99 = get_percentile(sorted_data, 99)
                table_rows.append((name, unit, avg, tp50, tp90, tp99))

            print("\n=== Performance ===")
            print(f"{'Metric':<15} | {'AVG':>10} | {'TP50':>10} | {'TP90':>10} | {'TP99':>10}")
            print("-" * 66)
            for name, unit, avg, tp50, tp90, tp99 in table_rows:
                metric_display = f"{name} ({unit})" if unit else name
                avg_str = f"{avg:.2f}"
                tp50_str = f"{tp50:.2f}"
                tp90_str = f"{tp90:.2f}"
                tp99_str = f"{tp99:.2f}"
                print(f"{metric_display:<15} | {avg_str:>10} | {tp50_str:>10} | {tp90_str:>10} | {tp99_str:>10}")

    def save_records(self, records: List[RequestRecord]) -> None:
        records.sort(key=lambda record: record.id)
        
        records_dicts = []
        for r in records:
            record_dict = {
                "id": r.id,
                "duration": round(r.duration, 4),
                "ttft": round(r.ttft, 2),
                "tpot": round(r.tpot, 2),
                "input_len": r.input_len,
                "output_len": r.output_len,
                "status": r.status,
                "output": r.output
            }
            records_dicts.append(record_dict)

        with open(self.record_file, "w", encoding="utf-8") as f:
            json.dump(records_dicts, f, ensure_ascii=False, indent=2)

async def main(config: Config):
    client = LLMClient(config)
    reporter = RecordReporter(config.record_file)

    prompts = None
    input_ids = None
    if config.dataset:
        with open(config.dataset, "r") as f:
            prompts = [json.loads(line)["input"] for line in f.readlines()]
            #prompts = [json.loads(line)["question"]*40 for line in f.readlines()]
            #prompts = [json.load(f)["queryList"][0]]
        if config.request_num is not None:
            prompt_num = len(prompts)
            if config.request_num > prompt_num:
                for i in range(config.request_num - prompt_num):
                    id = i % prompt_num
                    prompts.append(prompts[id])
            else:
                prompts = prompts[:config.request_num]
    elif config.input_text:
        prompts = [config.input_text]
    else:
        start_id = 1
        input_ids = [[random.randint(start_id, config.input_len + start_id) for _ in range(config.input_len)] for _ in range(config.request_num)]
        # input_ids = [list(range(start_id, config.input_len + start_id)) for _ in range(config.request_num)]
    if prompts:
        print(f"Pressing stream endpoint {config.url} with {len(prompts)} prompts by qps {config.qps}...")
    else:
        print(f"Pressing stream endpoint {config.url} with {len(input_ids)} input_ids by qps {config.qps}...")
    
    inputs = prompts if prompts else input_ids

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256)) as session:
        start_time = time.time()

        if config.qps is None:
            records = []
            for id, input in enumerate(inputs):
                record = await client.send_request(session, id, input)
                records.append(record)
        else:
            async with client._progress_tracker(len(inputs)) as _:
                tasks = [client.send_request(session, id, input) for id, input in enumerate(inputs)]
                records = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time

    reporter.save_records(records)
    reporter.report_summary(records, total_duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u",
                        "--url",
                        type=str,
                        default="localhost:8000",
                        required=False,
                        help="Inference server URL.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        required=False,
                        help="Specify input length")
    parser.add_argument("--output-len",
                        type=int,
                        default=128,
                        required=False,
                        help="Specify max output length")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.6,
                        required=False,
                        help="Temperature value")
    parser.add_argument("--qps",
                        type=float,
                        default=None,
                        required=False,
                        help="Used in async infer mode.")
    parser.add_argument("--request-num",
                        type=int,
                        default=None,
                        help="If not None, extend prompt num from len(prompts) to request_num.")
    parser.add_argument("--input-text",
                        type=str,
                        default=None,
                        required=False,
                        help="If not None, prompt will be set as the input-text.")
    parser.add_argument("--ignore-eos",
                        action="store_true",
                        help="Ignore eos token.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        required=False,
                        help="If not None, prompt in the dataset will be used to call the model.")
    parser.add_argument("--resdata",
                        type=str,
                        default=".",
                        required=False,
                        help="Response data path used to save results.")
    args = parser.parse_args()

    try:
        config = Config(url=args.url,
                        input_len=args.input_len,
                        output_len=args.output_len,
                        temperature=args.temperature,
                        qps=args.qps,
                        request_num=args.request_num,
                        input_text=args.input_text,
                        ignore_eos=args.ignore_eos,
                        dataset=args.dataset,
                        record_file=args.resdata)
    except ValueError as e:
        print(f"Configuration error: {e}")
        exit(1)

    asyncio.run(main(config))

