"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import dataclasses
import logging
import random
import urllib
from itertools import chain
from typing import Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.utils import PDRegistryRequest


def setup_logger():
    logger = logging.getLogger("pdlb")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None


class MiniLoadBalancer:
    def __init__(self, prefill_configs: list[PrefillConfig], decode_servers: list[str], enable_cache_report: bool = False):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers
        self.bootstrap_room_cnt = 0
        self.bootstrap_room_lock = asyncio.Lock()
        self.enable_cache_report = enable_cache_report
        self.prefill_index = 0  # 用于prefill_configs的轮询计数
        self.decode_index = 0  # 用于decode_servers的轮询计数

    async def get_bootstrap_room(self):
        async with self.bootstrap_room_lock:
            self.bootstrap_room_cnt += 1
            return self.bootstrap_room_cnt

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(new_decode_server)

    def select_pair_round_robin(self):
        # 校验服务列表非空
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        # 轮询选择prefill_config
        prefill_config = self.prefill_configs[self.prefill_index]
        # 更新prefill索引（循环递增，超过长度后重置为0）
        self.prefill_index = (self.prefill_index + 1) % len(self.prefill_configs)

        # 轮询选择decode_server
        decode_server = self.decode_servers[self.decode_index]
        # 更新decode索引
        self.decode_index = (self.decode_index + 1) % len(self.decode_servers)

        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    def select_pair(self):
        # TODO: return some message instead of panic
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        prefill_config = random.choice(self.prefill_configs)  # noqa: S311
        decode_server = random.choice(self.decode_servers)  # noqa: S311
        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    async def generate(self, modified_request, prefill_server, decode_server, endpoint) -> ORJSONResponse:
        ret_json, status_code = await self.generate_raw(modified_request, prefill_server, decode_server, endpoint)
        return ORJSONResponse(content=ret_json, status_code=status_code)

    async def generate_raw(self, modified_request, prefill_server, decode_server, endpoint):
        """Run one prefill+decode pair and return (merged_json, status_code)."""
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600)  # Add timeout for request reliability
        ) as session:
            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                session.post(f"{decode_server}/{endpoint}", json=modified_request),
            ]

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if modified_request is not None:
                if "return_logprob" in modified_request or self.enable_cache_report:
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()
                    # merge input logprobs (computed on prefill) into decode response
                    if "meta_info" in ret_json and "meta_info" in prefill_json:
                        for key in (
                            "input_token_logprobs",
                            "input_top_logprobs",
                            "input_token_ids_logprobs",
                        ):
                            if key in ret_json["meta_info"] and key in prefill_json["meta_info"]:
                                ret_json["meta_info"][key] = prefill_json["meta_info"][key] + ret_json["meta_info"][key]

                    # Inject cached_tokens from prefill if needed
                    if self.enable_cache_report:
                        # Extract cached_tokens from prefill response (which may have usage field)
                        cached_tokens = 0

                        # Try to get cached_tokens from prefill's usage field (new format)
                        if "usage" in prefill_json and "prompt_tokens_details" in prefill_json["usage"]:
                            if prefill_json["usage"]["prompt_tokens_details"] is not None:
                                cached_tokens = prefill_json["usage"]["prompt_tokens_details"].get("cached_tokens", 0)

                        # If not found in usage, try to get from meta_info (fallback for compatibility)
                        if cached_tokens == 0 and "meta_info" in prefill_json:
                            cached_tokens = prefill_json["meta_info"].get("cached_tokens", 0)

                        # Inject cached_tokens into decode response
                        if cached_tokens > 0 or "usage" in ret_json:
                            if ret_json.get("usage") is None:
                                ret_json["usage"] = {}
                            if ret_json["usage"].get("prompt_tokens_details") is None:
                                ret_json["usage"]["prompt_tokens_details"] = {}
                            ret_json["usage"]["prompt_tokens_details"]["cached_tokens"] = cached_tokens
                else:
                    # For non-logprob requests, also get prefill response to merge cached_tokens
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()
                    # Merge cached_tokens from both prefill and decode responses
                    if "meta_info" in ret_json and "meta_info" in prefill_json:
                        prefill_cached = prefill_json["meta_info"].get("cached_tokens", 0)
                        ret_json["meta_info"]["cached_tokens"] = prefill_cached

            else:
                ret_json = "profile"

            return ret_json, decode_response.status

    async def generate_stream(self, modified_request, prefill_server, decode_server, endpoint="generate"):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3600),  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                    session.post(f"{decode_server}/{endpoint}", json=modified_request),
                ]
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                # aiohttp's StreamReader readline() has a hard 64KB limit that
                # triggers ValueError("Chunk too big") when an SSE line
                # contains many logprobs.  Use readany()+manual line splitting
                # to avoid this limit entirely.
                async def _safe_readlines(resp):
                    buf = b""
                    while True:
                        chunk = await resp.content.readany()
                        if not chunk:
                            if buf:
                                yield buf
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            yield line + b"\n"

                if modified_request.get("return_logprob", False) or self.enable_cache_report:
                    prefill_chunks = []
                    async for chunk in _safe_readlines(prefill_response):
                        prefill_chunks.append(chunk)

                    # Process all prefill chunks, collect cached_tokens and first chunk's logprobs
                    first_prefill_chunk_json = None
                    cached_from_prefill = {}
                    prefill_usage = None
                    for chunk in prefill_chunks:
                        decoded_chunk = chunk.decode("utf-8")
                        if decoded_chunk and decoded_chunk.startswith("data:") and "[DONE]" not in decoded_chunk:
                            chunk_json = orjson.loads(decoded_chunk[5:].strip("\n"))

                            # Collect first chunk's logprobs (for existing logic)
                            if first_prefill_chunk_json is None:
                                first_prefill_chunk_json = chunk_json

                            # Collect cached_tokens from usage (from new usage field)
                            if "usage" in chunk_json and self.enable_cache_report:
                                prefill_usage = chunk_json.get("usage")

                            # Collect each choice's cached_tokens (from meta_info, for compatibility)
                            if "choices" in chunk_json:
                                for choice in chunk_json["choices"]:
                                    if "index" in choice and "meta_info" in choice and "cached_tokens" in choice["meta_info"]:
                                        idx = choice["index"]
                                        cached_from_prefill[idx] = choice["meta_info"]["cached_tokens"]

                    async for chunk in _safe_readlines(decode_response):
                        decoded_chunk = chunk.decode("utf-8")
                        if decoded_chunk and decoded_chunk.startswith("data:") and "[DONE]" not in decoded_chunk:
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))

                            # Merge input logprobs (computed on prefill) into decode
                            if first_prefill_chunk_json and "meta_info" in ret_json and "meta_info" in first_prefill_chunk_json:
                                for key in (
                                    "input_token_logprobs",
                                    "input_top_logprobs",
                                    "input_token_ids_logprobs",
                                ):
                                    if key in ret_json["meta_info"] and key in first_prefill_chunk_json["meta_info"]:
                                        ret_json["meta_info"][key] = first_prefill_chunk_json["meta_info"][key] + ret_json["meta_info"][key]

                            # Check if it's the last chunk (has usage) and inject cached_tokens
                            if "usage" in ret_json and ret_json["usage"] is not None and self.enable_cache_report:
                                # Extract cached_tokens from prefill's usage
                                cached_tokens = 0
                                if prefill_usage and "prompt_tokens_details" in prefill_usage:
                                    if prefill_usage["prompt_tokens_details"] is not None:
                                        cached_tokens = prefill_usage["prompt_tokens_details"].get("cached_tokens", 0)

                                # If not obtained from usage, try from cached_from_prefill (compatibility)
                                if cached_tokens == 0 and cached_from_prefill:
                                    cached_tokens = cached_from_prefill.get(0, 0)

                                # Inject cached_tokens into decode response's usage
                                if cached_tokens > 0 or self.enable_cache_report:
                                    if ret_json["usage"].get("prompt_tokens_details") is None:
                                        ret_json["usage"]["prompt_tokens_details"] = {}
                                    ret_json["usage"]["prompt_tokens_details"]["cached_tokens"] = cached_tokens

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        elif decoded_chunk.strip() == "[DONE]":
                            yield b"data: [DONE]\n\n"
                        elif decoded_chunk.strip():
                            yield chunk
                else:
                    # Pure passthrough: forward the decode backend's SSE bytes
                    # verbatim. Do NOT line-split/strip here — splitting on single
                    # "\n" and dropping blank lines collapses the "\n\n" that
                    # terminates each SSE event, which breaks strict clients that
                    # frame events on "\n\n" (e.g. evalscope perf).
                    while True:
                        chunk = await decode_response.content.readany()
                        if not chunk:
                            break
                        yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )

    async def profile(self, prefill_server, decode_server, endpoint="start_profile"):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600)  # Add timeout for request reliability
        ) as session:
            tasks = [
                session.post(f"{prefill_server}/{endpoint}"),
                session.post(f"{decode_server}/{endpoint}"),
            ]
            # Wait for both responses to complete.
            await asyncio.gather(*tasks)

            return Response(
                content=("Start profiling.\n" if endpoint == "start_profile" else "Stop profiling. This will take some time.\n"),
                status_code=200,
            )


app = FastAPI()
load_balancer: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    """Simple health check - always returns 200"""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_check_generate():
    """Health check for generate - checks if backend servers are ready"""
    if load_balancer is None:
        return Response(status_code=503)  # Service Unavailable

    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.get(f"{server}/health_generate"))
        for _i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))
        for _i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    async with aiohttp.ClientSession() as session:
        for server in chain(prefill_servers):
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in chain(decode_servers):
            server_info = await session.get(f"{server}/get_server_info")
            decode_infos.append(await server_info.json())

    return {"prefill": prefill_infos, "decode": decode_infos}


@app.get("/get_model_info")
async def get_model_info():
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile():
    """Start profiling."""
    prefill_server, _, decode_server = load_balancer.select_pair_round_robin()
    return await load_balancer.profile(prefill_server, decode_server, "start_profile")


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile():
    """Stop profiling."""
    prefill_server, _, decode_server = load_balancer.select_pair_round_robin()
    return await load_balancer.profile(prefill_server, decode_server, "stop_profile")


@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
async def start_expert_distribution_record():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/start_expert_distribution_record"))
        for _i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
async def stop_expert_distribution_record():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/stop_expert_distribution_record"))
        for _i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
async def dump_expert_distribution_record():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/dump_expert_distribution_record"))
        for _i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    # Log incoming request
    text_length = len(request_data.get("text", "")) if isinstance(request_data.get("text"), str) else "batch"
    logger.debug(f"LB received generate request: stream={request_data.get('stream', False)}, text_length={text_length}")

    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair_round_robin()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    _inject_bootstrap(modified_request, hostname, bootstrap_port)

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(modified_request, prefill_server, decode_server, "generate")
    else:
        return await load_balancer.generate(modified_request, prefill_server, decode_server, "generate")


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair_round_robin()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    _inject_bootstrap(modified_request, hostname, bootstrap_port)

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    # The backend /v1/chat/completions only accepts a single conversation
    # (messages: list[message]). If the caller passes a batch — messages as a
    # list of conversations (list[list[message]]) — fan out into N single
    # requests so each gets its own prefill/decode pair and bootstrap_room.
    sub_messages = _split_chat_batch(request_data.get("messages"))
    if sub_messages is None:
        return await _forward_to_backend(request_data, "v1/chat/completions")

    if request_data.get("stream", False):
        raise HTTPException(
            status_code=400,
            detail="Batched messages are not supported with stream=true.",
        )

    async def run_one(messages):
        sub_request = request_data.copy()
        sub_request["messages"] = messages
        prefill_server, bootstrap_port, decode_server = load_balancer.select_pair_round_robin()
        hostname = urllib.parse.urlparse(prefill_server).hostname
        _inject_bootstrap(sub_request, hostname, bootstrap_port)
        return await load_balancer.generate_raw(sub_request, prefill_server, decode_server, "v1/chat/completions")

    results = await asyncio.gather(*[run_one(m) for m in sub_messages])
    status_code = max((status for _, status in results), default=200)
    return ORJSONResponse(
        content=[ret_json for ret_json, _ in results],
        status_code=status_code,
    )


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _split_chat_batch(messages):
    """Return a list of conversations if `messages` is a batch, else None.

    A single conversation is list[message] where each message is a dict.
    A batch is list[conversation] where each conversation is itself a list.
    """
    if not isinstance(messages, list) or not messages:
        return None
    if isinstance(messages[0], list):
        return messages
    return None


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)  # noqa: S311


def _inject_bootstrap(modified_request, hostname, bootstrap_port):
    """Inject PD bootstrap fields, matching the request's batch shape.

    For batched requests every sub-request needs its OWN bootstrap_room,
    otherwise their KV-transfer rooms collide on the prefill/decode side.
    """
    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [_generate_bootstrap_room() for _ in range(batch_size)],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    if (prompt := request.get("prompt")) is not None:
        # OpenAI /v1/completions: str or list[str] is a real batch;
        # list[int]/list[list[int]] are token-id prompts.
        if isinstance(prompt, str):
            return None
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return None
        return len(prompt)
    # /v1/chat/completions always maps a single `messages` list to one prompt.
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/register")
async def register(obj: PDRegistryRequest):
    if obj.mode == "prefill":
        load_balancer.add_prefill_server(PrefillConfig(obj.registry_url, obj.bootstrap_port))
        logger.info(f"Registered prefill server: {obj.registry_url} with bootstrap port: {obj.bootstrap_port}")
    elif obj.mode == "decode":
        load_balancer.add_decode_server(obj.registry_url)
        logger.info(f"Registered decode server: {obj.registry_url}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be either PREFILL or DECODE.",
        )

    logger.info(f"#Prefill servers: {len(load_balancer.prefill_configs)}, #Decode servers: {len(load_balancer.decode_servers)}")

    return Response(status_code=200)


def run(prefill_configs, decode_addrs, host, port, enable_cache_report=False):
    global load_balancer
    load_balancer = MiniLoadBalancer(prefill_configs, decode_addrs, enable_cache_report)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument("--prefill", type=str, default=[], nargs="+", help="URLs for prefill servers")
    parser.add_argument("--decode", type=str, default=[], nargs="+", help="URLs for decode servers")
    parser.add_argument(
        "--prefill-bootstrap-ports",
        type=int,
        nargs="+",
        help="Bootstrap ports for prefill servers",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server (default: 8000)")
    parser.add_argument("--enable-cache-report", action="store_true", help="Enable cached tokens report in usage information")
    args = parser.parse_args()

    bootstrap_ports = args.prefill_bootstrap_ports
    if bootstrap_ports is None:
        bootstrap_ports = [None] * len(args.prefill)
    elif len(bootstrap_ports) == 1:
        bootstrap_ports = bootstrap_ports * len(args.prefill)
    else:
        if len(bootstrap_ports) != len(args.prefill):
            raise ValueError("Number of prefill URLs must match number of bootstrap ports")

    prefill_configs = [PrefillConfig(url, port) for url, port in zip(args.prefill, bootstrap_ports)]

    run(prefill_configs, args.decode, args.host, args.port, args.enable_cache_report)
