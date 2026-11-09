import json
import logging
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)

DEFAULT_PROVIDER_PORT = 12001
DEFAULT_LOCAL_BUFFER_SIZE = 1 * 1024 * 1024 * 1024  # 1GB


def parse_global_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            return int(s[:-2].strip()) * 1024 * 1024 * 1024
        if s.endswith("mb"):
            return int(s[:-2].strip()) * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: str
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        master_server_address = os.getenv("MOONCAKE_MASTER")
        if not master_server_address:
            raise ValueError("The environment variable 'MOONCAKE_MASTER' is not set.")
        return MooncakeStoreConfig(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", os.getenv("LOCAL_HOSTNAME", "localhost")),
            metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
            global_segment_size=os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "900gb"),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "rdma"),
            device_name=os.getenv("MOONCAKE_DEVICE", ""),
            master_server_address=master_server_address,
            master_metrics_port=int(os.getenv("MOONCAKE_MASTER_METRICS_PORT", "8003")),
        )


def get_provider_port() -> int:
    return int(os.getenv("MOONCAKE_PROVIDER_PORT", DEFAULT_PROVIDER_PORT))


def normalize_local_hostname(local_hostname: str, port: int) -> str:
    local_hostname = local_hostname.strip()
    if not local_hostname:
        raise ValueError("local_hostname must not be empty")
    if local_hostname.endswith(f":{port}"):
        return local_hostname
    return f"{local_hostname}:{port}"


def get_master_ip(config: MooncakeStoreConfig) -> str:
    addr = config.master_server_address
    # HA: addr = "etcd://ip1:2379;ip2:2379;..." — for HTTP metadata/metrics we
    # need a concrete master IP. Allow override via MOONCAKE_MASTER_HTTP_IP
    # (set by launcher script to a known live master), else fall back to the
    # first etcd endpoint host (each master also runs an embedded HTTP
    # metadata server on 8087, so any reachable host works).
    override = os.getenv("MOONCAKE_MASTER_HTTP_IP")
    if override:
        return override.strip()
    if addr.startswith("etcd://"):
        endpoints = addr[len("etcd://") :]
        first = endpoints.split(";")[0].strip()
        return first.split(":")[0]
    return addr.split(":")[0]


def get_master_segments_url(config: MooncakeStoreConfig) -> str:
    master_server_ip = get_master_ip(config)
    return f"http://{master_server_ip}:{config.master_metrics_port}/get_all_segments"


def get_master_segments(config: MooncakeStoreConfig) -> list[Any]:
    resp = requests.get(get_master_segments_url(config), timeout=3)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(":" in line for line in lines):
        return lines
    try:
        payload = resp.json()
    except ValueError:
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("segments", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unexpected master segments payload type: {type(payload).__name__}")


def check_local_port_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def cleanup_rpc_meta(config: MooncakeStoreConfig, hostname: str):
    key = f"mooncake/rpc_meta/{hostname}"
    url = f"http://{get_master_ip(config)}:8087/metadata?key={quote(key, safe='')}"
    req = urllib.request.Request(url, method="DELETE")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="ignore").strip()
        logger.info("Deleted zombie rpc_meta before provider setup: %s, body=%s", hostname, body)
    except Exception as exc:
        logger.warning("Failed to delete zombie rpc_meta before provider setup: %s, exc=%s", hostname, exc)


def segment_mentions_provider(segment: Any, provider_addr: str) -> bool:
    if isinstance(segment, str):
        return provider_addr in segment
    if isinstance(segment, dict):
        return any(segment_mentions_provider(value, provider_addr) for value in segment.values())
    if isinstance(segment, list):
        return any(segment_mentions_provider(value, provider_addr) for value in segment)
    return False


def is_provider_registered(config: MooncakeStoreConfig, provider_addr: str) -> bool:
    try:
        segments = get_master_segments(config)
    except Exception as exc:
        logger.warning("Failed to query master segments: %s", exc)
        return False
    return any(segment_mentions_provider(segment, provider_addr) for segment in segments)


def wait_for_provider_registration(
    config: MooncakeStoreConfig,
    provider_addr: str,
    timeout: int = 600,
    poll_interval: float = 3.0,
) -> bool:
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < timeout:
        if is_provider_registered(config, provider_addr):
            return True
        logger.info(
            "waiting mooncake provider registered on master, provider=%s, cost_time: %.2f seconds.",
            provider_addr,
            time.perf_counter() - start_time,
        )
        time.sleep(poll_interval)
    return False


def maybe_init_npu(protocol: str):
    if protocol != "ascend":
        return
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise ImportError("Failed to import torch_npu for ascend provider") from e

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available for ascend provider")

    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    first_device = int(visible_devices.split(",")[0].strip())
    torch.npu.set_device(first_device)
    logger.info("Initialized torch.npu device for provider: %s", first_device)


def main():
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as e:
        raise ImportError(
            "Please install mooncake by following the instructions at https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
        ) from e

    config = MooncakeStoreConfig.load_from_env()
    maybe_init_npu(config.protocol)
    provider_port = get_provider_port()
    provider_addr = normalize_local_hostname(config.local_hostname, provider_port)

    provider_host, provider_port_str = provider_addr.rsplit(":", 1)
    provider_port = int(provider_port_str)

    if is_provider_registered(config, provider_addr):
        logger.info("Mooncake provider already registered on master, skip startup: %s", provider_addr)
        return

    cleanup_rpc_meta(config, provider_addr)

    if check_local_port_listening(provider_host, provider_port):
        raise RuntimeError(f"Provider port already in use before setup: {provider_addr}")

    logger.info(
        "Mooncake provider not found on master, starting provider: %s",
        provider_addr,
    )
    store = MooncakeDistributedStore()
    ret = store.setup(
        config.local_hostname,
        config.metadata_server,
        parse_global_segment_size(config.global_segment_size),
        DEFAULT_LOCAL_BUFFER_SIZE,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if ret:
        raise RuntimeError(f"MooncakeDistributedStore.setup failed, code={ret}")

    if not wait_for_provider_registration(config, provider_addr):
        raise RuntimeError(f"Mooncake provider registration timeout for {provider_addr}")

    logger.info("Mooncake provider is ready: %s", provider_addr)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
