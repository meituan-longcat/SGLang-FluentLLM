import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import socket
import struct
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FastQueue,
    PageTransferMetadata,
    StepCounter,
    group_concurrent_contiguous,
)
from sglang.srt.layers.attention.dp_attention import get_attention_dp_rank, get_attention_tp_rank, get_attention_tp_size
from sglang.srt.mem_cache.dsa_indexer_layout import check_stage_transfer_layout, check_transfer_layout
from sglang.srt.observability.metrics.collector import KVTransferMetricsCollector
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_colorful_logger, get_free_port, get_int_env_var, get_ip, get_local_ip_by_remote, is_npu

__is_npu__ = is_npu()


logger = get_colorful_logger(__name__)

# Sentinel status values reused over the existing sync_status wire channel to
# deliver a decode-initiated abort acknowledgement. Values intentionally sit
# outside KVPoll's 0..4 range so decode_thread can dispatch unambiguously.
ABORT_ACK_OK = 100
ABORT_ACK_FAIL = 101


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str, remote_endpoint: str = None):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.remote_endpoint = remote_endpoint

    def __str__(self):
        if self.remote_endpoint:
            return f"KVTransferError(bootstrap_room={self.bootstrap_room}, remote_endpoint={self.remote_endpoint}): {self.failure_reason}"
        else:
            return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int64]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]
    mla_l1_5_args: Optional[PageTransferMetadata]
    state_indices: Optional[list[int]]


@dataclasses.dataclass
class TransferIndexResolution:
    src_indices: npt.NDArray[np.int64]
    dst_indices: npt.NDArray[np.int64]
    # draft 层的页坐标。draft pool 不被 l1.5 包装时页索引是全局的且不分片, 与上面两组
    # (target 的 l1.5 局部坐标, 已过 KVP mask) 坐标系不同, 须分开传。None 表示同坐标系复用上面两组。
    draft_src_indices: npt.NDArray[np.int64] | None = None
    draft_dst_indices: npt.NDArray[np.int64] | None = None


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_index: int
    required_dst_info_num: int
    decode_prefix_len: int
    dst_indices_are_local: bool
    dst_page_transfer_mask: Optional[npt.NDArray[np.bool_]]
    dst_page_local_indices: Optional[npt.NDArray[np.int64]]
    dst_page_indices_mapping: Optional[npt.NDArray[np.int64]]
    is_dummy: bool
    dst_state_indices: list[int]

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            dst_kv_indices = np.array([], dtype=np.int64)
            dst_aux_index = None
            decode_prefix_len = 0
            dst_indices_are_local = False
            dst_page_transfer_mask = None
            dst_page_local_indices = None
            dst_page_indices_mapping = None
            is_dummy = True
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int64)
            dst_aux_index = int(msg[5].decode("ascii"))
            # decode_prefix_len is now msg[7] (we added it as additional message part)
            decode_prefix_len = int(msg[7].decode("ascii")) if len(msg) > 7 else 0
            dst_indices_are_local = bool(int(msg[8].decode("ascii"))) if len(msg) > 8 else False
            dst_page_transfer_mask = np.frombuffer(msg[9], dtype=np.bool_) if len(msg) > 9 else None
            dst_page_indices_mapping = np.cumsum(dst_page_transfer_mask) - 1 if dst_page_transfer_mask is not None else None
            dst_page_local_indices = np.frombuffer(msg[10], dtype=np.int64) if len(msg) > 10 else None
            is_dummy = False
            dst_state_indices = list(np.frombuffer(msg[11], dtype=np.int32))
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[6].decode("ascii")),
            decode_prefix_len=decode_prefix_len,
            dst_indices_are_local=dst_indices_are_local,
            dst_page_transfer_mask=dst_page_transfer_mask,
            dst_page_local_indices=dst_page_local_indices,
            dst_page_indices_mapping=dst_page_indices_mapping,
            is_dummy=is_dummy,
            dst_state_indices=dst_state_indices,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_offsets: list[list[int]]
    dst_aux_ptrs: list[int]
    decode_prefix_len: int
    dst_state_data_ptrs: list[int]
    # for mamba state different tp slice transfer
    dst_state_item_lens: list[int]
    dst_state_dim_per_tensor: list[int]
    dst_attn_tp_rank: int = -1
    dst_attn_tp_size: int = -1

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        if len(msg) not in (8, 9, 10, 11, 13):
            raise ValueError(f"Unsupported KV registration message: {len(msg)} parts")
        decode_prefix_len = int(msg[6].decode("ascii"))
        has_state_metadata = len(msg) >= 11
        if has_state_metadata:
            dst_state_data_ptrs = list(struct.unpack(f"{len(msg[7]) // 8}Q", msg[7]))
            dst_state_item_lens = list(struct.unpack(f"{len(msg[8]) // 4}I", msg[8]))
            dst_state_dim_per_tensor = list(struct.unpack(f"{len(msg[9]) // 4}I", msg[9]))
            offsets_data = msg[10]
            tp_metadata_start = 11
        else:
            dst_state_data_ptrs = []
            dst_state_item_lens = []
            dst_state_dim_per_tensor = []
            offsets_data = msg[7]
            tp_metadata_start = 8

        dst_attn_tp_rank = int(msg[tp_metadata_start].decode("ascii")) if len(msg) > tp_metadata_start else -1
        dst_attn_tp_size = int(msg[tp_metadata_start + 1].decode("ascii")) if len(msg) > tp_metadata_start + 1 else -1
        offset = 0
        num_layers = struct.unpack_from("I", offsets_data, offset)[0]
        offset += 4
        dst_offsets = []
        for _ in range(num_layers):
            layer_len = struct.unpack_from("I", offsets_data, offset)[0]
            offset += 4
            layer_offsets = list(struct.unpack_from(f"{layer_len}I", offsets_data, offset))
            offset += 4 * layer_len
            dst_offsets.append(layer_offsets)

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            decode_prefix_len=decode_prefix_len,
            dst_state_data_ptrs=dst_state_data_ptrs,
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
            dst_offsets=dst_offsets,
            dst_attn_tp_rank=dst_attn_tp_rank,
            dst_attn_tp_size=dst_attn_tp_size,
        )


class MooncakeKVManager(BaseKVManager):
    @dataclasses.dataclass
    class _AbortState:
        expected: int
        ack_success: set[int] = dataclasses.field(default_factory=set)
        ack_failed: set[int] = dataclasses.field(default_factory=set)

        @property
        def ack_count(self) -> int:
            return len(self.ack_success) + len(self.ack_failed)

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        draft_is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.server_args = server_args
        local_ip = get_local_ip_by_remote()
        self.engine = MooncakeTransferEngine(
            hostname=local_ip,
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )
        logger.info(
            f"[MOONCAKE_INIT] mode={disaggregation_mode.value} gpu_id={self.kv_args.gpu_id} "
            f"local_ip={local_ip} session_id={self.engine.session_id} "
            f"ib_device={self.kv_args.ib_device}"
        )
        # logger.info(f"[MoonCakeTransferEngine][Topology] {self.engine.engine.get_local_topology()}")
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.src_mode = "ON" if bool(server_args.enable_mla_l1_5_cache) else "OFF"
        # 仅当 target 开 l1.5 而 draft pool 未开时为 True (DFLASH GQA draft 不被包装; MTP draft 同为 MLA 被包装故为 False)。
        self.draft_needs_global_indices = (
            bool(server_args.enable_mla_l1_5_cache) and self.kv_args.draft_layer_num > 0 and not self.kv_args.draft_enable_mla_l1_5_cache
        )
        self.is_mla_backend = is_mla_backend
        self.draft_is_mla_backend = draft_is_mla_backend
        # 非 MLA draft (DFlash GQA) 的 pool 按 attn_tp 分 head, 每 rank 只持一段, 传输须按 head 配对; MTP 各 rank 语义相同。
        self.draft_needs_head_match = self.kv_args.draft_layer_num > 0 and not draft_is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.dp_rank = get_attention_dp_rank()
        self.enable_dp_attention = server_args.enable_dp_attention
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError("If dp_attention is not enabled, dp size must be 1 in disaggregation mode.")
        self.request_status: dict[int, KVPoll] = {}
        self.rank_port = None
        self.request_state_lock = threading.RLock()
        # Use a shared ZMQ context to avoid creating one IO thread per Context instance
        self._zmq_ctx = zmq.Context(2)
        self._zmq_ctx.set(zmq.MAX_SOCKETS, 1000000)
        self.server_socket = self._zmq_ctx.socket(zmq.PULL)
        if hasattr(server_args, "enable_metrics") and server_args.enable_metrics:
            labels = {
                "model_name": server_args.served_model_name,
                "app_key": server_args.app_key,
            }
            self.kv_transfer_metrics = KVTransferMetricsCollector(labels, server_args.metrics_reporters)
        else:
            self.kv_transfer_metrics = None

        self.register_buffer_to_engine()
        self.pp_rank = self.kv_args.pp_rank
        self.pp_size = server_args.pp_size

        # Negative cache for peers that recently failed, shared by both roles.
        # PREFILL keys it by mooncake_session_id (the decode peer's RDMA rpc
        # endpoint); DECODE keys it by bootstrap_addr (the prefill peer's
        # bootstrap endpoint, :8998). Both are "peer id -> time it failed", and
        # the two modes are mutually exclusive (if/elif below), so a single set
        # of structures serves both without the keys ever coexisting.
        self.session_lock = threading.Lock()
        self.session_failures = defaultdict(int)
        self.failed_sessions: dict[str, float] = {}
        # TTL=0 disables the negative cache entirely (elapsed < 0 is never
        # true), which is the intended way to turn this off.
        self.failed_session_ttl = max(get_int_env_var("SGLANG_DISAGGREGATION_FAILED_SESSION_TTL", 30), 0)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._push_zmq_ctx = zmq.Context(2)
            self._push_zmq_ctx.set(zmq.MAX_SOCKETS, 1000000)
            # Per-instance PUSH socket cache (keyed by endpoint) to reuse connections
            self._push_socket_lock = threading.Lock()
            self._push_socket_cache: dict[str, Any] = {}
            self._push_socket_locks: dict[str, threading.Lock] = {}

            self.transfer_infos: dict[int, dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: dict[str, KVArgsRegisterInfo] = {}
            # Rooms that have ever been handed to a transfer worker (sync or
            # async) and are not yet cleaned up. Membership determines whether
            # _handle_abort_req can locally mark the request Failed or must
            # leave it to finish. Guarded by request_state_lock.
            self.rooms_in_transfer_queue: set[int] = set()
            self.start_prefill_thread()
            self._register_to_bootstrap()

            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.75 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.start_transfer_thread(transfer_thread_pool_size, transfer_queue_size)
            self.bootstrap_time_out = get_int_env_var("SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 1000)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.connection_lock = threading.Lock()
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0)
            # Heartbeat failure should be at least 1
            self.max_failures = max(get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1)
            self.start_decode_thread()
            self.connection_pool: dict[str, dict[str, Union[str, int]]] = {}
            self.required_prefill_response_num_table: dict[int, int] = {}
            self.prefill_response_tracker: dict[int, set[int]] = {}
            self.prefill_parallel_info_table: dict[str, tuple[int, int, bool, int]] = {}
            # The receiver handshake (HTTP to the bootstrap server + ZMQ
            # registration) used to run inline in MooncakeKVReceiver.__init__,
            # i.e. on the scheduler main loop via
            # decode.py process_input_requests() -> DecodePreallocQueue.add().
            # A prefill node that dies at the machine/network level blackholes
            # the SYN, so that handshake blocked ~131s (tcp_syn_retries=6) and
            # stalled the WHOLE decode instance, not just the one request.
            # Offload it to a small thread pool: the room stays in
            # KVPoll.Bootstrapping (which decode.py already treats as
            # "not ready yet, check again next loop") until the worker
            # finishes. Set SGLANG_DISAGGREGATION_ASYNC_HANDSHAKE=0 to fall
            # back to the old synchronous behaviour without redeploying.
            self.defer_handshake = get_int_env_var("SGLANG_DISAGGREGATION_ASYNC_HANDSHAKE", 1) != 0
            # Upper bound for a deferred handshake. Without this a request whose
            # handshake never completes would sit in Bootstrapping forever
            # (nothing else times it out now that it no longer blocks the loop).
            self.bootstrap_time_out = get_int_env_var("SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 1000)
            self.handshake_executor = None
            if self.defer_handshake:
                self.handshake_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(get_int_env_var("SGLANG_DISAGGREGATION_HANDSHAKE_THREADS", 4), 1),
                    thread_name_prefix="kv-handshake",
                )
            logger.info(
                "[handshake] defer=%s workers=%s bootstrap_time_out=%ss",
                self.defer_handshake,
                getattr(self.handshake_executor, "_max_workers", None),
                self.bootstrap_time_out,
            )
            # Decode-initiated abort state, keyed by bootstrap_room.
            # Populated by MooncakeKVReceiver.abort(), consumed by the
            # dedicated ABORT_ACK wire path (_handle_abort_ack). Presence in
            # this dict silences organic Success and short-circuits organic
            # Failed so the KV slot is only released on full ACK quorum.
            # Guarded by request_state_lock.
            self.abort_state: dict[int, MooncakeKVManager._AbortState] = {}
            # Restart anchor for poll()'s waiting_timeout loop: if an abort
            # round returned with any ack=failed, re-arm the timeout from
            # that moment so abort() can fan out another round later.
            # Guarded by request_state_lock.
            self.abort_restart_time: dict[int, float] = {}
            # Rooms whose KV indices may have reached prefill. Guarded by
            # request_state_lock together with failure and init paths.
            self.dst_info_sent_rooms: set[int] = set()
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = get_int_env_var("SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300)
        else:
            raise ValueError(f"Unsupported DisaggregationMode: {self.disaggregation_mode}")

        self.failure_lock = threading.Lock()
        self.failure_records: dict[int, str] = {}

    def _clear_failed_session(self, mooncake_session_id: str) -> None:
        with self.session_lock:
            if mooncake_session_id in self.failed_sessions:
                del self.failed_sessions[mooncake_session_id]
                logger.info(
                    "Session %s failed state cleared due to KVArgs registration.",
                    mooncake_session_id,
                )

            if mooncake_session_id in self.session_failures:
                del self.session_failures[mooncake_session_id]

    def _mark_session_failed(self, mooncake_session_id: str, reason: str = "transfer_failed") -> None:
        self.failed_sessions[mooncake_session_id] = time.monotonic()
        logger.warning(
            "Session %s marked failed (reason=%s).",
            mooncake_session_id,
            reason,
        )

    def _mark_bootstrap_failed(self, bootstrap_addr: str) -> None:
        """Thread-safe wrapper around _mark_session_failed for the decode side.

        _mark_session_failed() does NOT take session_lock itself -- it assumes
        the caller already holds it (see _fail_session_and_chunk). The decode
        handshake runs on a pool of workers, so it must lock explicitly.
        """
        with self.session_lock:
            self._mark_session_failed(bootstrap_addr, reason="bootstrap_unreachable")

    def _fail_session_and_chunk(self, mooncake_session_id, endpoint, dst_port, bootstrap_room, reason):
        with self.session_lock:
            self.session_failures[mooncake_session_id] += 1
            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
            if self.session_failures[mooncake_session_id] >= 1:
                self._mark_session_failed(
                    mooncake_session_id,
                    reason=reason,
                )
                logger.error(f"Session {mooncake_session_id} failed.")

        self.record_failure(
            bootstrap_room,
            f"Failed to send kv chunk of {bootstrap_room} to {endpoint}:{dst_port}",
        )

    def _is_session_failed(self, mooncake_session_id: str) -> bool:
        with self.session_lock:
            failed_at = self.failed_sessions.get(mooncake_session_id)
            if failed_at is None:
                return False

            elapsed = time.monotonic() - failed_at
            logger.info(
                "Session %s failed for %.2fs (TTL=%ds).",
                mooncake_session_id,
                elapsed,
                self.failed_session_ttl,
            )
            if elapsed < self.failed_session_ttl:
                return True

            del self.failed_sessions[mooncake_session_id]
            logger.info(
                "Session %s failed TTL expired (%.2fs >= %ds), reset.",
                mooncake_session_id,
                elapsed,
                self.failed_session_ttl,
            )
            return False

    def _check_session(self, mooncake_session_id, bootstrap_room):
        if self._is_session_failed(mooncake_session_id):
            logger.info(
                "Blocked transfer due to failed session (room=%s, session=%s).",
                bootstrap_room,
                mooncake_session_id,
            )
            self.record_failure(
                bootstrap_room,
                f"Decode instance could be dead, remote mooncake session {mooncake_session_id} is not alive",
            )
            return True
        return False

    def register_buffer_to_engine(self):
        device_types = getattr(self.kv_args, 'buffer_device_types', None)
        for i, (kv_data_ptr, kv_data_len) in enumerate(zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)):
            is_cpu = (device_types is not None and device_types[i] == "cpu")
            self.engine.register(kv_data_ptr, kv_data_len, is_cpu=is_cpu)

        for aux_data_ptr, aux_data_len in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            self.engine.register(aux_data_ptr, aux_data_len)

        for state_data_ptr, state_data_len in zip(self.kv_args.state_data_ptrs, self.kv_args.state_data_lens):
            self.engine.register(state_data_ptr, state_data_len)

    def _connect(self, endpoint: str):
        # Reuse sockets across calls to avoid spawning a new ZMQ IO thread per endpoint.
        # Returns (socket, per_endpoint_lock) — caller must hold the lock while sending.
        with self._push_socket_lock:
            if endpoint not in self._push_socket_cache:
                sock = self._push_zmq_ctx.socket(zmq.PUSH)
                # Prevent send_multipart from blocking forever when the peer is dead.
                # Timeout raises zmq.error.Again.
                sock.setsockopt(zmq.SNDTIMEO, 5000)  # 5 s send timeout
                # Discard unsent messages immediately on close
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                self._push_socket_cache[endpoint] = sock
                self._push_socket_locks[endpoint] = threading.Lock()
                logger.info(
                    "Added new PUSH socket for endpoint %s. _push_socket_cache size: %d",
                    endpoint,
                    len(self._push_socket_cache),
                )
            return self._push_socket_cache[endpoint], self._push_socket_locks[endpoint]

    def resolve_transfer_indices(
        self,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
    ) -> TransferIndexResolution:
        src_indices = kv_chunk.prefill_kv_indices
        dst_indices = req.dst_kv_indices[kv_chunk.index_slice]

        valid_len = min(len(src_indices), len(dst_indices))
        # Fast path: empty transfer chunk. Avoid MLA assertions/index ops on empty payload.
        if valid_len == 0:
            empty = np.array([], dtype=np.int64)
            return TransferIndexResolution(src_indices=empty, dst_indices=empty)

        if valid_len < len(src_indices) or valid_len < len(dst_indices):
            logger.warning(
                "Mismatched transfer indices, truncating to %s (src=%s, dst=%s)",
                valid_len,
                len(src_indices),
                len(dst_indices),
            )
        src_indices = src_indices[:valid_len]
        dst_indices = dst_indices[:valid_len]

        # draft pool 不被 l1.5 包装, 页号即全局页号, 故用未过 KVP mask 的原始全局页索引 (head 配对在 submit_one_cache 做)。
        draft_src = src_indices if self.draft_needs_global_indices else None
        draft_dst = dst_indices if self.draft_needs_global_indices else None

        src_mode = self.src_mode
        dst_mode = "ON" if req.dst_indices_are_local else "OFF"
        src_args = kv_chunk.mla_l1_5_args

        # Prefill OFF/Decode OFF: use original indices.
        if src_mode == "OFF" and dst_mode == "OFF":
            return TransferIndexResolution(src_indices, dst_indices, draft_src, draft_dst)

        # Prefill ON/Decode OFF: prefill-side only has partial kv cache, only send local part.
        if src_mode == "ON" and dst_mode == "OFF":
            assert src_args is not None, "Prefill MLA L1.5 cache is enabled but no transfer metadata provided"
            src_mask = src_args.page_transfer_mask
            src_local = src_args.page_local_indices
            return TransferIndexResolution(
                src_indices=src_local,
                dst_indices=dst_indices[src_mask],
                draft_src_indices=draft_src,
                draft_dst_indices=draft_dst,
            )

        # Prefill OFF/Decode ON: decode-side only has partial kv cache, only send requested part.
        if src_mode == "OFF" and dst_mode == "ON":
            assert req.dst_page_transfer_mask is not None, "OFF/ON expects decode ownership mask when destination reports local indices."
            dst_mapping = req.dst_page_indices_mapping[kv_chunk.index_slice]
            dst_mask = req.dst_page_transfer_mask[kv_chunk.index_slice]
            dst_mapping = dst_mapping[dst_mask]
            dst_local = req.dst_page_local_indices[dst_mapping]
            return TransferIndexResolution(
                src_indices=src_indices[dst_mask],
                dst_indices=dst_local,
                draft_src_indices=draft_src,
                draft_dst_indices=draft_dst,
            )

        # Prefill ON/Decode ON: both sides hold partial kv cache, find the intersection part.
        assert src_args is not None, "ON/ON expects prefill metadata (page_transfer_mask/page_local_indices)."
        assert req.dst_page_transfer_mask is not None, "ON/ON expects decode ownership mask when destination uses local index space."

        # src_args.page_transfer_mask is generated from current chunk page_indices,
        # so it is already in chunk-local coordinates.
        src_mask = src_args.page_transfer_mask[:valid_len]
        src_local = src_args.page_local_indices

        # req.dst_page_transfer_mask is request-global and should be sliced by chunk.
        dst_mask = req.dst_page_transfer_mask[kv_chunk.index_slice][:valid_len]

        # Only positions owned by both sides should be transferred.
        common_mask = src_mask & dst_mask

        dst_mapping = req.dst_page_indices_mapping[kv_chunk.index_slice][:valid_len]
        dst_mapping = dst_mapping[common_mask]
        dst_local = req.dst_page_local_indices[dst_mapping]

        src_mapping = np.cumsum(src_args.page_transfer_mask) - 1
        src_mapping = src_mapping[:valid_len]
        src_mapping = src_mapping[common_mask]
        src_local = src_args.page_local_indices[src_mapping]

        return TransferIndexResolution(
            src_indices=src_local,
            dst_indices=dst_local,
            draft_src_indices=draft_src,
            draft_dst_indices=draft_dst,
        )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths))

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        draft_prefill_kv_indices: npt.NDArray[np.int64] | None = None,
        draft_dst_kv_indices: npt.NDArray[np.int64] | None = None,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(prefill_kv_indices, dst_kv_indices)

        num_layers = len(self.kv_args.kv_data_ptrs)
        # draft 层的 ptr 下标从 offsets 尾部 draft_layer_num 条推导, 不假设每层 buffer 个数 (target MLA split-kv 3 个, draft GQA 2 个)。
        draft_ptr_ids: set = set()
        if self.draft_needs_global_indices and draft_prefill_kv_indices is not None and self.kv_args.draft_layer_num > 0:
            for grp in self.kv_args.offsets[-self.kv_args.draft_layer_num :]:
                draft_ptr_ids.update(int(o) for o in grp)
            draft_prefill_blocks, draft_dst_blocks = group_concurrent_contiguous(draft_prefill_kv_indices, draft_dst_kv_indices)
        else:
            draft_prefill_blocks, draft_dst_blocks = prefill_kv_blocks, dst_kv_blocks
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id + self.kv_args.prefill_start_layer],
                self.kv_args.kv_item_lens[layer_id],
                layer_id in draft_ptr_ids,
            )
            for layer_id in range(num_layers)
        ]

        def set_transfer_blocks(src_ptr: int, dst_ptr: int, item_len: int, is_draft: bool) -> list[tuple[int, int, int]]:
            # 同 async 路径: draft 按 head 分片, 非同号 rank 无交集
            if is_draft and self.draft_needs_head_match:
                reg = self.decode_kv_args_table[mooncake_session_id]
                if reg.dst_attn_tp_size > 0:
                    if reg.dst_attn_tp_size != self.attn_tp_size:
                        raise RuntimeError(
                            f"[DFLASH] PD 两侧 attn_tp 不等 (prefill={self.attn_tp_size}, "
                            f"decode={reg.dst_attn_tp_size}), GQA draft 按 head 分片, 暂不支持重分片。"
                        )
                    if self.attn_tp_rank != reg.dst_attn_tp_rank:
                        return []
            src_blocks, dst_blocks = (draft_prefill_blocks, draft_dst_blocks) if is_draft else (prefill_kv_blocks, dst_kv_blocks)
            transfer_blocks = []
            for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: list[tuple[int, int, int, bool]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len, is_draft in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len, is_draft))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        return process_layers(layers_params)

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(dst_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: list[int],
        dst_state_data_ptrs: list[int],
    ):
        """Send state or extra pool data with type-specific handling."""
        state_type = getattr(self.kv_args, "state_type", "none")
        if state_type == "none":
            return 0

        if state_type == "mamba":
            # Check if we need slice transfer for different TP sizes
            # if (
            #     target_rank_registration_info is not None
            #     and self.attn_tp_size != target_rank_registration_info.dst_attn_tp_size
            # ):
            #     assert False, "P and D should use the same TP when Mamba"
            # return self._send_mamba_state_slice(
            #     req,
            #     prefill_state_indices,
            #     dst_state_data_ptrs,
            #     ...
            # )
            return self._send_mamba_state(
                req,
                prefill_state_indices,
                dst_state_data_ptrs,
            )

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list[int],
        dst_state_data_ptrs: list[int],
    ):
        """Transfer Mamba states."""
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = prefill_state_item_lens[i]
            src_addr = prefill_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(req.dst_state_indices[0])
            transfer_blocks.append((src_addr, dst_addr, length))

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(req.mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths))

    def sync_status_to_decode_endpoint(self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        endpoint = "tcp://" + remote + ":" + str(dst_port)
        payload = [
            str(room).encode("ascii"),
            str(status).encode("ascii"),
            str(prefill_rank).encode("ascii"),
        ]
        sock, lock = self._connect(endpoint)
        try:
            with lock:
                sock.send_multipart(payload)
        except Exception as e:
            logger.warning(f"Failed to sync status to {endpoint} (room={room}), evicting socket and retrying: {e}")

    def transfer_worker(self, queue: FastQueue):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                logger.debug(
                    f"[TRANSFER_WORKER] Got transfer request for room {kv_chunk.room}, "
                    f"is_last={kv_chunk.is_last}, "
                    f"kv_indices_len={len(kv_chunk.prefill_kv_indices)}"
                )
                reqs_to_be_processed = self.transfer_infos[kv_chunk.room].values() if kv_chunk.room in self.transfer_infos else []
                polls = []
                dst_ranks_infos = []
                # Unique id per prefill sender so decode's response set size matches expected_response_num.
                prefill_unique_rank = self.attn_tp_size * self.pp_rank + self.attn_tp_rank
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        failed = self._check_session(req.mooncake_session_id, kv_chunk.room)
                        if failed:
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                prefill_unique_rank,
                            )
                            break

                        resolved = self.resolve_transfer_indices(kv_chunk, req)

                        target_rank_registration_info: KVArgsRegisterInfo = self.decode_kv_args_table[req.mooncake_session_id]

                        logger.debug(f"[TRANSFER_WORKER] Calling send_kvcache for room {kv_chunk.room}, session {req.mooncake_session_id}")
                        tm_start = time.monotonic()
                        ret = self.send_kvcache(
                            req.mooncake_session_id,
                            resolved.src_indices,
                            self.decode_kv_args_table[req.mooncake_session_id].dst_kv_ptrs,
                            resolved.dst_indices,
                            resolved.draft_src_indices,
                            resolved.draft_dst_indices,
                        )
                        logger.debug(f"[TRANSFER_WORKER] send_kvcache returned {ret} for room {kv_chunk.room}")
                        if ret != 0:
                            self._fail_session_and_chunk(
                                req.mooncake_session_id,
                                req.endpoint,
                                req.dst_port,
                                kv_chunk.room,
                                reason="send_kvcache",
                            )

                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(req.endpoint, req.dst_port, req.room, KVPoll.Failed, prefill_unique_rank)
                            break

                        if kv_chunk.is_last:
                            # Only the last chunk we need to send the aux data
                            self.maybe_send_extra(
                                req,
                                kv_chunk.state_indices,
                                target_rank_registration_info.dst_state_data_ptrs,
                            )
                            if self.pp_rank == self.pp_size - 1:
                                ret = self.send_aux(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_aux_index,
                                    self.decode_kv_args_table[req.mooncake_session_id].dst_aux_ptrs,
                                    req.dst_aux_index,
                                )
                                polls.append(ret == 0)
                            else:
                                polls.append(True)
                            dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(endpoint, dst_port, room, status, prefill_unique_rank)
                        elapsed_ms = time.monotonic() - tm_start
                        if self.kv_transfer_metrics:
                            self.kv_transfer_metrics.log_kv_transfer_time(elapsed_ms)
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if kv_chunk.room not in self.request_status or self.check_status(kv_chunk.room) == KVPoll.Success:
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)
                    with self.request_state_lock:
                        self.rooms_in_transfer_queue.discard(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                ) from e

    def start_prefill_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def init_pp0_sock(zmq_context: zmq.Context):
            """PP1~PPn fetch PP0's rank_ip:rank_port from bootstrap server and connect."""
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0]) if self.dist_init_addr else get_ip()
            bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"

            url = f"http://{bootstrap_server_url}/route?tp_rank={self.attn_tp_rank}&target_dp_rank={self.dp_rank}&target_pp_rank=0"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch PP0 address from bootstrap server: {response.status_code}")
            info = response.json()
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            logger.info(f"PP{self.pp_rank} fetched PP0 address: {info['rank_ip']}:{info['rank_port']}")
            sock = zmq_context.socket(zmq.PUSH)
            sock.connect(endpoint)
            return sock

        def send_pp_bootstrap_confirmation(zmq_context, room: int, mooncake_session_id: str):
            """PP1~PPn send bootstrap confirmation to PP0 (lazy init socket)."""
            if not hasattr(self, "pp0_socket"):
                self.pp0_socket = init_pp0_sock(zmq_context)
            try:
                self.pp0_socket.send_multipart(
                    [
                        str(room).encode("ascii"),
                        b"PP_BOOTSTRAP_CONFIRM",
                        str(self.pp_rank).encode("ascii"),
                        mooncake_session_id.encode("ascii"),
                    ]
                )
            except Exception as e:
                logger.warning(f"Failed to send_pp_bootstrap_confirmation {room=} {self.pp_rank=}: {e}")

        def try_mark_waiting_for_input(room, pp_bootstrap_confirmations, pp_decode_infos_enough):
            """PP0: mark WaitingForInput when both decode infos and PP confirmations are complete."""
            if not pp_decode_infos_enough.get(room, False):
                return
            if len(pp_bootstrap_confirmations.get(room, set())) < self.pp_size - 1:
                return
            # All decode infos + all PP confirmations received for this room
            self.update_status(room, KVPoll.WaitingForInput)
            pp_decode_infos_enough.pop(room, None)
            pp_bootstrap_confirmations.pop(room, None)

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            zmq_context = zmq.Context()
            # room -> set of confirmed pp_ranks
            pp_bootstrap_confirmations: dict[int, set] = {}
            # room -> bool whether all decode infos received
            pp_decode_infos_enough: dict[int, bool] = {}

            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    self._clear_failed_session(mooncake_session_id)
                    logger.debug(f"Register KVArgs from {mooncake_session_id} successfully")
                    continue
                else:
                    room = int(room)
                    msg_tag = waiting_req_bytes[1] if len(waiting_req_bytes) > 1 else b""
                    msg_tag = msg_tag.decode("ascii")
                    if msg_tag == "PP_BOOTSTRAP_CONFIRM":
                        assert self.pp_rank == 0
                        peer_pp_rank = int(waiting_req_bytes[2].decode("ascii"))
                        if room not in pp_bootstrap_confirmations:
                            pp_bootstrap_confirmations[room] = set()
                        pp_bootstrap_confirmations[room].add(peer_pp_rank)
                    elif msg_tag == "ABORT_REQ":
                        decode_endpoint = waiting_req_bytes[2].decode("ascii")
                        decode_dst_port = int(waiting_req_bytes[4].decode("ascii"))
                        self._handle_abort_req(room, decode_endpoint, decode_dst_port)
                        continue
                    else:
                        required_dst_info_num = int(waiting_req_bytes[6].decode("ascii"))
                        if room not in self.transfer_infos:
                            self.transfer_infos[room] = {}
                        self.transfer_infos[room][mooncake_session_id] = TransferInfo.from_zmq(waiting_req_bytes)

                        if len(self.transfer_infos[room]) == required_dst_info_num:
                            if self.pp_rank == 0:
                                pp_decode_infos_enough[room] = True
                            else:
                                self.update_status(room, KVPoll.WaitingForInput)
                                send_pp_bootstrap_confirmation(zmq_context, room, "")

                    if self.pp_rank == 0:
                        try_mark_waiting_for_input(room, pp_bootstrap_confirmations, pp_decode_infos_enough)

        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def start_transfer_thread(self, transfer_thread_pool_size: int, transfer_queue_size: int):
        self.transfer_queues: list[FastQueue] = [FastQueue() for _ in range(transfer_queue_size)]
        for queue in self.transfer_queues:
            threading.Thread(target=self.transfer_worker, args=(queue,), daemon=True).start()

    def start_decode_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def decode_thread():
            while True:
                (bootstrap_room, status, prefill_rank) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status in (ABORT_ACK_OK, ABORT_ACK_FAIL):
                    self._handle_abort_ack(bootstrap_room, prefill_rank, status == ABORT_ACK_OK)
                    continue

                # While a decode-initiated abort is in flight for this room,
                # organic Success must be ignored (user has abandoned the
                # request) and organic Failed takes precedence over the ACK
                # quorum. Only when no abort is pending do we fall through to
                # the normal accounting path.
                if self._consume_status_during_abort(bootstrap_room, status, prefill_rank):
                    continue

                handled = self.handle_decode_status_message(bootstrap_room, status, prefill_rank)
                if not handled and status in (KVPoll.Success, KVPoll.Failed):
                    logger.debug(
                        "Ignoring stale decode status room=%s status=%s prefill_rank=%s",
                        bootstrap_room,
                        status,
                        prefill_rank,
                    )

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_parallel_info_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[bootstrap_addr].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the map
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(bootstrap_room)
                        else:
                            logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                            self.heartbeat_failures[bootstrap_addr] = self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = self.heartbeat_failures.get(bootstrap_addr, 0) + 1

                    if self.heartbeat_failures.get(bootstrap_addr, 0) >= self.max_failures:
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        state_indices: Optional[list[int]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)
        if bootstrap_room not in self.request_status or self.check_status(bootstrap_room) == KVPoll.Failed:
            logger.debug("Request with bootstrap_room=%s already failed", bootstrap_room)
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        with self.request_state_lock:
            self.rooms_in_transfer_queue.add(bootstrap_room)
        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                mla_l1_5_args=mla_l1_5_args,
                state_indices=state_indices,
            )
        )

    def _update_status_unlocked(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            elif self.request_status[bootstrap_room] != KVPoll.Failed:
                self.request_status[bootstrap_room] = max(self.request_status[bootstrap_room], status)

    def register_decode_response_tracker(self, bootstrap_room: int, required_prefill_response_num: int):
        with self.request_state_lock:
            self.required_prefill_response_num_table[bootstrap_room] = required_prefill_response_num
            self.prefill_response_tracker[bootstrap_room] = set()

    def clear_request_state(self, bootstrap_room: int):
        with self.request_state_lock:
            self.request_status.pop(bootstrap_room, None)
            self.rooms_in_transfer_queue.discard(bootstrap_room)

    def clear_decode_response_state(self, bootstrap_room: int):
        with self.request_state_lock:
            self.request_status.pop(bootstrap_room, None)
            self.required_prefill_response_num_table.pop(bootstrap_room, None)
            self.prefill_response_tracker.pop(bootstrap_room, None)
            self.abort_state.pop(bootstrap_room, None)
            self.abort_restart_time.pop(bootstrap_room, None)
            self.dst_info_sent_rooms.discard(bootstrap_room)

    def _consume_status_during_abort(self, bootstrap_room: int, status: int, prefill_rank: int) -> bool:
        """Intercept sync_status while a decode abort is in flight.

        Returns True if the message was consumed here (caller should not
        dispatch to handle_decode_status_message). Returns False when no
        abort is pending — caller proceeds with normal accounting.
        """
        with self.request_state_lock:
            if bootstrap_room not in self.abort_state:
                return False
            current_status = self.request_status.get(bootstrap_room)
            if current_status is None:
                return True
            if status == KVPoll.Success:
                # User already abandoned the request; ignore.
                return True
            if status == KVPoll.Failed:
                if current_status == KVPoll.Success:
                    return True
                self.record_failure(
                    bootstrap_room,
                    "Prefill reported Failed during decode abort",
                )
                self._update_status_unlocked(bootstrap_room, KVPoll.Failed)
                self.abort_state.pop(bootstrap_room, None)
                self.abort_restart_time.pop(bootstrap_room, None)
                return True
            return True

    def _handle_abort_ack(self, bootstrap_room: int, prefill_rank: int, ack_ok: bool) -> None:
        """Consume an ABORT_ACK_OK / ABORT_ACK_FAIL reply.

        * All ranks ack_ok  → every prefill rank has guaranteed it will not
          touch the decode buffer; finalize the room as Failed.
        * Any rank ack_fail → that rank is still transferring; reset the
          waiting_timeout anchor so poll() will re-fan-out a fresh round
          later, but do NOT terminate the room.
        * Missing abort_state entry → ACK arrived after the room already
          concluded (organic Failed, heartbeat, etc.); drop silently.
        """
        with self.request_state_lock:
            state = self.abort_state.get(bootstrap_room)
            if state is None:
                return
            if ack_ok:
                state.ack_success.add(prefill_rank)
            else:
                state.ack_failed.add(prefill_rank)
            if state.ack_count < state.expected:
                return
            if not state.ack_failed:
                self.record_failure(
                    bootstrap_room,
                    "Abort due to timeout",
                )
                self._update_status_unlocked(bootstrap_room, KVPoll.Failed)
                self.abort_state.pop(bootstrap_room, None)
                self.abort_restart_time.pop(bootstrap_room, None)
            else:
                # Some rank is still transferring. Arm a new waiting window
                # so poll() can fan out another ABORT_REQ round on timeout.
                self.abort_restart_time[bootstrap_room] = time.time()
                state.ack_success.clear()
                state.ack_failed.clear()

    def handle_decode_status_message(self, bootstrap_room: int, status: int, prefill_rank: int) -> bool:
        with self.request_state_lock:
            current_status = self.request_status.get(bootstrap_room)
            if current_status is None:
                return False

            if status == KVPoll.Success:
                if current_status in (KVPoll.Failed, KVPoll.Success):
                    return False

                expected_response_num = self.required_prefill_response_num_table.get(bootstrap_room)
                responses = self.prefill_response_tracker.get(bootstrap_room)
                if expected_response_num is None or responses is None:
                    return False

                responses.add(prefill_rank)
                if len(responses) == expected_response_num:
                    self._update_status_unlocked(bootstrap_room, KVPoll.Success)
                return True

            if status == KVPoll.Failed:
                if current_status == KVPoll.Success:
                    return False
                self.record_failure(
                    bootstrap_room,
                    "Failed to get kvcache from prefill instance, it might be dead",
                )
                self._update_status_unlocked(bootstrap_room, KVPoll.Failed)
                return True

            return False

    def _handle_abort_req(self, bootstrap_room: int, decode_endpoint: str, decode_dst_port: int) -> None:
        """Handle a decode-initiated ABORT_REQ on the prefill side.

        Reuses sync_status_to_decode_endpoint; the ack flavour is encoded in
        the status slot as ABORT_ACK_OK / ABORT_ACK_FAIL (outside KVPoll's
        0..4 range so decode_thread can dispatch unambiguously).

          ack_ok   — prefill guarantees it will not write into decode's
                     buffer for this room anymore: already Success, already
                     Failed, cleared, or still in WaitingForInput/
                     Bootstrapping (flipped to Failed in-place so
                     transfer_worker never picks it up).
          ack_fail — the room is in the sync/async transfer queue; we do
                     not interrupt it. Decode will re-fan-out on its next
                     waiting_timeout.
        """
        prefill_unique_rank = self.attn_tp_size * self.pp_rank + self.attn_tp_rank
        with self.request_state_lock:
            current = self.request_status.get(bootstrap_room)
            in_transfer = bootstrap_room in self.rooms_in_transfer_queue
            if current is None or current == KVPoll.Success or current == KVPoll.Failed:
                ack_status = ABORT_ACK_OK
            elif in_transfer:
                ack_status = ABORT_ACK_FAIL
            else:
                self.record_failure(
                    bootstrap_room,
                    "Decode requested abort",
                )
                self._update_status_unlocked(bootstrap_room, KVPoll.Failed)
                ack_status = ABORT_ACK_OK
        try:
            self.sync_status_to_decode_endpoint(
                decode_endpoint,
                decode_dst_port,
                bootstrap_room,
                ack_status,
                prefill_unique_rank,
            )
        except Exception as e:
            logger.warning(f"Failed to send ABORT_ACK for room={bootstrap_room} to {decode_endpoint}:{decode_dst_port}: {e}")

    def check_status(self, bootstrap_room: int):
        with self.request_state_lock:
            return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        with self.request_state_lock:
            self._update_status_unlocked(bootstrap_room, status)

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def submit_handshake(self, receiver):
        """Run receiver._do_handshake() on the handshake thread pool.

        The room is already in KVPoll.Bootstrapping when we get here, so the
        scheduler just keeps polling until the worker flips it to
        WaitingForInput (success) or Failed (any error).
        """

        def _run():
            try:
                receiver._do_handshake()
            except BaseException as e:
                # BaseException on purpose: NPU / C++ extensions do not always
                # raise Exception subclasses, and a swallowed error here would
                # leave the room stuck in Bootstrapping until bootstrap_time_out.
                logger.error(
                    "[handshake] failed room=%s addr=%s: %r",
                    receiver.bootstrap_room,
                    receiver.bootstrap_addr,
                    e,
                )
                try:
                    receiver._fail_handshake(f"handshake raised: {e!r}")
                except BaseException as e2:
                    logger.error("[handshake] could not mark room Failed: %r", e2)

        try:
            self.handshake_executor.submit(_run)
        except BaseException as e:
            # Pool rejected the task (shutting down / saturated queue). Do not
            # leave the room hanging in Bootstrapping -- fail it right here.
            logger.error("[handshake] submit rejected room=%s: %r", receiver.bootstrap_room, e)
            receiver._fail_handshake(f"handshake submit failed: {e!r}")

    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        # In mooncake implementation, decode_prefix_len is handled via ZMQ messages
        # Check the stored transfer info for this room
        if bootstrap_room in self.transfer_infos:
            for transfer_info in self.transfer_infos[bootstrap_room].values():
                if hasattr(transfer_info, "decode_prefix_len") and transfer_info.decode_prefix_len > 0:
                    logger.debug(f"Found decode_prefix_len={transfer_info.decode_prefix_len} for room {bootstrap_room}")
                    return transfer_info.decode_prefix_len
        logger.debug(f"No decode_prefix_len found for room {bootstrap_room}, using 0")
        return 0

    def get_session_id(self):
        return self.engine.get_session_id()

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0]) if self.dist_init_addr else get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "tp_rank": self.kv_args.tp_rank,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "enable_mla_l1_5_cache": self.server_args.enable_mla_l1_5_cache,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            # Per-pp-rank layer ownership, so a decode (PP1) writing cross-PP L3
            # objects can slice each pp{i}of{N}_ to this rank's exact layer range
            # (supports non-even PP layer splits, not just floor division).
            "prefill_start_layer": self.kv_args.prefill_start_layer,
            "prefill_end_layer": self.kv_args.prefill_end_layer,
        }

        logger.info(f"Sending POST request to bootstrap server: {url} {payload}")

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Prefill instance failed to register to bootstrap server: {e}")

    def _handle_node_failure(self, failed_bootstrap_addr):
        endpoints_to_close = []
        with self.connection_lock:
            keys_to_remove = [k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)]
            for k in keys_to_remove:
                for info in self.connection_pool[k]:
                    endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                    endpoints_to_close.append(endpoint)
            for k in keys_to_remove:
                for info in self.connection_pool[k]:
                    endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                    endpoints_to_close.append(endpoint)
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_parallel_info_table:
                del self.prefill_parallel_info_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(failed_bootstrap_addr, [])
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        if endpoints_to_close:
            MooncakeKVReceiver._close_sockets(endpoints_to_close)

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if room in self.request_status and self.check_status(room) != KVPoll.Success:
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                # Heartbeat is the ultimate fallback when prefill dies mid-
                # abort. Clean abort bookkeeping so the room can be released
                # even though no prefill rank will ever ACK.
                with self.request_state_lock:
                    self.abort_state.pop(room, None)
                    self.abort_restart_time.pop(room, None)
                affected_rooms.append(room)
        logger.error(f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests")


@dataclasses.dataclass
class WriteRequest:
    trans_info: TransferInfo
    dst_ranks_info: tuple[str, int, int]
    prefill_kv_blocks: npt.NDArray[np.int64]
    dst_kv_blocks: npt.NDArray[np.int64]
    # DFLASH: draft 层专用的全局页坐标分组 (未过 KVP mask)。None 表示与 target 同坐标系。
    draft_prefill_kv_blocks: npt.NDArray[np.int64] | None = None
    draft_dst_kv_blocks: npt.NDArray[np.int64] | None = None
    submit_bids: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LayerWiseTask:
    kv_chunk: TransferKVChunk
    begin_cache_step: int
    aux_step: int
    next_layer_id: int = 0
    write_requests: list[WriteRequest] = dataclasses.field(default_factory=list)
    polls: list[bool] = dataclasses.field(default_factory=list)


class MooncakeAsyncKVManager(MooncakeKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        draft_is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend, draft_is_mla_backend)
        self.target_layer_num = self.kv_args.target_layer_num
        self.draft_layer_num = self.kv_args.draft_layer_num
        self.prefill_start_layer = self.kv_args.prefill_start_layer
        self.is_mla_backend = is_mla_backend
        self.is_last_pp_rank = self.kv_args.pp_rank == self.pp_size - 1
        if self.is_last_pp_rank:
            self.effective_layer_num = self.kv_args.prefill_end_layer - self.kv_args.prefill_start_layer + self.draft_layer_num
        else:
            self.effective_layer_num = self.kv_args.prefill_end_layer - self.kv_args.prefill_start_layer
        self.draft_is_mla_backend = draft_is_mla_backend
        self.submit_interval = server_args.disaggregation_layerwise_interval
        assert self.submit_interval > 0, "submit_interval must be positive"
        self.current_transfer_batch: list[tuple[TransferKVChunk, int]] = []
        # Pool construction has finished; this stage's logical layout is fixed.
        self.offsets = tuple(tuple(row) for row in args.offsets)
        self._checked_stage_registrations = {}

    def start_transfer_thread(self, transfer_thread_pool_size: int, transfer_queue_size: int):
        self.transfer_queues: list[FastQueue] = [FastQueue()]
        for queue in self.transfer_queues:
            threading.Thread(target=self.async_transfer_worker, args=(queue,), daemon=True).start()

    def register_step_counter(self, step_counter: StepCounter):
        self.step_counter = step_counter

    @contextmanager
    def add_batch(self, is_idle: bool):
        yield  # add transfer request

        if not is_idle:
            begin_cache_step, begin_aux_step = self.step_counter.current_step()
            for kv_chunk, shard_idx in self.current_transfer_batch:
                self.transfer_queues[shard_idx].put(
                    LayerWiseTask(
                        kv_chunk=kv_chunk,
                        begin_cache_step=begin_cache_step,
                        aux_step=begin_aux_step if kv_chunk.is_last and self.is_last_pp_rank else None,
                    )
                )

            # advance to step of next batch no matter if batch is empty
            delta_aux_step = 1 if self.is_last_pp_rank else 0
            self.step_counter.advance_step(delta_cache_step=self.effective_layer_num, delta_aux_step=delta_aux_step)

        self.current_transfer_batch.clear()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        mla_l1_5_args: Optional[tuple[Any, Any]] = None,
        state_indices: Optional[list[int]] = None,
    ):
        logger.debug("async manager add_transfer_request")
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if bootstrap_room not in self.request_status or self.check_status(bootstrap_room) == KVPoll.Failed:
            logger.debug("Request with bootstrap_room=%s already failed", bootstrap_room)
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        kv_chunk = TransferKVChunk(
            room=bootstrap_room,
            prefill_kv_indices=kv_indices,
            index_slice=index_slice,
            is_last=is_last,
            prefill_aux_index=aux_index,
            mla_l1_5_args=mla_l1_5_args,
            state_indices=state_indices,
        )
        with self.request_state_lock:
            self.rooms_in_transfer_queue.add(bootstrap_room)
        self.current_transfer_batch.append((kv_chunk, shard_idx))

    def submit_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):
        # Submit transfer for all aux buffers (output_ids, logprobs, cached_tokens, etc.)
        batch_ids = []
        for aux_data_ptr, aux_item_len, dst_aux_ptr in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_item_lens, dst_aux_ptrs):
            prefill_aux_addr = aux_data_ptr + prefill_aux_index * aux_item_len
            decode_aux_addr = dst_aux_ptr + dst_aux_index * aux_item_len
            bid = self.engine.transfer_submit_write(mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len)
            batch_ids.append(bid)
        # Return the last batch_id for compatibility (or could return list)
        return batch_ids[-1] if batch_ids else -1

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths))

    def maybe_submit_extra(
        self,
        mooncake_session_id: str,
        prefill_state_indices: int,
        dst_state_data_ptrs: list[int],
        dst_state_indices: list[int],
    ):
        state_type = getattr(self.kv_args, "state_type", "none")
        if state_type == "none":
            return []

        batch_ids = []
        if state_type == "mamba":
            prefill_state_data_ptrs = self.kv_args.state_data_ptrs
            prefill_state_item_lens = self.kv_args.state_item_lens

            for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
                length = prefill_state_item_lens[i]
                src_addr = prefill_state_data_ptrs[i] + length * int(prefill_state_indices[0])
                dst_addr = dst_state_ptr + length * int(dst_state_indices[0])
                bid = self.engine.transfer_submit_write(mooncake_session_id, src_addr, dst_addr, length)
                batch_ids.append(bid)

        return batch_ids

    def submit_layer_cache(
        self,
        mooncake_session_id: str,
        begin_layer_id: int,  # [begin_layer_id, end_layer_id)
        end_layer_id: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
        prefill_start_layer: int,
        draft_prefill_kv_blocks: npt.NDArray[np.int64] | None = None,
        draft_dst_kv_blocks: npt.NDArray[np.int64] | None = None,
    ) -> int:
        registration = self.decode_kv_args_table[mooncake_session_id]
        dst_kv_ptrs = registration.dst_kv_ptrs
        dst_attn_tp_rank = registration.dst_attn_tp_rank
        dst_attn_tp_size = registration.dst_attn_tp_size
        if (prefill_start_layer != self.prefill_start_layer
                or not 0 <= begin_layer_id <= end_layer_id <= self.effective_layer_num):
            raise ValueError("Layerwise transfer does not match this P stage's global range")
        cached = self._checked_stage_registrations.get(mooncake_session_id)
        if cached is None or cached[0] is not registration:
            stage_offsets = check_stage_transfer_layout(
                self.offsets, registration.dst_offsets, self.prefill_start_layer,
                self.effective_layer_num,
                min(len(self.kv_args.kv_data_ptrs), len(self.kv_args.kv_item_lens)),
                len(dst_kv_ptrs),
            )
            # A new wire registration creates a new object and is rechecked.
            # Keep immutable validated rows, not references to mutable wire lists.
            self._checked_stage_registrations[mooncake_session_id] = (registration, stage_offsets)
        else:
            stage_offsets = cached[1]
        transfer_blocks = []

        def submit_one_cache(ptr_offset, dst_ptr_offset, src_blocks, dst_blocks, is_draft=False):
            src_ptr = self.kv_args.kv_data_ptrs[ptr_offset]
            dst_ptr = dst_kv_ptrs[dst_ptr_offset]
            item_len = self.kv_args.kv_item_lens[ptr_offset]
            # draft 按 attn_tp 分 head, 每 rank 只持一段; 不配对会让各 rank 的分片写同一地址互相覆盖
            if is_draft and self.draft_needs_head_match and dst_attn_tp_size > 0:
                if dst_attn_tp_size != self.attn_tp_size:
                    raise RuntimeError(
                        f"[DFLASH] PD 两侧 attn_tp 不等 (prefill={self.attn_tp_size}, "
                        f"decode={dst_attn_tp_size}), GQA draft 按 head 分片, 暂不支持重分片。"
                    )
                if self.attn_tp_rank != dst_attn_tp_rank:
                    return
            for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))

        # effective_layer_num 尾部 draft_layer_num 条是 draft 层, 用 draft_* 那组坐标。
        draft_begin = self.effective_layer_num - self.draft_layer_num if self.draft_layer_num > 0 else end_layer_id
        use_draft_coords = self.draft_needs_global_indices and draft_prefill_kv_blocks is not None
        for layer_id in range(begin_layer_id, end_layer_id):
            is_draft_layer = layer_id >= draft_begin
            if is_draft_layer and use_draft_coords:
                src_blocks, dst_blocks = draft_prefill_kv_blocks, draft_dst_kv_blocks
            else:
                src_blocks, dst_blocks = prefill_kv_blocks, dst_kv_blocks
            src_layer_offsets, dst_layer_offsets = stage_offsets[layer_id]
            check_transfer_layout(src_layer_offsets, dst_layer_offsets, layer_id + prefill_start_layer)
            for offset, dst_offset in zip(src_layer_offsets, dst_layer_offsets):
                submit_one_cache(offset, dst_offset, src_blocks, dst_blocks, is_draft=is_draft_layer)
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def async_transfer_worker(self, transfer_queue: FastQueue):

        def discard_tasks(tasks: list[LayerWiseTask], droped: list[LayerWiseTask]) -> list[LayerWiseTask]:
            droped_rooms = set(id(task) for task in droped)
            return [task for task in tasks if id(task) not in droped_rooms]

        def query_ready_step(task: LayerWiseTask) -> tuple[int, int]:
            if task.next_layer_id < self.effective_layer_num:
                ready_cache_step = self.step_counter.query_ready_cache_step()
                if not StepCounter.is_step_ready(ready_cache_step, task.begin_cache_step + task.next_layer_id):
                    time.sleep(1e-3)
            elif task.kv_chunk.is_last:
                ready_aux_step = self.step_counter.query_ready_aux_step()
                if not StepCounter.is_step_ready(ready_aux_step, task.aux_step):
                    time.sleep(1e-3)

            ready_cache_step = self.step_counter.query_ready_cache_step()
            ready_aux_step = self.step_counter.query_ready_aux_step()
            return ready_cache_step, ready_aux_step

        def get_new_tasks(blocking: bool) -> list[LayerWiseTask]:
            new_tasks: list[LayerWiseTask] = []
            if blocking:
                new_tasks.append(transfer_queue.get())
            while True:
                try:
                    new_tasks.append(transfer_queue.get_nowait())
                except FastQueue.EmptyError:
                    break

            return new_tasks

        def initialize(tasks: list[LayerWiseTask]) -> list[LayerWiseTask]:
            abort_tasks: list[LayerWiseTask] = []

            for task in tasks:
                kv_chunk = task.kv_chunk
                reqs_to_be_processed = self.transfer_infos[kv_chunk.room].values() if kv_chunk.room in self.transfer_infos else []

                for req in reqs_to_be_processed:
                    if req.is_dummy:
                        task.polls.append(True)
                        abort_tasks.append(task)
                        break

                    failed = self._check_session(req.mooncake_session_id, kv_chunk.room)
                    if failed:
                        task.polls.append(False)
                        abort_tasks.append(task)
                        break

                    resolved = self.resolve_transfer_indices(kv_chunk, req)

                    # Group by indices
                    prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(resolved.src_indices, resolved.dst_indices)
                    # draft 层用全局页坐标独立分组 (未过 KVP mask), 与 target 的 l1.5 局部坐标分开。
                    draft_prefill_kv_blocks, draft_dst_kv_blocks = None, None
                    if self.draft_needs_global_indices and resolved.draft_src_indices is not None:
                        draft_prefill_kv_blocks, draft_dst_kv_blocks = group_concurrent_contiguous(
                            resolved.draft_src_indices, resolved.draft_dst_indices
                        )
                    task.write_requests.append(
                        WriteRequest(
                            trans_info=req,
                            dst_ranks_info=(req.endpoint, req.dst_port, req.room),
                            prefill_kv_blocks=prefill_kv_blocks,
                            dst_kv_blocks=dst_kv_blocks,
                            draft_prefill_kv_blocks=draft_prefill_kv_blocks,
                            draft_dst_kv_blocks=draft_dst_kv_blocks,
                        )
                    )

            return abort_tasks

        def handle_failed_task(req, task: LayerWiseTask, abort_tasks):
            logger.error("Transfer failed detected!")

            self._fail_session_and_chunk(
                req.trans_info.mooncake_session_id,
                req.trans_info.endpoint,
                req.trans_info.dst_port,
                task.kv_chunk.room,
                reason="submit_layer_cache",
            )

            task.polls.append(False)
            abort_tasks.append(task)

        def submit_transfer(
            tasks: list[LayerWiseTask], ready_cache_step: int, ready_aux_step: int
        ) -> tuple[list[LayerWiseTask], list[LayerWiseTask]]:
            abort_tasks: list[LayerWiseTask] = []
            complete_tasks: list[LayerWiseTask] = []

            for task in tasks:
                is_task_failed = False
                kv_chunk = task.kv_chunk
                # submit layer cache
                if task.next_layer_id < self.effective_layer_num and StepCounter.is_step_ready(
                    ready_cache_step, task.begin_cache_step + task.next_layer_id
                ):
                    for req in task.write_requests:
                        if (task.next_layer_id + 1) % self.submit_interval == 0 or task.next_layer_id == self.effective_layer_num - 1:
                            ret = self.submit_layer_cache(
                                req.trans_info.mooncake_session_id,
                                (task.next_layer_id // self.submit_interval) * self.submit_interval,
                                task.next_layer_id + 1,
                                req.prefill_kv_blocks,
                                req.dst_kv_blocks,
                                self.prefill_start_layer,
                                req.draft_prefill_kv_blocks,
                                req.draft_dst_kv_blocks,
                            )

                            if ret != 0:
                                handle_failed_task(req, task, abort_tasks)
                                is_task_failed = True
                                break

                    task.next_layer_id += 1

                # submit aux data
                if not is_task_failed and kv_chunk.is_last and task.aux_step is not None and StepCounter.is_step_ready(ready_aux_step, task.aux_step):
                    task.aux_step = None  # reset to None to mark aux has been submitted
                    for req in task.write_requests:
                        ret = self.send_aux(
                            req.trans_info.mooncake_session_id,
                            kv_chunk.prefill_aux_index,
                            self.decode_kv_args_table[req.trans_info.mooncake_session_id].dst_aux_ptrs,
                            req.trans_info.dst_aux_index,
                        )

                        if ret != 0:
                            handle_failed_task(req, task, abort_tasks)
                            is_task_failed = True
                            break

                        target_rank_registration_info: KVArgsRegisterInfo = self.decode_kv_args_table[req.trans_info.mooncake_session_id]
                        ret = self.maybe_send_extra(
                            req.trans_info,
                            kv_chunk.state_indices,
                            target_rank_registration_info.dst_state_data_ptrs,
                        )
                        if ret != 0:
                            handle_failed_task(req, task, abort_tasks)
                            is_task_failed = True
                            break

                if not is_task_failed and task.next_layer_id == self.effective_layer_num and task.aux_step is None:
                    complete_tasks.append(task)

            return complete_tasks, abort_tasks

        def finalize(tasks: list[LayerWiseTask]) -> None:
            for task in tasks:
                kv_chunk = task.kv_chunk
                status = KVPoll.Success if all(task.polls) else KVPoll.Failed
                if status == KVPoll.Failed or kv_chunk.is_last:  # last chunk or any failed
                    self.update_status(kv_chunk.room, status)
                    prefill_unique_rank = self.attn_tp_size * self.pp_rank + self.attn_tp_rank
                    for packed_req in task.write_requests:
                        endpoint, dst_port, room = packed_req.dst_ranks_info
                        self.sync_status_to_decode_endpoint(endpoint, dst_port, room, status, prefill_unique_rank)

                if kv_chunk.room not in self.request_status or self.check_status(kv_chunk.room) == KVPoll.Success:
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)
                    with self.request_state_lock:
                        self.rooms_in_transfer_queue.discard(kv_chunk.room)

        pending_tasks: list[LayerWiseTask] = []
        while True:
            try:
                if len(new_tasks := get_new_tasks(blocking=(len(pending_tasks) == 0))) > 0:
                    if len(abort_tasks := initialize(new_tasks)) > 0:
                        finalize(abort_tasks)
                        new_tasks = discard_tasks(new_tasks, abort_tasks)

                    pending_tasks.extend(new_tasks)

                if len(pending_tasks) > 0:
                    ready_cache_step, ready_aux_step = query_ready_step(pending_tasks[0])  # only wait the first task
                    complete_tasks, abort_tasks = submit_transfer(pending_tasks, ready_cache_step, ready_aux_step)
                    finalize(abort_tasks)
                    finalize(complete_tasks)
                    pending_tasks = discard_tasks(pending_tasks, complete_tasks + abort_tasks)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                ) from e


class MooncakeKVSender(BaseKVSender):
    def __init__(self, mgr: MooncakeKVManager, bootstrap_addr: str, bootstrap_room: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.init_time = time.time()
        self.conclude_state = None
        self._kv_transfer_result_lock = threading.Lock()
        self._kv_transfer_result_reported = False
        # inner state
        self.curr_idx = 0

    def _log_kv_transfer_result_once(self, success: bool, failure_type: Optional[str] = None) -> None:
        with self._kv_transfer_result_lock:
            if self._kv_transfer_result_reported:
                return
            self._kv_transfer_result_reported = True

        metrics = self.kv_mgr.kv_transfer_metrics
        if not metrics:
            return

        metrics.log_kv_transfer_result(success)
        if failure_type == "handshake":
            metrics.log_kv_handshake_failed()
        elif failure_type == "timeout":
            metrics.log_kv_transfer_timeout()
        elif failure_type == "transfer":
            metrics.log_kv_transfer_failed()

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None, decode_prefix_len: Optional[int] = 0):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        self.decode_prefix_len = decode_prefix_len

        # Get decode_prefix_len from manager if available (priority over parameter)
        manager_decode_prefix_len = self.kv_mgr.receive_decode_prefix_info(self.bootstrap_room)
        if manager_decode_prefix_len > 0:
            self.decode_prefix_len = manager_decode_prefix_len
            logger.debug(f"MooncakeKVSender updated decode_prefix_len from manager: {manager_decode_prefix_len} for room {self.bootstrap_room}")
        else:
            logger.debug(f"MooncakeKVSender using parameter decode_prefix_len: {self.decode_prefix_len} for room {self.bootstrap_room}")
        logger.debug(f"MooncakeKVSender init {num_kv_indices=} {aux_index} {decode_prefix_len=}")

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        start_idx: Optional[int] = 0,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        state_indices: Optional[list[int]] = None,
    ):
        """
        Send the kv cache at the given kv indices to the decoder server
        mla_l1_5_args: optional (page_transfer_mask, page_local_indices)
            page_transfer_mask: boolean mask to select decode pages that will receive data from this prefill rank
            page_local_indices: remapped local page indices that this prefill rank will send
        """
        if self.curr_idx < start_idx:
            self.curr_idx = start_idx
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
        logger.debug(f"kv sender send {is_last=} {kv_indices=} {index_slice=} {self.curr_idx=} {self.num_kv_indices=} {self.decode_prefix_len=}")
        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room, kv_indices, index_slice, False, mla_l1_5_args=mla_l1_5_args, state_indices=state_indices
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                mla_l1_5_args=mla_l1_5_args,
                state_indices=state_indices,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self._log_kv_transfer_result_once(
                    status == KVPoll.Success,
                    None if status == KVPoll.Success else "transfer",
                )
            elif status == KVPoll.Bootstrapping:
                elapsed = time.time() - self.init_time
                if elapsed >= self.kv_mgr.bootstrap_time_out:
                    logger.warning_once(
                        "Some requests timed out when bootstrapping, "
                        "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                        "If a greater mean TTFT is acceptable, you can "
                        "'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' "
                        "(10 minutes) to relax the timeout condition. "
                    )
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                    )
                    self._log_kv_transfer_result_once(False, "handshake")
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    self.conclude_state = KVPoll.Failed
                    return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        self.kv_mgr.clear_request_state(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self._log_kv_transfer_result_once(False, "transfer")

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, "Failed due to an unknown reason from another rank")
        raise KVTransferError(self.bootstrap_room, failure_reason, self.bootstrap_server_url)


class MooncakeKVReceiver(BaseKVReceiver):
    _ctx: Optional[zmq.Context] = None
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    @classmethod
    def _get_ctx(cls) -> zmq.Context:
        if cls._ctx is None:
            cls._ctx = zmq.Context(2)
            cls._ctx.set(zmq.MAX_SOCKETS, 1000000)
        return cls._ctx

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.init_time = None
        self.prefill_enable_mla_l1_5_cache = None
        self.dst_enable_mla_l1_5_cache = bool(self.kv_mgr.server_args.enable_mla_l1_5_cache)
        # When the handshake is deferred, poll() uses this to bound how long a
        # room may stay in Bootstrapping.
        self.handshake_start = time.monotonic()
        self.bootstrap_infos = []
        self.async_handshake = getattr(self.kv_mgr, "defer_handshake", False)
        self._kv_transfer_result_lock = threading.Lock()
        self._kv_transfer_result_reported = False

        if self.async_handshake:
            self.kv_mgr.submit_handshake(self)
            return

        try:
            self._do_handshake()
        except BaseException as e:
            logger.error(
                "[handshake] failed room=%s addr=%s: %r",
                self.bootstrap_room,
                self.bootstrap_addr,
                e,
            )
            self._fail_handshake(f"handshake raised: {e!r}")
            raise

    def _fail_handshake(
        self,
        failure_reason: str,
        mark_bootstrap_failed: bool = False,
    ) -> None:
        """Fail a decode-side handshake and report KVHandshakeFailed once."""
        with self.kv_mgr.request_state_lock:
            if self.kv_mgr.request_status.get(self.bootstrap_room) in (None, KVPoll.Failed):
                return
            self.kv_mgr.record_failure(self.bootstrap_room, failure_reason)
            should_report = self._claim_kv_transfer_result_once()
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        if mark_bootstrap_failed:
            self.kv_mgr._mark_bootstrap_failed(self.bootstrap_addr)

        if should_report:
            self._emit_kv_transfer_result(False, "handshake")

    def _claim_kv_transfer_result_once(self) -> bool:
        with self._kv_transfer_result_lock:
            if self._kv_transfer_result_reported:
                return False
            self._kv_transfer_result_reported = True
            return True

    def _emit_kv_transfer_result(self, success: bool, failure_type: Optional[str] = None) -> None:
        metrics = self.kv_mgr.kv_transfer_metrics
        if not metrics:
            return

        metrics.log_kv_transfer_result(success)
        if failure_type == "handshake":
            metrics.log_kv_handshake_failed()
        elif failure_type == "timeout":
            metrics.log_kv_transfer_timeout()
        elif failure_type == "transfer":
            metrics.log_kv_transfer_failed()

    def _log_kv_transfer_result_once(self, success: bool, failure_type: Optional[str] = None) -> None:
        if not self._claim_kv_transfer_result_once():
            return

        self._emit_kv_transfer_result(success, failure_type)

    def _do_handshake(self):
        """Bootstrap handshake: fetch prefill topology, build the connection
        pool entry and register our kv_args with every prefill rank.

        Runs either inline (defer_handshake=0, legacy behaviour) or on the
        handshake thread pool. Every exit path must leave the room in a
        terminal-ish state: Failed on error, WaitingForInput on success.
        """
        # Skip the handshake entirely if this bootstrap addr failed within the
        # TTL. Failures are otherwise never remembered -- the early return
        # below happens before prefill_parallel_info_table is written -- so
        # every request to a dead prefill pays the full timeout again.
        if self.kv_mgr._is_session_failed(self.bootstrap_addr):
            self._fail_handshake(
                f"Skipped bootstrap handshake: {self.bootstrap_addr} failed within the last {self.kv_mgr.failed_session_ttl}s",
            )
            return

        # Double-checked read/write. The lock only guards the dict --
        # the fetch below can take up to 131s against a blackholed prefill, and
        # holding connection_lock across it would serialize every handshake
        # worker (and stall the heartbeat), defeating the point of the pool.
        # Two workers racing on the same dead addr each pay one fetch; that is
        # benign and setdefault keeps the table consistent.
        with self.kv_mgr.connection_lock:
            cached_parallel_info = self.kv_mgr.prefill_parallel_info_table.get(self.bootstrap_addr)
        if cached_parallel_info is None:
            self.prefill_tp_size, self.prefill_dp_size, self.prefill_enable_mla_l1_5_cache, self.prefill_pp_size = (
                self._get_prefill_parallel_info_from_server()
            )
            if (
                self.prefill_tp_size is None
                or self.prefill_dp_size is None
                or self.prefill_pp_size is None
                or self.prefill_enable_mla_l1_5_cache is None
            ):
                self._fail_handshake(
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                    mark_bootstrap_failed=True,
                )
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_tp_size}"
                )
                with self.kv_mgr.connection_lock:
                    self.kv_mgr.prefill_parallel_info_table.setdefault(
                        self.bootstrap_addr,
                        (self.prefill_tp_size, self.prefill_dp_size, self.prefill_enable_mla_l1_5_cache, self.prefill_pp_size),
                    )
        else:
            self.prefill_tp_size, self.prefill_dp_size, self.prefill_enable_mla_l1_5_cache, self.prefill_pp_size = cached_parallel_info

        # Currently, we don't allow prefill instance and decode instance to
        # have different TP sizes per DP rank, except for models using MLA.
        local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size

        # if not self.kv_mgr.draft_is_mla_backend and self.kv_mgr.server_args.speculative_algorithm is not None:
        #    assert local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank, \
        #        "PD with different TP sizes per DP rank is not yet supported for non-MLA draft models"
        if self.prefill_enable_mla_l1_5_cache:
            assert self.kv_mgr.is_mla_backend
            self.target_tp_ranks = [i for i in range(prefill_tp_size_per_dp_rank)]
            self.target_tp_rank = None  # make all tp ranks not dummy rank
            self.required_dst_info_num = local_tp_size_per_dp_rank
            self.required_prefill_response_num = prefill_tp_size_per_dp_rank
        elif local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
            self.required_prefill_response_num = 1
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            assert self.kv_mgr.is_mla_backend, "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
            self.target_tp_rank = (self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank) // (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.required_dst_info_num = local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            self.target_tp_ranks = [self.target_tp_rank]
            self.required_prefill_response_num = 1
        else:
            assert self.kv_mgr.is_mla_backend, "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank) * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank + 1) * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            # we equally select fixed prefill tp ranks for each dp rank
            range_size = prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank
            self.target_tp_rank = self.target_tp_ranks[self.kv_mgr.dp_rank % range_size]
            self.required_dst_info_num = 1
            self.required_prefill_response_num = 1

        assert self.kv_mgr.pp_size == 1, "decode not support pp now"
        self.target_pp_ranks = [rank for rank in range(self.prefill_pp_size)]
        self.required_prefill_response_num *= self.prefill_pp_size
        self.target_dp_rank = self.bootstrap_room % self.prefill_dp_size
        with self.kv_mgr.request_state_lock:
            if self.kv_mgr.request_status.get(self.bootstrap_room) in (None, KVPoll.Failed):
                return
            self.kv_mgr.register_decode_response_tracker(
                self.bootstrap_room,
                self.required_prefill_response_num,
            )

        # NOTE: key distinguished by bootstrap_addr, target_dp_rank, and target_tp_rank
        bootstrap_key = f"{self.bootstrap_addr}_{self.target_dp_rank}_{self.target_tp_rank}"

        # Same double-checked pattern as the parallel-info table above: read the
        # pool under the lock, do the (slow) fan-out + ZMQ registration outside
        # it, then publish under the lock again.
        with self.kv_mgr.connection_lock:
            cached_bootstrap_infos = self.kv_mgr.connection_pool.get(bootstrap_key)

        if cached_bootstrap_infos is None:
            bootstrap_infos = []
            # prefill pp_rank -> (start_layer, end_layer), collected per real rank
            # below; consumed for cross-PP L3 write layer slicing.
            pp_layer_ranges = {}
            # Enable higher PP ranks to be bootstrapped earlier to make PP PD requests bootstrap more robust
            for target_pp_rank in reversed(self.target_pp_ranks):
                for target_tp_rank in self.target_tp_ranks:
                    bootstrap_info = self._get_bootstrap_info_from_server(target_tp_rank, self.target_dp_rank, target_pp_rank)
                    if bootstrap_info is not None:
                        # NOTE: only support MLA for now: select one prefill rank as real rank
                        bootstrap_info["is_dummy"] = not bool(target_tp_rank == self.target_tp_rank or self.target_tp_rank is None)
                        # Record this pp_rank's layer ownership once (from the real,
                        # non-dummy rank), keyed by prefill pp_rank for fan-out.
                        start = bootstrap_info.get("prefill_start_layer")
                        end = bootstrap_info.get("prefill_end_layer")
                        if not bootstrap_info["is_dummy"] and start is not None and end is not None:
                            pp_layer_ranges[int(target_pp_rank)] = (int(start), int(end))
                        logger.debug(f"Fetched bootstrap info: {bootstrap_info} for PP {target_pp_rank} DP {self.target_dp_rank} TP {target_tp_rank}")
                        bootstrap_infos.append(bootstrap_info)
                    else:
                        self._fail_handshake(
                            f"Could not fetch bootstrap info for tp_rank: {target_tp_rank} "
                            f"and target_dp_rank: {self.target_dp_rank} {target_pp_rank=}",
                            mark_bootstrap_failed=True,
                        )
                        return

            self.bootstrap_infos = bootstrap_infos

            if not getattr(self.kv_mgr, "prefill_pp_layer_ranges", None) and len(pp_layer_ranges) == len(self.target_pp_ranks):
                self.kv_mgr.prefill_pp_layer_ranges = pp_layer_ranges

            # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
            try:
                self._register_kv_args()
            except zmq.error.ZMQError as e:
                logger.warning(f"Failed to register kv_args to prefill: {e}, prefill may be dead")
                self._fail_handshake(
                    f"ZMQ send failed during kv_args registration: {e}",
                    mark_bootstrap_failed=True,
                )
                return

            with self.kv_mgr.connection_lock:
                self.kv_mgr.connection_pool.setdefault(bootstrap_key, self.bootstrap_infos)
        else:
            self.bootstrap_infos = cached_bootstrap_infos

        assert len(self.bootstrap_infos) > 0
        with self.kv_mgr.request_state_lock:
            if self.kv_mgr.request_status.get(self.bootstrap_room) in (None, KVPoll.Failed):
                return
            with self.kv_mgr.connection_lock:
                self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, tp_rank, target_dp_rank, target_pp_rank):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?tp_rank={tp_rank}&target_dp_rank={target_dp_rank}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(f"Failed to get prefill server info: {response.status_code}, {response.text}")
                return None
        except requests.exceptions.Timeout as e:
            logger.error(f"Timed out fetching prefill info from bootstrap: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_prefill_parallel_info_from_server(self) -> tuple[int, int, bool, int]:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?tp_rank={-1}&target_dp_rank={-1}&target_pp_rank={-1}"
            # This call had no timeout, unlike every other HTTP call in this
            # file. When a prefill node dies at the machine/network level the
            # SYN is blackholed rather than refused, so it blocked ~131s
            # (tcp_syn_retries=6) instead of failing fast.
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return (
                    int(prefill_parallel_info["prefill_tp_size"]),
                    int(prefill_parallel_info["prefill_dp_size"]),
                    bool(prefill_parallel_info["enable_mla_l1_5_cache"]),
                    int(prefill_parallel_info["prefill_pp_size"]),
                )
            else:
                logger.error(f"Failed to get prefill parallel info: {response.status_code}, {response.text}")
                return None, None, None, None
        except requests.exceptions.Timeout as e:
            logger.error(f"Timed out fetching prefill parallel info from bootstrap: {e}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None, None, None

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            packed_kv_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs)
            packed_aux_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs)
            packed_state_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs)
            packed_state_item_lens = b"".join(struct.pack("I", item_len) for item_len in self.kv_mgr.kv_args.state_item_lens)
            state_dim_per_tensor = getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", [])
            packed_state_dim_per_tensor = b"".join(struct.pack("I", dim) for dim in state_dim_per_tensor)

            offsets = self.kv_mgr.kv_args.offsets
            packed_offsets_parts = []
            packed_offsets_parts.append(struct.pack("I", len(offsets)))  # num_layers
            for layer_offsets in offsets:
                packed_offsets_parts.append(struct.pack("I", len(layer_offsets)))  # num_offset
                for offset in layer_offsets:
                    packed_offsets_parts.append(struct.pack("I", offset))  # offset
            packed_offsets = b"".join(packed_offsets_parts)

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        # Include decode_prefix_len for kv_args registration
                        str(getattr(self, "decode_prefix_len", 0)).encode("ascii"),
                        packed_state_data_ptrs,
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_offsets,
                        str(self.kv_mgr.attn_tp_rank).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                    ]
                )

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._get_ctx().socket(zmq.PUSH)
                sock.setsockopt(zmq.SNDTIMEO, 5000)  # 5 s send timeout
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def _close_sockets(cls, endpoints: list):
        # Collect sockets and their per-endpoint locks under global lock,
        # then close each socket while holding its per-endpoint lock to
        # avoid racing with threads that are mid-send on the same socket.
        to_close: list = []
        with cls._global_lock:
            for endpoint in endpoints:
                if endpoint in cls._socket_cache:
                    sock = cls._socket_cache.pop(endpoint)
                    lock = cls._socket_locks.pop(endpoint, None)
                    to_close.append((endpoint, sock, lock))

        for endpoint, sock, lock in to_close:
            try:
                if lock is not None:
                    with lock:
                        sock.close(linger=0)
                else:
                    sock.close(linger=0)
            except Exception as e:
                logger.debug(f"Failed to close cached ZMQ socket for {endpoint}: {e}")
            logger.info(f"Closed cached ZMQ socket for {endpoint}")

    def init(
        self,
        kv_indices: npt.NDArray[np.int64],
        aux_index: Optional[int] = None,
        decode_prefix_len: Optional[int] = 0,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        state_indices: Optional[list[int]] = None,
    ):
        # Store decode_prefix_len to be sent back to prefill
        self.decode_prefix_len = decode_prefix_len
        dst_page_transfer_mask = None
        dst_page_local_indices = None
        if mla_l1_5_args is not None:
            dst_page_transfer_mask = mla_l1_5_args.page_transfer_mask
            dst_page_local_indices = mla_l1_5_args.page_local_indices

        # Checked and recorded under one lock with the failure paths: a room
        # that failed while waiting for its KV slot (with nothing to stop,
        # since no indices went out) must never hand prefill a write target
        # afterwards, or its KV would be released while prefill may write.
        with self.kv_mgr.request_state_lock:
            if self.kv_mgr.request_status.get(self.bootstrap_room) in (None, KVPoll.Failed):
                return
            self.kv_mgr.dst_info_sent_rooms.add(self.bootstrap_room)
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            is_dummy = bootstrap_info["is_dummy"]

            try:
                sock, lock = self._connect("tcp://" + self.prefill_server_url)
                with lock:
                    sock.send_multipart(
                        [
                            str(self.bootstrap_room).encode("ascii"),
                            get_local_ip_by_remote().encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.session_id.encode("ascii"),
                            kv_indices.tobytes() if not is_dummy else b"",
                            str(aux_index).encode("ascii") if not is_dummy else b"",
                            str(self.required_dst_info_num).encode("ascii"),
                            # Send decode_prefix_len as additional message part
                            str(self.decode_prefix_len).encode("ascii") if not is_dummy else b"",
                            str(int(self.dst_enable_mla_l1_5_cache)).encode("ascii") if not is_dummy else b"",
                            dst_page_transfer_mask.tobytes() if (not is_dummy and dst_page_transfer_mask is not None) else b"",
                            dst_page_local_indices.tobytes() if (not is_dummy and dst_page_local_indices is not None) else b"",
                            (np.array(state_indices, dtype=np.int32).tobytes() if not is_dummy and state_indices is not None else b""),
                        ]
                    )
            except zmq.error.ZMQError as e:
                logger.warning(f"Failed to send init to prefill {self.prefill_server_url}: {e}, room={self.bootstrap_room}, prefill may be dead")
                self._fail_handshake(
                    f"ZMQ send failed to {self.prefill_server_url}: {e}",
                )
                return
            self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self._log_kv_transfer_result_once(
                    status == KVPoll.Success,
                    None if status == KVPoll.Success else "transfer",
                )
            elif status == KVPoll.Bootstrapping:
                # With a deferred handshake nothing else bounds this state: the
                # worker may be stuck in a 131s connect against a blackholed
                # prefill, or the pool may be saturated. Fail the room once it
                # exceeds bootstrap_time_out so it cannot linger forever.
                if self.async_handshake:
                    elapsed = time.monotonic() - self.handshake_start
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
                        logger.warning(
                            "[handshake] room=%s addr=%s stuck in Bootstrapping for %.1fs (>= %ss), marking Failed",
                            self.bootstrap_room,
                            self.bootstrap_addr,
                            elapsed,
                            self.kv_mgr.bootstrap_time_out,
                        )
                        self._fail_handshake(
                            f"Bootstrap handshake timed out after {elapsed:.1f}s (addr={self.bootstrap_addr})",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed
            elif status == KVPoll.WaitingForInput:
                if self.init_time is None:
                    # Bound handshake plus resource waiting on decode itself.
                    # Prefill has its own start time and may expire earlier or
                    # later; matching timeout values do not align deadlines.
                    elapsed = time.monotonic() - self.handshake_start
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
                        logger.warning(
                            "[decode-prealloc] room=%s waited %.1fs for a KV slot (>= %ss), marking Failed",
                            self.bootstrap_room,
                            elapsed,
                            self.kv_mgr.bootstrap_time_out,
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Abort due to timeout before decode KV allocation after {elapsed:.1f}s (room={self.bootstrap_room})",
                        )
                        self._log_kv_transfer_result_once(False, "timeout")
                        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                        self.conclude_state = KVPoll.Failed
                        # Prefill may still hold the room: let it drop it.
                        # The scheduler clears our state via failure_exception().
                        self._send_abort_req()
                        return KVPoll.Failed
                else:
                    now = time.time()
                    with self.kv_mgr.request_state_lock:
                        start = self.kv_mgr.abort_restart_time.get(self.bootstrap_room, self.init_time)
                    elapsed = now - start
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal "
                            "after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export "
                            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) "
                            "to relax the timeout condition. "
                        )
                        self._log_kv_transfer_result_once(False, "timeout")
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        # Do NOT conclude Failed locally: prefill may still
                        # be holding the decode buffer for RDMA writes. Go
                        # through abort() so the KV slot is only released
                        # after every prefill rank has ACKed (or the
                        # heartbeat has confirmed prefill is dead).
                        logger.info(
                            f"[decode-abort] waiting_timeout exceeded "
                            f"room={self.bootstrap_room} elapsed={elapsed:.1f}s "
                            f"-> trigger abort() from poll()"
                        )
                        self.abort()
                        with self.kv_mgr.request_state_lock:
                            cur = self.kv_mgr.request_status.get(self.bootstrap_room)
                        return cur if cur is not None else KVPoll.Failed
            elif status == KVPoll.Transferring:
                logger.warning(f"Req(room={self.bootstrap_room}) in Transferring, which is unexpected")

            return status
        else:
            return self.conclude_state

    def abort(self) -> None:
        """Request prefill side to stop writing into our KV buffers.

        Fans ABORT_REQ out to every prefill rank we registered with. The
        room stays in its current KVPoll state (typically WaitingForInput)
        until a quorum of ABORT_ACK replies arrives; only a full ack_ok
        sweep finalizes the room as Failed. Any ack_fail reply resets the
        waiting window so a subsequent poll() timeout can fan out a fresh
        round.

        Caller contract: until poll() returns Failed the KV slot must stay
        reserved for this request.
        """
        if self.conclude_state in (KVPoll.Success, KVPoll.Failed):
            return

        with self.kv_mgr.request_state_lock:
            current = self.kv_mgr.request_status.get(self.bootstrap_room)
            if current in (KVPoll.Success, KVPoll.Failed):
                return
            if current is None:
                # Receiver never registered — nothing for prefill to abort.
                return
            # If the decode response tracker hasn't been registered yet the
            # ACK quorum can never fire; finalize locally.
            expected = self.kv_mgr.required_prefill_response_num_table.get(self.bootstrap_room)
            if expected is None:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    "Abort before decode response tracker registration",
                )
                self.kv_mgr._update_status_unlocked(self.bootstrap_room, KVPoll.Failed)
                return
            # Re-entrant: re-fanout is intentional (prefill's _handle_abort_req
            # is idempotent). Keep existing AbortState if present so previously
            # collected ack counters are preserved across rounds.
            if self.bootstrap_room not in self.kv_mgr.abort_state:
                self.kv_mgr.abort_state[self.bootstrap_room] = MooncakeKVManager._AbortState(expected=expected)

        # Fan ABORT_REQ out to every prefill rank we've registered with,
        # matching init()'s fan-out so required_prefill_response_num is the
        # right quorum size.
        for bootstrap_info in self.bootstrap_infos:
            prefill_url = f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            try:
                sock, lock = self._connect("tcp://" + prefill_url)
                with lock:
                    sock.send_multipart(
                        [
                            str(self.bootstrap_room).encode("ascii"),
                            b"ABORT_REQ",
                            get_local_ip_by_remote().encode("ascii"),
                            b"",
                            str(self.kv_mgr.rank_port).encode("ascii"),
                        ]
                    )
            except Exception as e:
                # broaden beyond zmq.error.ZMQError. _connect() /
                # get_local_ip_by_remote() can raise other exceptions; we
                # must not let them escape and break the heartbeat fallback.
                logger.warning(f"ABORT_REQ send failed to {prefill_url} (room={self.bootstrap_room}): {e}")
                # Heartbeat path will eventually force the room to Failed
                # via _handle_node_failure.

    def _send_abort_req(self) -> None:
        """Best-effort cancellation before publishing KV indices.

        No remote writer has our buffer addresses yet, so local failure need
        not wait for ACKs. Prefill's own bootstrap timeout remains the fallback
        if this notification cannot be delivered.
        """
        for info in self.bootstrap_infos:
            if info["is_dummy"]:
                continue
            prefill_url = f"{info['rank_ip']}:{info['rank_port']}"
            try:
                sock, lock = self._connect("tcp://" + prefill_url)
                with lock:
                    sock.send_multipart(
                        [
                            str(self.bootstrap_room).encode("ascii"),
                            b"ABORT_REQ",
                            get_local_ip_by_remote().encode("ascii"),
                            b"",
                            str(self.kv_mgr.rank_port).encode("ascii"),
                        ],
                        flags=zmq.NOBLOCK,
                    )
            except Exception as e:
                logger.warning(f"ABORT_REQ send failed to {prefill_url} (room={self.bootstrap_room}): {e}")

    def cancel_before_transfer(self) -> None:
        with self.kv_mgr.request_state_lock:
            if self.bootstrap_room in self.kv_mgr.dst_info_sent_rooms:
                raise RuntimeError("Cannot use preallocation cleanup after publishing KV targets")
            self.conclude_state = KVPoll.Failed
            self.clear()
        self._send_abort_req()

    def clear(self) -> None:
        self.kv_mgr.clear_decode_response_state(self.bootstrap_room)
        with self.kv_mgr.connection_lock:
            rooms = self.kv_mgr.addr_to_rooms_tracker.get(self.bootstrap_addr)
            if rooms is not None:
                rooms.discard(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self._log_kv_transfer_result_once(False, "transfer")

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, "Failed due to an unknown reason from another rank")
        raise KVTransferError(self.bootstrap_room, failure_reason, self.bootstrap_addr)


class MooncakeKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.pp_size = None
        self.tp_size_per_dp_rank = None
        self.prefill_port_table: dict[int, dict[int, dict[int, dict[str, Union[str, int]]]]] = {}
        self.enable_mla_l1_5_cache = False

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(text="Method not allowed", status=405, content_type="application/json")

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        tp_size = data["tp_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        tp_rank = int(data["tp_rank"])
        self.enable_mla_l1_5_cache = bool(data["enable_mla_l1_5_cache"])
        pp_rank = int(data["pp_rank"])
        pp_size = int(data["pp_size"])
        prefill_start_layer = data.get("prefill_start_layer")
        prefill_end_layer = data.get("prefill_end_layer")

        if self.tp_size is None:
            self.tp_size = tp_size

        if self.dp_size is None:
            self.dp_size = dp_size

        if self.pp_size is None:
            self.pp_size = pp_size

        tp_size_per_dp_rank = tp_size // dp_size
        if self.tp_size_per_dp_rank is None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        if role == "Prefill":
            dp_rank = tp_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = tp_rank % tp_size_per_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                dp_group_table = self.prefill_port_table.setdefault(dp_rank, {})
                tp_group_table = dp_group_table.setdefault(tp_rank_in_dp_group, {})

                tp_group_table[pp_rank] = {
                    "rank_ip": rank_ip,
                    "rank_port": rank_port,
                    "prefill_start_layer": prefill_start_layer,
                    "prefill_end_layer": prefill_end_layer,
                }

            logger.info(f"Register prefill bootstrap: {pp_rank=} {tp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}")

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        tp_rank = request.query.get("tp_rank")
        target_dp_rank = request.query.get("target_dp_rank")
        target_pp_rank = request.query.get("target_pp_rank")
        if not tp_rank or not target_dp_rank or not target_pp_rank:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use tp_rank == -1 and target_dp_rank == -1 to sync dp size
        if int(tp_rank) == -1 and int(target_dp_rank) == -1 and int(target_pp_rank) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.tp_size,
                "prefill_dp_size": self.dp_size,
                "prefill_pp_size": self.pp_size,
                "enable_mla_l1_5_cache": self.enable_mla_l1_5_cache,
            }
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_rank)][int(tp_rank)][int(target_pp_rank)]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
