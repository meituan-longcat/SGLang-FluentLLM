import asyncio
import dataclasses
import logging
import os
import socket
import struct
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web
from contextlib import contextmanager

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
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank, get_attention_tp_rank, get_attention_tp_size
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_by_remote,
    get_colorful_logger
)
from sglang.srt.metrics.collector import KVTransferMetricsCollector
from sglang.srt.utils import is_npu
__is_npu__ = is_npu()


logger = get_colorful_logger(__name__)

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

@dataclasses.dataclass
class TransferIndexResolution:
    src_indices: npt.NDArray[np.int64]
    dst_indices: npt.NDArray[np.int64]


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


    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            dst_kv_indices = np.array([], dtype=np.int64)
            dst_aux_index = None
            decode_prefix_len = 0
            dst_indices_are_local = False
            dst_page_transfer_mask = None
            dst_page_local_indices = None
            dst_page_indices_mapping = None
            is_dummy = True
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int64)
            dst_aux_index = int(msg[5].decode("ascii"))
            # decode_prefix_len is now msg[7] (we added it as additional message part)
            decode_prefix_len = int(msg[7].decode("ascii")) if len(msg) > 7 else 0
            dst_indices_are_local = bool(int(msg[8].decode("ascii"))) if len(msg) > 8 else False
            dst_page_transfer_mask = (
                np.frombuffer(msg[9], dtype=np.bool_) if len(msg) > 9 else None
            )
            dst_page_local_indices = (
                np.frombuffer(msg[10], dtype=np.int64) if len(msg) > 10 else None
            )
            dst_page_indices_mapping = (
                np.cumsum(dst_page_transfer_mask) - 1 if dst_page_transfer_mask is not None else None
            )
            is_dummy = False
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
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_offsets: list[List[int]]
    dst_aux_ptrs: list[int]
    decode_prefix_len : int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Handle both old format (6 parts) and new format (7 parts with decode_prefix_len)
        if len(msg) >= 7:
            decode_prefix_len = int(msg[6].decode("ascii"))
        else:
            decode_prefix_len = 0

        offsets_data=msg[7]
        offset=0
        num_layers=struct.unpack_from("I", offsets_data, offset)[0]
        offset+=4
        dst_offsets=[]
        for _ in range(num_layers):
            layer_len=struct.unpack_from("I", offsets_data, offset)[0]
            offset+=4
            layer_offsets=list(struct.unpack_from(f"{layer_len}I", offsets_data, offset))
            offset+=4*layer_len
            dst_offsets.append(layer_offsets)

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            decode_prefix_len=decode_prefix_len,
            dst_offsets=dst_offsets,
        )


class MooncakeKVManager(BaseKVManager):
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
        self.engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )
        # logger.info(f"[MoonCakeTransferEngine][Topology] {self.engine.engine.get_local_topology()}")
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.src_mode = "ON" if bool(server_args.enable_mla_l1_5_cache) else "OFF"
        self.is_mla_backend = is_mla_backend
        self.draft_is_mla_backend = draft_is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.dp_rank = get_attention_dp_rank()
        self.enable_dp_attention = server_args.enable_dp_attention
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )
        self.request_status: Dict[int, KVPoll] = {}
        self.rank_port = None
        self.request_state_lock = threading.RLock()
        # Use a shared ZMQ context to avoid creating one IO thread per Context instance
        self._zmq_ctx = zmq.Context(2)
        self._zmq_ctx.set(zmq.MAX_SOCKETS, 1000000)
        self.server_socket = self._zmq_ctx.socket(zmq.PULL)
        if hasattr(server_args, 'enable_metrics') and server_args.enable_metrics:
            labels = {
                'model_name': server_args.served_model_name,
                'app_key': server_args.app_key,
            }
            self.kv_transfer_metrics = KVTransferMetricsCollector(labels, server_args.metrics_reporters)
        else:
            self.kv_transfer_metrics = None

        self.register_buffer_to_engine()
        self.pp_rank=self.kv_args.pp_rank
        self.pp_size=server_args.pp_size

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._push_zmq_ctx=zmq.Context(2)
            self._push_zmq_ctx.set(zmq.MAX_SOCKETS, 1000000)
            # Per-instance PUSH socket cache (keyed by endpoint) to reuse connections
            self._push_socket_lock = threading.Lock()
            self._push_socket_cache: Dict[str, Any] = {}
            self._push_socket_locks: Dict[str, threading.Lock] = {}

            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
            self.start_prefill_thread()
            self._register_to_bootstrap()

            self.session_lock = threading.Lock()
            self.session_failures = defaultdict(int)
            self.failed_sessions: Dict[str, float] = {}
            self.failed_session_ttl = max(
                get_int_env_var("SGLANG_DISAGGREGATION_FAILED_SESSION_TTL", 30), 0
            )

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
            self.bootstrap_time_out = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 120
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.connection_lock = threading.Lock()
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.required_prefill_response_num_table: Dict[int, int] = {}
            self.prefill_response_tracker: Dict[int, Set[int]] = {}
            self.prefill_parallel_info_table: Dict[str, Tuple[int, int, bool, int]] = {}
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
            )
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

        self.failure_lock = threading.Lock()
        self.failure_records: Dict[int, str] = {}

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

    def _fail_session_and_chunk(self, mooncake_session_id, endpoint, dst_port, bootstrap_room, reason):
        with self.session_lock:
            self.session_failures[mooncake_session_id] += 1
            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
            if self.session_failures[mooncake_session_id] >= 1:
                self._mark_session_failed(
                    mooncake_session_id,
                    reason=reason,
                )
                logger.error(
                    f"Session {mooncake_session_id} failed."
                )

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
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

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
                    "Added new PUSH socket for endpoint %s. "
                    "_push_socket_cache size: %d",
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

        src_mode = self.src_mode
        dst_mode = (
            "ON"
            if req.dst_indices_are_local
            else "OFF"
        )
        src_args = kv_chunk.mla_l1_5_args

        # Prefill OFF/Decode OFF: use original indices.
        if src_mode == "OFF" and dst_mode == "OFF":
            return TransferIndexResolution(src_indices, dst_indices)

        # Prefill ON/Decode OFF: prefill-side only has partial kv cache, only send local part.
        if src_mode == "ON" and dst_mode == "OFF":
            assert src_args is not None, "Prefill MLA L1.5 cache is enabled but no transfer metadata provided"
            src_mask = src_args.page_transfer_mask
            src_local = src_args.page_local_indices
            return TransferIndexResolution(
                src_indices=src_local,
                dst_indices=dst_indices[src_mask],
            )

        # Prefill OFF/Decode ON: decode-side only has partial kv cache, only send requested part.
        if src_mode == "OFF" and dst_mode == "ON":
            assert req.dst_page_transfer_mask is not None, (
                "OFF/ON expects decode ownership mask when destination reports local indices."
            )
            dst_mapping = req.dst_page_indices_mapping[kv_chunk.index_slice]
            dst_mask = req.dst_page_transfer_mask[kv_chunk.index_slice]
            dst_mapping = dst_mapping[dst_mask]
            dst_local = req.dst_page_local_indices[dst_mapping]
            return TransferIndexResolution(
                src_indices=src_indices[dst_mask],
                dst_indices=dst_local,
            )

        # Prefill ON/Decode ON: both sides hold partial kv cache, find the intersection part.
        assert src_args is not None, (
            "ON/ON expects prefill metadata (page_transfer_mask/page_local_indices)."
        )
        assert req.dst_page_transfer_mask is not None, (
            "ON/ON expects decode ownership mask when destination uses local index space."
        )

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
        )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id+self.kv_args.prefill_start_layer],
                self.kv_args.kv_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
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

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
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
            logger.warning(
                f"Failed to sync status to {endpoint} (room={room}), evicting socket and retrying: {e}"
            )

    def transfer_worker(
        self, queue: FastQueue
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                logger.debug(f"[TRANSFER_WORKER] Got transfer request for room {kv_chunk.room}, is_last={kv_chunk.is_last}, kv_indices_len={len(kv_chunk.prefill_kv_indices)}")
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
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

                        logger.debug(f"[TRANSFER_WORKER] Calling send_kvcache for room {kv_chunk.room}, session {req.mooncake_session_id}")
                        tm_start = time.monotonic()
                        ret = self.send_kvcache(
                            req.mooncake_session_id,
                            resolved.src_indices,
                            self.decode_kv_args_table[
                                req.mooncake_session_id
                            ].dst_kv_ptrs,
                            resolved.dst_indices,
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
                            self.sync_status_to_decode_endpoint(
                                req.endpoint, req.dst_port, req.room, KVPoll.Failed, prefill_unique_rank
                            )
                            break

                        if kv_chunk.is_last:
                            # Only the last chunk we need to send the aux data
                            if self.pp_rank==self.pp_size-1:
                                ret=self.send_aux(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_aux_index,
                                    self.decode_kv_args_table[
                                        req.mooncake_session_id
                                    ].dst_aux_ptrs,
                                    req.dst_aux_index,
                                )
                                polls.append(True if ret== 0 else False)
                            else:
                                polls.append(True)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, prefill_unique_rank
                                    )
                        elapsed_ms = time.monotonic() - tm_start
                        if self.kv_transfer_metrics:
                            self.kv_transfer_metrics.log_kv_transfer_time(elapsed_ms)
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    self._clear_failed_session(mooncake_session_id)
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[6].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def start_transfer_thread(self, transfer_thread_pool_size: int, transfer_queue_size: int):
        self.transfer_queues: List[FastQueue] = [
            FastQueue() for _ in range(transfer_queue_size)
        ]
        for queue in self.transfer_queues:
            threading.Thread(
                target=self.transfer_worker, args=(queue, ), daemon=True
            ).start()

    def start_decode_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def decode_thread():
            while True:
                (bootstrap_room, status, prefill_rank) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                handled = self.handle_decode_status_message(
                    bootstrap_room, status, prefill_rank
                )
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

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the map
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
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
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
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

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                mla_l1_5_args=mla_l1_5_args,
            )
        )

    def _update_status_unlocked(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def register_decode_response_tracker(
        self, bootstrap_room: int, required_prefill_response_num: int
    ):
        with self.request_state_lock:
            self.required_prefill_response_num_table[bootstrap_room] = (
                required_prefill_response_num
            )
            self.prefill_response_tracker[bootstrap_room] = set()

    def clear_request_state(self, bootstrap_room: int):
        with self.request_state_lock:
            self.request_status.pop(bootstrap_room, None)

    def clear_decode_response_state(self, bootstrap_room: int):
        with self.request_state_lock:
            self.request_status.pop(bootstrap_room, None)
            self.required_prefill_response_num_table.pop(bootstrap_room, None)
            self.prefill_response_tracker.pop(bootstrap_room, None)

    def handle_decode_status_message(
        self, bootstrap_room: int, status: int, prefill_rank: int
    ) -> bool:
        with self.request_state_lock:
            current_status = self.request_status.get(bootstrap_room)
            if current_status is None:
                return False

            if status == KVPoll.Success:
                if current_status in (KVPoll.Failed, KVPoll.Success):
                    return False

                expected_response_num = self.required_prefill_response_num_table.get(
                    bootstrap_room
                )
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

    def check_status(self, bootstrap_room: int):
        with self.request_state_lock:
            return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        with self.request_state_lock:
            self._update_status_unlocked(bootstrap_room, status)

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        # In mooncake implementation, decode_prefix_len is handled via ZMQ messages
        # Check the stored transfer info for this room
        if bootstrap_room in self.transfer_infos:
            for transfer_info in self.transfer_infos[bootstrap_room].values():
                if hasattr(transfer_info, 'decode_prefix_len') and transfer_info.decode_prefix_len > 0:
                    logger.debug(f"Found decode_prefix_len={transfer_info.decode_prefix_len} for room {bootstrap_room}")
                    return transfer_info.decode_prefix_len
        logger.debug(f"No decode_prefix_len found for room {bootstrap_room}, using 0")
        return 0

    def get_session_id(self):
        return self.engine.get_session_id()

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

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
        }

        logger.info(f"Sending POST request to bootstrap server: {url} {payload}")

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Prefill instance failed to register to bootstrap server: {e}"
            )

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            endpoints_to_close = []
            for k in keys_to_remove:
                for info in self.connection_pool[k]:
                    endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                    endpoints_to_close.append(endpoint)
            if endpoints_to_close:
                MooncakeKVReceiver._close_sockets(endpoints_to_close)
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_parallel_info_table:
                del self.prefill_parallel_info_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )

@dataclasses.dataclass
class WriteRequest:
    trans_info: TransferInfo
    dst_ranks_info: Tuple[str, int, int]
    prefill_kv_blocks: npt.NDArray[np.int64]
    dst_kv_blocks: npt.NDArray[np.int64]
    submit_bids: List[int] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class LayerWiseTask:
    kv_chunk: TransferKVChunk
    begin_cache_step: int
    aux_step: int
    next_layer_id: int = 0
    write_requests: List[WriteRequest] = dataclasses.field(default_factory=list)
    polls: List[bool] = dataclasses.field(default_factory=list)

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
        self.is_last_pp_rank = (self.kv_args.pp_rank == self.pp_size-1)
        if self.is_last_pp_rank:
            self.effective_layer_num=self.kv_args.prefill_end_layer-self.kv_args.prefill_start_layer+self.draft_layer_num
        else:
            self.effective_layer_num=self.kv_args.prefill_end_layer-self.kv_args.prefill_start_layer
        self.draft_is_mla_backend = draft_is_mla_backend
        self.kv_cache_quant_method = server_args.kv_cache_quant_method
        self.submit_interval = server_args.disaggregation_layerwise_interval
        assert self.submit_interval > 0, "submit_interval must be positive"
        self.current_transfer_batch: List[Tuple[TransferKVChunk, int]] = []
        self.offsets = args.offsets

    def start_transfer_thread(self, transfer_thread_pool_size: int, transfer_queue_size: int):
        self.transfer_queues: List[FastQueue] = [FastQueue()]
        for queue in self.transfer_queues:
            threading.Thread(
                target=self.async_transfer_worker, args=(queue,), daemon=True
            ).start()

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
            delta_aux_step=1 if self.is_last_pp_rank else 0
            self.step_counter.advance_step(delta_cache_step=self.effective_layer_num, delta_aux_step=delta_aux_step)

        self.current_transfer_batch.clear()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        mla_l1_5_args: Optional[Tuple[Any, Any]] = None,
    ):
        logger.debug('async manager add_transfer_request')
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
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
        )
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
        for aux_data_ptr, aux_item_len, dst_aux_ptr in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_item_lens, dst_aux_ptrs
        ):
            prefill_aux_addr = aux_data_ptr + prefill_aux_index * aux_item_len
            decode_aux_addr = dst_aux_ptr + dst_aux_index * aux_item_len
            bid = self.engine.transfer_submit_write(
                mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len
            )
            batch_ids.append(bid)
        # Return the last batch_id for compatibility (or could return list)
        return batch_ids[-1] if batch_ids else -1

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def submit_layer_cache(
        self,
        mooncake_session_id: str,
        begin_layer_id: int,  # [begin_layer_id, end_layer_id)
        end_layer_id: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
        prefill_start_layer: int
    ) -> int:
        dst_kv_ptrs = self.decode_kv_args_table[mooncake_session_id].dst_kv_ptrs
        dst_offset = self.decode_kv_args_table[mooncake_session_id].dst_offsets
        transfer_blocks = []

        def submit_one_cache(ptr_offset, dst_ptr_offset):
            src_ptr = self.kv_args.kv_data_ptrs[ptr_offset]
            dst_ptr = dst_kv_ptrs[dst_ptr_offset]
            item_len = self.kv_args.kv_item_lens[ptr_offset]
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
        for layer_id in range(begin_layer_id, end_layer_id):
            for offset, dst_offset in zip(self.offsets[layer_id], dst_offset[layer_id+prefill_start_layer]):
                submit_one_cache(offset, dst_offset)
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def async_transfer_worker(self, transfer_queue: FastQueue):
        def discard_finished_bid_inplace(submit_bids: List[int]):
            finished_cnt = 0
            failed = False
            for bid in submit_bids:
                status = self.engine.transfer_check_status(bid)
                if status == 1:
                    finished_cnt += 1
                elif status == -1:
                    failed = True
                    if self.kv_transfer_metrics:
                        self.kv_transfer_metrics.log_kv_transfer_timeout()
                    logger.error(f"Transfer timeout detected!")
                    break
                elif status == -2:
                    failed = True
                    if self.kv_transfer_metrics:
                        self.kv_transfer_metrics.log_kv_transfer_failed()
                    logger.error(f"Transfer failed detected")
                    break
                else:
                    failed = (status != 0)
                    break
            submit_bids[:] = submit_bids[finished_cnt:]
            return not failed

        def discard_tasks(tasks: List[LayerWiseTask], droped: List[LayerWiseTask]) -> List[LayerWiseTask]:
            droped_rooms = set(id(task)for task in droped)
            return [task for task in tasks if id(task) not in droped_rooms]

        def query_ready_step(task: LayerWiseTask) -> Tuple[int, int]:
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

        def get_new_tasks(blocking: bool) -> List[LayerWiseTask]:
            new_tasks: List[LayerWiseTask] = []
            if blocking:
                new_tasks.append(transfer_queue.get())
            while True:
                try:
                    new_tasks.append(transfer_queue.get_nowait())
                except FastQueue.Empty:
                    break

            return new_tasks

        def initialize(tasks: List[LayerWiseTask]) -> List[LayerWiseTask]:
            abort_tasks: List[LayerWiseTask] = []

            for task in tasks:
                kv_chunk = task.kv_chunk
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )

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
                    prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
                        resolved.src_indices, resolved.dst_indices
                    )
                    task.write_requests.append(WriteRequest(
                        trans_info=req,
                        dst_ranks_info=(req.endpoint, req.dst_port, req.room),
                        prefill_kv_blocks=prefill_kv_blocks,
                        dst_kv_blocks=dst_kv_blocks
                    ))

            return abort_tasks

        def submit_transfer(
            tasks: List[LayerWiseTask],
            ready_cache_step: int,
            ready_aux_step: int
        ) -> Tuple[List[LayerWiseTask], List[LayerWiseTask]]:
            abort_tasks: List[LayerWiseTask] = []
            complete_tasks: List[LayerWiseTask] = []

            for task in tasks:
                kv_chunk = task.kv_chunk
                # submit layer cache
                if (
                    task.next_layer_id < self.effective_layer_num
                    and StepCounter.is_step_ready(ready_cache_step, task.begin_cache_step + task.next_layer_id)
                ):
                    for req in task.write_requests:
                        if (
                            (task.next_layer_id + 1) % self.submit_interval == 0
                            or task.next_layer_id == self.effective_layer_num - 1
                        ):
                            ret = self.submit_layer_cache(
                                req.trans_info.mooncake_session_id,
                                (task.next_layer_id // self.submit_interval) * self.submit_interval,
                                task.next_layer_id + 1,
                                req.prefill_kv_blocks,
                                req.dst_kv_blocks,
                                self.prefill_start_layer
                            )

                            if ret != 0:
                                if self.kv_transfer_metrics:
                                    self.kv_transfer_metrics.log_kv_transfer_failed()
                                    logger.error(f"Transfer failed detected!")

                                self._fail_session_and_chunk(
                                    req.trans_info.mooncake_session_id,
                                    req.trans_info.endpoint,
                                    req.trans_info.dst_port,
                                    kv_chunk.room,
                                    reason="submit_layer_cache",
                                )

                                task.polls.append(False)
                                abort_tasks.append(task)
                                break

                    task.next_layer_id += 1

                # submit aux data
                if (
                    kv_chunk.is_last
                    and task.aux_step is not None
                    and StepCounter.is_step_ready(ready_aux_step, task.aux_step)
                ):
                    task.aux_step = None  # reset to None to mark aux has been submitted
                    if kv_chunk.is_last:
                        for req in task.write_requests:
                            aux_bid = self.submit_aux(
                                req.trans_info.mooncake_session_id,
                                kv_chunk.prefill_aux_index,
                                self.decode_kv_args_table[req.trans_info.mooncake_session_id].dst_aux_ptrs,
                                req.trans_info.dst_aux_index
                            )
                            req.submit_bids.append(aux_bid)

                if task.next_layer_id == self.effective_layer_num and task.aux_step is None:
                    complete_tasks.append(task)

            return complete_tasks, abort_tasks

        def pop_transfered(tasks: List[LayerWiseTask]) -> List[LayerWiseTask]:
            complete_tasks: List[LayerWiseTask] = []
            for task in tasks:
                kv_chunk = task.kv_chunk
                for req in task.write_requests[len(task.polls):]:  # only check the uncompleted requests
                    success = discard_finished_bid_inplace(req.submit_bids)
                    if success:
                        if len(req.submit_bids) == 0:
                            task.polls.append(True)
                    else:
                        self._fail_session_and_chunk(
                            req.trans_info.mooncake_session_id,
                            req.trans_info.endpoint,
                            req.trans_info.dst_port,
                            kv_chunk.room,
                            reason="submit_status",
                        )

                        task.polls.append(False)
                        break

                # all finished or any failed
                if (
                    len(task.polls) == len(task.write_requests)
                    or (len(task.polls) > 0 and not all(task.polls))
                ):
                    complete_tasks.append(task)

            return complete_tasks

        def finalize(tasks: List[LayerWiseTask]) -> None:
            for task in tasks:
                kv_chunk = task.kv_chunk
                status = KVPoll.Success if all(task.polls) else KVPoll.Failed
                if (status == KVPoll.Failed or kv_chunk.is_last): # last chunk or any failed
                    self.update_status(kv_chunk.room, status)
                    prefill_unique_rank=self.attn_tp_size*self.pp_rank+self.attn_tp_rank
                    for packed_req in task.write_requests:
                        endpoint, dst_port, room = packed_req.dst_ranks_info
                        self.sync_status_to_decode_endpoint(
                            endpoint, dst_port, room, status, prefill_unique_rank
                        )

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

        pending_tasks: List[LayerWiseTask] = []
        inflight_tasks: List[LayerWiseTask] = []
        while True:
            try:
                if len(new_tasks := get_new_tasks(blocking=(len(pending_tasks) + len(inflight_tasks) == 0))) > 0:
                    if len(abort_tasks := initialize(new_tasks)) > 0:
                        finalize(abort_tasks)
                        new_tasks = discard_tasks(new_tasks, abort_tasks)

                    pending_tasks.extend(new_tasks)

                if len(pending_tasks) > 0:
                    ready_cache_step, ready_aux_step = query_ready_step(pending_tasks[0]) # only wait the first task
                    submited_tasks, abort_tasks = submit_transfer(pending_tasks, ready_cache_step, ready_aux_step)
                    finalize(abort_tasks)
                    pending_tasks = discard_tasks(pending_tasks, submited_tasks + abort_tasks)
                    inflight_tasks.extend(submited_tasks)

                if len(complete_tasks := pop_transfered(inflight_tasks)) > 0:
                    finalize(complete_tasks)
                    inflight_tasks = discard_tasks(inflight_tasks, complete_tasks)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )


class MooncakeKVSender(BaseKVSender):

    def __init__(
        self, mgr: MooncakeKVManager, bootstrap_addr: str, bootstrap_room: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.init_time = None
        self.conclude_state = None
        # inner state
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None, decode_prefix_len: Optional[int] = 0):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        self.decode_prefix_len = decode_prefix_len
        self.init_time = time.time()

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
                self.bootstrap_room, kv_indices, index_slice, False, mla_l1_5_args=mla_l1_5_args
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                mla_l1_5_args=mla_l1_5_args,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
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

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
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

        if self.bootstrap_addr not in self.kv_mgr.prefill_parallel_info_table:
            self.prefill_tp_size, self.prefill_dp_size, self.prefill_enable_mla_l1_5_cache, self.prefill_pp_size = (
                self._get_prefill_parallel_info_from_server()
            )
            if (self.prefill_tp_size is None or self.prefill_dp_size is None or self.prefill_pp_size is None
                or self.prefill_enable_mla_l1_5_cache is None):
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_tp_size}"
                )
                self.kv_mgr.prefill_parallel_info_table[self.bootstrap_addr] = (
                    self.prefill_tp_size, self.prefill_dp_size, self.prefill_enable_mla_l1_5_cache, self.prefill_pp_size
                )
        else:
            self.prefill_tp_size, self.prefill_dp_size, self.prefill_enable_mla_l1_5_cache, self.prefill_pp_size \
                = self.kv_mgr.prefill_parallel_info_table[self.bootstrap_addr]

        # Currently, we don't allow prefill instance and decode instance to
        # have different TP sizes per DP rank, except for models using MLA.
        local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size

        #if not self.kv_mgr.draft_is_mla_backend and self.kv_mgr.server_args.speculative_algorithm is not None:
        #    assert local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank, "PD with different TP sizes per DP rank is not yet supported for non-MLA draft models"
        if self.prefill_enable_mla_l1_5_cache:
            assert (
                self.kv_mgr.is_mla_backend
            )
            self.target_tp_ranks = [i for i in range(prefill_tp_size_per_dp_rank)]
            self.target_tp_rank = None # make all tp ranks not dummy rank
            self.required_dst_info_num = local_tp_size_per_dp_rank
            self.required_prefill_response_num = prefill_tp_size_per_dp_rank
        elif local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
            self.required_prefill_response_num = 1
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
            self.target_tp_rank = (
                self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank
            ) // (local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank)
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.target_tp_ranks = [self.target_tp_rank]
            self.required_prefill_response_num = 1
        else:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.tp_rank % local_tp_size_per_dp_rank + 1)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
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

        assert self.kv_mgr.pp_size==1, "decode not support pp now"
        self.target_pp_ranks=[rank for rank in range(self.prefill_pp_size)]
        self.required_prefill_response_num*=self.prefill_pp_size
        self.target_dp_rank = self.bootstrap_room % self.prefill_dp_size
        self.kv_mgr.register_decode_response_tracker(
            self.bootstrap_room,
            self.required_prefill_response_num,
        )

        # NOTE: key distinguished by bootstrap_addr, target_dp_rank, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_rank}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            # Enable higher PP ranks to be bootstrapped earlier to make PP PD requests bootstrap more robust
            for target_pp_rank in reversed(self.target_pp_ranks):
                for target_tp_rank in self.target_tp_ranks:
                    bootstrap_info = self._get_bootstrap_info_from_server(
                        target_tp_rank,
                        self.target_dp_rank,
                        target_pp_rank
                    )
                    if bootstrap_info is not None:
                        # NOTE: only support MLA for now: select one prefill rank as real rank
                        bootstrap_info["is_dummy"] = not bool(
                            target_tp_rank == self.target_tp_rank
                            or self.target_tp_rank is None
                        )
                        logger.debug(
                            f"Fetched bootstrap info: {bootstrap_info} for PP {target_pp_rank} DP {self.target_dp_rank} TP {target_tp_rank}"
                        )
                        bootstrap_infos.append(bootstrap_info)
                    else:
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Could not fetch bootstrap info for tp_rank: {target_tp_rank} and target_dp_rank: {self.target_dp_rank} {target_pp_rank=}",
                        )
                        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                        return

            self.bootstrap_infos = bootstrap_infos

            # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
            try:
                self._register_kv_args()
            except zmq.error.ZMQError as e:
                logger.warning(
                    f"Failed to register kv_args to prefill: {e}, prefill may be dead"
                )
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"ZMQ send failed during kv_args registration: {e}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return

            self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0
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
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_prefill_parallel_info_from_server(self) -> Tuple[int, int, bool, int]:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?tp_rank={-1}&target_dp_rank={-1}&target_pp_rank={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return (
                    int(prefill_parallel_info["prefill_tp_size"]),
                    int(prefill_parallel_info["prefill_dp_size"]),
                    bool(prefill_parallel_info["enable_mla_l1_5_cache"]),
                    int(prefill_parallel_info["prefill_pp_size"]),
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None, None, None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None, None, None

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            offsets=self.kv_mgr.kv_args.offsets
            packed_offsets_parts=[]
            packed_offsets_parts.append(struct.pack("I", len(offsets)))  # num_layers
            for layer_offsets in offsets:
                packed_offsets_parts.append(struct.pack("I", len(layer_offsets)))  # num_offset
                for offset in layer_offsets:
                    packed_offsets_parts.append(struct.pack("I", offset))  # offset
            packed_offsets=b"".join(packed_offsets_parts)

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
                        str(getattr(self, 'decode_prefix_len', 0)).encode("ascii"),
                        packed_offsets
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
            except Exception:
                pass
            logger.info(f"Closed cached ZMQ socket for {endpoint}")

    def init(
        self,
        kv_indices: npt.NDArray[np.int64],
        aux_index: Optional[int] = None,
        decode_prefix_len: Optional[int] = 0,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
    ):
        # Store decode_prefix_len to be sent back to prefill
        self.decode_prefix_len = decode_prefix_len
        dst_page_transfer_mask = None
        dst_page_local_indices = None
        if mla_l1_5_args is not None:
            dst_page_transfer_mask = mla_l1_5_args.page_transfer_mask
            dst_page_local_indices = mla_l1_5_args.page_local_indices

        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
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
                        ]
                    )
            except zmq.error.ZMQError as e:
                logger.warning(
                    f"Failed to send init to prefill {self.prefill_server_url}: {e}, "
                    f"room={self.bootstrap_room}, prefill may be dead"
                )
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"ZMQ send failed to {self.prefill_server_url}: {e}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed
            elif status == KVPoll.Transferring:
                logger.warning(f"Req(room={self.bootstrap_room}) in Transferring, which is unexpected")

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        self.kv_mgr.clear_decode_response_state(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
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
        self.prefill_port_table: Dict[int, Dict[int, Dict[int, Dict[str, Union[str, int]]]]] = {}
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
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        tp_size = data["tp_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        tp_rank = int(data["tp_rank"])
        self.enable_mla_l1_5_cache = bool(data["enable_mla_l1_5_cache"])
        pp_rank=int(data["pp_rank"])
        pp_size=int(data["pp_size"])

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
                dp_group_table=self.prefill_port_table.setdefault(dp_rank, {})
                tp_group_table=dp_group_table.setdefault(tp_rank_in_dp_group, {})

                tp_group_table[pp_rank] = {
                    "rank_ip": rank_ip,
                    "rank_port": rank_port,
                }

            logger.info(
                f"Register prefill bootstrap: {pp_rank=} {tp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        tp_rank = request.query.get("tp_rank")
        target_dp_rank = request.query.get("target_dp_rank")
        target_pp_rank=request.query.get("target_pp_rank")
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
