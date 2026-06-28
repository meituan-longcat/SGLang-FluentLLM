import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, PageTransferMetadata
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# For warmup reqs, we don't kv transfer, we use the fake sender and receiver
class FakeKVSender(BaseKVSender):
    def __init__(self, mgr: BaseKVManager, bootstrap_addr: str, bootstrap_room: int):
        self.has_sent = False

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.debug("FakeKVSender poll success")
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
        decode_prefix_len: Optional[int] = 0,
    ):
        self.decode_prefix_len = decode_prefix_len
        logger.debug(
            f"FakeKVSender init with kv_indices: {kv_indices}, aux_index: {aux_index}, decode_prefix_len: {decode_prefix_len}"
        )
        pass

    def send(
        self, 
        kv_indices: npt.NDArray[np.int64],
        start_idx: Optional[int] = 0,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
    ):
        self.has_sent = True
        logger.debug(f"FakeKVSender send with kv_indices: {kv_indices}")

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.has_init = False
        self.decode_prefix_len = 0

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.debug("FakeKVReceiver poll success")
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
        decode_prefix_len: Optional[int] = 0,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
    ):
        self.has_init = True
        self.decode_prefix_len = decode_prefix_len
        logger.debug(
            f"FakeKVReceiver init with kv_indices: {kv_indices}, aux_index: {aux_index}, decode_prefix_len: {decode_prefix_len}"
        )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class FakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        draft_is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.is_mla_backend = is_mla_backend
        self.draft_is_mla_backend = draft_is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.decode_prefix_lengths = {}
    
    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        return self.decode_prefix_lengths.get(bootstrap_room, 0)
    
    def store_decode_prefix_info(self, bootstrap_room: int, decode_prefix_len: int):
        """Store decode prefix info from decode side"""
        self.decode_prefix_lengths[bootstrap_room] = decode_prefix_len
    
    def send_decode_prefix_info(self, bootstrap_room: int, decode_prefix_len: int):
        """Send decode prefix info to prefill side - in fake backend this is handled via direct method calls"""
        # In fake implementation, decode prefix info is handled via direct method calls
        # This method is kept for compatibility with base interface
        pass
