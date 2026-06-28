from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import fastapi
import zmq

from sglang.srt.managers.io_struct import (
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SetSchedulerMaxRunningRequestsOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Future] = deque()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj:
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj:
                self._sender.send_pyobj(obj)

        await self._result_event.wait()
        result_values = copy.deepcopy(self._result_values)
        self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()


class TokenizerCommunicatorMixin:
    """Mixin class for TokenizerManager to handle communication with the scheduler."""

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        # Communicators
        self.init_weights_update_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_tensor_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_weights_by_name_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.flush_cache_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.profile_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.health_check_communitcator = _Communicator(self.send_to_scheduler, 1)
        self.get_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        # Add new communicator
        self.set_scheduler_max_running_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        self.expert_distribution_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_load_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size, mode="watching"
        )


        self._result_dispatcher += self._get_communicator_dispatcher()

    def _get_communicator_dispatcher(self: TokenizerManager):
        return TypeBasedDispatcher(
            [
                (
                    InitWeightsUpdateGroupReqOutput,
                    self.init_weights_update_group_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromTensorReqOutput,
                    self.update_weights_from_tensor_communicator.handle_recv,
                ),
                (
                    GetWeightsByNameReqOutput,
                    self.get_weights_by_name_communicator.handle_recv,
                ),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    SetInternalStateReqOutput,
                    self.set_internal_state_communicator.handle_recv,
                ),
                (
                    SetSchedulerMaxRunningRequestsOutput,
                    self.set_scheduler_max_running_communicator.handle_recv,
                ),
                (
                    ExpertDistributionReqOutput,
                    self.expert_distribution_communicator.handle_recv,
                ),
                (
                    GetLoadReqOutput,
                    self.get_load_communicator.handle_recv,
                ),
            ]
        )

    async def flush_cache(self: TokenizerManager) -> FlushCacheReqOutput:
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    async def start_profile(
        self: TokenizerManager,
        output_dir: Optional[str] = None,
        start_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        profile_by_stage: bool = False,
    ):
        self.auto_create_handle_loop()
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        with_stack = False if with_stack is False or env_with_stack is False else True
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=profile_by_stage,
            profile_id=str(time.time()),
        )
        return await self._execute_profile(req)

    async def stop_profile(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self: TokenizerManager, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def start_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.START_RECORD)

    async def stop_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.STOP_RECORD)

    async def dump_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.DUMP_RECORD)

    async def init_weights_update_group(
        self: TokenizerManager,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.init_weights_update_group_communicator(obj))[0]
        return result.success, result.message

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be for update weights from distributed"

        # This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            result = (await self.update_weights_from_distributed_communicator(obj))[0]
            return result.success, result.message

    async def update_weights_from_tensor(
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"

        # This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            result = (await self.update_weights_from_tensor_communicator(obj))[0]
            return result.success, result.message

    async def get_weights_by_name(
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self: TokenizerManager,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self: TokenizerManager,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
        req = GetInternalStateReq()
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
        return [res.updated for res in responses]

    async def get_load(self: TokenizerManager) -> List[GetLoadReqOutput]:
        req = GetLoadReqInput()
        return await self.get_load_communicator(req)
