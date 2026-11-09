import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.io_struct import HeapDumpReq, HeapDumpReqOutput, ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_int_env_var, get_str_env_var, is_cuda, is_npu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduling.scheduler import Scheduler


class SchedulerProfilerMixin:
    def init_profiler(self: "Scheduler"):
        self.torch_profiler = None
        self.torch_npu_profiler = None
        self.torch_profiler_output_dir: Optional[str] = None
        self.profiler_activities: Optional[list[str]] = None
        self.profile_id: Optional[str] = None
        self.profiler_start_forward_ct: Optional[int] = None
        self.profiler_target_forward_ct: Optional[int] = None
        self.profiler_target_prefill_ct: Optional[int] = None
        self.profiler_target_decode_ct: Optional[int] = None
        self.profiler_prefill_ct: Optional[int] = None
        self.profiler_decode_ct: Optional[int] = None
        self.profile_by_stage: bool = False
        self.profile_steps: Optional[int] = None
        self.profile_in_progress: bool = False
        self.rpd_profiler = None
        self.npu_start_profile = False

    def _init_npu_profiling(self):
        npu_profiling_save_path = get_str_env_var("NPU_PROF_SAVE_DIR", "")
        if is_npu() and npu_profiling_save_path:
            logger.info("NPU Profiling is activated.")
            import torch_npu

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            )
            self.npu_custom_profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU,
                ],
                with_stack=False,  # Whether to record the call stack
                record_shapes=True,  # Whether to record operator shapes
                profile_memory=False,  # Whether to profile GPU memory
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(
                    wait=get_int_env_var("NPU_PROF_WAIT_TIME", 10),  # Number of steps to skip
                    warmup=get_int_env_var("NPU_PROF_WARMUP_TIME", 1),  # Number of steps for warmup after wait
                    active=get_int_env_var("NPU_PROF_ACTIVE_TIME", 5),  # Number of steps to profile after warmup
                    repeat=get_int_env_var("NPU_PROF_REPEAT_TIME", 1),  # How many times to repeat the above process
                ),
                # Save path for profile log, must use tensorboard_trace_handler for viewing with tb
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(npu_profiling_save_path),
            )
            self.npu_custom_profiler.prof_if.prof_if = self.npu_custom_profiler.prof_if
        else:
            self.npu_custom_profiler = None

    def init_profile(
        self: "Scheduler",
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[list[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
    ) -> ProfileReqOutput:
        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        self.profile_by_stage = profile_by_stage

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/workdir/prof")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = output_dir
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)

        if num_steps:
            self.profile_steps = num_steps
            if self.profile_by_stage:
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
            elif start_step:
                self.profiler_target_forward_ct = self.profiler_start_forward_ct + num_steps
            else:
                self.profiler_target_forward_ct = self.forward_ct + num_steps
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def start_profile(self: "Scheduler", stage: Optional[ForwardMode] = None) -> Optional[ProfileReqOutput]:
        stage_str = f" for {stage.__str__()}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [activity_map[a] for a in activities if a in activity_map] if not is_npu() else None

        if "RPD" in activities:
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
            )

            if self.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.tp_cpu_group)

            self.rpd_profiler = rpdTracerControl()
            self.rpd_profiler.setPythonTrace(True)
            self.rpd_profiler.start()
            self.rpd_profiler.rangePush("", "rpd profile range", "")
            self.profile_in_progress = True
        elif torchprof_activities is not None:
            assert is_cuda(), "Profiling on GPU."
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else False,
                record_shapes=record_shapes if record_shapes is not None else False,
            )
            self.torch_profiler.start()
            self.profile_in_progress = True
            # Capture grammar host-func timing so it can be spliced onto the
            # timeline at stop; the CUDA driver callback thread is invisible to
            # torch.profiler otherwise (see grammar_hostfunc module header).
            try:
                from sglang.srt.speculative.grammar_hostfunc import (
                    enable_span_capture,
                )

                enable_span_capture()
            except Exception:
                logger.warning("Failed to enable span capture for grammar hostfunc profiling.")
        else:
            assert is_npu(), "Profiling on NPU."
            logger.info("NPU Profiling is activated.")
            import torch_npu

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            )
            self.torch_npu_profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU,
                ],
                with_stack=with_stack if with_stack is not None else True,  # 是否记录调用栈
                record_shapes=True,  # 是否记录运算符的形状
                profile_memory=False,  # 是否对显存进行 profile
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(
                    wait=0,  # 需要跳过的 steps
                    warmup=0,  # wait 之后用几个 step 来 warmup
                    active=1,  # 对 warmup 之后的几个 step 进行 profile
                    repeat=1,  # 以上过程重复几遍
                ),
                # 保存 profile log 的地址，想用 tb 看这里必须是 tensorboard_trace_handler
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    self.torch_profiler_output_dir  # 统一存储路径
                ),
            )
            self.torch_npu_profiler.prof_if.prof_if = self.torch_npu_profiler.prof_if
            # Recording
            self.torch_npu_profiler.start()

            self.profile_in_progress = True

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self.profile_in_progress = True

        if "CUDA_PROFILER" in activities:
            torch.cuda.cudart().cudaProfilerStart()
            self.profile_in_progress = True

        return ProfileReqOutput(success=True, message="Succeeded")

    def stop_profile(self: "Scheduler", stage: Optional[ForwardMode] = None) -> Optional[ProfileReqOutput]:
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        if not Path(self.torch_profiler_output_dir).exists():
            Path(self.torch_profiler_output_dir).mkdir(parents=True, exist_ok=True)

        stage_suffix = f"-{stage.__str__()}" if stage else ""
        logger.info("Stop profiling" + stage_suffix + "...")
        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            trace_path = os.path.join(
                self.torch_profiler_output_dir,
                self.profile_id + f"-TP-{self.tp_rank}" + stage_suffix + ".trace.json.gz",
            )
            self.torch_profiler.export_chrome_trace(trace_path)
            # Splice grammar host-func spans onto the exported timeline.
            try:
                from sglang.srt.speculative.grammar_hostfunc import (
                    disable_span_capture,
                    drain_spans,
                    inject_spans_into_trace,
                )

                disable_span_capture()
                inject_spans_into_trace(trace_path, drain_spans())
            except Exception as e:
                logger.warning("Failed to inject grammar host-func spans: %s", e)
            torch.distributed.barrier(self.tp_cpu_group)

        if self.torch_npu_profiler is not None:
            torch.npu.synchronize()

            try:
                logger.info("Triggering flush (step)...")
                # Flush
                self.torch_npu_profiler.step()
            except Exception as e:
                logger.warning(f"Profiler step failed (possibly already stopped): {e}")

            time.sleep(5)

            try:
                self.torch_npu_profiler.stop()
            except Exception as e:
                logger.info(f"Profiler stop called (final cleanup): {e}")

        if self.rpd_profiler is not None:
            self.rpd_profiler.rangePop()
            self.rpd_profiler.stop()
            self.rpd_profiler.flush()

            torch.distributed.barrier(self.tp_cpu_group)
            if self.tp_rank == 0:
                from sglang.srt.utils import rpd_to_chrome_trace

                rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
            self.rpd_profiler = None
            self.rpd_profiler_path = None

        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                str(time.time()) + f"-TP-{self.tp_rank}-memory" + stage_suffix + ".pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            torch.cuda.cudart().cudaProfilerStop()

        logger.info(
            "Profiling done. Traces are saved to: %s",
            self.torch_profiler_output_dir,
        )
        self.torch_profiler = None
        self.torch_npu_profiler = None
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded.")

    def profile_batch_predicate(self: "Scheduler", batch):
        if self.profile_by_stage:
            if batch.forward_mode.is_prefill():
                if self.profiler_prefill_ct == 0:
                    self.start_profile(batch.forward_mode)
                self.profiler_prefill_ct += 1
                if self.profiler_prefill_ct > self.profiler_target_prefill_ct and self.profile_in_progress:
                    self.stop_profile(stage=ForwardMode.EXTEND)
            elif batch.forward_mode.is_decode():
                if self.profiler_decode_ct == 0:
                    if self.profile_in_progress:
                        # force trace flush
                        self.stop_profile(ForwardMode.EXTEND)
                    self.start_profile(batch.forward_mode)
                self.profiler_decode_ct += 1
                if self.profiler_decode_ct > self.profiler_target_decode_ct and self.profile_in_progress:
                    self.stop_profile(stage=ForwardMode.DECODE)
            elif batch.forward_mode.is_idle():
                pass
            else:
                raise RuntimeError(f"unsupported profile stage: {batch.forward_mode}")
        else:
            # Check profiler
            if self.profiler_target_forward_ct and self.profiler_target_forward_ct <= self.forward_ct:
                self.stop_profile()
            if self.profiler_start_forward_ct and self.profiler_start_forward_ct == self.forward_ct:
                self.start_profile()

    def heap_dump(self, recv_req: HeapDumpReq):
        """Trigger a jemalloc heap profile dump via mallctl."""
        import ctypes

        logger.info("Heap dump triggered.")

        try:
            libc = ctypes.CDLL(None)
            mallctl = libc.mallctl
            mallctl.restype = ctypes.c_int
            mallctl.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]

            ret = mallctl(b"prof.dump", None, None, None, 0)
            if ret != 0:
                msg = f"mallctl prof.dump failed with ret={ret}"
                logger.warning(msg)
                return HeapDumpReqOutput(success=False, message=msg)

            msg = f"Heap profile dumped for pid={os.getpid()}"
            logger.info(msg)
            return HeapDumpReqOutput(success=True, message=msg)
        except Exception as e:
            msg = f"Failed to dump heap: {e}"
            logger.warning(msg)
            return HeapDumpReqOutput(success=False, message=msg)

    def profile(self: "Scheduler", recv_req: ProfileReq):
        if recv_req.type == ProfileReqType.START_PROFILE:
            if recv_req.profile_by_stage or recv_req.start_step:
                return self.init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                )
            else:
                self.init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                )
                return self.start_profile(True)
        else:
            return self.stop_profile()
