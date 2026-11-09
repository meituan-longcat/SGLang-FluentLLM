import os
from typing import Optional

import torch

from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.utils import reset_cp_metadata
from sglang.srt.managers.scheduling.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecDeocdingWorker
from sglang.srt.utils import get_colorful_logger, is_npu
from sglang.srt.utils.oe_utils import update_token_table

__is_npu__ = is_npu()
_DFLASH_CHECK_SLOTS = bool(os.environ.get("DFLASH_CHECK_SLOTS"))



logger = get_colorful_logger(__name__)


class DFlashWorker(BaseSpecDeocdingWorker, TpModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        attn_tp_rank: int,
        moe_ep_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
        global_rank: int,
    ):
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        TpModelWorker.__init__(
            self,
            server_args=server_args,
            gpu_id=gpu_id,
            attn_tp_rank=attn_tp_rank,
            moe_ep_rank=moe_ep_rank,
            global_rank=global_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=target_worker.model_runner.req_to_token_pool,
            kv_allocator=target_worker.model_runner.kv_allocator,
            oe_token_table=target_worker.model_runner.oe_token_table,
        )

        BaseSpecDeocdingWorker.__init__(self, server_args=server_args, gpu_id=gpu_id, target_worker=target_worker, drafter_use_oe=False)

        if __is_npu__:
            self.start_grammar_update_thread()

        self.block_size = int(server_args.speculative_num_draft_tokens)
        self.draft_model = self.model_runner.model
        self.mask_token_id = self._resolve_mask_token_id()
        captured = server_args.target_layers_need_capture
        expected = list(self.draft_model.target_layer_ids)
        if captured is None or list(captured) != expected:
            raise ValueError(
                f"DFLASH --target-layers-need-capture={captured} 与 draft config 的 "
                f"target_layer_ids={expected} 不一致 (顺序也须相同, fc 按此顺序 cat)。"
            )
        assert hasattr(self, "model_runner_list")
        draft_attn0 = self.draft_model.layers[0].self_attn

        _rope = draft_attn0.rotary_emb
        _cache = getattr(_rope, "cos_sin_cache", None)
        if _cache is None:
            _cache = getattr(_rope, "cos_cached", None)  # prepare_for_npu 后的形态
        self._max_rope_position = int(_cache.shape[0]) - 1
        logger.info(
            f"DFLASH draft rope 类={type(_rope).__name__}, 表长={_cache.shape[0]}, "
            f"最大合法 position={self._max_rope_position}"
        )

        for runner in self.model_runner_list:
            self.init_drafter_embedding(drafter_model_runner=runner)
            runner.attn_backend.dflash_swa_enabled = draft_attn0.use_swa_sink_mode
            runner.attn_backend.dflash_swa_first_kv = draft_attn0.context_sink_size
            runner.attn_backend.dflash_swa_window = draft_attn0.sliding_window_size

        self.cuda_graph_runner = None
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph

        # 图模式下材化走 target 图; eager (prefill / 排查) 下仍走散算子。
        self.materialize_in_graph: bool = __is_npu__ and ENV.npu_enable_graph
        if self.materialize_in_graph:
            self._attach_materializer()

        if global_server_args_dict["enable_speculative_sampling"]:
            logger.info(
                "DFLASH speculative sampling 已启用: 非 greedy 批 draft 侧 top-k 采样并记录 q, "
                "verify 走 p/q 校验; greedy 批仍为 argmax 链式比对"
            )

        if not __is_npu__:
            from sglang.srt.speculative.dflash_cuda_graph_runner import DFlashCudaGraphRunner

            self.init_cuda_graphs(graph_runner_cls=DFlashCudaGraphRunner)
        # NPU: 非 overlap 走 draft eager; overlap 下由 EagleWorkerOverlapped 捕获 draft 图。

    def _attach_materializer(self) -> None:
        """挂材化模块到 target 的 logits_processor, 供 target ge 图 trace。

        须在 target 图 capture 前完成 (EagleWorkerOverlapped 先构造 worker 再
        init_npu_graphs, 时序天然成立)。
        """
        from sglang.srt.speculative.dflash_materializer import DFlashMaterializer

        compile_bs = self.server_args.npu_compile_bs
        max_bs = max(int(x) for x in compile_bs.split(",")) if compile_bs else self.max_running_requests
        materializer = DFlashMaterializer(
            draft_model=self.draft_model,
            draft_pool=self.model_runner.token_to_kv_pool,
            max_rope_position=self._max_rope_position,
            max_num_tokens=max_bs * self.block_size,
        )
        logits_processor = self.target_worker.model_runner.model.logits_processor
        object.__setattr__(logits_processor, "dflash_materializer", materializer)
        logger.info(f"DFLASH 材化入 target 图: max_num_tokens={max_bs * self.block_size}")

    def _resolve_mask_token_id(self) -> int:
        mask_token_id = getattr(self.draft_model, "mask_token_id", None)
        if mask_token_id is None:
            config = self.model_runner.model_config.hf_config
            mask_token_id = getattr(config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("DFLASH 需要 mask_token_id(配在 draft config 的 dflash_config.mask_token_id 或 model config 里)。")
        assert 0 <= int(mask_token_id) < self.vocab_size
        return int(mask_token_id)

    def forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch, launch_done=None):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)
        if bool(
            __is_npu__
            and self.npu_graph_runner_for_draft_extend
            and (forward_batch.all_decode_or_idle or global_server_args_dict.get("npu_disable_all_gather", False))
        ):
            output = self._forward_decode_spec_npu_graph(forward_batch)
        elif forward_batch.forward_mode.is_target_verify():
            vocab_masks = self.preprocess_for_verify(forward_batch)
            if self.cuda_graph_runner is not None and self.cuda_graph_runner.can_run(forward_batch):
                output = self.cuda_graph_runner.replay(forward_batch, vocab_masks)
            else:
                self.init_attn_backends(forward_batch)
                output = self.forward_decode_spec(forward_batch, vocab_masks)
        elif forward_batch.forward_mode.is_extend():
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            output = self.forward_prefill_spec(model_worker_batch, forward_batch)
        elif forward_batch.forward_mode.is_idle():
            output = self.forward_idle(forward_batch)
        else:
            raise ValueError(f"DFLASH 不支持的 forward mode: {forward_batch.forward_mode}")
        if launch_done:
            launch_done.set()
        return output

    def multi_batch_select(self, max_input_batch, compile_bs_list):
        max_compile_bs = max(compile_bs_list)
        assert max_input_batch <= max_compile_bs, (
            f"max input batch ({max_input_batch}) should less equal than max compile bs({max_compile_bs}) in graph mode"
        )
        return min(bs for bs in compile_bs_list if bs >= max_input_batch)

    # 须用 no_grad 而非 inference_mode: 后者创建的 inference tensor 与 capture dummy
    # (普通 tensor) dispatch key 不同, guard 失配重编译。
    @torch.no_grad()
    def _forward_decode_spec_npu_graph(self, forward_batch: ForwardBatch):
        self.req_to_token_pool.verified_lens[0] = 0
        vocab_masks = None
        if forward_batch.forward_mode.is_target_verify():
            vocab_masks = self.preprocess_for_verify(forward_batch)
        if global_server_args_dict.get("npu_disable_all_gather", False):
            padding_size = max(self.npu_graph_runner_for_draft_extend.compile_bs)
        else:
            max_input_bs = max(forward_batch.global_num_tokens) / self.server_args.speculative_num_draft_tokens
            padding_size = self.multi_batch_select(max_input_bs, self.npu_graph_runner_for_draft_extend.compile_bs)

        forward_batch.padding_for_graph_mtp(self.target_worker.model_runner, padding_size)
        if self.target_worker.use_over_embedding and forward_batch.oe_token_table is None:
            forward_batch.oe_token_table = self.target_worker.model_runner.oe_token_table
        forward_batch.can_run_all2all = ENV.npu_enable_all2all_comm
        self.init_attn_backends(forward_batch)
        out = self.forward_decode_spec(forward_batch, vocab_masks)

        bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
        mtpn_factor = self.server_args.speculative_num_steps + 1

        def unpad(ori, n):
            if isinstance(ori, torch.Tensor):
                return ori[:n]
            if isinstance(ori, list):
                return [x[:n] for x in ori]
            return ori

        logits_output, output_ids, commit_lens, new_verified_id, token_list, grammar_terminated_list = out
        logits_output.next_token_logits = unpad(logits_output.next_token_logits, bs * mtpn_factor)
        logits_output.hidden_states = unpad(logits_output.hidden_states, bs * mtpn_factor)
        return (
            logits_output,
            unpad(output_ids, bs * mtpn_factor),
            unpad(commit_lens, bs),
            unpad(new_verified_id, bs),
            unpad(token_list, bs),
            grammar_terminated_list,
        )

    def forward_prefill_spec(self, model_worker_batch: ModelWorkerBatch, forward_batch: ForwardBatch):
        if self.target_worker.use_over_embedding:
            batch_size = forward_batch.batch_size
            forward_batch.oe_column_starts[:batch_size] = forward_batch.extend_prefix_lens
            forward_batch.oe_req_lens[:batch_size] = forward_batch.extend_seq_lens

        if __is_npu__:
            target_logits_output = self.target_worker.model_runner.forward(forward_batch)
        else:
            self.target_worker.model_runner.attn_backend.init_forward_metadata(forward_batch)
            target_logits_output = self.target_worker.model_runner.forward_extend(forward_batch, skip_metadata_init=True)

        # 非末 stage 只跨 stage 传 PPProxyTensors,不做 spec。
        if self.pp_group.is_last_rank:
            next_token_ids = self.target_worker.model_runner.sample(target_logits_output, forward_batch)
            if model_worker_batch.disagg_set_aux_fn is not None:
                model_worker_batch.disagg_set_aux_fn(next_token_ids, target_logits_output)
            if target_logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH prefill 需要 target 多层 aux 抓取(capture_hidden_mode=FULL + set_target_layers_need_capture),但 hidden_states 为 None。"
                )

            draft_cache_loc = self._resolve_draft_cache_loc(forward_batch)
            self._materialize_target_hidden_to_draft_kv(
                target_hidden=target_logits_output.hidden_states,
                positions=forward_batch.positions,
                cache_loc=draft_cache_loc,
                tag="prefill",
            )
            target_logits_output.hidden_states = None

            self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] = (
                forward_batch.extend_prefix_lens + forward_batch.extend_seq_lens
            )

            token_list = self._propose_block(forward_batch, verified_id=next_token_ids)
            return (
                target_logits_output,
                next_token_ids,
                None,
                next_token_ids,
                token_list,
                None,  # prefill 无 grammar_terminated_list (grammar accept 走 scheduler 常规路径)
            )
        else:
            return (target_logits_output, None, None, None, None, None)

    def forward_decode_spec(self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor]):
        if __is_npu__ and not ENV.npu_enable_graph:
            forward_batch.padding_for_dflash_eager(self.target_worker.model_runner)

        if self.target_worker.use_over_embedding:
            batch_size = forward_batch.batch_size
            forward_batch.oe_column_starts[:batch_size] = self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
            forward_batch.oe_req_lens[:batch_size] = self.block_size

        # 前移到 verify 前: 只依赖本轮尚未 += 的 verified_lens, 顺序约束仍成立。
        # int32 与 capture dummy 对齐, 否则静默重编译。
        if self.materialize_in_graph:
            forward_batch.dflash_materialize_cache_loc = self._resolve_draft_cache_loc(
                forward_batch, num_slots=forward_batch.batch_size * self.block_size
            ).to(torch.int32)

        logits_output = self.forward_target_verify(forward_batch)
        target_predict, logits_output, accept_length, accept_index = self.rejection_sampling(forward_batch, logits_output, vocab_masks)

        if self.target_worker.use_over_embedding:
            batch_size = forward_batch.batch_size
            forward_batch.oe_out_column_starts[:batch_size] = self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] + 1
            forward_batch.oe_out_req_lens[:batch_size] = self.block_size
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=target_predict,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        output_ids = target_predict[accept_index]
        commit_lens = accept_length + 1

        need_grammar_thread = __is_npu__ and not self.server_args.disable_overlap_schedule
        if need_grammar_thread:
            output_ids_copy_done = torch.get_device_module(self.device).Event()
            output_ids_cpu = output_ids.clone().to("cpu", non_blocking=True, copy=True)
            commit_lens_cpu = commit_lens.clone().to("cpu", non_blocking=True, copy=True)
            output_ids_copy_done.record()

        self.compute_spec_logprobs(forward_batch, logits_output, commit_lens, output_ids)

        new_verified_id = self._committed_bonus(target_predict, accept_index, commit_lens)

        ori_bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
        padded_bs = forward_batch.batch_size
        need_unpad = __is_npu__ and not ENV.npu_enable_graph and ori_bs < padded_bs
        if need_unpad:
            output_ids = output_ids[: ori_bs * self.block_size]
            commit_lens_out = commit_lens[:ori_bs]
            new_verified_id_out = new_verified_id[:ori_bs]
            if logits_output.next_token_logits is not None:
                logits_output.next_token_logits = logits_output.next_token_logits[: ori_bs * self.block_size]
        else:
            commit_lens_out = commit_lens
            new_verified_id_out = new_verified_id

        # 图路径的材化已在 target 图内完成 (并已置空 hidden_states)。
        if not self.materialize_in_graph:
            draft_cache_loc = self._resolve_draft_cache_loc(
                forward_batch, num_slots=forward_batch.batch_size * self.block_size
            )
            self._materialize_target_hidden_to_draft_kv(
                target_hidden=logits_output.hidden_states,
                positions=forward_batch.positions,
                cache_loc=draft_cache_loc,
                tag="decode",
            )
            logits_output.hidden_states = None

        # pad 行 req_pool_indices 恒为 0 (保留槽), 其累加落在 verified_lens[0], 每轮开头清零。
        self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] += commit_lens

        # 以 padded_bs 进入, 与 idle rank 的 forward_idle -> _propose_block 对齐。
        token_list = self._propose_block(forward_batch, verified_id=new_verified_id)
        if need_unpad:
            token_list = token_list[:ori_bs]

        grammar_terminated_list_or_event = None
        if need_grammar_thread:
            grammar_terminated_list_or_event = self.submit_grammar_update(
                forward_batch, output_ids_copy_done, output_ids_cpu, commit_lens_cpu
            )
        return (logits_output, output_ids, commit_lens_out, new_verified_id_out, token_list, grammar_terminated_list_or_event)


    def forward_idle(self, forward_batch: ForwardBatch):
        if __is_npu__:
            forward_batch.padding_for_dflash_eager(self.target_worker.model_runner)
            logits_output = self.target_worker.model_runner.forward(forward_batch)
        else:
            logits_output = self.target_worker.model_runner.forward_idle(forward_batch)
        # 非末 stage 不跑 draft。
        if self.pp_group.is_last_rank and self.server_args.enable_dp_attention:
            bs = forward_batch.batch_size
            dummy_verified_id = torch.zeros(
                bs, dtype=torch.int32, device=forward_batch.positions.device
            )
            self._propose_block(forward_batch, verified_id=dummy_verified_id, is_idle_pad=True)
        return logits_output, None, None, None, None, None  # 末位 grammar_terminated_list 占位


    def forward_target_verify(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_target_verify()
        forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        if __is_npu__:
            logits_output = self.target_worker.model_runner.forward(forward_batch)
        else:
            logits_output = self.target_worker.model_runner.forward_extend(forward_batch, skip_metadata_init=True)
        return logits_output

    def _propose_block(
        self,
        forward_batch: ForwardBatch,
        verified_id: torch.Tensor,
        skip_metadata_init: bool = False,
        is_idle_pad: bool = False,
    ) -> torch.Tensor:
        batch_size = verified_id.shape[0]
        block = self.block_size
        device = verified_id.device

        seq_lens = self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]

        block_ids = torch.full((batch_size, block), self.mask_token_id, dtype=torch.int32, device=device)
        block_ids[:, 0] = verified_id.to(torch.int32)
        input_ids = block_ids.reshape(-1)

        positions = (
            seq_lens.unsqueeze(1) + torch.arange(block, device=device, dtype=torch.int32)
        ).flatten()
        positions.clamp_(max=self._max_rope_position)

        real_bs = 0 if is_idle_pad else getattr(forward_batch, "ori_bs", forward_batch.batch_size)
        block_out_cache_loc = self._draft_block_slots(forward_batch.req_pool_indices, seq_lens, batch_size, block, device, real_bs)

        if skip_metadata_init or __is_npu__:
            prefix_lens_cpu: Optional[list] = None
            seq_lens_sum: int = 0
        else:
            prefix_lens_cpu = seq_lens.tolist()
            seq_lens_sum = int((seq_lens + block).sum().item())

        draft_forward_batch = self._make_draft_forward_batch(
            forward_batch,
            input_ids,
            positions,
            block_out_cache_loc,
            seq_lens,
            prefix_lens_cpu,
            seq_lens_sum,
        )
        reset_cp_metadata()
        if __is_npu__:
            from sglang.srt.layers.attention.npu_mla_backend import update_dflash_extend_meta

            can_npu_graph = self.npu_graph_runner_for_draft_extend and self.npu_graph_runner_for_draft_extend.can_run(draft_forward_batch)
            if not can_npu_graph:
                self._materialize_draft_kv_lens(draft_forward_batch)
            update_dflash_extend_meta(draft_forward_batch, self.model_runner)

            if can_npu_graph:
                draft_output = self.npu_graph_runner_for_draft_extend.replay(draft_forward_batch)
            else:
                draft_output = self.model_runner.model.forward(
                    input_ids=draft_forward_batch.input_ids,
                    positions=draft_forward_batch.positions,
                    forward_batch=draft_forward_batch,
                )
        else:
            if not skip_metadata_init:
                self.model_runner.attn_backend.init_forward_metadata(draft_forward_batch)
            draft_output = self.model_runner.model.forward(
                input_ids=draft_forward_batch.input_ids,
                positions=draft_forward_batch.positions,
                forward_batch=draft_forward_batch,
                input_embeds=None,
            )
        if self._use_speculative_sampling(forward_batch, is_idle_pad):
            block_tokens = self._draft_block_sample(
                forward_batch, draft_output.next_token_logits, batch_size, block
            )
        else:
            _argmax_full = torch.argmax(draft_output.next_token_logits, dim=-1).reshape(batch_size, block)
            block_tokens = _argmax_full[:, 1:].to(torch.int32).contiguous()
        if self.target_worker.use_over_embedding and forward_batch.oe_token_table is not None:
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=block_tokens.reshape(-1),
                row_indices=forward_batch.req_pool_indices,
                column_starts=(seq_lens + 1).to(torch.int32),
                oe_req_lens=torch.full((batch_size,), block - 1, dtype=torch.int32, device=device),
            )
        return block_tokens

    def _use_speculative_sampling(
        self, forward_batch: ForwardBatch, is_idle_pad: bool
    ) -> bool:
        """draft 侧 speculative sampling 开关: flag 开 + 非 idle + 批内非全 greedy。

        verify 侧 npu_verify 按 is_all_greedy 对称分派; 两端判据不一致的轮次
        (如 batch N 全 greedy、batch N+1 非 greedy) q 会读到默认值, accept 退化一轮
        —— 与 MTP 的 draft_sample 同一暴露面, 不额外处理。
        仅 NPU: GPU 侧 verify (gpu_eagle_utils) 不消费 speculative_sampling_info,
        放开采样会静默改变 CUDA 路径的提案分布而不被校验。
        """
        if (
            is_idle_pad
            or not __is_npu__
            or not global_server_args_dict["enable_speculative_sampling"]
        ):
            return False
        sampling_info = forward_batch.sampling_info
        return (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and forward_batch.speculative_sampling_info is not None
        )

    def _draft_block_sample(
        self,
        forward_batch: ForwardBatch,
        next_token_logits: torch.Tensor,  # [bs*block, V]
        batch_size: int,
        block: int,
    ) -> torch.Tensor:
        """块内 top-k 采样替代 argmax, 并把 q 分布记录进 speculative_sampling_info。

        与 eagle_worker.draft_sample 同构; DFLASH 单次前向出整块, 一次写满
        slot 0..block-2 (slot j ↔ logits 行 j+1 ↔ 下轮 candidates[:, j+1],
        末 slot 按设计保持零 padding)。全程 device 算子, 可留在 launch-ahead 窗口。
        """
        spec_info = forward_batch.speculative_sampling_info
        sampling_info = forward_batch.sampling_info
        real_bs = getattr(forward_batch, "ori_bs", batch_size)
        assert block - 1 <= spec_info.probs_selected.shape[1], (
            f"DFLASH block={block} 超出 speculative_sampling_info 的 "
            f"num_draft_tokens={spec_info.probs_selected.shape[1] + 1}"
        )

        # 行 0 是 anchor 自身预测 (与 argmax 路径丢弃第 0 列同语义), 采样行 1..block-1
        logits = next_token_logits.reshape(batch_size, block, -1)[:, 1:].reshape(
            batch_size * (block - 1), -1
        )
        top_k = spec_info.probs.shape[-1]
        topk_logits, probs_idx = torch.topk(
            logits, k=top_k, dim=-1, largest=True, sorted=True
        )  # [bs*(block-1), k]

        # 温度逐行展开 (pad 行填 1.0), 与 npu_sample_tokens 同款。clamp 是纯防御:
        # SamplingParams 已把 temperature<eps 归一成 1.0 (sampling_params.py:96-99),
        # 张量内不会出现 0, 但该归一化逻辑变动时这里不应静默产出 NaN
        temperatures = sampling_info.temperatures.view(-1, 1)  # [n, 1], n 为真实行数
        if temperatures.shape[0] < batch_size:
            temperatures = torch.cat(
                [
                    temperatures,
                    temperatures.new_full((batch_size - temperatures.shape[0], 1), 1.0),
                ],
                dim=0,
            )
        expanded_temperature = (
            temperatures.expand(batch_size, block - 1).reshape(-1, 1).clamp(min=1e-6)
        )  # [bs*(block-1), 1]
        # softmax 作用于 top-k 子集即截断重归一, 和为 1, 无需再归一 (MTP 同款)
        probs_sort = torch.softmax(topk_logits / expanded_temperature, dim=-1)

        # greedy 行 (top_k<=1) 混入采样批: 温度全熵采样会让该行 accept 塌缩到 ~q(argmax)
        # 且每轮如此; 改为 argmax 提案 + one-hot q, 与 verify 侧被 top_k=1 mask 成
        # one-hot 的 p 配对后, 逐位接受退化为确定性比较, 等价 greedy 链式。
        top_ks = sampling_info.top_ks.view(-1, 1)  # [n, 1]
        if top_ks.shape[0] < batch_size:
            top_ks = torch.cat(
                [top_ks, top_ks.new_full((batch_size - top_ks.shape[0], 1), 1 << 30)],
                dim=0,
            )
        greedy_rows = (top_ks <= 1).expand(batch_size, block - 1).reshape(
            -1
        )  # [bs*(block-1)], pad 行按非 greedy 走采样 (结果被丢弃)
        one_hot = torch.zeros_like(probs_sort)
        one_hot[:, 0] = 1.0  # topk 已排序, 第 0 列即全词表 argmax; one-hot 行 multinomial 恒采 col 0
        probs_sort = torch.where(greedy_rows.unsqueeze(-1), one_hot, probs_sort)

        sampled_index = torch.multinomial(probs_sort, num_samples=1)  # [bs*(block-1), 1]
        block_tokens = (
            torch.gather(probs_idx, dim=1, index=sampled_index)
            .view(-1)
            .to(torch.int32)
        )  # [bs*(block-1)]
        probs_selected = torch.gather(probs_sort, dim=1, index=sampled_index).view(
            -1
        )  # [bs*(block-1)]

        # q 记录: 只写真实行 (pad 行 req_pool_indices=0 是保留槽, 不写)
        req_pool_indices = forward_batch.req_pool_indices[:real_bs]
        spec_info.probs[req_pool_indices, : block - 1] = probs_sort.view(
            batch_size, block - 1, top_k
        )[:real_bs]
        spec_info.probs_idx[req_pool_indices, : block - 1] = probs_idx.to(
            torch.int32
        ).view(batch_size, block - 1, top_k)[:real_bs]
        spec_info.probs_selected[req_pool_indices, : block - 1] = probs_selected.view(
            batch_size, block - 1
        )[:real_bs]
        return block_tokens.view(batch_size, block - 1).contiguous()

    def _draft_block_slots(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        batch_size: int,
        block: int,
        device: torch.device,
        real_bs: int,
    ) -> torch.Tensor:
        req_to_token = self.req_to_token_pool.req_to_token
        column_indices = seq_lens.unsqueeze(1) + torch.arange(block, device=device)
        row_indices = req_pool_indices.unsqueeze(1).expand(batch_size, block)
        slots = req_to_token[row_indices.to(torch.long), column_indices.to(torch.long)]
        if _DFLASH_CHECK_SLOTS and real_bs > 0:
            _rpi_max = int(req_pool_indices[:real_bs].max().item())
            _rpi_min = int(req_pool_indices[:real_bs].min().item())
            if _rpi_max >= self.req_to_token_pool.size or _rpi_min < 0:
                raise RuntimeError(
                    f"[DFLASH] req_pool_indices 越界: "
                    f"[{_rpi_min}, {_rpi_max}] vs pool.size={self.req_to_token_pool.size} "
                    f"req_pool_indices={req_pool_indices[:real_bs].tolist()}"
                )
            _slots_max = int(slots[:real_bs].max().item())
            if _slots_max >= self.model_runner.token_to_kv_pool.size:
                raise RuntimeError(
                    f"[DFLASH] draft block slot 越界: slot_max={_slots_max} "
                    f">= draft_pool.size={self.model_runner.token_to_kv_pool.size} "
                    f"req_pool_indices={req_pool_indices[:real_bs].tolist()} "
                    f"seq_lens={seq_lens[:real_bs].tolist()}"
                )
            assert (slots[:real_bs] != 0).all(), (
                f"DFLASH draft block slot 未分配: seq_lens={seq_lens[:real_bs].tolist()} "
                f"alloced_lens={self.req_to_token_pool.alloced_lens[req_pool_indices[:real_bs]].tolist()} "
                f"slots={slots[:real_bs].reshape(-1).tolist()[:block]}"
            )
        return slots.reshape(-1)

    def _materialize_draft_kv_lens(self, draft_forward_batch: ForwardBatch) -> None:
        seq_lens = draft_forward_batch.seq_lens
        draft_forward_batch.seq_lens = torch.tensor(
            seq_lens.tolist(), dtype=torch.int64, device=seq_lens.device
        )

    def _make_draft_forward_batch(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_out_cache_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: Optional[list],
        seq_lens_sum: int,
    ) -> ForwardBatch:
        batch_size = prefix_lens.shape[0]
        block = self.block_size
        device = prefix_lens.device
        extend_seq_lens_tensor = torch.full((batch_size,), block, dtype=torch.int64, device=device)

        draft_forward_batch = forward_batch
        draft_forward_batch.forward_mode = ForwardMode.DFLASH_EXTEND
        draft_forward_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        draft_forward_batch.input_ids = input_ids
        draft_forward_batch.positions = positions
        draft_forward_batch.out_cache_loc = block_out_cache_loc
        draft_forward_batch.extend_prefix_lens = prefix_lens
        draft_forward_batch.extend_seq_lens = extend_seq_lens_tensor
        draft_forward_batch.extend_prefix_lens_cpu = prefix_lens_cpu
        draft_forward_batch.extend_seq_lens_cpu = [block] * batch_size
        draft_forward_batch.seq_lens = prefix_lens + block
        draft_forward_batch.seq_lens_sum = seq_lens_sum
        draft_forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        draft_forward_batch.attn_backend = self.model_runner.attn_backend
        draft_forward_batch.extend_num_tokens = batch_size * block
        draft_forward_batch.new_tokens_total = batch_size * block
        if draft_forward_batch.global_batch_size is not None:
            draft_forward_batch.global_num_tokens = [b * block for b in draft_forward_batch.global_batch_size]
        draft_forward_batch.global_sp_num_tokens = None
        return draft_forward_batch


    def _committed_bonus(
        self,
        target_predict: torch.Tensor,
        accept_index: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> torch.Tensor:
        last_positions = (torch.cumsum(commit_lens, dim=0) - 1).to(torch.long)
        bonus_flat_index = accept_index[last_positions]
        return target_predict[bonus_flat_index.to(torch.long)].to(torch.int32)

    def _resolve_draft_cache_loc(
        self,
        forward_batch: ForwardBatch,
        num_slots: Optional[int] = None,
    ) -> torch.Tensor:
        """材化写 draft pool 用的 slot; l1.5 下从 req_to_token 重读未 remap 的全局 slot。"""
        if not (__is_npu__ and global_server_args_dict.get("enable_mla_l1_5_cache", False)):
            return forward_batch.out_cache_loc
        if num_slots is None:
            draft_cache_loc = torch.zeros_like(forward_batch.out_cache_loc)
            forward_batch.get_out_cache_loc_kernel_wrapper(forward_batch.batch_size, draft_cache_loc)
            return draft_cache_loc
        draft_cache_loc = torch.zeros(
            (num_slots,),
            dtype=forward_batch.out_cache_loc.dtype,
            device=forward_batch.out_cache_loc.device,
        )
        forward_batch.get_out_cache_loc_kernel_wrapper(forward_batch.batch_size, draft_cache_loc)
        return draft_cache_loc[: forward_batch.positions.shape[0]]

    @torch.no_grad()
    def _materialize_target_hidden_to_draft_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        tag: str,
    ):
        """把 target 多层 aux hidden 投影成 draft 各层的 KV 写入 draft pool。

        draft 的历史 KV 不是自己算的, 只走 kv_proj + k_norm + k_rope, 不跑 attention/MLP。
        """
        positions = positions.clamp(max=self._max_rope_position)
        if _DFLASH_CHECK_SLOTS:
            self._assert_slots_in_pool(cache_loc, self.model_runner.token_to_kv_pool, tag)
        context_hidden = self.draft_model.project_target_hidden(target_hidden)
        draft_pool = self.model_runner.token_to_kv_pool
        for layer in self.draft_model.layers:
            attention_module = layer.self_attn
            radix_attention_layer = attention_module.attn
            k, v = attention_module.kv_proj_only(context_hidden)
            k = k.reshape(-1, attention_module.num_kv_heads, attention_module.head_dim).contiguous()
            v = v.reshape(-1, attention_module.num_kv_heads, attention_module.head_dim).contiguous()
            k = attention_module.apply_k_norm(k)
            k = attention_module.apply_k_rope(positions, k)
            if __is_npu__:
                k = k.contiguous()
                draft_pool.set_kv_buffer(radix_attention_layer, cache_loc, k, v)
            else:
                draft_pool.set_kv_buffer(radix_attention_layer, cache_loc, k, v, layer_id_override=radix_attention_layer.layer_id)

    @torch.no_grad()
    def _assert_slots_in_pool(
        self,
        cache_loc: torch.Tensor,
        draft_pool,
        tag: str,
    ) -> None:
        """校验材化写入的 slot 未越界 (越界在 NPU 上不报错, 只静默掉 accept)。"""
        pool_size = draft_pool.size
        max_slot = int(cache_loc.max().item())
        min_slot = int(cache_loc.min().item())
        if max_slot >= pool_size or min_slot < 0:
            raise RuntimeError(
                f"[DFLASH] 材化写入越界 ({tag}): slot=[{min_slot}, {max_slot}] "
                f"draft_pool.size={pool_size} cache_loc[:8]={cache_loc[:8].tolist()}"
            )

    def propose(self, *args, **kwargs):
        raise NotImplementedError("DFLASH 单步出整块, 应调用 _propose_block")
