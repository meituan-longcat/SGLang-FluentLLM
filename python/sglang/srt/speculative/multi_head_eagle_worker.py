from __future__ import annotations
from typing_extensions import override
from typing import Optional, TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.managers.scheduler import logger
from sglang.srt.model_executor.forward_batch_info import ForwardMode, CaptureHiddenMode
from sglang.srt.speculative.eagle_worker import EAGLEWorker

from sglang.srt.oe_utils import update_token_table
from sglang.srt.speculative.eagle_utils import update_oe_metadata, fast_topk

from sglang.srt.utils import is_npu
__is_npu__ = is_npu()

if not __is_npu__:
    from flashinfer.sampling import softmax

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@triton.jit
def renew_input_ids(
    input_ids_ptr,
    out_ids_ptr,
    valid_lengths_ptr,
    new_draft_ids_ptr,
    spec_num_tokens: tl.constexpr,
    bs: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= bs:
        return
    new_draft_id = tl.load(new_draft_ids_ptr + pid)
    valid_length = tl.load(valid_lengths_ptr + pid)
    # shift input ids
    for i in range(valid_length):
        if i == valid_length - 1:
            # set new id
            tl.store(out_ids_ptr + pid * spec_num_tokens + i, new_draft_id)
        else:
            input_id = tl.load(input_ids_ptr + pid * spec_num_tokens + i + 1)
            tl.store(out_ids_ptr + pid * spec_num_tokens + i, input_id)

def torch_renew_input_ids(
    input_ids: torch.Tensor,      # [bs * spec_num_tokens]
    valid_lengths: torch.Tensor,  # [bs]
    new_draft_ids: torch.Tensor,  # [bs]
    spec_num_tokens: int,
) -> torch.Tensor:
    bs = valid_lengths.numel()
    input_ids = input_ids.view(bs, -1)
    assert spec_num_tokens == input_ids.shape[1]
    out=torch.roll(input_ids, shifts=-1, dims=1)
    out[:, -1]=input_ids[:, -1]

    # 按 valid_lengths - 1 位置设置为 new_draft_ids
    valid_lengths=valid_lengths.to(torch.long)
    last_idx=valid_lengths-1
    row_indices=torch.arange(bs, device=input_ids.device, dtype=torch.long)

    valid_mask=valid_lengths>0
    out[row_indices, last_idx]=torch.where(
        valid_mask,
        new_draft_ids.to(out.dtype),
        out[row_indices, last_idx]
    )
    return out.view(-1)

class MultiHeadEAGLEWorker(EAGLEWorker):
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
        super().__init__(
            server_args,
            gpu_id,
            attn_tp_rank,
            moe_ep_rank,
            nccl_port,
            target_worker,
            global_rank,
        )

    def _set_input_ids_for_draft_extend(
        self, forward_batch: ForwardBatch, step: int, valid_lengths: torch.Tensor
    ):
        # The input ids is of draft step 0 is already set.
        if step == 0:
            return

        new_draft_ids = forward_batch.spec_info.topk_index.flatten()
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            assert forward_batch.extend_seq_lens is not None
            pt = 0
            for i, extend_len in enumerate(forward_batch.extend_seq_lens):
                input_ids = forward_batch.input_ids[pt : pt + extend_len]
                forward_batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], new_draft_ids[i].reshape(1))
                )
                pt += extend_len
        else:
            new_ids = torch.zeros_like(forward_batch.input_ids)
            if not __is_npu__:
                renew_input_ids[(forward_batch.batch_size,)](
                    input_ids_ptr=forward_batch.input_ids,
                    out_ids_ptr=new_ids,
                    valid_lengths_ptr=valid_lengths,
                    new_draft_ids_ptr=new_draft_ids,
                    spec_num_tokens=self.speculative_num_steps + 1,
                    bs=forward_batch.batch_size,
                )
            else:
                new_ids = torch_renew_input_ids(forward_batch.input_ids, valid_lengths, new_draft_ids, self.speculative_num_steps + 1)
            forward_batch.input_ids = new_ids

    @override
    def propose(self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor):
        # Update Here for OE Table Update
        forward_batch.req_to_token_pool.verified_lens[
            forward_batch.req_pool_indices
        ] += accept_lengths

        # Stores Draft tokens
        token_list: torch.Tensor = torch.empty(
            (forward_batch.batch_size, self.server_args.speculative_num_steps),
            dtype=torch.int32,
            device=accept_lengths.device,
        )

        for i in range(self.server_args.speculative_num_steps):
            # Update input ids for Draft Model
            model_runner = self.model_runner_list[i]
            self._set_input_ids_for_draft_extend(
                forward_batch, step=i, valid_lengths=accept_lengths
            )
            forward_batch.attn_backend = model_runner.attn_backend
            forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            if i == 0:
                forward_batch.topk_indices = None

            if __is_npu__:
                can_npu_graph=(
                    len(self.npu_graph_runner_for_draft_extends) > 0
                    and self.npu_graph_runner_for_draft_extends[i].can_run(forward_batch)
                )
                if can_npu_graph:
                    logits_output=self.npu_graph_runner_for_draft_extends[i].replay(forward_batch)
                else:
                    logits_output=model_runner.forward(
                        forward_batch
                    )
                probs = torch.nn.functional.softmax(logits_output.next_token_logits, dim=-1)
            else:
                logits_output=model_runner.forward_extend(
                    forward_batch, skip_metadata_init=True
                )
                probs = softmax(logits_output.next_token_logits)

            spec_info = forward_batch.spec_info
            _, spec_info.topk_index = fast_topk(probs, self.topk, dim=-1)

            # NOTE: Sglang didn't capture this hidden states and directly used hidden_states from target model, but
            # according to Megatron implemention of Flash-3B we need this hidden_states, otherwise it results in
            # low accept rate.
            spec_info.hidden_states = logits_output.hidden_states

            # new_draft_ids at step i
            new_draft_ids = spec_info.topk_index.flatten()
            if self.use_over_embedding:
                update_oe_metadata(forward_batch, i, self.speculative_num_steps)
                update_token_table(
                    forward_batch.oe_token_table,
                    new_draft_ids.to(torch.int32),
                    forward_batch.req_pool_indices,
                    forward_batch.oe_out_column_starts,
                    forward_batch.oe_out_req_lens,
                )
            token_list[:, i] = new_draft_ids

        return token_list
