from sglang.srt.utils import get_colorful_logger
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.env import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda_available, is_npu

if is_cuda_available():
    from flashinfer.sampling import (
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
        min_p_sampling_from_probs
    )


logger = get_colorful_logger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        batch_next_token_ids: Optional[torch.Tensor] = None,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            if batch_next_token_ids is None:
                batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                # To return logits, uncomment the following
                # logprobs = logits
        else:
            # Post process logits
            logits = logits.div(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                if return_logprob:
                    # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
                    # https://github.com/flashinfer-ai/flashinfer/issues/708
                    # so we use the torch implementation.

                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)

                if batch_next_token_ids is None:
                    if sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        # Check Nan will throw exception, only check when crash_on_warnings is True
                        check_nan = self.use_nan_detection and crash_on_warnings()
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs,
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=check_nan,
                        )
            elif global_server_args_dict["sampling_backend"] == "pytorch":
                if batch_next_token_ids is None:
                    # A slower fallback implementation with torch native operations.
                    if is_npu():
                        batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch_npu(
                            probs,
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            sampling_info.min_ps,
                            sampling_info.need_min_p_sampling,
                        )
                    else:
                        batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                            probs,
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            sampling_info.min_ps,
                            sampling_info.need_min_p_sampling,
                        )

                if return_logprob:
                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        return batch_next_token_ids.to(torch.int32)

    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = torch.arange(0, probs.shape[-1], device=probs.device
                        ).view(1, -1) >= top_ks.view(-1, 1)
    mask = mask | ((probs_sum - probs_sort) > top_ps.view(-1, 1))

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        mask = mask | (probs_sort < min_p_thresholds.view(-1, 1))

    probs_sort = torch.where(mask, 0, probs_sort)
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_probs_torch_npu(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    top_k_white_list = {1: 4096, 2: 2048, 3: 2048, 4: 2048}
    # probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sort, probs_idx = torch.topk(probs, k=top_k_white_list.get(probs.shape[0], 1024), dim=-1, largest=True, sorted=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask_top_k = torch.arange(0, probs_sort.shape[-1], device=probs_sort.device).view(1, -1) >= top_ks.view(-1, 1)
    mask_top_p = torch.zeros_like(mask_top_k, dtype=torch.bool)
    if need_min_p_sampling:
        mask_top_p = (probs_sum - probs_sort) > top_ps.view(-1, 1)

    probs_sort = torch.where(mask_top_k | mask_top_p, 0, probs_sort)
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    assert len(top_logprobs_nums) == logprobs.shape[0], (
        len(top_logprobs_nums),
        logprobs.shape[0],
    )
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx
