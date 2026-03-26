from __future__ import annotations

import time
from typing import Optional, Union, List, Tuple, TYPE_CHECKING
  
from sglang.srt.env import global_server_args_dict
from sglang.srt.utils import get_colorful_logger
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.managers.req import Req, BaseFinishReason, FINISH_ABORT
from sglang.srt.managers.io_struct import HealthCheckOutput, BatchTokenIDOut, BatchEmbeddingOut

if TYPE_CHECKING:
    import threading
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
    )
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

logger = get_colorful_logger(__name__)


DEFAULT_FORCE_STREAM_INTERVAL = 50

class SchedulerPostProcessMixin:

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result, launch_done)
            if batch.is_empty():
                self.running_batch = None
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result, launch_done)
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                if self.draft_worker:
                    self.draft_worker.resolve_batch_result(result.bid)
                else:
                    self.tp_worker.resolve_last_batch_result(launch_done)
                self.set_next_batch_sampling_info_done(batch)
        elif batch.forward_mode.is_dummy_first():
            self.set_next_batch_sampling_info_done(batch)

        if self.return_health_check_ct:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.return_health_check_ct -= 1
            self.send_to_tokenizer.send_pyobj(HealthCheckOutput())

        # Publish KV cache events
        self._publish_kv_events()

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        skip_stream_req = None

        if self.is_generation:
            (
                logits_output,
                next_token_ids,
                next_token_multi_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
                bid,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.next_token_multi_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
                result.bid,
            )
            if self.enable_overlap:
                # Here CPU wait for GPU
                # Then do GPU -> CPU
                if self.draft_worker:
                    (
                        logits_output,
                        next_token_ids,
                        _,
                        _,
                        _,
                    ) = self.draft_worker.resolve_batch_result(bid)
                else:
                    logits_output, next_token_ids, next_token_multi_ids = (
                        self.tp_worker.resolve_last_batch_result(launch_done)
                    )
            else:
                # Move next_token_ids and logprobs to cpu
                next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    if logits_output.next_token_logprobs is not None:
                        logits_output.next_token_logprobs = (
                            logits_output.next_token_logprobs.tolist()
                        )
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = tuple(
                            logits_output.input_token_logprobs.tolist()
                        )

            hidden_state_offset = 0

            # When speculation + overlap, if Prefill is interspersed in consecutive Decodes,
            # it will break the logic of dynamically reserving cache based on accept length for requests.
            # Need to correct the reserve of running batch to ensure sufficient slots.
            if (
                self.running_batch is not None
                and self.enable_overlap
                and not self.spec_algorithm.is_none()
            ):
                for req in self.running_batch.reqs:
                    req.reserve_num_tokens = (
                        self.server_args.speculative_num_draft_tokens
                    )

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    # During overlap scheduling, if we reach here, it means the Prefill batch
                    # was retracted after merge_batch, and the request no longer exists
                    # in the last launch.
                    assert (
                        req.req_pool_idx is not None
                        and req.req_pool_idx not in self.req_to_token_pool.free_slots
                    ), f"{req.req_pool_idx=} {self.req_to_token_pool.free_slots=}"
                    logger.info(f"Release Retracted req {req.req_pool_idx}")
                    self.req_to_token_pool.free(req.req_pool_idx)
                    req.req_pool_idx = None
                    continue

                if req.is_chunked <= 0:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    if next_token_multi_ids is not None:
                        req.output_multi_ids.append(next_token_multi_ids[i].tolist())

                    req.check_finished()

                    if req.finished():
                        if not self.enable_overlap or (self.server_args.request_cache_size <= 0 \
                                and isinstance(req.finished_reason, FINISH_ABORT)):
                            self.tree_cache.cache_finished_req(req)
                        elif self.cur_batch is not None and self.cur_batch.forward_mode == ForwardMode.EXTEND:
                            # If prefill ends immediately and there's no delayed token, release resources here
                            # Requests with delayed tokens will be merged into running batch and released
                            # in process_batch_result_decode
                            self.tree_cache.cache_finished_req(req)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    if req.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                        )

                    if req.grammar is not None:
                        req.grammar.accept_token(next_token_id)
                        req.grammar.finished = req.finished()
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if req.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = (
                                extend_input_len - extend_logprob_start_len
                            )
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                            logprob_pt += num_input_logprobs

            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()

        else:  # embedding or reward model
            embeddings, bid = result.embeddings, result.bid
            if isinstance(embeddings, dict):
                embeddings_dict = []
                length = len(next(iter(embeddings.values())))
                for i in range(length):
                    item = {key: tensor[i].tolist() for key, tensor in embeddings.items()}
                    embeddings_dict.append(item)
            else:
                embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                if isinstance(embeddings, dict):
                    req.embedding = embeddings_dict[i]
                else:
                    req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        self.kv_allocator.free_group_end()
        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        logits_output, next_token_ids, next_token_multi_ids, bid, accept_lengths_cpu = (
            result.logits_output,
            result.next_token_ids,
            result.next_token_multi_ids,
            result.bid,
            result.accept_lengths_cpu,
        )

        # Update Metrics
        if not self.draft_worker:
            self.num_generated_tokens += len(batch.reqs)
        elif not self.enable_overlap:
            num_accepted_tokens = sum(accept_lengths_cpu)
            self.spec_num_total_accepted_tokens += num_accepted_tokens
            self.spec_num_total_forward_ct += batch.batch_size()
            self.num_generated_tokens += num_accepted_tokens
            for req in batch.reqs:
                req.spec_verify_ct += 1

        if self.enable_overlap:
            # Here CPU wait for GPU
            # Then do GPU -> CPU
            if self.draft_worker:
                (
                    logits_output,
                    next_token_ids,
                    accept_lengths,
                    _,
                    _,
                ) = self.draft_worker.resolve_batch_result(bid)
                accept_lengths_cpu = accept_lengths.tolist()
                num_accepted_tokens = sum(accept_lengths_cpu)
                self.spec_num_total_accepted_tokens += num_accepted_tokens
                self.spec_num_total_forward_ct += batch.batch_size()
                self.num_generated_tokens += num_accepted_tokens
                for req in batch.reqs:
                    req.spec_verify_ct += 1
            else:
                logits_output, next_token_ids, next_token_multi_ids = (
                    self.tp_worker.resolve_last_batch_result(launch_done)
                )
            next_token_logprobs = logits_output.next_token_logprobs
        else:
            next_token_ids = next_token_ids.tolist()
            if next_token_multi_ids is not None:
                next_token_multi_ids = next_token_multi_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()

        # Update pre-allocated slot count
        batch.update_reserve_num_tokens(accept_lengths_cpu)

        self.kv_allocator.free_group_begin()

        # Check finish condition
        pt = 0
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                # During overlap scheduling, retracted requests may enter post-processing
                # while the previously launched batch no longer has this request.
                # Release req_pool resources here.
                assert (
                    req.req_pool_idx is not None
                    and req.req_pool_idx not in self.req_to_token_pool.free_slots
                ), f"{req.req_pool_idx=} {self.req_to_token_pool.free_slots=}"
                logger.info(f"Release Retracted req {req.req_pool_idx}")
                self.req_to_token_pool.free(req.req_pool_idx)
                req.req_pool_idx = None
                continue

            # For cases without speculative inference, accept_lengths_cpu is 1
            ids = next_token_ids[pt : pt + accept_lengths_cpu[i]]
            if next_token_multi_ids is not None:
                multi_ids = next_token_multi_ids[pt: pt + accept_lengths_cpu[i]]
            pt += accept_lengths_cpu[i]

            if self.enable_overlap and req.finished():
                if req.spec_verify_ct > 0 and not self.spec_algorithm.is_none():
                    req.accept_draft_tokens = (len(req.output_ids)-1) / req.spec_verify_ct
                # This indicates a delayed token, actually release its resources here
                self.tree_cache.cache_finished_req(req)
                continue

            for token_idx, next_token_id in enumerate(ids):
                req.output_ids.append(next_token_id)
                if next_token_multi_ids is not None:
                    req.output_multi_ids.append(multi_ids[token_idx])
                # Check if finished after appending token id
                req.check_finished()
                if req.return_logprob and batch.spec_algorithm.is_none():
                    # speculative worker handles logprob in speculative decoding
                    req.output_token_logprobs_val.append(next_token_logprobs[i])
                    req.output_token_logprobs_idx.append(next_token_id)
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            logits_output.next_token_top_logprobs_val[i]
                        )
                        req.output_top_logprobs_idx.append(
                            logits_output.next_token_top_logprobs_idx[i]
                        )
                    if req.token_ids_logprob is not None:
                        req.output_token_ids_logprobs_val.append(
                            logits_output.next_token_token_ids_logprobs_val[i]
                        )
                        req.output_token_ids_logprobs_idx.append(
                            logits_output.next_token_token_ids_logprobs_idx[i]
                        )

                if req.return_hidden_states and logits_output.hidden_states is not None:
                    req.hidden_states.append(
                        logits_output.hidden_states[i].cpu().clone()
                    )

                if req.grammar is not None:
                    req.grammar.accept_token(next_token_id)
                    req.grammar.finished = req.finished()

                if req.finished():
                    if req.spec_verify_ct > 0 and not self.spec_algorithm.is_none():
                        req.accept_draft_tokens = (len(req.output_ids)-1) / req.spec_verify_ct
                    if not self.enable_overlap:
                        self.tree_cache.cache_finished_req(req)
                    elif self.cur_batch is not None and self.cur_batch.forward_mode == ForwardMode.EXTEND:
                        # In this case, since a new Prefill batch was launched before, finished requests are directly
                        # filtered without delayed tokens. So cache and release related resources here.
                        self.tree_cache.cache_finished_req(req)
                    # Do not process tokens after EOS or reaching max length
                    break

        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

        self.stream_output(batch.reqs, batch.return_logprob)

        self.kv_allocator.free_group_end()
        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.attn_tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats()

    def stream_output(
        self, reqs: List[Req], return_logprob: bool, skip_req: Optional[Req] = None
    ):
        """Stream the output to detokenizer."""
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        if self.is_generation:
            decoded_texts = []
            decode_ids_list = []
            read_offsets = []
            output_ids = []
            output_multi_ids = []

            skip_special_tokens = []
            spaces_between_special_tokens = []
            no_stop_trim = []
            prompt_tokens = []
            completion_tokens = []
            cached_tokens = []
            spec_verify_ct = []
            batch_accept_draft_tokens = []
            output_hidden_states = None
            output_extra_infos = []

            if return_logprob:
                input_token_logprobs_val = []
                input_token_logprobs_idx = []
                output_token_logprobs_val = []
                output_token_logprobs_idx = []
                input_top_logprobs_val = []
                input_top_logprobs_idx = []
                output_top_logprobs_val = []
                output_top_logprobs_idx = []
                input_token_ids_logprobs_val = []
                input_token_ids_logprobs_idx = []
                output_token_ids_logprobs_val = []
                output_token_ids_logprobs_idx = []
            else:
                input_token_logprobs_val = input_token_logprobs_idx = (
                    output_token_logprobs_val
                ) = output_token_logprobs_idx = input_top_logprobs_val = (
                    input_top_logprobs_idx
                ) = output_top_logprobs_val = output_top_logprobs_idx = (
                    input_token_ids_logprobs_val
                ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                    output_token_ids_logprobs_idx
                ) = None

            for req in reqs:
                if req is skip_req:
                    continue

                # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
                if self.model_config.is_multimodal_gen and req.to_abort:
                    continue

                if req.finished():
                    if req.finished_output:
                        # With the overlap schedule, a request will try to output twice and hit this line twice
                        # because of the one additional delayed token. This "continue" prevented the dummy output.
                        continue
                    req.finished_output = True
                    should_output = True

                    # Log request time stats for Dynamo compatibility
                    if (
                        self.tp_rank == 0
                        and self.server_args.enable_request_time_stats_logging
                        and hasattr(req, 'log_time_stats')
                    ):
                        try:
                            req.log_time_stats()
                        except Exception as e:
                            logger.warning(f"Failed to log time stats: {e}")
                else:
                    if req.stream:
                        stream_interval = (
                            req.sampling_params.stream_interval or self.stream_interval
                        )
                        should_output = (
                            len(req.output_ids) % stream_interval == 1
                            if not self.model_config.is_multimodal_gen
                            and stream_interval > 1
                            else len(req.output_ids) % stream_interval == 0
                        )
                    else:
                        should_output = (
                            len(req.output_ids) == 1 or len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                            if not self.model_config.is_multimodal_gen
                            else False
                        )

                if should_output:
                    send_token_offset = req.send_token_offset
                    send_output_token_logprobs_offset = (
                        req.send_output_token_logprobs_offset
                    )
                    rids.append(req.rid)
                    finished_reasons.append(
                        req.finished_reason.to_json() if req.finished_reason else None
                    )
                    decoded_texts.append(req.decoded_text)
                    decode_ids, read_offset = req.init_incremental_detokenize()

                    if self.model_config.is_multimodal_gen:
                        decode_ids_list.append(decode_ids)
                    else:
                        decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                    req.send_decode_id_offset = len(decode_ids)
                    read_offsets.append(read_offset)
                    output_ids.append(req.output_ids[send_token_offset:])
                    req.send_token_offset = len(req.output_ids)
                    output_multi_ids.append(req.output_multi_ids)
                    skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                    spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )
                    no_stop_trim.append(req.sampling_params.no_stop_trim)
                    prompt_tokens.append(len(req.origin_input_ids))
                    completion_tokens.append(len(req.output_ids))
                    cached_tokens.append(req.cached_tokens)
                    logger.debug(f"Stream output: rid={req.rid} id={id(req)} cached_tokens={req.cached_tokens} input_len={len(req.origin_input_ids)}")
                    if not self.spec_algorithm.is_none():
                        spec_verify_ct.append(req.spec_verify_ct)
                        batch_accept_draft_tokens.append(req.accept_draft_tokens)

                    if return_logprob:
                        if (
                            req.return_logprob
                            and not req.input_logprob_sent
                            # Decode server does not send input logprobs
                            and self.disaggregation_mode != DisaggregationMode.DECODE
                        ):
                            input_token_logprobs_val.append(req.input_token_logprobs_val)
                            input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                            input_top_logprobs_val.append(req.input_top_logprobs_val)
                            input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                            input_token_ids_logprobs_val.append(
                                req.input_token_ids_logprobs_val
                            )
                            input_token_ids_logprobs_idx.append(
                                req.input_token_ids_logprobs_idx
                            )
                            req.input_logprob_sent = True
                        else:
                            input_token_logprobs_val.append([])
                            input_token_logprobs_idx.append([])
                            input_top_logprobs_val.append([])
                            input_top_logprobs_idx.append([])
                            input_token_ids_logprobs_val.append([])
                            input_token_ids_logprobs_idx.append([])

                        if req.return_logprob:
                            output_token_logprobs_val.append(
                                req.output_token_logprobs_val[
                                    send_output_token_logprobs_offset:
                                ]
                            )
                            output_token_logprobs_idx.append(
                                req.output_token_logprobs_idx[
                                    send_output_token_logprobs_offset:
                                ]
                            )
                            output_top_logprobs_val.append(
                                req.output_top_logprobs_val[
                                    send_output_token_logprobs_offset:
                                ]
                            )
                            output_top_logprobs_idx.append(
                                req.output_top_logprobs_idx[
                                    send_output_token_logprobs_offset:
                                ]
                            )
                            output_token_ids_logprobs_val.append(
                                req.output_token_ids_logprobs_val[
                                    send_output_token_logprobs_offset:
                                ]
                            )
                            output_token_ids_logprobs_idx.append(
                                req.output_token_ids_logprobs_idx[
                                    send_output_token_logprobs_offset:
                                ]
                            )
                            req.send_output_token_logprobs_offset = len(
                                req.output_token_logprobs_val
                            )
                        else:
                            output_token_logprobs_val.append([])
                            output_token_logprobs_idx.append([])
                            output_top_logprobs_val.append([])
                            output_top_logprobs_idx.append([])
                            output_token_ids_logprobs_val.append([])
                            output_token_ids_logprobs_idx.append([])

                    if req.return_hidden_states:
                        if output_hidden_states is None:
                            output_hidden_states = []
                        output_hidden_states.append(req.hidden_states)

                    req.output_extra_info['decode_prefix_len'] = req.prefix_len
                    output_extra_infos.append(req.output_extra_info)

            # Send to detokenizer
            if rids:
                if self.model_config.is_multimodal_gen:
                    raise NotImplementedError()
                batch_id_out = BatchTokenIDOut(
                    rids,
                    finished_reasons,
                    decoded_texts,
                    decode_ids_list,
                    read_offsets,
                    output_ids,
                    output_multi_ids,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                    no_stop_trim,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    spec_verify_ct,
                    input_token_logprobs_val,
                    input_token_logprobs_idx,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx,
                    output_hidden_states,
                    batch_accept_draft_tokens,
                    output_extra_infos,
                    generated_time=time.time()
                )
                if "multi_ids" in global_server_args_dict["mm_mode"]:
                    # In multimodal scenarios, if multi-level tokens need to be returned,
                    # original token IDs are usually needed. Here we choose to return the original values directly.
                    self.send_to_tokenizer.send_pyobj(batch_id_out)
                else:
                    self.send_to_detokenizer.send_pyobj(batch_id_out)

        else:  # embedding or reward model
            embeddings = []
            prompt_tokens = []
            for req in reqs:
                if req.finished():
                    rids.append(req.rid)
                    finished_reasons.append(req.finished_reason.to_json())
                    embeddings.append(req.embedding)
                    prompt_tokens.append(len(req.origin_input_ids))
            self.send_to_detokenizer.send_pyobj(
                BatchEmbeddingOut(rids, finished_reasons, embeddings, prompt_tokens)
            )

    def add_input_logprob_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Compute input_token_logprobs_val
            # Always pad the first one with None.
            req.input_token_logprobs_val = [None]
            req.input_token_logprobs_val.extend(input_token_logprobs)
            # The last input logprob is for sampling, so just pop it out.
            req.input_token_logprobs_val.pop()

            # Compute input_token_logprobs_idx
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]
            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_logprobs_idx = [
                x if x < self.model_config.vocab_size - 1 else 0
                for x in input_token_logprobs_idx
            ]
            req.input_token_logprobs_idx = input_token_logprobs_idx

            if req.top_logprobs_num > 0:
                req.input_top_logprobs_val = [None]
                req.input_top_logprobs_idx = [None]

                assert len(req.temp_input_top_logprobs_val) == len(
                    req.temp_input_top_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_top_logprobs_val,
                    req.temp_input_top_logprobs_idx,
                    # strict=True,
                ):
                    req.input_top_logprobs_val.extend(val)
                    req.input_top_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_top_logprobs_val.pop()
                req.input_top_logprobs_idx.pop()
                req.temp_input_top_logprobs_idx = None
                req.temp_input_top_logprobs_val = None

            if req.token_ids_logprob is not None:
                req.input_token_ids_logprobs_val = [None]
                req.input_token_ids_logprobs_idx = [None]

                assert len(req.temp_input_token_ids_logprobs_val) == len(
                    req.temp_input_token_ids_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_token_ids_logprobs_val,
                    req.temp_input_token_ids_logprobs_idx,
                ):
                    req.input_token_ids_logprobs_val.extend(val)
                    req.input_token_ids_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_token_ids_logprobs_val.pop()
                req.input_token_ids_logprobs_idx.pop()
                req.temp_input_token_ids_logprobs_idx = None
                req.temp_input_token_ids_logprobs_val = None

            if req.return_logprob:
                relevant_tokens_len = len(req.origin_input_ids) - req.logprob_start_len
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs_val.append(output.next_token_logprobs[i])
        req.output_token_logprobs_idx.append(next_token_ids[i])

        self.add_input_logprob_return_values(
            i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
        )

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.output_token_ids_logprobs_val.append(
                output.next_token_token_ids_logprobs_val[i]
            )
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs