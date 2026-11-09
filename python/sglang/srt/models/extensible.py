import importlib
from typing import Any

from torch import Tensor, nn

from sglang.srt.layers.attention.dp_attention import get_attention_tp_rank
from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


# Used for Input/Output Processor sharing
class ContextBase:
    def __init__(self, base_lm, config_dict: dict[str, Any]):
        pass


class InputProcessorBase(nn.Module):
    def __init__(self, base_lm, ctx, config_dict: dict[str, Any]):
        super().__init__()
        self.base_lm = base_lm

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Tensor = None,
    ) -> Tensor:
        if input_embeds is not None:
            return input_embeds
        else:
            return self.base_lm.model.embed_tokens(input_ids)


class OutputProcessorBase(nn.Module):
    def __init__(self, base_lm, ctx, config_dict: dict[str, Any]):
        super().__init__()
        self.base_lm = base_lm

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        output_hidden_states: Tensor,
    ) -> LogitsProcessorOutput:
        return self.base_lm.logits_processor(
            input_ids,
            output_hidden_states,
            self.base_lm.lm_head,
            forward_batch,
        )


_EXT_CLS_REGISTRY: dict[str, type] = {}


def register_ext_cls(name: str, cls: type) -> None:
    global _EXT_CLS_REGISTRY
    _EXT_CLS_REGISTRY[name] = cls


def get_ext_cls(name: str) -> type:
    if name not in _EXT_CLS_REGISTRY:
        raise ValueError(f"Input module {name} not found in registry. {_EXT_CLS_REGISTRY=}")
    return _EXT_CLS_REGISTRY[name]


register_ext_cls("ContextBase", ContextBase)
register_ext_cls("InputProcessorBase", InputProcessorBase)
register_ext_cls("OutputProcessorBase", OutputProcessorBase)


class ExtensibleLM(nn.Module):
    def __init__(
        self,
        base_lm: nn.Module,
        ext_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.base_lm = base_lm

        if "ext_def_file" in ext_config:
            ext_def_file = ext_config["ext_def_file"]
            import os
            import sys
            from pathlib import Path

            ext_def_dir = os.path.dirname(os.path.abspath(ext_def_file))
            sys.path.insert(0, (ext_def_dir))
            ext_def_module = f"{Path(ext_def_file).stem}"
            logger.info(f"\033[32m[[ExtensibleLM] Loading {ext_def_dir=}, {ext_def_module=}]\033[0m")
            importlib.import_module(ext_def_module)

        ctx_config = ext_config["context"]
        ctx_name = ctx_config.pop("cls")
        ctx_cls = get_ext_cls(ctx_name)
        self.ctx: ContextBase = ctx_cls(base_lm, ctx_config)

        input_processor_config = ext_config["input_processor"]
        input_processor_name = input_processor_config.pop("cls")
        input_processor_cls = get_ext_cls(input_processor_name)
        self.input_processor: InputProcessorBase = input_processor_cls(
            self.base_lm,
            self.ctx,
            input_processor_config,
        ).eval()

        output_processor_config = ext_config["output_processor"]
        output_processor_name = output_processor_config.pop("cls")
        output_processor_cls = get_ext_cls(output_processor_name)
        self.output_processor: OutputProcessorBase = output_processor_cls(
            self.base_lm,
            self.ctx,
            output_processor_config,
        ).eval()

        self.attn_tp_rank = get_attention_tp_rank()
        self.step = 0

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Tensor = None,
    ) -> LogitsProcessorOutput:
        # input processor: get input hidden states
        input_embeds = self.input_processor(input_ids, positions, forward_batch, input_embeds)

        # base model forward
        out_hidden_states = self.base_lm.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        # output processor: lm hidden states to logits
        logits_output: LogitsProcessorOutput = self.output_processor(input_ids, positions, forward_batch, out_hidden_states)
        self.step += 1
        return logits_output
