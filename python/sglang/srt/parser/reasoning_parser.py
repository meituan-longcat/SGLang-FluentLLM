# ruff: noqa: S105
import logging
from typing import Optional

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.harmony_parser import HarmonyParser

logger = logging.getLogger(__name__)


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if any(token.startswith(current_text) and token != current_text for token in [self.think_start_token, self.think_end_token]):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Supported models:
      - DeepSeek-R1: Always generates thinking content without <think> start tag
      - DeepSeek-R1-0528: Generates thinking content with <think> start tag

    Format patterns:
      - DeepSeek-R1: "I need to think about this...</think>The answer is 42."
      - DeepSeek-R1-0528: "<think>I need to think about this...</think>The answer is 42."

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class KimiK2d5Detector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )

        self.tool_call_start_token = "<|tool_calls_section_begin|>"

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token in processed_text:
            # Extract reasoning content
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1].strip()
        elif self.tool_call_start_token in processed_text:
            # Extract tool call content
            splits = processed_text.split(self.tool_call_start_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = self.tool_call_start_token + splits[1].strip()
        else:
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if any(token.startswith(current_text) and token != current_text for token in [self.think_start_token, self.think_end_token]):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning:
            if self.think_end_token in current_text:
                end_idx = current_text.find(self.think_end_token)

                reasoning_text = current_text[:end_idx]

                self._buffer = ""
                self._in_reasoning = False
                normal_text = current_text[end_idx + len(self.think_end_token) :]

                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())
            if self.tool_call_start_token in current_text:
                end_idx = current_text.find(self.tool_call_start_token)

                reasoning_text = current_text[:end_idx]

                self._buffer = ""
                self._in_reasoning = False
                normal_text = current_text[end_idx:]

                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class LongCatDetector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<longcat_think>)*(.*)</longcat_think>
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<longcat_think>",
            "</longcat_think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self.tool_call_start_token = "<longcat_tool_call>"
        self._skip_next_newline = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "")

        if self.think_end_token in processed_text:
            # Extract reasoning content
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0].rstrip()
            normal_text = splits[1]
            if normal_text.startswith("\n"):
                normal_text = normal_text[1:]
        elif self.tool_call_start_token in processed_text:
            # Extract tool call content
            splits = processed_text.split(self.tool_call_start_token, maxsplit=1)
            reasoning_text = splits[0].rstrip()
            normal_text = self.tool_call_start_token + splits[1]
        else:
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if any(token.startswith(current_text) and token != current_text for token in [self.think_start_token, self.think_end_token]):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning:
            if self.think_end_token in current_text:
                end_idx = current_text.find(self.think_end_token)

                reasoning_text = current_text[:end_idx]

                self._buffer = ""
                self._in_reasoning = False
                normal_text = current_text[end_idx + len(self.think_end_token) :]
                if normal_text.startswith("\n"):
                    normal_text = normal_text[1:]
                else:
                    self._skip_next_newline = True

                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())
            if self.tool_call_start_token in current_text:
                end_idx = current_text.find(self.tool_call_start_token)

                reasoning_text = current_text[:end_idx]

                self._buffer = ""
                self._in_reasoning = False
                normal_text = current_text[end_idx:]

                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            if self._skip_next_newline:
                self._skip_next_newline = False
                if current_text.startswith("\n"):
                    current_text = current_text[1:]
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    Detector for GLM-4.5/4.6/4.7/5.x models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    The GLM chat template pre-fills `<think>` in the generation prompt when
    thinking is enabled, so the generated text carries no opening tag —
    hence force_reasoning defaults to True (the serving layer overrides it
    when the request disables thinking via chat_template_kwargs).

    GLM may also start a tool call directly from reasoning mode without
    emitting `</think>`; `<tool_call>` therefore switches parsing back to
    normal text so the function-call detector can consume it.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self.tool_start_token = "<tool_call>"

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Reasoning mode may end by starting a tool call without `</think>`:
            # hand the tool-call section back as normal text.
            if self.tool_start_token in processed_text:
                tool_idx = processed_text.find(self.tool_start_token)
                return StreamingParseResult(
                    normal_text=processed_text[tool_idx:],
                    reasoning_text=processed_text[:tool_idx].rstrip(),
                )
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of a token that switches parsing
        # mode, keep buffering
        if any(
            token.startswith(current_text) and token != current_text
            for token in [
                self.think_start_token,
                self.think_end_token,
                self.tool_start_token,
            ]
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())

        # A tool call started while still in reasoning: switch to normal mode
        # so the tool-call tokens flow to the function-call detector.
        if self._in_reasoning and self.tool_start_token in current_text:
            tool_idx = current_text.find(self.tool_start_token)
            reasoning_text = current_text[:tool_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[tool_idx:]

            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())

        # Continue with reasoning content, but hold back a tail that could be
        # the beginning of `</think>` or `<tool_call>` split across chunks
        if self._in_reasoning:
            emit_text = current_text
            held_back = ""
            for token in [self.think_end_token, self.tool_start_token]:
                for i in range(len(token) - 1, 0, -1):
                    if current_text.endswith(token[:i]) and i > len(held_back):
                        held_back = current_text[len(current_text) - i :]
            if held_back:
                emit_text = current_text[: len(current_text) - len(held_back)]

            if self.stream_reasoning:
                self._buffer = held_back
                if emit_text:
                    return StreamingParseResult(reasoning_text=emit_text)
                return StreamingParseResult()
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        self._buffer = ""
        return StreamingParseResult(normal_text=current_text)


class DeepSeekV31Detector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False, request_has_tools: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self.request_has_tools = request_has_tools

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        if self.request_has_tools:
            in_reasoning = False
        else:
            in_reasoning = self._in_reasoning or any(token in text for token in [self.think_start_token, self.think_end_token])

        if not in_reasoning:
            text = text.replace(self.think_start_token, "").strip()
            text = text.replace(self.think_end_token, "").strip()
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if self.request_has_tools:
            self._in_reasoning = False

        if self._in_reasoning and any(
            token.startswith(current_text) and token != current_text for token in [self.think_start_token, self.think_end_token]
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and (self.think_start_token in current_text if self._in_reasoning else False):
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text.rstrip())

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            current_text = current_text.replace(self.think_start_token, "")
            current_text = current_text.replace(self.think_end_token, "")
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )


class GptOssDetector(BaseReasoningFormatDetector):
    """
    Detector for T4-style reasoning format (GPT-OSS), using the HarmonyParser.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self.parser = HarmonyParser()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        events = self.parser.parse(text)
        # Flush the buffer for one-shot parsing
        events += self.parser.parse("")

        reasoning_text = "".join([e.content for e in events if e.event_type == "reasoning"])
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)
        # Tool call events preserve raw text with structural markers

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        events = self.parser.parse(new_text)

        reasoning_text = "".join([e.content for e in events if e.event_type == "reasoning"])
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: dict[str, type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseek-v3": Qwen3Detector,
        "glm45": Glm45Detector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "kimi_k2": DeepSeekR1Detector,
        "kimi_k2d5": KimiK2d5Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "longcat": LongCatDetector,
        "longcat_xml": LongCatDetector,
        "step3": DeepSeekR1Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Special cases where we override force_reasoning
        if model_type.lower() in {"qwen3-thinking", "gpt-oss"}:
            force_reasoning = True

        # Only pass force_reasoning if explicitly set, let detectors use their defaults
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        if model_type.lower() == "deepseekv31":
            if request is not None and request.tools is not None:
                kwargs["request_has_tools"] = True
            else:
                kwargs["request_has_tools"] = False

        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> tuple[Optional[str], Optional[str]]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> tuple[Optional[str], Optional[str]]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
