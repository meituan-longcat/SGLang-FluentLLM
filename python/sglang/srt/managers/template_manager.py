# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import json
import os
import re
from typing import Optional

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.utils import get_colorful_logger
from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    register_completion_template,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    register_conv_template,
)
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

logger = get_colorful_logger(__name__)


class TemplateManager:
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self._chat_template_name: Optional[str] = None
        self._completion_template_name: Optional[str] = None
        self._jinja_template_content_format: Optional[str] = "openai"
        self._override_chat_template: Optional[str] = None

        # Launch tokenizer process
        self.tokenizer_manager = TokenizerManager(server_args, port_args)

        logger.warning(f"{server_args.chat_template=} {server_args.model_path=}")
        if server_args.chat_template:
            self._load_explicit_chat_template(server_args.chat_template)
        else:
            # If no pre-defined template was found, fallback to HuggingFace template
            if self._chat_template_name is None:
                # Default to string content format if no template was found
                self._jinja_template_content_format = "string"
                logger.warning("chat template will be detected by hf in function _apply_jinja_template")

        # Detect reasoning pattern from chat template
        self._force_reasoning = self._detect_reasoning_pattern(
            self.tokenizer_manager.query_actual_chat_template()
        )

        # Load completion template
        if server_args.completion_template:
            self.load_completion_template(server_args.completion_template)

    @property
    def chat_template_name(self) -> Optional[str]:
        """Get the current chat template name."""
        return self._chat_template_name

    @property
    def completion_template_name(self) -> Optional[str]:
        """Get the current completion template name."""
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> Optional[str]:
        """Get the detected template content format ('string' or 'openai' or None)."""
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool:
        """
        Check if the current chat template enforces reasoning/thinking.

        Returns:
            True if the template contains reasoning patterns like <think> tags
        """
        return self._force_reasoning

    def _detect_reasoning_pattern(self, template: Optional[str]) -> bool:
        """
        Detect if the chat template contains reasoning/thinking patterns.
        """
        if template is None:
            return False

        force_reasoning_pattern = r"<\|im_start\|>assistant\\n<think>\\n"
        has_reasoning = re.search(force_reasoning_pattern, template) is not None

        enable_thinking_pattern = r"enable_thinking\s*\|\s*default\((true|false|.*?)\)"
        enable_thinking_match = re.search(enable_thinking_pattern, template)
        if enable_thinking_match is not None:
            has_reasoning = enable_thinking_match.group(1) == "true"

        if has_reasoning:
            logger.warning("Detected the force reasoning pattern in chat template.")

        return has_reasoning

    def _load_jinja_template(self, template_path: str) -> None:
        """Load a Jinja template file."""
        with open(template_path, "r") as f:
            chat_template = "".join(f.readlines()).strip("\n")
        self._override_chat_template = chat_template.replace("\\n", "\n")
        self._chat_template_name = None
        # Detect content format from the loaded template
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            f"Detected user specified Jinja chat template with content format: {self._jinja_template_content_format}"
        )

    def _load_explicit_chat_template(
        self, chat_template_arg: str
    ) -> None:
        """Load explicitly specified chat template."""
        logger.warning(f"Loading chat template from argument: {chat_template_arg}")

        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )

        if chat_template_arg.endswith(".jinja"):
            self._load_jinja_template(chat_template_arg)
        else:
            self._load_json_chat_template(chat_template_arg)

    def load_completion_template(self, completion_template_arg: str) -> None:
        """
        Load completion template for code completion.

        Args:
            completion_template_arg: Template name or file path
        """
        logger.warning(f"Loading completion template: {completion_template_arg}")

        if not completion_template_exists(completion_template_arg):
            if not os.path.exists(completion_template_arg):
                raise RuntimeError(
                    f"Completion template {completion_template_arg} is not a built-in template name "
                    "or a valid completion template file path."
                )

            self._load_json_completion_template(completion_template_arg)
        else:
            self._completion_template_name = completion_template_arg

    def _load_json_chat_template(self, template_path: str) -> None:
        """Load a JSON chat template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of chat template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None

            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        self._chat_template_name = template["name"]

    def _load_json_completion_template(self, template_path: str) -> None:
        """Load a JSON completion template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of completion template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                fim_position = FimPosition[template["fim_position"]]
            except KeyError:
                raise ValueError(
                    f"Unknown fim position: {template['fim_position']}"
                ) from None

            register_completion_template(
                CompletionTemplate(
                    name=template["name"],
                    fim_begin_token=template["fim_begin_token"],
                    fim_middle_token=template["fim_middle_token"],
                    fim_end_token=template["fim_end_token"],
                    fim_position=fim_position,
                ),
                override=True,
            )
        self._completion_template_name = template["name"]
