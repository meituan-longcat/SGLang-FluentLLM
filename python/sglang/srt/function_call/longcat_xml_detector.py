import json
import ast
import logging
import re
from typing import List, Union, Literal


from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if arg_key not in tool.function.parameters["properties"]:
        return None
    return tool.function.parameters["properties"][arg_key].get("type", None)


def parse_arguments(json_value):
    try:
        try:
            parsed_value = json.loads(json_value)
        except:
            parsed_value = ast.literal_eval(json_value)
        return parsed_value, True
    except:
        return json_value, False


class LongCatXMLDetector(BaseFormatDetector):

    _ST_IDLE = 0
    _ST_FUNC_NAME = 1
    _ST_BODY = 2
    _ST_ARG_KEY = 3
    _ST_ARG_GAP = 4
    _ST_ARG_VALUE = 5

    _TAG_ARG_KEY_OPEN = "<longcat_arg_key>"
    _TAG_ARG_KEY_CLOSE = "</longcat_arg_key>"
    _TAG_ARG_VALUE_OPEN = "<longcat_arg_value>"
    _TAG_ARG_VALUE_CLOSE = "</longcat_arg_value>"

    def __init__(self):
        super().__init__()
        self.bot_token = "<longcat_tool_call>"
        self.eot_token = "</longcat_tool_call>"
        self.func_call_regex = r"<longcat_tool_call>.*?</longcat_tool_call>"
        self.func_call_regex_fix = r"<longcat_tool_call>.*?"
        self.func_detail_regex = r"<longcat_tool_call>([^\n]*)\n(.*)</longcat_tool_call>"
        self.func_detail_regex_fix = r"<longcat_tool_call>([^\n]*)\n(.*)"
        self.func_arg_regex = r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)</longcat_arg_value>"
        self.func_arg_regex_fix_1 = r"<longcat_arg_key>(.*?)<longcat_arg_value>(.*?)</longcat_arg_value>"
        self.func_arg_regex_fix_2 = r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)"
        self.func_arg_regex_fix_3 = r"<longcat_arg_key>(.*?)<longcat_arg_value>(.*?)"
        self._state = self._ST_IDLE
        self._current_func_name = ""
        self._current_arg_key = ""
        self._current_arg_value_buf = ""
        self._kv_count = 0
        self._value_sent = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _extract_pairs(self, func_args: str) -> list:
        for regex in (self.func_arg_regex, self.func_arg_regex_fix_1,
                      self.func_arg_regex_fix_2, self.func_arg_regex_fix_3):
            pairs = re.findall(regex, func_args, re.DOTALL)
            if pairs:
                return pairs
        return []

    def detect_and_parse(
        self,
        text: str,
        tools: List[Tool],
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none", "bypass_check"]]
    ) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        if not match_result_list:
            match_result_list = re.findall(self.func_call_regex_fix, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                if not func_detail:
                    func_detail = re.search(
                        self.func_detail_regex_fix,
                        match_result, re.DOTALL,
                    )
                if not func_detail:
                    continue
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
                pairs = self._extract_pairs(func_args)
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    arg_type = get_argument_type(func_name, arg_key, tools)
                    if arg_type != "string":
                        arg_value, _ = parse_arguments(arg_value)
                    arguments[arg_key] = arg_value
                final_match_result = {"name": func_name, "parameters": arguments}
                call = self.parse_base_json(final_match_result, tools, tool_choice)
                calls.extend(call)
                if func_name and arguments and len(call) == 0:
                    normal_text += match_result
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _escape_json_str_content(self, s: str) -> str:
        return json.dumps(s, ensure_ascii=False)[1:-1]

    def _try_consume_tag(self, tag: str) -> bool:
        if self._buffer.startswith(tag):
            self._buffer = self._buffer[len(tag):]
            return True
        return False

    def _buffer_might_start_with(self, tag: str) -> bool:
        buf = self._buffer
        check_len = min(len(buf), len(tag))
        return check_len > 0 and tag[:check_len] == buf[:check_len]

    def parse_streaming_increment(
        self,
        new_text: str,
        tools: List[Tool],
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none", "bypass_check"]]
    ) -> StreamingParseResult:

        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_text = ""
        argument_diff = ""

        while self._buffer:
            if self._state == self._ST_IDLE:
                bot_pos = self._buffer.find(self.bot_token)
                if bot_pos != -1:
                    normal_text += self._buffer[:bot_pos]
                    self._buffer = self._buffer[bot_pos + len(self.bot_token):]
                    self._state = self._ST_FUNC_NAME
                    continue
                partial = self._ends_with_partial_token(self._buffer, self.bot_token)
                if partial:
                    normal_text += self._buffer[:-partial]
                    self._buffer = self._buffer[-partial:]
                    break
                if self.current_tool_id > 0:
                    self._buffer = ""
                else:
                    normal_text += self._buffer
                    self._buffer = ""
                break

            elif self._state == self._ST_FUNC_NAME:
                nl_pos = self._buffer.find("\n")
                if nl_pos == -1:
                    break
                func_name = self._buffer[:nl_pos].strip()
                self._buffer = self._buffer[nl_pos + 1:]
                if not func_name:
                    self._state = self._ST_IDLE
                    continue
                if func_name not in self._tool_indices and tool_choice != "bypass_check":
                    self._state = self._ST_IDLE
                    self._buffer = ""
                    continue

                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                self.current_tool_name_sent = True
                self._current_func_name = func_name
                self._kv_count = 0
                self._state = self._ST_BODY

                result = StreamingParseResult(
                    normal_text=normal_text,
                    calls=[ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=func_name,
                        parameters="",
                    )],
                )
                return result

            elif self._state == self._ST_BODY:
                if not self._buffer:
                    break

                if self._try_consume_tag(self._TAG_ARG_KEY_OPEN):
                    self._current_arg_key = ""
                    self._state = self._ST_ARG_KEY
                    continue
                elif self._try_consume_tag(self.eot_token) or \
                     self._buffer.startswith(self.bot_token):
                    # Normal close, or fallback: next <longcat_tool_call> without </longcat_tool_call>
                    if self._buffer.startswith(self.bot_token):
                        pass  # don't consume bot_token, let _ST_IDLE handle it
                    if self._kv_count == 0:
                        argument_diff += "{}"
                    else:
                        argument_diff += "}"

                    full_args = self.streamed_args_for_tool[self.current_tool_id] + argument_diff
                    try:
                        parsed_args = json.loads(full_args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": self._current_func_name,
                        "arguments": parsed_args,
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = full_args

                    completing_id = self.current_tool_id
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self._current_func_name = ""
                    self._kv_count = 0
                    self._state = self._ST_IDLE

                    calls = []
                    if argument_diff:
                        calls.append(ToolCallItem(
                            tool_index=completing_id,
                            parameters=argument_diff,
                        ))
                    return StreamingParseResult(normal_text=normal_text, calls=calls)
                elif self._buffer_might_start_with(self._TAG_ARG_KEY_OPEN) or \
                     self._buffer_might_start_with(self.eot_token) or \
                     self._buffer_might_start_with(self.bot_token):
                    break
                else:
                    self._buffer = self._buffer[1:]
                    continue

            elif self._state == self._ST_ARG_KEY:
                close_pos = self._buffer.find(self._TAG_ARG_KEY_CLOSE)
                if close_pos != -1:
                    self._current_arg_key += self._buffer[:close_pos]
                    self._buffer = self._buffer[close_pos + len(self._TAG_ARG_KEY_CLOSE):]
                    self._current_arg_key = self._current_arg_key.strip()
                    self._state = self._ST_ARG_GAP
                    continue
                # Fallback: <longcat_arg_value> appeared without </longcat_arg_key>
                fallback_pos = self._buffer.find(self._TAG_ARG_VALUE_OPEN)
                if fallback_pos != -1:
                    self._current_arg_key += self._buffer[:fallback_pos]
                    self._buffer = self._buffer[fallback_pos:]
                    self._current_arg_key = self._current_arg_key.strip()
                    self._state = self._ST_ARG_GAP
                    continue
 
                partial = self._ends_with_partial_token(self._buffer, self._TAG_ARG_KEY_CLOSE)
                if not partial:
                    partial = self._ends_with_partial_token(self._buffer, self._TAG_ARG_VALUE_OPEN)

                if partial:
                    self._current_arg_key += self._buffer[:-partial]
                    self._buffer = self._buffer[-partial:]
                else:
                    self._current_arg_key += self._buffer
                    self._buffer = ""
                break

            elif self._state == self._ST_ARG_GAP:
                if not self._buffer:
                    break
                if self._try_consume_tag(self._TAG_ARG_VALUE_OPEN):
                    self._current_arg_value_buf = ""
                    self._value_sent = ""
                    key_json = json.dumps(self._current_arg_key, ensure_ascii=False)
                    if self._kv_count == 0:
                        argument_diff += "{" + key_json + ": "
                    else:
                        argument_diff += ", " + key_json + ": "

                    arg_type = get_argument_type(self._current_func_name, self._current_arg_key, tools)
                    if arg_type == "string":
                        argument_diff += '"'

                    self._state = self._ST_ARG_VALUE
                    continue
                elif self._buffer_might_start_with(self._TAG_ARG_VALUE_OPEN):
                    break
                else:
                    self._buffer = self._buffer[1:]
                    continue

            elif self._state == self._ST_ARG_VALUE:
                close_pos = self._buffer.find(self._TAG_ARG_VALUE_CLOSE)
                # Fallback: next tag appeared without </longcat_arg_value>
                fallback_pos = -1
                for next_tag in (self._TAG_ARG_KEY_OPEN, self.eot_token, self.bot_token):
                    pos = self._buffer.find(next_tag)
                    if pos != -1 and (fallback_pos == -1 or pos < fallback_pos):
                        fallback_pos = pos
                # Use whichever comes first: normal close or fallback
                effective_close = -1
                is_fallback = False
                if close_pos != -1 and (fallback_pos == -1 or close_pos <= fallback_pos):
                    effective_close = close_pos
                elif fallback_pos != -1:
                    effective_close = fallback_pos
                    is_fallback = True

                if effective_close != -1:
                    last_chunk = self._buffer[:effective_close]
                    if is_fallback:
                        self._buffer = self._buffer[effective_close:]
                    else:
                        self._buffer = self._buffer[effective_close + len(self._TAG_ARG_VALUE_CLOSE):]
                    self._current_arg_value_buf += last_chunk

                    full_value = self._current_arg_value_buf
                    arg_type = get_argument_type(self._current_func_name, self._current_arg_key, tools)

                    if arg_type == "string":
                        unsent = full_value[len(self._value_sent):]
                        argument_diff += self._escape_json_str_content(unsent) + '"'
                    else:
                        parsed_val, _ = parse_arguments(full_value)
                        val_json = json.dumps(parsed_val, ensure_ascii=False)
                        argument_diff += val_json

                    self._kv_count += 1
                    self._state = self._ST_BODY
                    continue

                partial = self._ends_with_partial_token(self._buffer, self._TAG_ARG_VALUE_CLOSE)
                if not partial:
                    for next_tag in (self._TAG_ARG_KEY_OPEN, self.eot_token, self.bot_token):
                        partial = self._ends_with_partial_token(self._buffer, next_tag)
                        if partial:
                            break
                if partial:
                    streamable = self._buffer[:-partial]
                    self._current_arg_value_buf += streamable
                    self._buffer = self._buffer[-partial:]
                else:
                    streamable = self._buffer
                    self._current_arg_value_buf += streamable
                    self._buffer = ""

                arg_type = get_argument_type(self._current_func_name, self._current_arg_key, tools)
                if arg_type == "string" and streamable:
                    current_full = self._current_arg_value_buf
                    unsent = current_full[len(self._value_sent):]
                    if unsent:
                        argument_diff += self._escape_json_str_content(unsent)
                        self._value_sent = current_full
                break

        if argument_diff:
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            return StreamingParseResult(
                normal_text=normal_text,
                calls=[ToolCallItem(
                    tool_index=self.current_tool_id,
                    parameters=argument_diff,
                )],
            )

        return StreamingParseResult(normal_text=normal_text)

    def flush(self, tools: List[Tool]) -> StreamingParseResult:
        """Finalize any incomplete tool call at end-of-stream.

        Handles three cases where the model stopped without emitting a closing tag:
        - _ST_ARG_KEY: </longcat_arg_key> missing — discard the incomplete kv pair, close the tool
        - _ST_ARG_GAP: <longcat_arg_value> never arrived — same treatment
        - _ST_ARG_VALUE: </longcat_arg_value> missing — complete the value with what we have
        - _ST_BODY: </longcat_tool_call> missing — close the tool with accumulated args
        """
        if self._state == self._ST_IDLE or self.current_tool_id < 0:
            return StreamingParseResult()

        argument_diff = ""

        if self._state == self._ST_ARG_VALUE:
            remaining = self._buffer
            self._current_arg_value_buf += remaining
            self._buffer = ""

            full_value = self._current_arg_value_buf
            arg_type = get_argument_type(self._current_func_name, self._current_arg_key, tools)

            if arg_type == "string":
                unsent = full_value[len(self._value_sent):]
                argument_diff += self._escape_json_str_content(unsent) + '"'
            else:
                parsed_val, _ = parse_arguments(full_value)
                val_json = json.dumps(parsed_val, ensure_ascii=False)
                argument_diff += val_json

            self._kv_count += 1

        if self._kv_count == 0:
            argument_diff += "{}"
        else:
            argument_diff += "}"

        full_args = self.streamed_args_for_tool[self.current_tool_id] + argument_diff
        try:
            parsed_args = json.loads(full_args)
        except json.JSONDecodeError:
            parsed_args = {}
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": self._current_func_name,
            "arguments": parsed_args,
        }
        self.streamed_args_for_tool[self.current_tool_id] = full_args

        completing_id = self.current_tool_id
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self._current_func_name = ""
        self._kv_count = 0
        self._buffer = ""
        self._state = self._ST_IDLE

        calls = []
        if argument_diff:
            calls.append(ToolCallItem(
                tool_index=completing_id,
                parameters=argument_diff,
            ))
        return StreamingParseResult(calls=calls)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()