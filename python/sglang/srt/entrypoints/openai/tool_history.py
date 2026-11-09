"""Helpers for normalizing tool-call history before chat-template rendering."""

import json


def normalize_longcat_tool_history(messages: list[dict]) -> None:
    """Normalize OpenAI tool history to the shape expected by LongCat Jinja.

    OpenAI-compatible clients encode ``function.arguments`` as a JSON string and
    identify tool results by ``tool_call_id``. The LongCat template iterates over
    an argument mapping and renders the tool name on the tool-result turn, so
    normalize both fields before the first template render.
    """

    tool_names_by_id: dict[str, str] = {}

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue

                tool_call_id = tool_call.get("id")
                tool_name = function.get("name")
                if tool_call_id and tool_name:
                    tool_names_by_id[tool_call_id] = tool_name

                arguments = function.get("arguments")
                if arguments is None:
                    function["arguments"] = {}
                elif isinstance(arguments, str):
                    stripped_arguments = arguments.strip()
                    if stripped_arguments in ("", "null", "None"):
                        function["arguments"] = {}
                    else:
                        try:
                            parsed_arguments = json.loads(stripped_arguments)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                "function_call arguments JSON parse error: "
                                f"{exc} | Original request arguments: "
                                f"{arguments[:200]!r}"
                            ) from exc
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError(
                                "function_call arguments must decode to a JSON "
                                f"object, got {type(parsed_arguments).__name__}"
                            )
                        function["arguments"] = parsed_arguments
                elif not isinstance(arguments, dict):
                    raise ValueError(
                        "function_call arguments must be a JSON string or object, "
                        f"got {type(arguments).__name__}"
                    )

        elif role == "tool" and not message.get("name"):
            tool_call_id = message.get("tool_call_id")
            tool_name = tool_names_by_id.get(tool_call_id)
            if not tool_name:
                raise ValueError(
                    f"Cannot resolve tool name for tool_call_id={tool_call_id!r}. "
                    "Either 'tool_calls[].function.name' in assistant message or "
                    "'name' in tool message must be provided."
                )
            message["name"] = tool_name
