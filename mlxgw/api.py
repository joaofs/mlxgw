"""OpenAI wire format: errors, request validation, response/chunk shapes."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field


class APIError(Exception):
    """An error rendered as {"error": {"message", "type", "code", "param"}}."""

    def __init__(self, status: int, message: str, type: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type, code, param

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "code": self.code, "param": self.param}}


def invalid(message: str, param: str | None = None, code: str = "invalid_value") -> APIError:
    return APIError(400, message, code=code, param=param)


# Fields we act on.
SUPPORTED = {
    "model", "messages", "stream", "stream_options", "temperature", "top_p",
    "max_tokens", "max_completion_tokens", "stop", "seed", "n", "tools", "tool_choice",
}
# OpenAI fields that have no effect on generation; accepted and ignored.
IGNORED = {"user", "metadata", "store", "service_tier", "prompt_cache_key", "safety_identifier", "parallel_tool_calls"}
# OpenAI fields we do not implement. Rejected unless set to their no-op value, because
# clients often send defaults explicitly. reasoning_effort "none" is a no-op for the
# non-thinking models configured here (OpenCode sends it for title generation).
UNSUPPORTED_NOOP_VALUES = {
    "functions": (None, []),
    "function_call": (None, "none"), "response_format": (None, {"type": "text"}),
    "logprobs": (None, False), "top_logprobs": (None, 0), "logit_bias": (None, {}),
    "presence_penalty": (None, 0), "frequency_penalty": (None, 0), "audio": (None,),
    "modalities": (None, ["text"]), "prediction": (None,), "reasoning_effort": (None, "none"),
    "web_search_options": (None,), "verbosity": (None,),
}
# Fields of a function tool definition. `strict` (OpenCode sends false) only matters for
# constrained decoding, which this gateway does not do; accepted and ignored.
TOOL_FUNCTION_FIELDS = {"name", "description", "parameters"}
TOOL_FUNCTION_IGNORED = {"strict"}
ROLES = {"system", "developer", "user", "assistant", "tool"}
MAX_STOP = 4


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]  # normalised for the chat template; see _messages
    stream: bool = False
    tools: list[dict] | None = None  # function tools to render and parse; None means plain chat
    include_usage: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    received_at: float = field(default_factory=time.perf_counter)


def _number(body: dict, key: str, lo: float, hi: float) -> float | None:
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise invalid(f"'{key}' must be a number between {lo} and {hi}", key)
    return float(v)


def _int(body: dict, key: str, lo: int) -> int | None:
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise invalid(f"'{key}' must be an integer >= {lo}", key)
    return v


def _content_text(content, param: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for i, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise invalid("only text content parts are supported", f"{param}[{i}]", "unsupported_content")
            parts.append(part["text"])
        return "".join(parts)
    raise invalid("'content' must be a string or a list of text parts", param)


def _tool_calls(raw, p: str) -> list[dict]:
    """Assistant tool calls -> the shape chat templates expect (arguments as an object)."""
    if not isinstance(raw, list):
        raise invalid("'tool_calls' must be an array", p)
    out = []
    for j, tc in enumerate(raw):
        q = f"{p}[{j}]"
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict) or tc.get("type", "function") != "function" \
                or not isinstance(tc.get("id"), str) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise invalid("each tool call needs 'id', 'type': 'function' and 'function.name'", q)
        args = fn.get("arguments") or "{}"  # missing or empty: a call without arguments
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            raise invalid("'function.arguments' must be a JSON object encoded as a string",
                          f"{q}.function.arguments")
        out.append({"id": tc["id"], "type": "function", "function": {"name": fn["name"], "arguments": args}})
    return out


def _messages(raw) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise invalid("'messages' must be a non-empty array", "messages")
    out = []
    for i, m in enumerate(raw):
        p = f"messages[{i}]"
        if not isinstance(m, dict):
            raise invalid("each message must be an object", p)
        role = m.get("role")
        if role == "function" or m.get("function_call"):
            raise invalid("legacy function messages are not supported; use tools and tool_calls",
                          p, "unsupported_parameter")
        if role not in ROLES:
            raise invalid(f"'role' must be one of {sorted(ROLES)}", f"{p}.role")
        msg = {"role": "system" if role == "developer" else role,
               "content": _content_text(m.get("content"), f"{p}.content")}
        if role == "tool":
            if not isinstance(m.get("tool_call_id"), str) or not m["tool_call_id"]:
                raise invalid("a tool message needs 'tool_call_id'", f"{p}.tool_call_id")
            msg["tool_call_id"] = m["tool_call_id"]
        elif m.get("tool_calls"):
            if role != "assistant":
                raise invalid("only assistant messages can carry 'tool_calls'", f"{p}.tool_calls")
            msg["tool_calls"] = _tool_calls(m["tool_calls"], f"{p}.tool_calls")
        out.append(msg)
    return out


def _tools(body: dict) -> list[dict] | None:
    """Validated function tools, or None when none are offered or tool_choice is 'none'."""
    choice = body.get("tool_choice")
    if choice not in (None, "auto", "none"):
        raise invalid("'tool_choice' values other than 'auto' and 'none' are not supported by this "
                      "gateway ('required' and named functions need constrained decoding)",
                      "tool_choice", "unsupported_parameter")
    raw = body.get("tools")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise invalid("'tools' must be an array", "tools")
    tools = []
    for i, t in enumerate(raw):
        p = f"tools[{i}]"
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or t.get("type") != "function" or set(t) - {"type", "function"}:
            raise invalid("each tool must be {'type': 'function', 'function': {...}}", p)
        if extra := set(fn) - TOOL_FUNCTION_FIELDS - TOOL_FUNCTION_IGNORED:
            raise invalid(f"unrecognized tool function field: '{sorted(extra)[0]}'", f"{p}.function",
                          "unknown_parameter")
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            raise invalid("'function.name' is required", f"{p}.function.name")
        if not isinstance(fn.get("description", ""), str):
            raise invalid("'function.description' must be a string", f"{p}.function.description")
        if not isinstance(fn.get("parameters", {}), dict):
            raise invalid("'function.parameters' must be a JSON Schema object", f"{p}.function.parameters")
        tools.append({"type": "function", "function": {k: fn[k] for k in TOOL_FUNCTION_FIELDS if k in fn}})
    return tools if tools and choice != "none" else None


def parse_chat_request(body) -> ChatRequest:
    if not isinstance(body, dict):
        raise invalid("request body must be a JSON object")
    for key in body:
        if key in UNSUPPORTED_NOOP_VALUES:
            if body[key] not in UNSUPPORTED_NOOP_VALUES[key]:
                raise invalid(f"'{key}' is not supported by this gateway", key, "unsupported_parameter")
        elif key not in SUPPORTED and key not in IGNORED:
            raise invalid(f"unrecognized request argument: '{key}'", key, "unknown_parameter")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise invalid("'model' is required", "model")
    n = _int(body, "n", 1)
    if n not in (None, 1):
        raise invalid("only n=1 is supported", "n", "unsupported_parameter")

    stream = body.get("stream") or False
    if not isinstance(stream, bool):
        raise invalid("'stream' must be a boolean", "stream")
    include_usage = False
    opts = body.get("stream_options")
    if opts is not None:
        if not stream:
            raise invalid("'stream_options' is only allowed when 'stream' is true", "stream_options")
        if not isinstance(opts, dict) or set(opts) - {"include_usage"}:
            raise invalid("'stream_options' supports only 'include_usage'", "stream_options")
        include_usage = bool(opts.get("include_usage", False))

    stop = body.get("stop")
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > MAX_STOP or not all(isinstance(s, str) and s for s in stop):
        raise invalid(f"'stop' must be a non-empty string or a list of up to {MAX_STOP} non-empty strings", "stop")

    max_tokens = _int(body, "max_completion_tokens", 1) or _int(body, "max_tokens", 1)
    seed = _int(body, "seed", -(2**63))
    temperature = _number(body, "temperature", 0, 2)
    top_p = _number(body, "top_p", 0, 1)
    return ChatRequest(
        model=model, messages=_messages(body.get("messages")), stream=stream,
        tools=_tools(body), include_usage=include_usage,
        temperature=1.0 if temperature is None else temperature,
        top_p=1.0 if top_p is None else top_p,
        max_tokens=max_tokens, stop=stop, seed=seed,
    )


# --- responses -------------------------------------------------------------

def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def tool_call(call, index: int | None = None) -> dict:
    """OpenAI tool call from an engine ToolCall; `index` is set in stream deltas."""
    out = {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
    return out if index is None else {"index": index, **out}


def completion(id: str, model: str, created: int, text: str, finish_reason: str,
               usage_: dict, timings: dict, tool_calls: list | None = None) -> dict:
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message = {"role": "assistant", "content": text or None,
                   "tool_calls": [tool_call(c) for c in tool_calls]}
    return {
        "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message,
                     "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage_,
        "x_gateway": timings,
    }


def chunk(id: str, model: str, created: int, delta: dict | None, finish_reason: str | None = None,
          **extra) -> dict:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "logprobs": None,
                                         "finish_reason": finish_reason}]
    return {"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": choices, **extra}
