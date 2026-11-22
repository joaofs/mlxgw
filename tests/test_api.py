import pytest

from mlxgw.api import APIError, parse_chat_request

MSG = [{"role": "user", "content": "hi"}]


def parse(**fields):
    return parse_chat_request({"model": "tiny", "messages": MSG, **fields})


def test_minimal_request_defaults():
    r = parse()
    assert (r.model, r.stream, r.include_usage, r.temperature, r.top_p, r.max_tokens, r.stop, r.seed) == \
        ("tiny", False, False, 1.0, 1.0, None, [], None)


def test_supported_fields():
    r = parse(stream=True, stream_options={"include_usage": True}, temperature=0.2, top_p=0.9,
              max_tokens=10, stop="END", seed=42, n=1)
    assert (r.stream, r.include_usage, r.temperature, r.top_p, r.max_tokens, r.stop, r.seed) == \
        (True, True, 0.2, 0.9, 10, ["END"], 42)


def test_max_completion_tokens_takes_precedence():
    assert parse(max_tokens=10, max_completion_tokens=20).max_tokens == 20
    assert parse(max_completion_tokens=5).max_tokens == 5


def test_message_normalisation():
    r = parse_chat_request({"model": "tiny", "messages": [
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None},
    ]})
    assert r.messages == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "ab"},
                          {"role": "assistant", "content": ""}]


@pytest.mark.parametrize("field,value", [
    ("tool_choice", "required"),
    ("tool_choice", {"type": "function", "function": {"name": "f"}}),
    ("response_format", {"type": "json_object"}),
    ("logprobs", True),
    ("top_logprobs", 3),
    ("logit_bias", {"1": 5}),
    ("presence_penalty", 0.5),
    ("frequency_penalty", 0.5),
    ("functions", [{"name": "f"}]),
    ("reasoning_effort", "low"),
    ("modalities", ["text", "audio"]),
])
def test_unsupported_fields_rejected(field, value):
    with pytest.raises(APIError) as e:
        parse(**{field: value})
    assert e.value.status == 400 and e.value.param == field and e.value.code == "unsupported_parameter"
    assert "not supported" in e.value.message


def test_n_greater_than_one_rejected():
    with pytest.raises(APIError) as e:
        parse(n=2)
    assert (e.value.status, e.value.param, e.value.code) == (400, "n", "unsupported_parameter")


@pytest.mark.parametrize("field,value", [
    ("tools", []), ("tool_choice", "none"), ("response_format", {"type": "text"}),
    ("logprobs", False), ("presence_penalty", 0), ("frequency_penalty", 0.0), ("n", 1),
    ("user", "u1"), ("metadata", {"a": "b"}), ("parallel_tool_calls", True), ("tools", None),
])
def test_noop_values_accepted(field, value):
    parse(**{field: value})


def test_unknown_field_rejected():
    with pytest.raises(APIError) as e:
        parse(top_k=40)
    assert (e.value.status, e.value.param, e.value.code) == (400, "top_k", "unknown_parameter")


@pytest.mark.parametrize("messages,param", [
    ([{"role": "tool", "content": "x"}], "messages[0].tool_call_id"),
    ([{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}], "messages[0].tool_calls[0]"),
    ([{"role": "function", "name": "f", "content": "x"}], "messages[0]"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}], "messages[0].content[0]"),
    ([{"role": "wizard", "content": "x"}], "messages[0].role"),
    ([], "messages"),
    ("hi", "messages"),
])
def test_bad_messages(messages, param):
    with pytest.raises(APIError) as e:
        parse_chat_request({"model": "tiny", "messages": messages})
    assert e.value.status == 400 and e.value.param == param


@pytest.mark.parametrize("fields,param", [
    ({"temperature": 3}, "temperature"),
    ({"temperature": "hot"}, "temperature"),
    ({"top_p": 1.5}, "top_p"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": True}, "max_tokens"),
    ({"stop": ["a", "b", "c", "d", "e"]}, "stop"),
    ({"stop": [""]}, "stop"),
    ({"stop": 5}, "stop"),
    ({"seed": 1.5}, "seed"),
    ({"stream": "yes"}, "stream"),
    ({"stream_options": {"include_usage": True}}, "stream_options"),  # without stream
    ({"stream": True, "stream_options": {"bogus": 1}}, "stream_options"),
    ({"model": ""}, "model"),
])
def test_invalid_values(fields, param):
    with pytest.raises(APIError) as e:
        parse(**fields)
    assert e.value.status == 400 and e.value.param == param


def test_body_must_be_object():
    with pytest.raises(APIError):
        parse_chat_request([1, 2])


# --- tools -------------------------------------------------------------------

TOOL = {"type": "function", "function": {"name": "read", "description": "Read a file",
                                         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}


def test_tools_accepted_and_strict_ignored():
    r = parse(tools=[{"type": "function", "function": {**TOOL["function"], "strict": False}}], tool_choice="auto")
    assert r.tools == [TOOL]  # `strict` is accepted but not passed on


def test_no_tools_means_plain_chat():
    assert parse().tools is None
    assert parse(tools=[]).tools is None
    assert parse(tools=None, tool_choice="auto").tools is None
    assert parse(tools=[TOOL], tool_choice="none").tools is None  # validated, then not rendered


@pytest.mark.parametrize("tools,param,code", [
    ({"type": "function"}, "tools", "invalid_value"),
    ([{"type": "function"}], "tools[0]", "invalid_value"),
    ([{"type": "custom", "custom": {"name": "f"}}], "tools[0]", "invalid_value"),
    ([{**TOOL, "extra": 1}], "tools[0]", "invalid_value"),
    ([{"type": "function", "function": {"description": "no name"}}], "tools[0].function.name", "invalid_value"),
    ([{"type": "function", "function": {"name": "f", "parameters": "x"}}], "tools[0].function.parameters",
     "invalid_value"),
    ([{"type": "function", "function": {"name": "f", "bogus": True}}], "tools[0].function", "unknown_parameter"),
])
def test_bad_tools(tools, param, code):
    with pytest.raises(APIError) as e:
        parse(tools=tools)
    assert (e.value.status, e.value.param, e.value.code) == (400, param, code)


def test_tool_turn_messages_normalised():
    r = parse_chat_request({"model": "tiny", "tools": [TOOL], "messages": [
        {"role": "user", "content": "read a.txt"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.txt"}'}},
            {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": ""}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "x = 1"}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "error"},
    ]})
    assert r.messages[1] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": {"path": "a.txt"}}},
        {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": {}}},
    ]}
    assert r.messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "x = 1"}
    assert r.messages[3] == {"role": "tool", "tool_call_id": "call_2", "content": "error"}


@pytest.mark.parametrize("message,param", [
    ({"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
                                           "function": {"name": "f", "arguments": "{bad"}}]},
     "messages[0].tool_calls[0].function.arguments"),
    ({"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
                                           "function": {"name": "f", "arguments": "[1]"}}]},
     "messages[0].tool_calls[0].function.arguments"),
    ({"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f"}}]},
     "messages[0].tool_calls[0]"),
    ({"role": "assistant", "tool_calls": {"id": "c"}}, "messages[0].tool_calls"),
    ({"role": "user", "content": "x", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f"}}]},
     "messages[0].tool_calls"),
    ({"role": "tool", "tool_call_id": "", "content": "x"}, "messages[0].tool_call_id"),
    ({"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": "{}"}}, "messages[0]"),
])
def test_bad_tool_messages(message, param):
    with pytest.raises(APIError) as e:
        parse_chat_request({"model": "tiny", "messages": [message]})
    assert e.value.status == 400 and e.value.param == param


def test_reasoning_effort_none_is_a_noop():
    parse(reasoning_effort="none")


def test_opencode_request_shape_is_accepted():
    """Field for field, the body OpenCode 2.3.2 sent (captured), with content shortened."""
    r = parse_chat_request({
        "model": "fast", "stream": True, "stream_options": {"include_usage": True},
        "max_completion_tokens": 16384,
        "tools": [{"type": "function", "function": {**TOOL["function"], "strict": False}}],
        "messages": [{"role": "system", "content": "You are an AI agent"},
                     {"role": "user", "content": "Read note.txt"}],
    })
    assert (r.stream, r.include_usage, r.max_tokens, r.tools) == (True, True, 16384, [TOOL])
