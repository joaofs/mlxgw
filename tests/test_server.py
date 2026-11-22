import json
import time

import httpx
import openai
import pytest
from starlette.testclient import TestClient

from mlxgw.server import create_app

from .conftest import FakeEngine, make_settings

MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine, make_settings()))


def assert_error_shape(resp, status, code=None, param=None):
    assert resp.status_code == status
    body = resp.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "code", "param"}
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    if code is not None:
        assert body["error"]["code"] == code
    assert body["error"]["param"] == param
    return body["error"]


def sse_events(text):
    """Parse an SSE body strictly: every event is a single 'data: ...' line followed by a blank line."""
    assert text.endswith("\n\n")
    events = text[:-2].split("\n\n")
    assert all(e.startswith("data: ") and "\n" not in e for e in events), events
    return [e[len("data: "):] for e in events]


# --- errors ------------------------------------------------------------------

def test_unsupported_field_is_400_with_openai_shape(client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG,
                                                  "response_format": {"type": "json_object"}})
    err = assert_error_shape(r, 400, "unsupported_parameter", "response_format")
    assert err["type"] == "invalid_request_error"


def test_unknown_model_is_404(client):
    r = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": MSG})
    assert_error_shape(r, 404, "model_not_found", "model")


def test_invalid_json_is_400(client):
    r = client.post("/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"})
    assert_error_shape(r, 400, "invalid_json")


def test_unknown_route_and_method(client):
    assert_error_shape(client.get("/v1/nothing"), 404, "not_found")
    assert_error_shape(client.get("/v1/chat/completions"), 405, "method_not_allowed")


def test_engine_error_mid_stream_is_reported_in_stream(engine, client):
    def broken(req, cancel):
        async def gen():
            yield "partial"
            raise RuntimeError("boom")
        return gen()
    engine.stream = broken
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "stream": True})
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    err = json.loads(events[-2])["error"]
    assert err["type"] == "server_error" and set(err) == {"message", "type", "code", "param"}


# --- models ------------------------------------------------------------------

def test_models_lists_aliases(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["tiny", "coder"]
    assert all(m["object"] == "model" for m in body["data"])


# --- non-streaming -----------------------------------------------------------

def test_non_streaming_completion(engine, client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "seed": 1,
                                                  "temperature": 0, "max_completion_tokens": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "Hello, world!"},
                                "logprobs": None, "finish_reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11,
                             "prompt_tokens_details": {"cached_tokens": 3}}
    assert body["x_gateway"]["ttft_s"] == 0.01
    req = engine.requests[0]
    assert (req.seed, req.temperature, req.max_tokens) == (1, 0.0, 50)


def test_stop_and_length(client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "stop": [", w"]})
    assert r.json()["choices"][0]["message"]["content"] == "Hello"
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "max_tokens": 2})
    assert r.json()["choices"][0]["message"]["content"] == "Hello,"
    assert r.json()["choices"][0]["finish_reason"] == "length"


# --- streaming ---------------------------------------------------------------

def test_sse_framing_with_usage(client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "stream": True,
                                                  "stream_options": {"include_usage": True}})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[:-1])
    assert text == "Hello, world!"
    # every chunk but the last carries usage: null; the last has empty choices and the usage
    assert all(c["usage"] is None for c in chunks[:-1])
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 4
    assert chunks[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
    assert "x_gateway" in chunks[-1]


def test_sse_without_usage(client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "stream": True})
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all("usage" not in c for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert "x_gateway" in chunks[-1]


def test_stream_validation_errors_are_plain_json(client):
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": MSG, "stream": True})
    assert_error_shape(r, 404, "model_not_found", "model")


# --- auth --------------------------------------------------------------------

def test_loopback_needs_no_token():
    c = TestClient(create_app(FakeEngine(), make_settings(host="127.0.0.1")))
    assert c.get("/v1/models").status_code == 200


def test_loopback_with_configured_token_enforces_it():
    c = TestClient(create_app(FakeEngine(), make_settings(host="127.0.0.1", api_key="s3cret")))
    assert_error_shape(c.get("/v1/models"), 401, "invalid_api_key")
    assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_non_loopback_without_token_refuses_to_start(host):
    with pytest.raises(SystemExit):
        create_app(FakeEngine(), make_settings(host=host))


def test_non_loopback_requires_bearer_token():
    c = TestClient(create_app(FakeEngine(), make_settings(host="0.0.0.0", api_key="s3cret")))
    assert_error_shape(c.get("/v1/models"), 401, "invalid_api_key")
    assert_error_shape(c.get("/v1/models", headers={"Authorization": "Bearer wrong"}), 401, "invalid_api_key")
    assert_error_shape(c.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG}), 401)
    ok = c.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG},
                headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


# --- live server: SDK contract, disconnect, responsiveness ---------------------

def test_openai_sdk_contract(live_server):
    base = live_server(FakeEngine())
    client = openai.OpenAI(base_url=base + "/v1", api_key="unused")
    assert [m.id for m in client.models.list()] == ["tiny", "coder"]
    r = client.chat.completions.create(model="tiny", messages=MSG, max_tokens=10)
    assert r.choices[0].message.content == "Hello, world!" and r.usage.total_tokens == 11
    stream = client.chat.completions.create(model="tiny", messages=MSG, stream=True,
                                            stream_options={"include_usage": True})
    chunks = list(stream)
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello, world!"
    assert chunks[-1].usage.completion_tokens == 4
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(model="tiny", messages=MSG, response_format={"type": "json_object"})
    assert e.value.body["param"] == "response_format"
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="missing", messages=MSG)


def test_client_disconnect_stops_generation(live_server):
    engine = FakeEngine(pieces=["tok"] * 200, delay=0.02)  # ~4 s if not cancelled
    base = live_server(engine)
    with httpx.Client(timeout=10) as http:
        with http.stream("POST", base + "/v1/chat/completions",
                         json={"model": "tiny", "messages": MSG, "stream": True}) as r:
            lines = r.iter_lines()
            for _ in range(6):
                next(lines)
        # leaving the block closes the connection mid-stream
    assert engine.cancelled.wait(2), "generation did not observe the disconnect"
    assert engine.finished.wait(2)


def test_non_streaming_disconnect_stops_generation(live_server):
    engine = FakeEngine(pieces=["tok"] * 200, delay=0.02)
    base = live_server(engine)
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(base + "/v1/chat/completions", json={"model": "tiny", "messages": MSG}, timeout=0.3)
    assert engine.cancelled.wait(2)


def test_event_loop_stays_responsive_and_generation_is_serialised(live_server):
    import threading
    engine = FakeEngine(pieces=["tok"] * 25, delay=0.02)  # ~0.5 s per request
    base = live_server(engine)
    results = []

    def run():
        t = time.perf_counter()
        httpx.post(base + "/v1/chat/completions", json={"model": "tiny", "messages": MSG}, timeout=10)
        results.append(time.perf_counter() - t)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    t0 = time.perf_counter()
    assert httpx.get(base + "/v1/models").status_code == 200
    assert time.perf_counter() - t0 < 0.2  # not blocked behind generation
    for t in threads:
        t.join()
    # two 0.5 s generations in a single worker: the second finishes after ~1 s
    assert max(results) > 0.9


# --- tool calls --------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                                           "strict": False}}]
CALL_A = '<tool_call>\n{"name": "read", "arguments": {"path": "a.txt"}}\n</tool_call>'
CALL_B = '<tool_call>\n{"name": "read", "arguments": {"path": "b.txt"}}\n</tool_call>'
# What a Qwen model streams: some text, two calls, markers split across pieces.
TOOL_PIECES = ["I will read both.", "\n<tool", "_call>\n{\"name\": \"read\", \"arg", "uments\": {\"path\": \"a.txt\"}}\n</",
               "tool_call>\n", CALL_B[:20], CALL_B[20:]]


def test_non_streaming_tool_calls(engine, client):
    engine.pieces = TOOL_PIECES
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["content"] == "I will read both.\n"
    assert [(c["type"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg["tool_calls"]] \
        == [("function", "read", {"path": "a.txt"}), ("function", "read", {"path": "b.txt"})]
    ids = [c["id"] for c in msg["tool_calls"]]
    assert len(set(ids)) == 2 and all(i.startswith("call_") for i in ids)
    assert all("index" not in c for c in msg["tool_calls"])
    assert engine.requests[0].tools[0]["function"]["name"] == "read"


def test_only_tool_calls_gives_null_content(engine, client):
    engine.pieces = [CALL_A]
    msg = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS}
                      ).json()["choices"][0]["message"]
    assert msg["content"] is None and len(msg["tool_calls"]) == 1


def test_streaming_tool_calls(engine, client):
    engine.pieces = TOOL_PIECES
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS, "stream": True,
                                                  "stream_options": {"include_usage": True}})
    chunks = [json.loads(e) for e in sse_events(r.text)[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    content = "".join(d.get("content") or "" for d in deltas)
    assert content == "I will read both.\n" and "tool_call" not in content and "{" not in content
    calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
    assert [(tc["index"], tc["type"], tc["function"]["name"], tc["function"]["arguments"]) for tc in calls] == \
        [(0, "function", "read", '{"path": "a.txt"}'), (1, "function", "read", '{"path": "b.txt"}')]
    assert all(tc["id"].startswith("call_") for tc in calls)
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["completion_tokens"] == len(TOOL_PIECES)


def test_text_answer_with_tools_finishes_with_stop(engine, client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"] == {"role": "assistant", "content": "Hello, world!"}


def test_length_inside_a_call_finishes_with_length(engine, client):
    engine.pieces = ["<tool_call>", '{"name": "read", ', '"arguments": {"path": "a.txt"}}', "</tool_call>"]
    choice = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS,
                                                       "max_tokens": 2}).json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"] == {"role": "assistant", "content": ""}  # truncated call dropped, nothing leaked


def test_tool_choice_none_leaves_markup_unparsed(engine, client):
    engine.pieces = [CALL_A]
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS,
                                                  "tool_choice": "none"})
    assert engine.requests[0].tools is None
    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_tools_for_a_model_without_tool_calling_is_400():
    client = TestClient(create_app(FakeEngine(tool_calling=False), make_settings()))
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS})
    assert_error_shape(r, 400, "tools_not_supported", "tools")


def test_tool_choice_required_is_400(client):
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "tools": TOOLS,
                                                  "tool_choice": "required"})
    assert_error_shape(r, 400, "unsupported_parameter", "tool_choice")


def test_openai_sdk_tool_round_trip(live_server):
    engine = FakeEngine(pieces=TOOL_PIECES)
    client = openai.OpenAI(base_url=live_server(engine) + "/v1", api_key="unused")
    r = client.chat.completions.create(model="tiny", messages=MSG, tools=TOOLS, tool_choice="auto")
    msg = r.choices[0].message
    assert r.choices[0].finish_reason == "tool_calls" and len(msg.tool_calls) == 2
    # The SDK's own streaming accumulator rebuilds the same calls.
    with client.chat.completions.stream(model="tiny", messages=MSG, tools=TOOLS) as s:
        streamed = s.get_final_completion().choices[0].message
    assert [(c.function.name, c.function.arguments) for c in streamed.tool_calls] == \
        [(c.function.name, c.function.arguments) for c in msg.tool_calls]
    # Send the results back, as an agent loop does.
    history = MSG + [msg.model_dump(exclude_none=True)] + [
        {"role": "tool", "tool_call_id": c.id, "content": [{"type": "text", "text": "contents"}]}
        for c in msg.tool_calls]
    engine.pieces = ["Both files say contents."]
    r2 = client.chat.completions.create(model="tiny", messages=history, tools=TOOLS)
    assert r2.choices[0].finish_reason == "stop" and r2.choices[0].message.content == "Both files say contents."
    sent = engine.requests[-1].messages
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}
    assert sent[2] == {"role": "tool", "tool_call_id": msg.tool_calls[0].id, "content": "contents"}
