import asyncio
import threading
import time

import pytest

from mlxgw.api import APIError
from mlxgw.config import GB
from mlxgw.engine import (ModelManager, OutputFilter, PromptCacheStore, StopMatcher, ToolCall, Worker,
                               render_prompt)

from .conftest import json_tool_splitter


# --- tool calls --------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
CALL = '<tool_call>\n{"name": "read", "arguments": {"path": "a.txt"}}\n</tool_call>'


def run_output(pieces, stops=(), tools=TOOLS):
    """Feed pieces through an OutputFilter; returns (content, calls, stop_hit, calls_count)."""
    out = OutputFilter(list(stops), json_tool_splitter(tools) if tools else None)
    items, hit = [], False
    for p in pieces:
        got, hit = out.feed(p)
        items += got
        if hit:
            break
    items += out.finish()
    content = "".join(i for i in items if isinstance(i, str))
    calls = [i for i in items if isinstance(i, ToolCall)]
    return content, [(c.name, c.arguments) for c in calls], hit, out.calls, items


def test_single_call_is_parsed_and_markup_removed():
    content, calls, _, n, items = run_output([CALL])
    assert content == "" and calls == [("read", '{"path": "a.txt"}')] and n == 1
    assert items[0].id.startswith("call_")


def test_markup_split_across_chunks_one_char_at_a_time():
    content, calls, *_ = run_output(list("Let me look.\n" + CALL))
    assert content == "Let me look.\n"
    assert calls == [("read", '{"path": "a.txt"}')]


def test_markers_split_across_chunk_boundaries():
    content, calls, *_ = run_output(["ok <tool_", "call>\n{\"name\": \"read\", \"argu", "ments\": {}}\n</tool",
                                     "_call>"])
    assert content == "ok " and calls == [("read", "{}")]


def test_multiple_calls_and_whitespace_between_them_dropped():
    two = CALL + "\n" + CALL.replace("a.txt", "b.txt") + "\n"
    content, calls, _, n, _ = run_output([two[i:i + 7] for i in range(0, len(two), 7)])
    assert content == "" and n == 2
    assert calls == [("read", '{"path": "a.txt"}'), ("read", '{"path": "b.txt"}')]


def test_text_after_a_call_is_kept():
    content, calls, *_ = run_output([CALL, "\n\nDone."])
    assert content == "\n\nDone." and len(calls) == 1


def test_malformed_arguments_are_dropped_without_leaking(caplog):
    bad = '<tool_call>\n{"name": "read", "arguments": {"path": }\n</tool_call>'
    content, calls, _, n, _ = run_output(["Before. ", bad, " after"])
    assert calls == [] and n == 0
    assert content == "Before.  after"
    assert "tool_call" not in content and "dropped a tool call" in caplog.text


def test_one_bad_call_does_not_drop_the_good_one():
    bad = '<tool_call>{"arguments": {}}</tool_call>'  # no name
    content, calls, *_ = run_output([bad, CALL])
    assert calls == [("read", '{"path": "a.txt"}')] and "tool_call" not in content


def test_call_without_end_marker_is_parsed_at_finish():
    content, calls, *_ = run_output(['<tool_call>\n{"name": "read", "arguments": {}}'])
    assert content == "" and calls == [("read", "{}")]


def test_truncated_call_is_dropped():
    content, calls, *_ = run_output(['<tool_call>\n{"name": "read", "argum'])
    assert content == "" and calls == []


def test_text_that_only_looks_like_a_marker_is_released():
    content, calls, *_ = run_output(["if a <tool", "s> and <tool_c"])
    assert content == "if a <tools> and <tool_c" and calls == []


def test_without_tools_markup_passes_through():
    content, calls, _, n, _ = run_output([CALL], tools=None)
    assert content == CALL and calls == [] and n == 0


def test_stop_strings_apply_to_content_not_call_arguments():
    content, calls, hit, *_ = run_output(['<tool_call>{"name": "read", "arguments": {"path": "STOP"}}</tool_call>',
                                          "text STOP more"], stops=["STOP"])
    assert calls == [("read", '{"path": "STOP"}')] and content == "text " and hit


def test_held_stop_prefix_is_released_before_a_call():
    *_, items = run_output(["a S", CALL], stops=["STOP"])
    assert items[:2] == ["a ", "S"] and isinstance(items[2], ToolCall)


class FakeTokenizer:
    has_chat_template, has_tool_calling = True, False

    def apply_chat_template(self, messages, **kw):
        return [1, 2, 3]


def test_tools_for_a_model_without_tool_calling_is_400():
    with pytest.raises(APIError) as e:
        render_prompt(FakeTokenizer(), [{"role": "user", "content": "hi"}], TOOLS)
    assert (e.value.status, e.value.code, e.value.param) == (400, "tools_not_supported", "tools")
    assert render_prompt(FakeTokenizer(), [{"role": "user", "content": "hi"}]) == [1, 2, 3]


# --- stop strings ------------------------------------------------------------

def feed_all(stops, pieces):
    m, out = StopMatcher(stops), []
    for p in pieces:
        text, hit = m.feed(p)
        out.append(text)
        if hit:
            return "".join(out), True
    return "".join(out) + m.flush(), False


def test_stop_string_split_across_pieces():
    assert feed_all(["</end>"], ["abc</", "en", "d> tail"]) == ("abc", True)


def test_partial_stop_prefix_is_released_when_it_does_not_match():
    m = StopMatcher(["STOP"])
    assert m.feed("xxST") == ("xx", False)  # "ST" held back
    assert m.feed("AY") == ("STAY", False)


def test_earliest_of_several_stops_wins():
    assert feed_all(["b", "a"], ["xxab"]) == ("xx", True)


def test_no_stops_passthrough():
    assert feed_all([], ["a", "b"]) == ("ab", False)


# --- prompt cache ------------------------------------------------------------

class FakeKV:
    def __init__(self, tokens, trimmable=True):
        self.tokens, self.trimmable = list(tokens), trimmable


def trim(cache, n):
    cache.tokens = cache.tokens[:-n]
    return n


def store(slots=4):
    return PromptCacheStore(slots, can_trim=lambda c: c.trimmable, trim=trim)


def test_multi_turn_extension_reuses_cache_in_place():
    s = store()
    turn1 = [1, 2, 3, 4, 5]  # prompt + generated
    kv = FakeKV(turn1)
    s.put(turn1, kv)
    turn2 = turn1 + [6, 7, 8]
    cache, reused = s.take(turn2)
    assert cache is kv and reused == 5
    assert s.entries == []  # handed over, will be put back after generation


def test_partial_match_trims_a_copy_and_keeps_original():
    s = store()
    kv = FakeKV([1, 2, 3, 4, 5, 6])
    s.put(kv.tokens, kv)
    cache, reused = s.take([1, 2, 3, 9, 9])
    assert reused == 3 and cache is not kv and cache.tokens == [1, 2, 3]
    assert kv.tokens == [1, 2, 3, 4, 5, 6] and len(s.entries) == 1


def test_identical_prompt_leaves_one_token_to_process():
    s = store()
    s.put([1, 2, 3], FakeKV([1, 2, 3]))
    cache, reused = s.take([1, 2, 3])
    assert reused == 2 and cache.tokens == [1, 2]


def test_longest_prefix_wins():
    s = store()
    a, b = FakeKV([1, 2, 7]), FakeKV([1, 2, 3, 4, 8])
    s.put(a.tokens, a)
    s.put(b.tokens, b)
    cache, reused = s.take([1, 2, 3, 4, 5, 6])
    assert reused == 4 and cache.tokens == [1, 2, 3, 4]


def test_non_trimmable_cache_only_reused_as_full_prefix():
    s = store()
    kv = FakeKV([1, 2, 3], trimmable=False)
    s.put(kv.tokens, kv)
    assert s.take([1, 2, 9, 9]) == (None, 0)
    assert s.take([1, 2, 3, 4]) == (kv, 3)


def test_no_match_returns_none():
    s = store()
    s.put([1, 2], FakeKV([1, 2]))
    assert s.take([5, 6, 7]) == (None, 0)


def test_slots_bound_evicts_oldest():
    s = store(slots=2)
    for first in (10, 20, 30):
        s.put([first, 1], FakeKV([first, 1]))
    assert [e.tokens[0] for e in s.entries] == [20, 30]


def test_put_drops_entries_that_are_prefixes_of_the_new_one():
    s = store()
    s.put([1, 2], FakeKV([1, 2]))
    s.put([5, 6], FakeKV([5, 6]))
    s.put([1, 2, 3], FakeKV([1, 2, 3]))
    assert [e.tokens for e in s.entries] == [[5, 6], [1, 2, 3]]


# --- model LRU ---------------------------------------------------------------

@pytest.fixture
def model_dirs(tmp_path):
    """Fake model dirs whose *.safetensors files (sparse) have the given GB sizes."""
    def make(name, gb):
        d = tmp_path / name
        d.mkdir()
        with open(d / "model.safetensors", "wb") as f:
            f.truncate(int(gb * GB))
        (d / "config.json").write_text("{}")  # not counted
        return d
    return {"a": make("a", 10), "b": make("b", 12), "c": make("c", 15), "huge": make("huge", 40)}


def manager(model_dirs, budget_gb=30):
    loads, evictions = [], []
    m = ModelManager(
        aliases={"a": "a", "b": "b", "c": "c", "huge": "huge"}, budget=int(budget_gb * GB),
        fetch=lambda repo: model_dirs[repo],
        load=lambda path: (loads.append(path.name) or f"model-{path.name}", "tok"),
        make_caches=lambda: None, on_evict=lambda: evictions.append(1))
    return m, loads, evictions


def test_size_estimated_from_weight_files(model_dirs):
    assert ModelManager.weights_size(model_dirs["b"]) == 12 * GB


def test_load_once_and_keep_resident(model_dirs):
    m, loads, _ = manager(model_dirs)
    assert m.get("a").model == "model-a"
    m.get("a")
    assert loads == ["a"]


def test_lru_eviction_against_budget(model_dirs):
    m, loads, evictions = manager(model_dirs, budget_gb=30)
    m.get("a")  # 10
    m.get("b")  # 22
    m.get("a")  # touch a -> b is now LRU
    m.get("c")  # 37 > 30 -> evict b
    assert list(m.loaded) == ["a", "c"] and m.resident_bytes() == 25 * GB
    assert len(evictions) == 1
    m.get("b")  # 37 > 30 -> evict a (LRU) -> 27
    assert list(m.loaded) == ["c", "b"]
    assert loads == ["a", "b", "c", "b"]


def test_evicts_several_when_needed(model_dirs):
    m, _, _ = manager(model_dirs, budget_gb=40)
    m.get("a"), m.get("b"), m.get("c")  # 37
    m.get("huge")  # 40 -> must evict all three
    assert list(m.loaded) == ["huge"]


def test_model_larger_than_budget_is_rejected(model_dirs):
    m, loads, _ = manager(model_dirs, budget_gb=30)
    m.get("a")
    with pytest.raises(APIError) as e:
        m.get("huge")
    assert e.value.status == 503 and e.value.code == "model_too_large"
    assert list(m.loaded) == ["a"] and loads == ["a"]  # nothing evicted for a hopeless load


# --- worker ------------------------------------------------------------------

def test_worker_serialises_jobs_and_forwards_errors():
    worker = Worker()
    active, peak = [0], [0]
    lock = threading.Lock()

    def job(emit, cancel):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        emit("x")
        with lock:
            active[0] -= 1

    def failing(emit, cancel):
        raise ValueError("bad")

    async def consume(j):
        return [x async for x in worker.run(j, threading.Event())]

    async def main():
        results = await asyncio.gather(*(consume(job) for _ in range(4)))
        assert results == [["x"]] * 4
        with pytest.raises(ValueError):
            await consume(failing)

    asyncio.run(main())
    assert peak[0] == 1


def test_worker_skips_job_cancelled_while_queued():
    worker = Worker()
    ran = []

    async def main():
        ev = threading.Event()
        ev.set()
        return [x async for x in worker.run(lambda emit, c: ran.append(1), ev)]

    assert asyncio.run(main()) == [] and ran == []
