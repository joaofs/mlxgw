"""Generation engine: one worker thread, model LRU, per-model prompt (KV) cache reuse.

The pieces that hold logic worth testing (Worker, StopMatcher, PromptCacheStore,
ModelManager) do not import MLX; only MLXEngine does, lazily.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .api import APIError, ChatRequest
from .config import GB, Settings

log = logging.getLogger("mlxgw")


@dataclass
class Final:
    """Last item of every generation stream (content deltas are plain str, calls ToolCall)."""
    finish_reason: str  # "stop" | "length" | "tool_calls" | "cancelled"
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    timings: dict


@dataclass
class ToolCall:
    """One parsed tool call; `arguments` is JSON text, as in the OpenAI shape."""
    id: str
    name: str
    arguments: str


class Cancelled(Exception):
    pass


_DONE = object()


class Worker:
    """Runs jobs one at a time on a single thread; bridges their output to asyncio.

    MLX generation is not safe to run concurrently, and its generation stream is
    thread-local, so every MLX call (load and generate) happens on this one thread.
    The executor's FIFO queue is the request queue.
    """

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")
        self._lock = threading.Lock()
        self.pending = 0  # jobs submitted and not finished (running + queued)
        self.running = False

    def _track(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        with self._lock:
            self.pending += 1

        def tracked():
            self.running = True
            try:
                return fn()
            finally:
                self.running = False
                with self._lock:
                    self.pending -= 1
        return tracked

    def busy(self) -> dict:
        """{"in_flight": 0|1, "queued": n} for status displays."""
        running = int(self.running)
        return {"in_flight": running, "queued": max(0, self.pending - running)}

    async def call(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the worker thread (after anything queued) and return its result."""
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._track(fn))

    async def run(self, job: Callable[[Callable[[Any], None], threading.Event], None],
                  cancel: threading.Event) -> AsyncIterator[Any]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(item):
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def wrapper():
            try:
                if not cancel.is_set():  # client left while queued
                    job(emit, cancel)
            except BaseException as e:  # noqa: BLE001 - forwarded to the request
                emit(e)
            finally:
                emit(_DONE)

        loop.run_in_executor(self._pool, self._track(wrapper))
        try:
            while (item := await queue.get()) is not _DONE:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            cancel.set()  # no-op if finished; stops the job if the consumer went away

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class StopMatcher:
    """Applies OpenAI `stop` strings to streamed text.

    Text that could be the start of a stop string is held back until it is
    disambiguated, so a stop string is never partially streamed.
    """

    def __init__(self, stops: list[str]):
        self.stops = stops
        self.buf = ""

    def feed(self, text: str) -> tuple[str, bool]:
        if not self.stops:
            return text, False
        self.buf += text
        hits = [i for s in self.stops if (i := self.buf.find(s)) >= 0]
        if hits:
            out, self.buf = self.buf[: min(hits)], ""
            return out, True
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), hold, -1):
                if self.buf.endswith(s[:k]):
                    hold = k
                    break
        cut = len(self.buf) - hold
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


class ToolCallSplitter:
    """Cuts tool-call markup out of streamed text and parses each call.

    The markers and the parser are the ones mlx-lm chose for the model's chat template
    (tokenizer.tool_call_start / tool_call_end / tool_parser), and marker matching is
    mlx-lm's TextStateMachine reader, which holds back text that could be the start of a
    marker. Whitespace between and after calls is dropped. A call that does not parse
    (truncated, invalid JSON) is dropped with a warning, as mlx-lm's own server does.
    """

    def __init__(self, start: str, end: str | None, parse: Callable, tools: list[dict] | None):
        from mlx_lm.generate import TextStateMachine
        transitions = {"normal": [(start, "tool")], "tool": [(end, "normal")] if end else []}
        self._reader = TextStateMachine(transitions).reader(state="normal")
        self._parse, self._tools = parse, tools
        self._held = ""  # whitespace after a call, dropped unless real text follows
        self.calls = 0

    def feed(self, text: str) -> list[str | ToolCall]:
        # One character at a time: a reader Delta keeps content and calls apart, so the
        # order between them is lost when one chunk holds both. A one-character Delta
        # holds one or the other. Chunks are single tokens, so this costs little.
        out: list[str | ToolCall] = []
        for ch in text:
            for item in self._items(self._reader.feed(ch)):
                if isinstance(item, str) and out and isinstance(out[-1], str):
                    out[-1] += item  # keep one content item per run of text
                else:
                    out.append(item)
        return out

    def finish(self) -> list[str | ToolCall]:
        """End of generation: release held text; parse a call left open (no end marker)."""
        return self._items(self._reader.finish())

    def _items(self, delta) -> list[str | ToolCall]:
        out = []
        if delta.content:
            text = delta.content
            if self.calls:
                self._held += text
                text = "" if not self._held.strip() else self._held
                if text:
                    self._held = ""
            if text:
                out.append(text)
        for call in delta.tool_calls:
            self._held = ""
            out += self._parsed(call)
        return out

    def _parsed(self, text: str) -> list[ToolCall]:
        try:
            parsed = self._parse(text, self._tools)
            out = []
            for c in parsed if isinstance(parsed, list) else [parsed]:
                if not isinstance(c.get("name"), str) or not c["name"]:
                    raise ValueError("no function name")
                args = c.get("arguments", {})
                out.append(ToolCall(c.get("id") or "call_" + uuid.uuid4().hex[:24], c["name"],
                                    args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)))
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            log.warning("dropped a tool call that could not be parsed (%s: %s): %.300r", type(e).__name__, e, text)
            return []
        self.calls += len(out)
        return out


class OutputFilter:
    """Generated text -> content deltas (str) and ToolCalls.

    Tool-call markup is cut out first (when the request has tools); stop strings then apply
    to the content only.
    """

    def __init__(self, stops: list[str], tools: ToolCallSplitter | None = None):
        self.stop, self.tools = StopMatcher(stops), tools

    @property
    def calls(self) -> int:
        return self.tools.calls if self.tools else 0

    def feed(self, text: str) -> tuple[list[str | ToolCall], bool]:
        """Returns (items, stop_hit)."""
        return self._content(self.tools.feed(text) if self.tools else [text])

    def finish(self) -> list[str | ToolCall]:
        out, hit = self._content(self.tools.finish() if self.tools else [])
        return out if hit else out + [t for t in [self.stop.flush()] if t]

    def _content(self, items) -> tuple[list[str | ToolCall], bool]:
        out = []
        for item in items:
            if isinstance(item, ToolCall):
                if held := self.stop.flush():  # a call interrupts the content; release it first
                    out.append(held)
                out.append(item)
                continue
            text, hit = self.stop.feed(item)
            if text:
                out.append(text)
            if hit:
                return out, True
        return out, False


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _CacheEntry:
    tokens: list[int]  # exactly the tokens whose KV state `cache` holds
    cache: Any


class PromptCacheStore:
    """Keeps the KV caches of the last `slots` conversations of one model.

    `take(tokens)` returns the cache with the longest usable common prefix and how many
    tokens of `tokens` it already covers. If the request extends an entry completely
    (the normal multi-turn case) the entry is handed over without copying. A partial
    match on a trimmable cache is served from a trimmed copy, leaving the entry intact
    for its own conversation. Non-trimmable caches (e.g. hybrid/SSM models) can only be
    reused when their tokens are a full prefix of the request.
    """

    def __init__(self, slots: int, can_trim: Callable | None = None, trim: Callable | None = None,
                 clone: Callable = copy.deepcopy):
        if can_trim is None or trim is None:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
            can_trim, trim = can_trim_prompt_cache, trim_prompt_cache
        self.slots, self._can_trim, self._trim, self._clone = slots, can_trim, trim, clone
        self.entries: list[_CacheEntry] = []  # oldest first

    def take(self, tokens: list[int]) -> tuple[Any, int]:
        limit = len(tokens) - 1  # at least one token must be left to process
        best, best_len = None, 0
        for e in self.entries:
            cp = min(common_prefix(e.tokens, tokens), limit)
            usable = cp if cp == len(e.tokens) or self._can_trim(e.cache) else 0
            if usable > best_len:
                best, best_len = e, usable
        if best is None:
            return None, 0
        self.entries.remove(best)
        if best_len == len(best.tokens):
            return best.cache, best_len
        self.entries.append(best)  # keep the original, mark it recently used
        cache = self._clone(best.cache)
        self._trim(cache, len(best.tokens) - best_len)
        return cache, best_len

    def put(self, tokens: list[int], cache: Any) -> None:
        if self._can_trim(cache):  # entries that are a prefix of this one are now redundant
            self.entries = [e for e in self.entries if common_prefix(e.tokens, tokens) < len(e.tokens)]
        self.entries.append(_CacheEntry(list(tokens), cache))
        del self.entries[: max(0, len(self.entries) - self.slots)]


@dataclass
class LoadedModel:
    repo: str
    model: Any
    tokenizer: Any
    size: int
    caches: PromptCacheStore = field(repr=False)


class ModelManager:
    """Loads models on first use and evicts least-recently-used ones to stay in budget.

    Size is estimated from the weight files on disk. `fetch` makes the files local
    (downloading if needed) and returns their directory; `load` returns (model, tokenizer).
    """

    def __init__(self, aliases: dict[str, str], budget: int, fetch: Callable[[str], Path],
                 load: Callable[[Path], tuple[Any, Any]], make_caches: Callable[[], PromptCacheStore],
                 on_evict: Callable[[], None] = lambda: None):
        self.aliases, self.budget = aliases, budget
        self._fetch, self._load, self._make_caches, self._on_evict = fetch, load, make_caches, on_evict
        self.loaded: OrderedDict[str, LoadedModel] = OrderedDict()  # repo -> model, LRU first

    @staticmethod
    def weights_size(path: Path) -> int:
        return sum(p.stat().st_size for p in Path(path).glob("*.safetensors"))

    def resident_bytes(self) -> int:
        return sum(m.size for m in self.loaded.values())

    def unload(self, alias: str) -> bool:
        """Free a resident model. Returns False if it was not loaded."""
        old = self.loaded.pop(self.aliases[alias], None)
        if old is None:
            return False
        log.info("unloaded %s (%.1f GB); resident %.1f/%.1f GB", old.repo, old.size / GB,
                 self.resident_bytes() / GB, self.budget / GB)
        del old
        gc.collect()
        self._on_evict()
        return True

    def get(self, alias: str) -> LoadedModel:
        repo = self.aliases[alias]
        if repo in self.loaded:
            self.loaded.move_to_end(repo)
            return self.loaded[repo]
        path = self._fetch(repo)
        size = self.weights_size(path)
        if size > self.budget:
            raise APIError(503, f"model '{alias}' ({size / GB:.1f} GB) exceeds the memory budget "
                                f"({self.budget / GB:.1f} GB)", type="server_error", code="model_too_large")
        while self.loaded and self.resident_bytes() + size > self.budget:
            old_repo, old = self.loaded.popitem(last=False)
            log.info("evicting %s (%.1f GB) to fit %s (%.1f GB)", old_repo, old.size / GB, repo, size / GB)
            del old
            gc.collect()
            self._on_evict()
        t = time.perf_counter()
        model, tokenizer = self._load(path)
        self.loaded[repo] = LoadedModel(repo, model, tokenizer, size, self._make_caches())
        log.info("loaded %s (%.1f GB) in %.1fs; resident %.1f/%.1f GB", repo, size / GB,
                 time.perf_counter() - t, self.resident_bytes() / GB, self.budget / GB)
        return self.loaded[repo]


def check_alias(aliases: dict[str, str], alias: str) -> None:
    if alias not in aliases:
        raise APIError(404, f"model '{alias}' not found; available: {', '.join(aliases)}",
                       code="model_not_found", param="model")


def render_prompt(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
    """Chat template (with tool definitions, if any) -> token ids."""
    if not tokenizer.has_chat_template:
        raise APIError(400, "this model has no chat template", code="unsupported_model")
    if tools and not tokenizer.has_tool_calling:
        raise APIError(400, "this model does not support tool calling: mlx-lm found no tool-call "
                            "format in its chat template", code="tools_not_supported", param="tools")
    return list(tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True))


def tool_splitter(tokenizer, tools: list[dict] | None) -> ToolCallSplitter | None:
    if not tools:
        return None
    return ToolCallSplitter(tokenizer.tool_call_start, tokenizer.tool_call_end, tokenizer.tool_parser, tools)


class MLXEngine:
    def __init__(self, settings: Settings):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

        def fetch(repo: str) -> Path:
            if Path(repo).expanduser().is_dir():
                return Path(repo).expanduser()
            self.fetching = repo  # visible to the admin API: a request is downloading it
            try:
                return Path(snapshot_download(repo, allow_patterns=DEFAULT_ALLOW_PATTERNS))
            finally:
                self.fetching = None

        self.settings = settings
        self.fetching: str | None = None
        self.worker = Worker()
        self.models = ModelManager(
            settings.aliases, settings.memory_budget_bytes, fetch=fetch,
            load=lambda p: load(str(p)),
            make_caches=lambda: PromptCacheStore(settings.prompt_cache_slots),
            on_evict=mx.clear_cache,
        )

    def aliases(self) -> dict[str, str]:
        return dict(self.settings.aliases)

    def check_model(self, alias: str) -> None:
        check_alias(self.settings.aliases, alias)

    def stream(self, req: ChatRequest, cancel: threading.Event) -> AsyncIterator[str | ToolCall | Final]:
        return self.worker.run(lambda emit, c: self._generate(req, emit, c), cancel)

    # --- admin (all MLX work goes through the worker thread) ---

    def resident(self) -> dict[str, int]:
        """repo -> weight bytes of the resident models."""
        return {repo: m.size for repo, m in list(self.models.loaded.items())}

    async def load(self, alias: str) -> None:
        await self.worker.call(lambda: self.models.get(alias))

    async def unload(self, alias: str) -> bool:
        return await self.worker.call(lambda: self.models.unload(alias))

    def _generate(self, req: ChatRequest, emit: Callable, cancel: threading.Event) -> None:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        t_start = time.perf_counter()
        lm = self.models.get(req.model)
        t_loaded = time.perf_counter()

        tokens = render_prompt(lm.tokenizer, req.messages, req.tools)
        output = OutputFilter(req.stop, tool_splitter(lm.tokenizer, req.tools))
        cache, reused = lm.caches.take(tokens)
        if cache is None:
            cache = make_prompt_cache(lm.model)
        if req.seed is not None:
            mx.random.seed(req.seed)
        temp = 0.0 if req.top_p == 0 else req.temperature  # top_p=0 means "most likely only"
        sampler = make_sampler(temp=temp, top_p=req.top_p)
        mx.reset_peak_memory()

        def progress(_done, _total):
            if cancel.is_set():
                raise Cancelled()

        generated: list[int] = []
        finish, t_first = "length", None
        gen = stream_generate(lm.model, lm.tokenizer, tokens[reused:],
                              max_tokens=req.max_tokens or self.settings.default_max_tokens,
                              sampler=sampler, prompt_cache=cache, prompt_progress_callback=progress)
        try:
            for r in gen:
                if t_first is None:
                    t_first = time.perf_counter()
                generated.append(r.token)
                items, hit = output.feed(r.text)
                for item in items:
                    emit(item)
                if hit:
                    finish = "stop"
                    break
                if r.finish_reason:
                    finish = r.finish_reason
                    break
                if cancel.is_set():
                    finish = "cancelled"
                    break
            for item in output.finish():  # held text; a call left open; nothing after a stop hit
                emit(item)
            if finish == "stop" and output.calls:
                finish = "tool_calls"
        except Cancelled:
            log.info("cancelled during prompt processing (%s)", lm.repo)
            emit(Final("cancelled", len(tokens), 0, reused, {}))
            return  # cache state is partial; drop it
        finally:
            gen.close()
        t_end = time.perf_counter()

        # Every token stream_generate yielded has been fed through the model, so the
        # cache now holds exactly tokens + generated. Check before keeping it.
        full = tokens + generated
        offset = next((c.offset for c in cache if isinstance(getattr(c, "offset", None), int)), None)
        if offset is None or offset == len(full):
            lm.caches.put(full, cache)
        else:
            log.warning("prompt cache offset %d != %d tokens; not keeping it", offset, len(full))

        eos = bool(generated) and generated[-1] in lm.tokenizer.eos_token_ids
        completion_tokens = len(generated) - int(eos)
        t_first = t_first or t_end
        prefill_s = t_first - t_loaded
        decode_s = t_end - t_first
        timings = {
            "model": lm.repo,
            "profile": self.settings.profile,
            "queue_s": round(t_start - req.received_at, 4),
            "load_s": round(t_loaded - t_start, 4),
            "ttft_s": round(t_first - req.received_at, 4),
            "prefill_s": round(prefill_s, 4),
            "prefill_tokens": len(tokens) - reused,
            "prompt_tps": round((len(tokens) - reused) / prefill_s, 1) if prefill_s > 0 else None,
            "decode_tps": round((len(generated) - 1) / decode_s, 1) if decode_s > 0 and len(generated) > 1 else None,
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        }
        log.info("%s: prompt %d tok (cached %d), completion %d tok, finish=%s, ttft %.3fs, "
                 "prefill %.3fs, decode %s tok/s, peak %.2f GB", lm.repo, len(tokens), reused,
                 completion_tokens, finish, timings["ttft_s"], prefill_s, timings["decode_tps"],
                 timings["peak_memory_gb"])
        emit(Final(finish, len(tokens), completion_tokens, reused, timings))
