import socket
import threading
import time

import pytest
import uvicorn

from mlxgw.admin import Admin
from mlxgw.config import GB, Hardware, Settings
from mlxgw.api import APIError
from mlxgw.engine import Final, OutputFilter, ToolCallSplitter, Worker, check_alias


def json_tool_splitter(tools):
    """The splitter MLXEngine builds for a Qwen-style template: mlx-lm's json_tools parser."""
    from mlx_lm.tool_parsers import json_tools
    return ToolCallSplitter(json_tools.tool_call_start, json_tools.tool_call_end, json_tools.parse_tool_call, tools)
from mlxgw.server import create_app


class FakeEngine:
    """Same interface as MLXEngine; emits fixed text pieces on the real Worker thread.

    The pieces go through the same OutputFilter as real generation, with Qwen-style
    <tool_call> markup parsed by mlx-lm's json_tools parser when the request has tools.
    """

    def __init__(self, pieces=("Hello", ",", " world", "!"), delay=0.0, tool_calling=True):
        self.pieces, self.delay, self.tool_calling = list(pieces), delay, tool_calling
        self.worker = Worker()
        self.requests = []
        self.cancelled = threading.Event()  # set when a job observed cancellation
        self.finished = threading.Event()
        self.loaded: dict[str, int] = {}  # repo -> bytes
        self.worker_threads: set[str] = set()  # threads that ran load/unload

    def aliases(self):
        return {"tiny": "fake/tiny-4bit", "coder": "fake/coder-4bit"}

    # admin interface
    def resident(self):
        return dict(self.loaded)

    async def load(self, alias):
        def job():
            self.worker_threads.add(threading.current_thread().name)
            self.loaded[self.aliases()[alias]] = GB // 2
        await self.worker.call(job)

    async def unload(self, alias):
        def job():
            self.worker_threads.add(threading.current_thread().name)
            return self.loaded.pop(self.aliases()[alias], None) is not None
        return await self.worker.call(job)

    def check_model(self, alias):
        check_alias(self.aliases(), alias)

    def stream(self, req, cancel):
        def job(emit, cancel):
            self.requests.append(req)
            if req.tools and not self.tool_calling:
                raise APIError(400, "this model does not support tool calling", code="tools_not_supported",
                               param="tools")
            output = OutputFilter(req.stop, json_tool_splitter(req.tools) if req.tools else None)
            n, finish = 0, "stop"
            for i, piece in enumerate(self.pieces):
                if req.max_tokens is not None and i >= req.max_tokens:
                    finish = "length"
                    break
                if cancel.is_set():
                    self.cancelled.set()
                    finish = "cancelled"
                    break
                time.sleep(self.delay)
                n += 1
                items, hit = output.feed(piece)
                for item in items:
                    emit(item)
                if hit:
                    break
            for item in output.finish():
                emit(item)
            if finish == "stop" and output.calls:
                finish = "tool_calls"
            self.finished.set()
            emit(Final(finish, 7, n, 3, {"ttft_s": 0.01, "decode_tps": 100.0}))

        return self.worker.run(job, cancel)


def make_settings(host="127.0.0.1", api_key=None):
    return Settings(host=host, port=0, api_key=api_key, profile="48gb",
                    aliases={"tiny": "fake/tiny-4bit"}, memory_budget_bytes=30 * GB,
                    prompt_cache_slots=4, default_max_tokens=64,
                    hardware=Hardware("Apple M-series", 48 * GB))


@pytest.fixture
def live_server():
    """Start a real uvicorn server around a FakeEngine; yields (base_url, engine)."""
    servers = []

    def start(engine, settings=None, admin=None, port=None):
        if port is None:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
        settings = settings or make_settings()
        # No network in tests: the default Admin would look up remote sizes on Hugging Face.
        admin = admin or Admin(settings, engine, local=lambda repo: None, remote_size=lambda repo: None)
        app = create_app(engine, settings, admin)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        deadline = time.time() + 5
        while not server.started and time.time() < deadline:
            time.sleep(0.01)
        servers.append((server, t))
        start.servers = servers
        return f"http://127.0.0.1:{port}"

    yield start
    for server, t in servers:
        server.should_exit = True
        t.join(timeout=5)
