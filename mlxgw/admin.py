"""Admin API for mlxgw-tui: status, load/unload, background downloads, log tail.

Routes live under /admin and use the same bearer-token auth as /v1. Nothing here runs
MLX: load and unload are handed to the engine, which runs them on its worker thread.
Downloads run in a subprocess watched by a background thread, so they never block
generation and can be cancelled by terminating the subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .api import APIError
from .config import Settings
from .engine import Final, ModelManager

log = logging.getLogger("mlxgw")

MAX_REQUESTS = 50
MAX_LOG_LINES = 500
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


# --- recent log lines --------------------------------------------------------------

class LogRing(logging.Handler):
    """Keeps the last `maxlen` formatted log lines, each with a sequence number."""

    def __init__(self, maxlen: int = MAX_LOG_LINES):
        super().__init__(logging.INFO)
        self.setFormatter(logging.Formatter(LOG_FORMAT))
        self.lines: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self.seq = 0
        self._ring_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 - logging must never raise
            return
        with self._ring_lock:
            self.seq += 1
            self.lines.append((self.seq, line))

    def since(self, after: int, limit: int) -> list[dict]:
        with self._ring_lock:
            out = [{"seq": s, "line": line} for s, line in self.lines if s > after]
        return out[-limit:]


class QuietPolling(logging.Filter):
    """Drops uvicorn access-log lines for the TUI's once-a-second polling."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        return not (len(args) >= 3 and isinstance(args[2], str)
                    and args[2].startswith(("/admin/status", "/admin/logs")))


# --- downloads -------------------------------------------------------------------

class Download:
    def __init__(self, repo: str, total: int | None):
        self.repo, self.total = repo, total
        self.state = "downloading"  # -> done | failed | cancelled
        self.done_bytes = 0
        self.error: str | None = None
        self.started = time.time()
        self.cancel = threading.Event()

    def as_dict(self) -> dict:
        return {"state": self.state, "done_bytes": self.done_bytes, "total_bytes": self.total,
                "error": self.error, "started": self.started}


# download(repo, cancel_event, report(done_bytes)) -> None; raises on failure.
# It must return promptly once cancel_event is set.
DownloadFn = Callable[[str, threading.Event, Callable[[int], None]], None]


class Downloads:
    """One background download per repo at most."""

    def __init__(self, download: DownloadFn):
        self._download = download
        self._lock = threading.Lock()
        self.items: dict[str, Download] = {}

    def active(self, repo: str) -> Download | None:
        d = self.items.get(repo)
        return d if d is not None and d.state == "downloading" else None

    def start(self, repo: str, total: int | None) -> Download:
        with self._lock:
            if self.active(repo):
                raise APIError(409, f"{repo} is already downloading", code="already_downloading")
            d = self.items[repo] = Download(repo, total)
        threading.Thread(target=self._run, args=(d,), name=f"download-{repo}", daemon=True).start()
        return d

    def _run(self, d: Download) -> None:
        log.info("download started: %s", d.repo)

        def report(done: int) -> None:
            d.done_bytes = done

        try:
            self._download(d.repo, d.cancel, report)
        except Exception as e:  # noqa: BLE001 - reported through status
            d.state, d.error = ("cancelled", None) if d.cancel.is_set() else ("failed", str(e) or type(e).__name__)
        else:
            d.state = "cancelled" if d.cancel.is_set() else "done"
        log.info("download %s: %s%s", d.state, d.repo, f" ({d.error})" if d.error else "")

    def cancel(self, repo: str) -> Download:
        d = self.active(repo)
        if d is None:
            raise APIError(409, f"{repo} is not downloading", code="not_downloading")
        d.cancel.set()
        return d

    def cancel_all(self) -> None:
        for d in list(self.items.values()):
            d.cancel.set()


def hf_repo_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    return Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def subprocess_download(repo: str, cancel: threading.Event, report: Callable[[int], None],
                        poll_s: float = 0.5) -> None:
    """Download with huggingface_hub in a child process (cancellable by terminating it).

    Progress is the number of bytes in the repo's cache directory, which counts partial
    (.incomplete) files; the same files mlx-lm's loader would fetch are requested.
    """
    from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS
    code = ("import sys, json; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1], allow_patterns=json.loads(sys.argv[2]))")
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    proc = subprocess.Popen([sys.executable, "-c", code, repo, json.dumps(DEFAULT_ALLOW_PATTERNS)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            env=env, text=True)
    # Drain stderr in the background so a chatty child cannot block on a full pipe.
    err_tail: deque[str] = deque(maxlen=20)
    reader = threading.Thread(target=lambda: err_tail.extend(proc.stderr), daemon=True)
    reader.start()
    repo_dir = hf_repo_dir(repo)
    base = _dir_bytes(repo_dir) if repo_dir.exists() else 0
    try:
        while proc.poll() is None:
            if cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return
            if repo_dir.exists():
                report(max(0, _dir_bytes(repo_dir) - base))
            cancel.wait(poll_s)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    reader.join(timeout=2)
    if proc.returncode != 0:
        last = next((line.strip() for line in reversed(err_tail) if line.strip()), "")
        raise RuntimeError(f"download exited with {proc.returncode}: {last}"[:500])
    if repo_dir.exists():
        report(max(0, _dir_bytes(repo_dir) - base))


# --- admin state -------------------------------------------------------------------

def _default_local(repo: str) -> Path | None:
    from .bench import local_snapshot
    return local_snapshot(repo)


def _default_remote_size(repo: str) -> int | None:
    from .bench import remote_size
    return remote_size(repo)


class Admin:
    """Gateway-side state the TUI reads. `local(repo)` -> snapshot dir or None (network-free),
    `remote_size(repo)` -> bytes or None (network; called from a background thread only)."""

    def __init__(self, settings: Settings, engine, download: DownloadFn = subprocess_download,
                 local: Callable[[str], Path | None] = _default_local,
                 remote_size: Callable[[str], int | None] = _default_remote_size):
        self.settings, self.engine = settings, engine
        self.downloads = Downloads(download)
        self.local, self._remote_size = local, remote_size
        self.started = time.time()
        self.requests: deque[dict] = deque(maxlen=MAX_REQUESTS)
        self.last_used: dict[str, float] = {}
        self.last_decode_tps: dict[str, float] = {}
        self.remote_sizes: dict[str, int | None] = {}
        self.logs = LogRing()

    # Called from the chat handler.
    def record(self, alias: str, final: Final | None, error: str | None = None) -> None:
        now = time.time()
        t = final.timings if final else {}
        self.requests.append({
            "time": now, "alias": alias,
            "prompt_tokens": final.prompt_tokens if final else None,
            "cached_tokens": final.cached_tokens if final else None,
            "completion_tokens": final.completion_tokens if final else None,
            "ttft_s": t.get("ttft_s"), "decode_tps": t.get("decode_tps"),
            "finish_reason": final.finish_reason if final else "error", "error": error,
        })
        self.last_used[alias] = now
        if t.get("decode_tps"):
            self.last_decode_tps[alias] = t["decode_tps"]

    def check_not_downloading(self, alias: str) -> None:
        repo = self.settings.aliases.get(alias)
        if repo and (d := self.downloads.active(repo)):
            pct = f" ({100 * d.done_bytes / d.total:.0f}%)" if d.total else ""
            raise APIError(409, f"model '{alias}' is being downloaded{pct}; retry when it finishes",
                           type="server_error", code="model_downloading", param="model")

    def fetch_remote_sizes(self) -> None:
        """Weight sizes of aliases not on disk, for the download confirmation. Background only."""
        for repo in set(self.settings.aliases.values()):
            if repo not in self.remote_sizes and self.local(repo) is None:
                self.remote_sizes[repo] = self._remote_size(repo)

    def model_state(self, alias: str, repo: str, resident: dict[str, int]) -> dict:
        download = self.downloads.items.get(repo)
        path = None
        if repo in resident:
            state = "loaded"
        elif self.downloads.active(repo) or getattr(self.engine, "fetching", None) == repo:
            state = "downloading"
        elif (path := self.local(repo)) is not None:
            state = "on disk"
        else:
            state = "not downloaded"
        if repo in resident:
            size = resident[repo]
        elif path is not None:
            size = ModelManager.weights_size(path)
        else:
            size = self.remote_sizes.get(repo)
        return {"alias": alias, "repo": repo, "state": state, "size_bytes": size,
                "last_used": self.last_used.get(alias), "last_decode_tps": self.last_decode_tps.get(alias),
                "download": download.as_dict() if download else None}

    def status(self) -> dict:
        s = self.settings
        resident = self.engine.resident()
        busy = self.engine.worker.busy()
        return {
            "gateway": {"pid": os.getpid(), "started": self.started,
                        "uptime_s": round(time.time() - self.started, 1), "host": s.host, "port": s.port},
            "machine": {"chip": s.hardware.chip, "memory_bytes": s.hardware.memory_bytes,
                        "profile": s.profile, "budget_bytes": s.memory_budget_bytes,
                        "resident_bytes": sum(resident.values())},
            "queue": busy,
            "models": [self.model_state(a, r, resident) for a, r in s.aliases.items()],
            "requests": list(self.requests),
        }


def admin_routes(admin: Admin, authorize: Callable[[Request], None]) -> list[Route]:
    def alias_of(request: Request) -> tuple[str, str]:
        authorize(request)
        alias = request.path_params["alias"]
        if alias not in admin.settings.aliases:
            raise APIError(404, f"model '{alias}' not found; available: {', '.join(admin.settings.aliases)}",
                           code="model_not_found")
        return alias, admin.settings.aliases[alias]

    async def status(request: Request):
        authorize(request)
        # Checking which models are on disk walks the HF cache; keep it off the event loop.
        return JSONResponse(await run_in_threadpool(admin.status))

    async def load(request: Request):
        alias, repo = alias_of(request)
        admin.check_not_downloading(alias)
        if repo not in admin.engine.resident() and await run_in_threadpool(admin.local, repo) is None:
            raise APIError(409, f"model '{alias}' is not downloaded; download it first", code="not_downloaded")
        t = time.perf_counter()
        await admin.engine.load(alias)
        return JSONResponse({"alias": alias, "state": "loaded", "load_s": round(time.perf_counter() - t, 3)})

    async def unload(request: Request):
        alias, _ = alias_of(request)
        was_loaded = await admin.engine.unload(alias)
        return JSONResponse({"alias": alias, "unloaded": was_loaded})

    async def download(request: Request):
        alias, repo = alias_of(request)
        if Path(repo).expanduser().is_dir():
            raise APIError(409, f"model '{alias}' is a local path; nothing to download", code="already_downloaded")
        if getattr(admin.engine, "fetching", None) == repo:
            raise APIError(409, f"model '{alias}' is already being downloaded by a request",
                           code="already_downloading")
        if admin.downloads.active(repo) is None and await run_in_threadpool(admin.local, repo) is not None:
            raise APIError(409, f"model '{alias}' is already downloaded", code="already_downloaded")
        d = admin.downloads.start(repo, admin.remote_sizes.get(repo))
        return JSONResponse({"alias": alias, "download": d.as_dict()}, status_code=202)

    async def cancel_download(request: Request):
        alias, repo = alias_of(request)
        d = admin.downloads.cancel(repo)
        return JSONResponse({"alias": alias, "download": d.as_dict()})

    async def logs(request: Request):
        authorize(request)
        try:
            after = int(request.query_params.get("after", 0))
            limit = min(int(request.query_params.get("limit", 200)), MAX_LOG_LINES)
        except ValueError:
            raise APIError(400, "'after' and 'limit' must be integers")
        return JSONResponse({"lines": admin.logs.since(after, limit), "last_seq": admin.logs.seq})

    return [
        Route("/admin/status", status, methods=["GET"]),
        Route("/admin/logs", logs, methods=["GET"]),
        Route("/admin/models/{alias}/load", load, methods=["POST"]),
        Route("/admin/models/{alias}/unload", unload, methods=["POST"]),
        Route("/admin/models/{alias}/download", download, methods=["POST"]),
        Route("/admin/models/{alias}/download/cancel", cancel_download, methods=["POST"]),
    ]
