"""Talking to a gateway (admin API client) and managing a local one (pid file lifecycle).

Stdlib only; used by mlxgw-tui. A local gateway started here runs detached in its
own session, logs to a file, and is recorded in a pid file under the user state dir.
`stop` only ever signals a process that the pid file names AND whose command line is
an mlxgw server on that port.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import LOOPBACK_HOSTS

SERVER_MODULE = "mlxgw.server"


# --- admin API client ----------------------------------------------------------------

class GatewayError(Exception):
    """The gateway answered with an error (status > 0) or did not answer (status 0)."""

    def __init__(self, status: int, message: str, code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code

    @property
    def unreachable(self) -> bool:
        return self.status == 0


class GatewayClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 3.0):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.base_url).hostname or ""

    @property
    def port(self) -> int:
        u = urllib.parse.urlsplit(self.base_url)
        return u.port or (443 if u.scheme == "https" else 80)

    @property
    def is_local(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def _request(self, method: str, path: str, timeout: float | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, method=method, headers=headers,
                                     data=b"" if method == "POST" else None)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())["error"]
                raise GatewayError(e.code, err.get("message") or str(e), err.get("code")) from None
            except (ValueError, KeyError, TypeError):
                raise GatewayError(e.code, f"HTTP {e.code} {e.reason}") from None
        except (OSError, ValueError) as e:  # refused, timeout, DNS, bad JSON
            reason = getattr(e, "reason", None) or e
            raise GatewayError(0, f"unreachable: {reason}") from None

    def status(self) -> dict:
        return self._request("GET", "/admin/status")

    def logs(self, after: int = 0, limit: int = 200) -> dict:
        return self._request("GET", f"/admin/logs?after={after}&limit={limit}")

    def load(self, alias: str) -> dict:  # loading a large model can take minutes
        return self._request("POST", f"/admin/models/{urllib.parse.quote(alias)}/load", timeout=900)

    def unload(self, alias: str) -> dict:
        return self._request("POST", f"/admin/models/{urllib.parse.quote(alias)}/unload", timeout=300)

    def download(self, alias: str) -> dict:
        return self._request("POST", f"/admin/models/{urllib.parse.quote(alias)}/download")

    def cancel_download(self, alias: str) -> dict:
        return self._request("POST", f"/admin/models/{urllib.parse.quote(alias)}/download/cancel")


# --- local lifecycle -----------------------------------------------------------------

class LifecycleError(Exception):
    pass


@dataclass
class PidInfo:
    pid: int
    host: str
    port: int
    config: str
    log: str
    started: float


def state_dir() -> Path:
    if d := os.environ.get("MLXGW_STATE_DIR"):
        return Path(d)
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "mlxgw"


def pid_path(port: int) -> Path:
    return state_dir() / f"gateway-{port}.pid"


def log_path(port: int) -> Path:
    return state_dir() / f"gateway-{port}.log"


def server_command(config: str, host: str, port: int) -> list[str]:
    return [sys.executable, "-m", SERVER_MODULE, "--config", config, "--host", host, "--port", str(port)]


_children: dict[int, subprocess.Popen] = {}  # processes this TUI started; polled to reap zombies


def _command_line(pid: int) -> str | None:
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _alive(pid: int) -> bool:
    if (child := _children.get(pid)) is not None and child.poll() is not None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_our_gateway(info: PidInfo) -> bool:
    """The pid is alive and is an mlxgw server for this port (guards against pid reuse)."""
    if not _alive(info.pid):
        return False
    cmd = _command_line(info.pid) or ""
    return SERVER_MODULE in cmd and f"--port {info.port}" in cmd


def read_pid(port: int) -> PidInfo | None:
    """PidInfo of a live gateway this tool started on `port`; removes a stale pid file."""
    path = pid_path(port)
    try:
        info = PidInfo(**json.loads(path.read_text()))
    except FileNotFoundError:
        return None
    except (ValueError, TypeError):
        path.unlink(missing_ok=True)  # unreadable: it cannot identify anything
        return None
    if is_our_gateway(info):
        return info
    if not _alive(info.pid) or _command_line(info.pid) is not None:  # dead, or pid reused
        path.unlink(missing_ok=True)
    return None


def start(config: str, port: int, host: str = "127.0.0.1", command: list[str] | None = None) -> PidInfo:
    """Start a detached gateway (survives the TUI) and record it in the pid file."""
    if host not in LOOPBACK_HOSTS:
        raise LifecycleError(f"refusing to start a gateway on non-loopback host {host}")
    if (info := read_pid(port)) is not None:
        raise LifecycleError(f"a gateway started earlier is already running (pid {info.pid})")
    config = str(Path(config).expanduser().resolve())
    if not Path(config).is_file():
        raise LifecycleError(f"config file not found: {config}")
    state_dir().mkdir(parents=True, exist_ok=True)
    log = log_path(port)
    with open(log, "a") as f:
        f.write(f"\n--- started by mlxgw-tui at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.flush()
        proc = subprocess.Popen(command or server_command(config, host, port), stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=str(Path(config).parent))
    _children[proc.pid] = proc
    info = PidInfo(proc.pid, host, port, config, str(log), time.time())
    tmp = pid_path(port).with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(info)))
    tmp.replace(pid_path(port))
    return info


def stop(port: int, timeout: float = 20.0) -> PidInfo:
    """SIGTERM the gateway named by our pid file (SIGKILL after `timeout`)."""
    info = read_pid(port)
    if info is None:
        raise LifecycleError(f"no gateway started by mlxgw-tui is running on port {port} "
                             f"(no live pid file at {pid_path(port)}); not stopping anything")
    os.kill(info.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _alive(info.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _alive(info.pid):
        os.kill(info.pid, signal.SIGKILL)
        while _alive(info.pid):
            time.sleep(0.05)
    if (child := _children.pop(info.pid, None)) is not None:
        child.wait()
    pid_path(port).unlink(missing_ok=True)
    return info


def tail(path: str | Path, lines: int = 30) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            return f.read().decode(errors="replace").splitlines()[-lines:]
    except OSError:
        return []
