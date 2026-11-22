import json
import os
import socket
import subprocess
import sys

import pytest

from mlxgw import control
from mlxgw.control import GatewayClient, GatewayError, LifecycleError


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXGW_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def fake_command(port):
    # Looks like a gateway for this port to is_our_gateway; does nothing but sleep.
    return [sys.executable, "-c", "import time; time.sleep(60)", "mlxgw.server", "--port", str(port)]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def config_file(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[profiles.x]\nmin_memory_gb = 0\n")
    return str(p)


def test_start_writes_pid_file_and_stop_removes_it(tmp_path):
    port = free_port()
    info = control.start(config_file(tmp_path), port, command=fake_command(port))
    try:
        data = json.loads(control.pid_path(port).read_text())
        assert data["pid"] == info.pid and data["port"] == port and data["log"] == str(control.log_path(port))
        assert control.read_pid(port) == info
        # The child runs in its own session, so it outlives the TUI and its terminal.
        assert os.getsid(info.pid) == info.pid
        with pytest.raises(LifecycleError, match="already running"):
            control.start(config_file(tmp_path), port, command=fake_command(port))
    finally:
        stopped = control.stop(port)
    assert stopped.pid == info.pid
    assert not alive(info.pid)
    assert not control.pid_path(port).exists()


def test_stop_without_pid_file_refuses():
    with pytest.raises(LifecycleError, match="not stopping anything"):
        control.stop(free_port())


def test_pid_file_naming_an_unrelated_process_is_never_signalled(state_dir):
    port = free_port()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state_dir.mkdir(parents=True)
        control.pid_path(port).write_text(json.dumps(
            {"pid": other.pid, "host": "127.0.0.1", "port": port, "config": "c", "log": "l", "started": 0}))
        assert control.read_pid(port) is None
        with pytest.raises(LifecycleError):
            control.stop(port)
        assert other.poll() is None  # still running
        assert not control.pid_path(port).exists()  # recognised as a reused pid
    finally:
        other.kill()
        other.wait()


def test_stale_pid_file_is_cleaned_up(state_dir):
    port = free_port()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    state_dir.mkdir(parents=True)
    control.pid_path(port).write_text(json.dumps(
        {"pid": dead.pid, "host": "127.0.0.1", "port": port, "config": "c", "log": "l", "started": 0}))
    assert control.read_pid(port) is None
    assert not control.pid_path(port).exists()


def test_start_refuses_non_loopback_and_missing_config(tmp_path):
    with pytest.raises(LifecycleError, match="non-loopback"):
        control.start(config_file(tmp_path), 1234, host="0.0.0.0")
    with pytest.raises(LifecycleError, match="config file not found"):
        control.start(str(tmp_path / "missing.toml"), 1234)


def test_client_reports_unreachable_and_http_errors(live_server):
    from .conftest import FakeEngine, make_settings
    c = GatewayClient(f"http://127.0.0.1:{free_port()}", timeout=1)
    with pytest.raises(GatewayError) as e:
        c.status()
    assert e.value.unreachable
    url = live_server(FakeEngine(), make_settings(api_key="k"))
    with pytest.raises(GatewayError) as e:
        GatewayClient(url).status()
    assert e.value.status == 401 and e.value.code == "invalid_api_key"
    assert GatewayClient(url, "k").status()["machine"]["profile"] == "48gb"


def test_client_locality():
    assert GatewayClient("http://127.0.0.1:8080").is_local
    assert GatewayClient("http://localhost:9000/").port == 9000
    assert not GatewayClient("http://gateway.example:8080").is_local


def test_tail(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("".join(f"line {i}\n" for i in range(100)))
    assert control.tail(p, 3) == ["line 97", "line 98", "line 99"]
    assert control.tail(tmp_path / "missing", 3) == []

