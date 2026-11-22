"""TUI tests: Textual's headless harness against an in-process gateway (fake engine)."""

import asyncio
import dataclasses
import socket
import sys
import time
from pathlib import Path

import httpx
import pytest

from mlxgw import control
from mlxgw.admin import Admin
from mlxgw.config import GB
from mlxgw.control import GatewayClient
from mlxgw.tui import ConfirmScreen, GatewayTUI, TextScreen, screen_text

from .conftest import FakeEngine, make_settings
from .test_admin import ALIASES, FakeDownloader

MSG = [{"role": "user", "content": "hi"}]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXGW_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


@pytest.fixture
def gw(live_server):
    """In-process gateway: tiny on disk, coder (8 GB) not downloaded, fake downloader."""
    on_disk = {"fake/tiny-4bit"}
    settings = dataclasses.replace(make_settings(), aliases=dict(ALIASES))
    engine = FakeEngine()
    dl = FakeDownloader()
    admin = Admin(settings, engine, download=dl,
                  local=lambda repo: Path("/nonexistent") if repo in on_disk else None,
                  remote_size=lambda repo: 8 * GB)
    admin.remote_sizes["fake/coder-4bit"] = 8 * GB
    url = live_server(engine, settings, admin)
    return url, engine, dl, admin


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


async def settle(app, pilot):
    """Let pending action workers finish, then poll once."""
    await pilot.pause()
    await app.workers.wait_for_complete()
    await app.refresh_now()
    await pilot.pause()


def text(app):
    return screen_text(app)


def model_states(app):
    return {m["alias"]: m["state"] for m in app.status["models"]}


def make_app(url, **kw):
    return GatewayTUI(GatewayClient(url, kw.pop("api_key", None), timeout=1), poll_interval=60, **kw)


def test_renders_status_in_80x24(gw):
    url, *_ = gw

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            screen = text(app)
            lines = screen.splitlines()
            assert len(lines) == 24 and all(len(line) <= 80 for line in lines)
            assert "running" in lines[0] and url in lines[0] and "(local)" in lines[0]
            assert "Apple M-series 48 GB" in lines[1] and "48gb" in lines[1] and "0.0/30.0 GB" in lines[1]
            assert "fake/tiny-4bit" in screen and "on disk" in screen
            assert "fake/coder-4bit" in screen and "not downloaded" in screen and "8.0 GB" in screen
            assert "in flight 0 · queued 0" in screen
            for key in ("Start", "Stop", "Download", "Load", "Unload", "Bench", "Quit"):
                assert key in lines[-1]
    run(go())


def test_reflects_requests_and_state_changes_from_outside(gw):
    url, engine, *_ = gw

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            await asyncio.to_thread(httpx.post, url + "/v1/chat/completions",
                                    json={"model": "tiny", "messages": MSG})
            await asyncio.to_thread(httpx.post, url + "/admin/models/tiny/load")
            await settle(app, pilot)
            screen = text(app)
            assert model_states(app)["tiny"] == "loaded"
            assert "0.5/30.0 GB" in screen  # memory bar follows resident bytes
            row = next(line for line in screen.splitlines() if " tiny " in line and ":" in line)
            assert "7" in row and "3" in row and "100" in row and "stop" in row  # prompt/cached/tok/s/finish
            assert "100" in next(line for line in screen.splitlines() if "fake/tiny-4bit" in line)
    run(go())


def test_unload_needs_confirmation(gw):
    url, engine, *_ = gw

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            await pilot.press("l")  # tiny is the first row; fits the budget, so no confirmation
            await settle(app, pilot)
            assert engine.loaded == {"fake/tiny-4bit": GB // 2}

            await pilot.press("u")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            assert "Unload tiny" in text(app)
            await settle(app, pilot)
            assert engine.loaded  # nothing happens while the question is open
            await pilot.press("n")
            await settle(app, pilot)
            assert not isinstance(app.screen, ConfirmScreen)
            assert engine.loaded and "cancelled" in text(app)

            await pilot.press("u", "y")
            await settle(app, pilot)
            assert engine.loaded == {}
            assert model_states(app)["tiny"] == "on disk"
    run(go())


def test_download_confirm_shows_size_then_cancel(gw):
    url, engine, dl, admin = gw

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            await pilot.press("down", "d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            assert "Download coder" in text(app) and "8.0 GB" in text(app)
            assert dl.calls == []
            await pilot.press("y")
            await settle(app, pilot)
            assert dl.calls == ["fake/coder-4bit"]
            dl.report(2 * GB)
            await settle(app, pilot)
            assert "downloading 25%" in text(app)

            await pilot.press("l")  # load is refused in the UI while downloading
            await settle(app, pilot)
            assert "download it first" in text(app) and engine.loaded == {}

            await pilot.press("c")
            await settle(app, pilot)
            for _ in range(100):
                if admin.downloads.items["fake/coder-4bit"].state == "cancelled":
                    break
                await asyncio.sleep(0.02)
            await app.refresh_now()
            await pilot.pause()
            assert "dl cancelled" in text(app)
    run(go())


def test_benchmark_shows_result(gw):
    url, engine, *_ = gw

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)  # not loaded yet: loading it is the expensive part
            await pilot.press("y")
            await settle(app, pilot)
            assert isinstance(app.screen, TextScreen)
            screen = text(app)
            assert "benchmark tiny" in screen and "decode tok/s     100.0" in screen
            await pilot.press("escape")
            await settle(app, pilot)
            # warm-up + 3 runs x (cold + follow-up)
            assert len(engine.requests) == 7
            assert "Activity" in text(app) and len(app.status["requests"]) == 7
    run(go())


def test_unreachable_gateway_keeps_polling_then_recovers(gw):
    url, *_ = gw

    async def go():
        dead = f"http://127.0.0.1:{free_port()}"
        app = GatewayTUI(GatewayClient(dead, timeout=1), poll_interval=0.1)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            assert app.state == "stopped"  # loopback, nothing running, no pid file
            assert "press s to start" in text(app)
            await pilot.press("l", "u", "d", "b", "x")  # every action degrades to a message
            await settle(app, pilot)
            assert "no gateway started by mlxgw-tui" in text(app)

            app.client.base_url = url  # the gateway comes up; the timer picks it up
            for _ in range(50):
                await pilot.pause(0.05)
                if app.state == "running":
                    break
            assert app.state == "running" and "fake/tiny-4bit" in text(app)
    run(go())


def test_remote_unreachable_and_start_stop_disabled():
    async def go():
        app = GatewayTUI(GatewayClient("http://192.0.2.1:8080", timeout=0.3), poll_interval=60)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            assert app.state == "unreachable" and "retrying" in text(app)
            await pilot.press("s")
            await pilot.pause()
            assert "only available for a gateway on this machine" in text(app)
            await pilot.press("x")
            await pilot.pause()
            assert "only available for a gateway on this machine" in text(app)
    run(go())


def test_auth_error_is_shown(live_server):
    url = live_server(FakeEngine(), make_settings(api_key="k"))

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            assert app.state == "error" and "HTTP 401" in text(app)
        app = make_app(url, api_key="k")
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            assert app.state == "running"
    run(go())


def test_stop_refuses_gateway_not_started_here(gw):
    url, *_ = gw

    async def go():
        app = make_app(url)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            await pilot.press("x")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert "cannot stop" in text(app) and "not started by mlxgw-tui" in text(app)
    run(go())


def test_local_start_stop_lifecycle(tmp_path, monkeypatch):
    """s starts a detached gateway (fake engine) recorded in the pid file; x (confirmed) stops it."""
    port = free_port()
    script = str(Path(__file__).with_name("fake_gateway.py"))
    monkeypatch.setattr(control, "server_command",
                        lambda config, host, p: [sys.executable, script, "mlxgw.server", "--port", str(p)])
    config = tmp_path / "config.toml"
    config.write_text("[profiles.x]\n")

    async def go():
        app = GatewayTUI(GatewayClient(f"http://127.0.0.1:{port}", timeout=1), config=str(config),
                         poll_interval=0.2)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            assert app.state == "stopped"
            await pilot.press("s")
            await settle(app, pilot)
            info = control.read_pid(port)
            assert info is not None
            for _ in range(150):
                await pilot.pause(0.1)
                if app.state == "running":
                    break
            assert app.state == "running"
            assert app.status["gateway"]["pid"] == info.pid
            assert "started here" in text(app)

            await pilot.press("x")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen) and control.read_pid(port) is not None
            await pilot.press("y")
            await settle(app, pilot)
            assert control.read_pid(port) is None
            assert app.state == "stopped"
        return info

    info = run(go())
    time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        import os
        os.kill(info.pid, 0)


def test_quit_leaves_detached_gateway_running(tmp_path, monkeypatch):
    port = free_port()
    script = str(Path(__file__).with_name("fake_gateway.py"))
    monkeypatch.setattr(control, "server_command",
                        lambda config, host, p: [sys.executable, script, "mlxgw.server", "--port", str(p)])
    config = tmp_path / "config.toml"
    config.write_text("[profiles.x]\n")

    async def go():
        app = GatewayTUI(GatewayClient(f"http://127.0.0.1:{port}", timeout=1), config=str(config),
                         poll_interval=60)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(app, pilot)
            await pilot.press("s")
            await settle(app, pilot)
            await pilot.press("q")
            await pilot.pause()

    run(go())
    try:
        info = control.read_pid(port)
        assert info is not None  # still alive after the TUI quit
        for _ in range(100):
            try:
                if GatewayClient(f"http://127.0.0.1:{port}").status()["gateway"]["pid"] == info.pid:
                    break
            except control.GatewayError:
                time.sleep(0.1)
        else:
            raise AssertionError("gateway did not answer")
    finally:
        control.stop(port)
    assert control.read_pid(port) is None
