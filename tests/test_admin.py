import dataclasses
import logging
import threading
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mlxgw.admin import Admin, LogRing
from mlxgw.config import GB
from mlxgw.server import create_app

from .conftest import FakeEngine, make_settings

MSG = [{"role": "user", "content": "hi"}]
ALIASES = {"tiny": "fake/tiny-4bit", "coder": "fake/coder-4bit"}


class FakeDownloader:
    """Blocks until released or cancelled; reports progress the test sets."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail: str | None = None
        self.calls: list[str] = []
        self.report = None

    def __call__(self, repo, cancel, report):
        self.calls.append(repo)
        self.report = report
        report(0)
        self.started.set()
        while not (self.release.is_set() or cancel.is_set()):
            time.sleep(0.01)
        if cancel.is_set():
            return
        if self.fail:
            raise RuntimeError(self.fail)
        report(8 * GB)
        on_disk.add(repo)


on_disk: set[str] = set()


@pytest.fixture
def env():
    on_disk.clear()
    on_disk.add("fake/tiny-4bit")
    settings = dataclasses.replace(make_settings(), aliases=dict(ALIASES))
    engine = FakeEngine()
    dl = FakeDownloader()
    admin = Admin(settings, engine, download=dl,
                  local=lambda repo: Path("/nonexistent") if repo in on_disk else None,
                  remote_size=lambda repo: 8 * GB)
    admin.remote_sizes["fake/coder-4bit"] = 8 * GB
    client = TestClient(create_app(engine, settings, admin))
    return client, engine, dl, admin


def model(status, alias):
    return next(m for m in status["models"] if m["alias"] == alias)


def wait_for(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    raise AssertionError("condition not reached")


# --- status ------------------------------------------------------------------------

def test_status_shape(env):
    client, *_ = env
    r = client.get("/admin/status")
    assert r.status_code == 200
    s = r.json()
    assert set(s) == {"gateway", "machine", "queue", "models", "requests"}
    assert set(s["gateway"]) == {"pid", "started", "uptime_s", "host", "port"}
    assert s["machine"] == {"chip": "Apple M-series", "memory_bytes": 48 * GB, "profile": "48gb",
                            "budget_bytes": 30 * GB, "resident_bytes": 0}
    assert s["queue"] == {"in_flight": 0, "queued": 0}
    assert [m["alias"] for m in s["models"]] == ["tiny", "coder"]
    assert set(model(s, "tiny")) == {"alias", "repo", "state", "size_bytes", "last_used",
                                     "last_decode_tps", "download"}
    assert model(s, "tiny")["state"] == "on disk"
    assert model(s, "coder")["state"] == "not downloaded"
    assert model(s, "coder")["size_bytes"] == 8 * GB  # remote size, for the download confirmation
    assert s["requests"] == []


def test_requests_are_recorded_in_bounded_ring(env):
    client, _, _, admin = env
    client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG})
    client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG, "stream": True})
    reqs = client.get("/admin/status").json()["requests"]
    assert len(reqs) == 2
    r = reqs[0]
    assert r["alias"] == "tiny" and r["prompt_tokens"] == 7 and r["cached_tokens"] == 3
    assert r["completion_tokens"] == 4 and r["ttft_s"] == 0.01 and r["decode_tps"] == 100.0
    assert r["finish_reason"] == "stop" and r["error"] is None
    tiny = model(client.get("/admin/status").json(), "tiny")
    assert tiny["last_used"] is not None and tiny["last_decode_tps"] == 100.0
    for _ in range(60):
        client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG})
    assert len(client.get("/admin/status").json()["requests"]) == admin.requests.maxlen == 50


def test_queue_depth_reflects_in_flight_and_queued(live_server):
    import httpx
    engine = FakeEngine(delay=0.2)
    url = live_server(engine)
    threads = [threading.Thread(target=httpx.post, args=(url + "/v1/chat/completions",),
                                kwargs={"json": {"model": "tiny", "messages": MSG}, "timeout": 10})
               for _ in range(3)]
    for t in threads:
        t.start()
    wait_for(lambda: httpx.get(url + "/admin/status").json()["queue"] == {"in_flight": 1, "queued": 2})
    for t in threads:
        t.join()
    assert httpx.get(url + "/admin/status").json()["queue"] == {"in_flight": 0, "queued": 0}


# --- load / unload -----------------------------------------------------------------

def test_load_and_unload_run_on_worker_thread(env):
    client, engine, _, _ = env
    r = client.post("/admin/models/tiny/load")
    assert r.status_code == 200 and r.json()["state"] == "loaded"
    s = client.get("/admin/status").json()
    assert model(s, "tiny")["state"] == "loaded" and model(s, "tiny")["size_bytes"] == GB // 2
    assert s["machine"]["resident_bytes"] == GB // 2
    r = client.post("/admin/models/tiny/unload")
    assert r.json() == {"alias": "tiny", "unloaded": True}
    assert model(client.get("/admin/status").json(), "tiny")["state"] == "on disk"
    assert client.post("/admin/models/tiny/unload").json()["unloaded"] is False
    assert engine.worker_threads and all(n.startswith("mlx-worker") for n in engine.worker_threads)


def test_load_refuses_model_not_on_disk(env):
    client, engine, *_ = env
    r = client.post("/admin/models/coder/load")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_downloaded"
    assert engine.loaded == {}


def test_unknown_alias_is_404(env):
    client, *_ = env
    for path in ("load", "unload", "download", "download/cancel"):
        r = client.post(f"/admin/models/nope/{path}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


# --- downloads ---------------------------------------------------------------------

def test_download_state_machine_to_done(env):
    client, _, dl, _ = env
    r = client.post("/admin/models/coder/download")
    assert r.status_code == 202 and r.json()["download"]["state"] == "downloading"
    assert dl.started.wait(2)
    dl.report(2 * GB)
    coder = model(client.get("/admin/status").json(), "coder")
    assert coder["state"] == "downloading"
    assert coder["download"]["done_bytes"] == 2 * GB and coder["download"]["total_bytes"] == 8 * GB

    # A second download and a chat/load request for the same model get clear errors.
    r = client.post("/admin/models/coder/download")
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_downloading"
    r = client.post("/v1/chat/completions", json={"model": "coder", "messages": MSG})
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_downloading"
    assert "25%" in r.json()["error"]["message"]
    r = client.post("/admin/models/coder/load")
    assert r.json()["error"]["code"] == "model_downloading"
    assert dl.calls == ["fake/coder-4bit"]

    # Other models keep generating while the download runs.
    r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MSG})
    assert r.status_code == 200

    dl.release.set()
    wait_for(lambda: model(client.get("/admin/status").json(), "coder")["download"]["state"] == "done")
    coder = model(client.get("/admin/status").json(), "coder")
    assert coder["state"] == "on disk"
    r = client.post("/admin/models/coder/download")
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_downloaded"


def test_download_cancel(env):
    client, _, dl, _ = env
    client.post("/admin/models/coder/download")
    assert dl.started.wait(2)
    r = client.post("/admin/models/coder/download/cancel")
    assert r.status_code == 200
    wait_for(lambda: model(client.get("/admin/status").json(), "coder")["download"]["state"] == "cancelled")
    coder = model(client.get("/admin/status").json(), "coder")
    assert coder["state"] == "not downloaded"
    r = client.post("/admin/models/coder/download/cancel")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_downloading"
    # After a cancel the model can be requested again (normal on-demand path) and re-downloaded.
    assert client.post("/v1/chat/completions", json={"model": "coder", "messages": MSG}).status_code == 200
    assert client.post("/admin/models/coder/download").status_code == 202


def test_download_failure_is_reported(env):
    client, _, dl, _ = env
    dl.fail = "network down"
    client.post("/admin/models/coder/download")
    dl.release.set()
    wait_for(lambda: model(client.get("/admin/status").json(), "coder")["download"]["state"] == "failed")
    coder = model(client.get("/admin/status").json(), "coder")
    assert coder["download"]["error"] == "network down" and coder["state"] == "not downloaded"


def test_download_refused_while_a_request_is_fetching(env):
    client, engine, dl, _ = env
    engine.fetching = "fake/coder-4bit"
    assert model(client.get("/admin/status").json(), "coder")["state"] == "downloading"
    r = client.post("/admin/models/coder/download")
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_downloading"
    assert dl.calls == []


def test_download_of_model_on_disk_is_refused(env):
    client, _, dl, _ = env
    r = client.post("/admin/models/tiny/download")
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_downloaded"
    assert dl.calls == []


# --- logs --------------------------------------------------------------------------

def test_log_ring_is_bounded_and_incremental():
    ring = LogRing(maxlen=3)
    lg = logging.getLogger("mlxgw.test-ring")
    lg.setLevel(logging.INFO)
    lg.addHandler(ring)
    try:
        for i in range(5):
            lg.info("line %d", i)
    finally:
        lg.removeHandler(ring)
    lines = ring.since(0, 100)
    assert [x["seq"] for x in lines] == [3, 4, 5]
    assert lines[-1]["line"].endswith("INFO line 4")
    assert ring.since(4, 100) == [{"seq": 5, "line": lines[-1]["line"]}]


def test_logs_endpoint_tails_gateway_log(env):
    client, *_ = env
    lg = logging.getLogger("mlxgw")
    old = lg.level
    lg.setLevel(logging.INFO)
    try:
        with client:  # runs the lifespan, which attaches the ring
            client.post("/admin/models/tiny/load")
            client.post("/admin/models/tiny/unload")
            logging.getLogger("mlxgw").info("hello from the gateway")
            body = client.get("/admin/logs").json()
            assert any("hello from the gateway" in x["line"] for x in body["lines"])
            after = body["last_seq"]
            assert client.get(f"/admin/logs?after={after}").json()["lines"] == []
            assert client.get("/admin/logs?after=x").status_code == 400
        assert admin_handler_removed()
    finally:
        lg.setLevel(old)


def admin_handler_removed():
    return not any(isinstance(h, LogRing) for h in logging.getLogger("mlxgw").handlers)


# --- auth --------------------------------------------------------------------------

def test_admin_requires_same_auth_as_v1():
    engine = FakeEngine()
    settings = make_settings(api_key="sekret")
    client = TestClient(create_app(engine, settings))
    for method, path in [("GET", "/admin/status"), ("GET", "/admin/logs"),
                         ("POST", "/admin/models/tiny/load"), ("POST", "/admin/models/tiny/unload"),
                         ("POST", "/admin/models/tiny/download"), ("POST", "/admin/models/tiny/download/cancel")]:
        r = client.request(method, path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key", path
        r = client.request(method, path, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401, path
    assert engine.loaded == {}
    r = client.get("/admin/status", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200
