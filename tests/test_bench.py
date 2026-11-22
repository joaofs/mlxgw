import json
import os
import socket
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mlxgw import bench
from mlxgw.config import GB

from .conftest import FakeEngine

CONFIG = str(Path(__file__).parent.parent / "config.toml")


# --- aggregation and table -----------------------------------------------------

def test_summarize_median_min_max_and_missing_values():
    runs = [{"decode_tps": 100.0, "prefill_tps": None, "cold_ttft_s": 0.3, "warm_ttft_s": 0.02,
             "warm_cached_tokens": 600, "peak_memory_gb": 1.0},
            {"decode_tps": 120.0, "prefill_tps": None, "cold_ttft_s": 0.1, "warm_ttft_s": 0.04,
             "warm_cached_tokens": 600, "peak_memory_gb": 1.2},
            {"decode_tps": 90.0, "prefill_tps": None, "cold_ttft_s": 0.2, "warm_ttft_s": 0.03,
             "warm_cached_tokens": 610, "peak_memory_gb": 1.1}]
    s = bench.summarize(runs)
    assert s["decode_tps"] == {"median": 100.0, "min": 90.0, "max": 120.0}
    assert s["cold_ttft_s"] == {"median": 0.2, "min": 0.1, "max": 0.3}
    assert s["warm_cached_tokens"]["median"] == 600
    assert s["prefill_tps"] is None


def test_summarize_even_count_median():
    s = bench.summarize([{"decode_tps": 10.0}, {"decode_tps": 20.0}])
    assert s["decode_tps"] == {"median": 15.0, "min": 10.0, "max": 20.0}


def test_format_table_is_aligned_and_shows_errors():
    stat = lambda med, lo, hi: {"median": med, "min": lo, "max": hi}  # noqa: E731
    ok = {"alias": "tiny", "repo": "org/tiny-4bit", "size_bytes": int(0.3 * GB), "summary": {
        "decode_tps": stat(424.5, 420.5, 427.4), "prefill_tps": stat(22210, 10990, 22435),
        "cold_ttft_s": stat(0.051, 0.05, 0.109), "warm_ttft_s": stat(0.012, 0.012, 0.012),
        "warm_cached_tokens": stat(1300, 1300, 1300), "peak_memory_gb": stat(0.86, 0.8, 0.92)}}
    bad = {"alias": "coder", "repo": "org/a-much-longer-coder-repo-name", "size_bytes": None, "error": "HTTP 503"}
    lines = bench.format_table([ok, bad]).splitlines()
    assert lines[0].split() [:2] == ["alias", "repo"]
    assert set(lines[1].replace(" ", "")) == {"-"}
    # every column starts at the same offset on the header and the data rows
    assert lines[2].index("org/tiny-4bit") == lines[0].index("repo") == lines[3].index("org/a-much")
    assert lines[2].index("424.5 (420.5-427.4)") == lines[0].index("decode tok/s")
    assert "0.012 " in lines[2] or lines[2].endswith("0.012")  # no range when min == max
    assert "1300" in lines[2] and "0.92" in lines[2]  # peak shows the max
    assert "error: HTTP 503" in lines[3] and "?" in lines[3]


# --- alias selection -----------------------------------------------------------

ALIASES = {"coder": "org/coder", "fast": "org/fast", "tiny": "org/tiny"}


def selection(tmp_path, requested, local_repos=("org/tiny",)):
    dirs = {}
    for repo in local_repos:
        d = tmp_path / repo.replace("/", "_")
        d.mkdir()
        with open(d / "model.safetensors", "wb") as f:
            f.truncate(GB // 2)
        dirs[repo] = d
    remote_calls = []

    def remote(repo):
        remote_calls.append(repo)
        return 17 * GB

    run, skipped = bench.select_aliases(ALIASES, requested, local=dirs.get, size_of_remote=remote)
    return run, skipped, remote_calls


def test_default_selection_runs_only_local_aliases(tmp_path):
    run, skipped, _ = selection(tmp_path, [])
    assert [(r["alias"], r["local"], r["size_bytes"]) for r in run] == [("tiny", True, GB // 2)]
    assert [(s["alias"], s["repo"], s["size_bytes"]) for s in skipped] == \
        [("coder", "org/coder", 17 * GB), ("fast", "org/fast", 17 * GB)]


def test_named_aliases_run_even_if_not_local_in_given_order(tmp_path):
    run, skipped, remote_calls = selection(tmp_path, ["fast", "tiny"])
    assert [(r["alias"], r["local"]) for r in run] == [("fast", False), ("tiny", True)]
    assert skipped == [] and remote_calls == []  # coder was not named: not even looked up


def test_unknown_alias_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="unknown alias"):
        selection(tmp_path, ["nope"])


def test_local_snapshot_requires_every_weight_file(tmp_path, monkeypatch):
    import huggingface_hub
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
    (snap / "model-00001.safetensors").write_bytes(b"x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    assert bench.local_snapshot("org/x") is None  # second shard missing
    (snap / "model-00002.safetensors").write_bytes(b"x")
    assert bench.local_snapshot("org/x") == snap

    def not_cached(*a, **k):
        assert k.get("local_files_only") is True
        raise huggingface_hub.errors.LocalEntryNotFoundError("not cached")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    assert bench.local_snapshot("org/x") is None
    assert bench.local_snapshot(str(tmp_path)) == tmp_path  # a local directory is used as-is


# --- spawn and always stop -----------------------------------------------------

FAKE_SERVER = textwrap.dedent("""
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, pidfile = int(sys.argv[1]), sys.argv[2]
    open(pidfile, "w").write(str(os.getpid()))
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "tiny"}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()
""")


@pytest.fixture
def fake_cmd(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    pidfile = tmp_path / "pid"
    calls = []

    def command(port):
        calls.append(port)
        return [sys.executable, str(script), str(port), str(pidfile)]
    command.calls, command.pidfile = calls, pidfile
    return command


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def test_spawns_when_nothing_answers_and_stops_afterwards(fake_cmd):
    dead_url = f"http://127.0.0.1:{bench.free_port()}"
    with bench.gateway(dead_url, None, command=fake_cmd) as (url, spawned):
        assert spawned and url != dead_url
        assert bench.probe(url, None) == ["tiny"]
        pid = int(fake_cmd.pidfile.read_text())
        assert alive(pid)
    assert not alive(pid) and port_free(fake_cmd.calls[0])


@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
def test_spawned_gateway_is_stopped_on_error_and_ctrl_c(fake_cmd, exc):
    with pytest.raises(exc):
        with bench.gateway(f"http://127.0.0.1:{bench.free_port()}", None, command=fake_cmd):
            pid = int(fake_cmd.pidfile.read_text())
            raise exc()
    assert not alive(pid) and port_free(fake_cmd.calls[0])


def test_existing_gateway_is_used_and_left_running(fake_cmd):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "tiny"}]}')

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with bench.gateway(url, None, command=fake_cmd) as (got, spawned):
            assert (got, spawned) == (url, False)
        assert fake_cmd.calls == []
        assert bench.probe(url, None) == ["tiny"]  # still up
    finally:
        server.shutdown()


def test_gateway_that_fails_to_start_is_reported(tmp_path):
    cmd = lambda port: [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]  # noqa: E731
    with pytest.raises(SystemExit, match="did not start(.|\n)*boom"):
        with bench.gateway(f"http://127.0.0.1:{bench.free_port()}", None, command=cmd):
            pass


# --- whole command against the fake engine --------------------------------------

def test_size_is_filled_in_after_warm_up_download(live_server, tmp_path, monkeypatch):
    url = live_server(FakeEngine())
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.truncate(GB)
    monkeypatch.setattr(bench, "local_snapshot", lambda repo: tmp_path)  # "downloaded" by the warm-up
    item = {"alias": "tiny", "repo": "org/tiny", "size_bytes": None, "local": False}
    args = type("A", (), {"runs": 1, "max_tokens": 4, "prompt_words": 10})
    res = bench.bench_alias(url, None, item, args, log=lambda *a: None)
    assert res["size_bytes"] == GB and "error" not in res

def test_main_end_to_end_against_fake_gateway(live_server, tmp_path, capsys):
    url = live_server(FakeEngine())
    out = tmp_path / "bench.json"
    code = bench.main(["--config", CONFIG, "--base-url", url, "--alias", "tiny", "--runs", "2",
                       "--max-tokens", "4", "--json", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "profile" in text and "already running, left running" in text
    assert "decode tok/s" in text and "100.0" in text
    doc = json.loads(out.read_text())
    assert doc["machine"]["profile"] and doc["machine"]["mlx_lm"]
    (res,) = doc["results"]
    assert res["alias"] == "tiny" and len(res["runs"]) == 2
    assert res["summary"]["decode_tps"]["median"] == 100.0
    assert res["runs"][0]["warm_cached_tokens"] == 3
