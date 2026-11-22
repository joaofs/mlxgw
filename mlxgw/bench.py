"""mlxgw-bench: measure decode/prefill tokens/sec, TTFT and peak memory per alias.

    uv run mlxgw-bench                     # every alias already downloaded locally
    uv run mlxgw-bench --alias coder       # named aliases (downloads if needed)

Uses a running gateway at --base-url if one answers; otherwise starts one on a free
loopback port for the duration of the run and always stops it. The HTTP client is
stdlib-only.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .config import GB, Settings, load_settings
from .engine import ModelManager

FILLER = ("The quick brown fox jumps over the lazy dog while the gateway measures how fast "
          "the model reads this sentence. ")
FOLLOW_UP = "Continue the story in two sentences."
FOLLOW_UP_MAX_TOKENS = 32  # the follow-up turn only measures time to first token


# --- local model check (never downloads) -----------------------------------------

def local_snapshot(repo: str) -> Path | None:
    """Directory of a fully downloaded model, or None. Network-free."""
    path = Path(repo).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    try:
        snap = Path(snapshot_download(repo, local_files_only=True, allow_patterns=["*.json"]))
    except Exception:
        return None
    index = snap / "model.safetensors.index.json"
    weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    return snap if all((snap / w).exists() for w in weights) else None


def remote_size(repo: str) -> int | None:
    """Weight size from Hugging Face metadata (no download); None if unavailable."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True, timeout=10)
        return sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    except Exception:
        return None


def select_aliases(aliases: dict[str, str], requested: list[str], local=local_snapshot,
                   size_of_remote=remote_size) -> tuple[list[dict], list[dict]]:
    """Return (to_run, skipped). Named aliases always run; otherwise only local ones."""
    unknown = [a for a in requested if a not in aliases]
    if unknown:
        raise SystemExit(f"unknown alias(es): {', '.join(unknown)}; this profile has: {', '.join(aliases)}")
    run, skipped = [], []
    for alias, repo in aliases.items():
        if requested and alias not in requested:
            continue
        path = local(repo)
        size = ModelManager.weights_size(path) if path else None
        if path or requested:
            run.append({"alias": alias, "repo": repo, "size_bytes": size, "local": path is not None})
        else:
            skipped.append({"alias": alias, "repo": repo, "size_bytes": size_of_remote(repo)})
    if requested:  # keep the order the user asked for
        run.sort(key=lambda r: requested.index(r["alias"]))
    return run, skipped


# --- gateway lifecycle -------------------------------------------------------------

def probe(base_url: str, api_key: str | None, timeout: float = 2.0) -> list[str] | None:
    """Aliases served at base_url, [] if something answers but not with a model list, None if nothing answers."""
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/models", headers=_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return [m["id"] for m in json.load(resp)["data"]]
    except urllib.error.HTTPError:
        return []
    except (OSError, ValueError, KeyError):
        return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gateway_command(config: str, port: int) -> list[str]:
    return [sys.executable, "-m", "mlxgw.server", "--config", config, "--host", "127.0.0.1", "--port", str(port)]


@contextlib.contextmanager
def gateway(base_url: str, api_key: str | None, command=None, start_timeout: float = 120.0):
    """Yield (url, spawned). Reuses a gateway answering at base_url; otherwise spawns
    one with command(port) and stops it on exit, including on errors and Ctrl-C."""
    if probe(base_url, api_key) is not None:
        yield base_url, False
        return
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log = tempfile.NamedTemporaryFile("w+", prefix="mlxgw-bench-", suffix=".log", delete=False)
    # Own session: a terminal Ctrl-C reaches only us, and we stop the child ourselves.
    proc = subprocess.Popen(command(port), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ok = False
    try:
        deadline = time.monotonic() + start_timeout
        while probe(url, api_key, timeout=1.0) is None:
            if proc.poll() is not None or time.monotonic() > deadline:
                log.seek(0)
                tail = "".join(log.readlines()[-15:])
                raise SystemExit(f"gateway did not start (log {log.name}):\n{tail}")
            time.sleep(0.2)
        yield url, True
        ok = True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        if ok:
            os.unlink(log.name)


# --- requests ----------------------------------------------------------------------

def _headers(api_key: str | None) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def chat_stream(base_url: str, api_key: str | None, alias: str, messages: list[dict], max_tokens: int,
                timeout: float = 3600) -> dict:
    """One streaming request; returns text, client-measured TTFT, usage and x_gateway."""
    body = {"model": alias, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=_headers(api_key))
    t0 = time.perf_counter()
    t_first, parts, usage, gw = None, [], None, None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from None
    with resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            if "error" in obj:
                raise RuntimeError(obj["error"]["message"])
            if obj.get("choices") and (text := obj["choices"][0]["delta"].get("content")):
                t_first = t_first or time.perf_counter()
                parts.append(text)
            usage = obj.get("usage") or usage
            gw = obj.get("x_gateway") or gw
    return {"text": "".join(parts), "ttft_s": (t_first or time.perf_counter()) - t0, "usage": usage, "gateway": gw}


def prompt_text(words: int, nonce: str) -> str:
    filler = FILLER.split()
    context = " ".join(filler[i % len(filler)] for i in range(words))
    return f"[{nonce}] {context}\n\nWrite a long story about a lighthouse."


def run_once(base_url, api_key, alias, max_tokens, prompt_words) -> dict:
    msgs = [{"role": "user", "content": prompt_text(prompt_words, uuid.uuid4().hex)}]
    cold = chat_stream(base_url, api_key, alias, msgs, max_tokens)
    msgs += [{"role": "assistant", "content": cold["text"]}, {"role": "user", "content": FOLLOW_UP}]
    warm = chat_stream(base_url, api_key, alias, msgs, FOLLOW_UP_MAX_TOKENS)
    g, u, wu = cold["gateway"], cold["usage"], warm["usage"]
    return {
        "decode_tps": g.get("decode_tps"),
        "prefill_tps": g.get("prompt_tps"),
        "cold_ttft_s": cold["ttft_s"],
        "warm_ttft_s": warm["ttft_s"],
        "prompt_tokens": u["prompt_tokens"],
        "prefill_tokens": g.get("prefill_tokens"),
        "completion_tokens": u["completion_tokens"],
        "warm_prompt_tokens": wu["prompt_tokens"],
        "warm_cached_tokens": wu["prompt_tokens_details"]["cached_tokens"],
        "peak_memory_gb": max(g.get("peak_memory_gb") or 0, warm["gateway"].get("peak_memory_gb") or 0),
    }


# --- aggregation and output --------------------------------------------------------

METRICS = ("decode_tps", "prefill_tps", "cold_ttft_s", "warm_ttft_s", "warm_cached_tokens", "peak_memory_gb")


def summarize(runs: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        vals = [r[key] for r in runs if r.get(key) is not None]
        out[key] = ({"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
                    if vals else None)
    return out


def _cell(stat: dict | None, fmt: str) -> str:
    if stat is None:
        return "-"
    if stat["min"] == stat["max"]:
        return format(stat["median"], fmt)
    return f"{format(stat['median'], fmt)} ({format(stat['min'], fmt)}-{format(stat['max'], fmt)})"


def _gb(n: int | None) -> str:
    return "?" if n is None else f"{n / GB:.1f} GB"


def format_table(results: list[dict]) -> str:
    headers = ["alias", "repo", "size", "decode tok/s", "prefill tok/s", "cold TTFT s", "warm TTFT s",
               "cached tok", "peak GB"]
    rows = []
    for r in results:
        s = r.get("summary")
        if r.get("error") or not s:
            rows.append([r["alias"], r["repo"], _gb(r.get("size_bytes")), f"error: {r.get('error')}"])
            continue
        peak = s["peak_memory_gb"]
        rows.append([r["alias"], r["repo"], _gb(r.get("size_bytes")),
                     _cell(s["decode_tps"], ".1f"), _cell(s["prefill_tps"], ".0f"),
                     _cell(s["cold_ttft_s"], ".3f"), _cell(s["warm_ttft_s"], ".3f"),
                     _cell(s["warm_cached_tokens"], ".0f"), "-" if peak is None else f"{peak['max']:.2f}"])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows if i < len(row))) for i in range(len(headers))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def machine_info(settings: Settings) -> dict:
    try:
        mlx_lm = importlib.metadata.version("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        mlx_lm = None
    return {"chip": settings.hardware.chip, "memory_gb": round(settings.hardware.memory_gb, 1),
            "profile": settings.profile, "memory_budget_gb": round(settings.memory_budget_bytes / GB, 1),
            "mlx_lm": mlx_lm, "python": platform.python_version(), "macos": platform.mac_ver()[0]}


def header_line(m: dict) -> str:
    return (f"{m['chip']}, {m['memory_gb']:.0f} GB RAM, profile {m['profile']}, "
            f"budget {m['memory_budget_gb']:.1f} GB, mlx-lm {m['mlx_lm']}")


# --- main --------------------------------------------------------------------------

def bench_alias(url, api_key, item, args, log) -> dict:
    result = dict(item)
    try:
        log(f"[{item['alias']}] warm-up (loads {item['repo']}{'' if item['local'] else ', downloading first'})")
        warm = chat_stream(url, api_key, item["alias"], [{"role": "user", "content": "Say hello."}], 8)
        result["load_s"] = warm["gateway"].get("load_s") or 0.0
        if result["size_bytes"] is None and (path := local_snapshot(item["repo"])):  # downloaded by warm-up
            result["size_bytes"] = ModelManager.weights_size(path)
        print(f"  {item['alias']}: loaded in {result['load_s']:.1f}s", flush=True)
        runs = []
        for i in range(args.runs):
            r = run_once(url, api_key, item["alias"], args.max_tokens, args.prompt_words)
            runs.append(r)
            log(f"  run {i + 1}: decode {r['decode_tps']} tok/s, prefill {r['prefill_tps']} tok/s "
                f"({r['prefill_tokens']} tok), cold TTFT {r['cold_ttft_s']:.3f}s, warm TTFT "
                f"{r['warm_ttft_s']:.3f}s ({r['warm_cached_tokens']}/{r['warm_prompt_tokens']} cached), "
                f"peak {r['peak_memory_gb']:.2f} GB")
        result["runs"], result["summary"] = runs, summarize(runs)
    except RuntimeError as e:
        result["error"] = str(e)
        print(f"  {item['alias']}: failed: {e}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlxgw-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alias", action="append", default=[], help="alias to benchmark (repeatable)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-words", type=int, default=500)
    ap.add_argument("--config", default=os.environ.get("MLXGW_CONFIG", "config.toml"))
    ap.add_argument("--base-url", default=os.environ.get("MLXGW_URL"))
    ap.add_argument("--api-key", default=os.environ.get("MLXGW_API_KEY"))
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be >= 1")

    settings = load_settings(args.config)
    api_key = args.api_key or settings.api_key
    base_url = args.base_url or f"http://127.0.0.1:{settings.port}"
    machine = machine_info(settings)
    log = print if args.verbose else (lambda *a, **k: None)
    print(header_line(machine))

    to_run, skipped = select_aliases(settings.aliases, args.alias)
    results = []
    try:
        with gateway(base_url, api_key, command=lambda port: gateway_command(args.config, port)) as (url, spawned):
            served = probe(url, api_key) or []
            print(f"gateway: {url} ({'started for this run' if spawned else 'already running, left running'})")
            for item in to_run:
                if item["alias"] not in served:
                    why = "not served by this gateway" if served else "gateway returned no model list (API key?)"
                    results.append({**item, "error": why})
                    continue
                results.append(bench_alias(url, api_key, item, args, log))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    print(f"\nmedian (min-max) of {args.runs} run(s); prompt ~{args.prompt_words} words, "
          f"{args.max_tokens} max tokens; warm TTFT = follow-up turn with prompt cache\n")
    print(format_table(results) if results else "(no aliases benchmarked)")
    if skipped:
        print("\nskipped (not downloaded locally):")
        aw, rw = max(len(s["alias"]) for s in skipped), max(len(s["repo"]) for s in skipped)
        for s in skipped:
            print(f"  {s['alias']:<{aw}}  {s['repo']:<{rw}}  {_gb(s['size_bytes']):>8}   "
                  f"include with: uv run mlxgw-bench --alias {s['alias']}")
    if args.json:
        doc = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "machine": machine,
               "params": {"runs": args.runs, "max_tokens": args.max_tokens, "prompt_words": args.prompt_words,
                          "follow_up_max_tokens": FOLLOW_UP_MAX_TOKENS, "base_url": base_url},
               "results": results, "skipped": skipped}
        Path(args.json).write_text(json.dumps(doc, indent=2))
        print(f"\nwrote {args.json}")
    return 1 if any(r.get("error") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
