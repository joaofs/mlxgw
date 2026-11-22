"""mlxgw-tui: one keyboard-driven screen to manage a local or remote gateway.

    uv sync --extra tui
    uv run mlxgw-tui                                   # this machine (port from config.toml)
    uv run mlxgw-tui --base-url http://<host>:8080 --api-key ...   # another machine

Quitting leaves the gateway running. Start/stop are local only and act only on a
gateway this tool started (identified by its pid file).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid

try:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.screen import ModalScreen
    from textual.widgets import DataTable, Footer, Log, Static
except ImportError:  # pragma: no cover
    sys.exit("mlxgw-tui needs Textual; install it with: uv sync --extra tui")

from . import bench, control
from .config import GB, load_settings
from .control import GatewayClient, GatewayError

BENCH_RUNS, BENCH_MAX_TOKENS, BENCH_PROMPT_WORDS = 3, 256, 500  # same defaults as mlxgw-bench


def gb(n) -> str:
    return "?" if n is None else f"{n / GB:.1f} GB"


def ago(ts, now=None) -> str:
    if not ts:
        return "-"
    s = max(0, (now or time.time()) - ts)
    if s < 60:
        return f"{s:.0f}s ago"
    if s < 3600:
        return f"{s / 60:.0f}m ago"
    if s < 86400:
        return f"{s / 3600:.0f}h ago"
    return f"{s / 86400:.0f}d ago"


def duration(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def mem_bar(used: int, total: int, width: int = 20) -> Text:
    frac = min(1.0, used / total) if total else 0.0
    filled = round(frac * width)
    t = Text("[")
    t.append("█" * filled, style="green" if frac < 0.85 else "yellow")
    t.append("░" * (width - filled), style="dim")
    t.append(f"] {used / GB:.1f}/{total / GB:.1f} GB")
    return t


def state_label(m: dict) -> str:
    d = m.get("download") or {}
    if m["state"] == "downloading":
        if d.get("state") == "downloading" and d.get("total_bytes"):
            return f"downloading {min(99, 100 * d['done_bytes'] / d['total_bytes']):.0f}%"
        if d.get("state") == "downloading":
            return f"downloading {gb(d.get('done_bytes'))}"
        return "downloading"
    if m["state"] == "not downloaded" and d.get("state") in ("failed", "cancelled"):
        return f"dl {d['state']}"
    return m["state"]


def short_line(line: str) -> str:
    """Log lines are clipped at the pane width; drop the date to keep more of the message."""
    return line[11:] if len(line) > 11 and line[4] == "-" and line[10] == " " else line


def tilde(path) -> str:
    home = os.path.expanduser("~")
    return "~" + str(path)[len(home):] if str(path).startswith(home + os.sep) else str(path)


def short_repo(repo: str, width: int = 26) -> str:
    return repo if len(repo) <= width else "…" + repo[-(width - 1):]


def run_benchmark(client: GatewayClient, alias: str, runs: int = BENCH_RUNS, max_tokens: int = BENCH_MAX_TOKENS,
                  prompt_words: int = BENCH_PROMPT_WORDS) -> dict:
    """mlxgw-bench's measurement for one alias, without its printing. Blocking."""
    url, key = client.base_url, client.api_key
    warm = bench.chat_stream(url, key, alias, [{"role": "user", "content": "Say hello."}], 8)
    results = [bench.run_once(url, key, alias, max_tokens, prompt_words) for _ in range(runs)]
    return {"alias": alias, "load_s": (warm["gateway"] or {}).get("load_s"), "runs": runs,
            "summary": bench.summarize(results)}


def format_benchmark(r: dict) -> str:
    s = r["summary"]

    def cell(key, fmt):
        return bench._cell(s.get(key), fmt)
    peak = s.get("peak_memory_gb")
    return "\n".join([
        f"benchmark {r['alias']}: median (min-max) of {r['runs']} runs, "
        f"~{BENCH_PROMPT_WORDS}-word prompt, {BENCH_MAX_TOKENS} max tokens",
        "",
        f"  warm-up load     {r['load_s'] or 0:.1f} s",
        f"  decode tok/s     {cell('decode_tps', '.1f')}",
        f"  prefill tok/s    {cell('prefill_tps', '.0f')}",
        f"  cold TTFT s      {cell('cold_ttft_s', '.3f')}",
        f"  warm TTFT s      {cell('warm_ttft_s', '.3f')}",
        f"  cached tok       {cell('warm_cached_tokens', '.0f')}",
        f"  peak GB          {'-' if peak is None else format(peak['max'], '.2f')}",
    ])


class ConfirmScreen(ModalScreen[bool]):
    """y / Enter accepts, n / Esc cancels. Nothing happens until one is pressed."""

    BINDINGS = [Binding("y,enter", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Static { width: 70; max-width: 100%; height: auto; padding: 1 2;
                             border: thick $warning; background: $surface; }
    """

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        yield Static(f"{self.question}\n\n[b]y[/b] yes   [b]n[/b] no", id="question", markup=True)

    def action_answer(self, yes: bool) -> None:
        self.dismiss(yes)


class TextScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,enter,q,space", "app.pop_screen", "Close")]
    DEFAULT_CSS = """
    TextScreen { align: center middle; }
    TextScreen > Static { width: 76; max-width: 100%; height: auto; padding: 1 2;
                          border: thick $accent; background: $surface; }
    """

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        yield Static(self.text + "\n\n(Enter or Esc to close)", id="text", markup=False)


class GatewayTUI(App):
    TITLE = "mlxgw"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { layout: vertical; }
    #header { height: 2; padding: 0 1; background: $panel; }
    #models { height: auto; max-height: 8; }
    .title { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #activity { height: 1fr; min-height: 3; }
    #log { height: 1fr; min-height: 3; overflow-x: hidden; }
    #message { height: 1; padding: 0 1; }
    """
    BINDINGS = [
        Binding("s", "start", "Start"),
        Binding("x", "stop", "Stop"),
        Binding("d", "download", "Download"),
        Binding("c", "cancel_download", "Cancel dl"),
        Binding("l", "load", "Load"),
        Binding("u", "unload", "Unload"),
        Binding("b", "bench", "Bench"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, client: GatewayClient, config: str | None = None, poll_interval: float = 1.0):
        super().__init__()
        self.client, self.config, self.poll_interval = client, config, poll_interval
        self.local = client.is_local
        self.state = "connecting"  # running | starting | stopped | unreachable | error
        self.status: dict | None = None
        self.error: str | None = None
        self.pid_info: control.PidInfo | None = None
        self.log_seq = 0
        self.gateway_pid: int | None = None
        self._file_tail: list[str] = []
        self._polling = False
        self.busy: str | None = None  # description of a running action

    # --- layout ---

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        models = DataTable(id="models", cursor_type="row", show_row_labels=False)
        yield models
        yield Static("Activity", id="activity-title", classes="title")
        yield DataTable(id="activity", cursor_type="none", show_cursor=False, show_row_labels=False)
        yield Static("Log", id="log-title", classes="title")
        yield Log(id="log", max_lines=500)
        yield Static(id="message")
        yield Footer()

    def on_mount(self) -> None:
        m = self.query_one("#models", DataTable)
        for label, key, width in [("alias", "alias", 6), ("repo", "repo", 26), ("size", "size", 7),
                                  ("state", "state", 15), ("tok/s", "tps", 6), ("used", "used", 8)]:
            m.add_column(label, key=key, width=width)
        a = self.query_one("#activity", DataTable)
        a.can_focus = False
        for label, width in [("time", 8), ("alias", 6), ("prompt", 6), ("cached", 6), ("compl", 5),
                             ("ttft s", 6), ("tok/s", 6), ("finish", 9)]:
            a.add_column(label, width=width)
        self.query_one("#log", Log).can_focus = False
        m.focus()
        self.say("q quits and leaves the gateway running")
        self.render_header()
        self.set_interval(self.poll_interval, self._tick)
        self.call_later(self.refresh_now)

    # --- polling ---

    def _tick(self) -> None:
        if not self._polling:
            self.run_worker(self.refresh_now(), group="poll", exit_on_error=False)

    def _fetch(self) -> tuple:
        """Blocking; runs in a thread. Returns (status, logs, error)."""
        pid_info = control.read_pid(self.client.port) if self.local else None
        try:
            status = self.client.status()
            after = self.log_seq if status["gateway"]["pid"] == self.gateway_pid else 0
            logs = self.client.logs(after=after)
            return status, logs, None, pid_info
        except GatewayError as e:
            return None, None, e, pid_info
        except Exception as e:  # noqa: BLE001 - e.g. an unexpected status shape; never crash the UI
            return None, None, GatewayError(-1, f"bad response: {type(e).__name__}: {e}"), pid_info

    async def refresh_now(self) -> None:
        if self._polling:
            return
        self._polling = True
        try:
            status, logs, err, pid_info = await asyncio.to_thread(self._fetch)
            self.apply(status, logs, err, pid_info)
        finally:
            self._polling = False

    def apply(self, status, logs, err: GatewayError | None, pid_info) -> None:
        self.pid_info = pid_info
        if err is None:
            self.state, self.status, self.error = "running", status, None
        elif err.unreachable:
            self.status, self.error = None, err.message
            self.state = ("starting" if pid_info else "stopped") if self.local else "unreachable"
        else:  # it answers but not usefully: wrong key, no admin API, ...
            self.status = None
            self.error = f"HTTP {err.status}: {err.message}" if err.status > 0 else err.message
            self.state = "error"
        self.render_header()
        self.render_models()
        self.render_activity()
        self.render_log(logs)

    def render_header(self) -> None:
        colors = {"running": "green", "starting": "yellow", "stopped": "red", "unreachable": "red",
                  "error": "red", "connecting": "yellow"}
        line1 = Text()
        line1.append(self.client.base_url, style="bold")
        line1.append("  ")
        line1.append(self.state, style=f"bold {colors.get(self.state, 'white')}")
        s = self.status
        if s:
            g = s["gateway"]
            ours = self.pid_info is not None and self.pid_info.pid == g["pid"]
            where = "local" if self.local else "remote"
            line1.append(f"  pid {g['pid']} ({where}{', started here' if ours else ''})  up {duration(g['uptime_s'])}")
            m = s["machine"]
            line2 = Text(f"{m['chip']} {m['memory_bytes'] / GB:.0f} GB · {m['profile']} · mem ")
            line2.append_text(mem_bar(m["resident_bytes"], m["budget_bytes"]))
        else:
            line2 = Text(self.error or "", style="dim")
            if self.state == "stopped":
                line2 = Text("no gateway running on this machine; press s to start one", style="yellow")
            elif self.state == "starting":
                line2 = Text(f"starting (pid {self.pid_info.pid}); waiting for it to answer", style="yellow")
            elif self.state == "unreachable":
                line2 = Text(f"{self.error}; retrying every {self.poll_interval:.0f}s", style="red")
        self.query_one("#header", Static).update(Text("\n").join([line1, line2]))

    def render_models(self) -> None:
        table = self.query_one("#models", DataTable)
        models = (self.status or {}).get("models", [])
        keys = [m["alias"] for m in models]
        existing = [r.value for r in table.rows]
        if keys != existing:
            cursor = table.cursor_row
            table.clear()
            for alias in keys:
                table.add_row(*[""] * 6, key=alias)
            if keys:
                table.move_cursor(row=min(max(cursor, 0), len(keys) - 1))
        now = time.time()
        for m in models:
            tps = m.get("last_decode_tps")
            for col, val in [("alias", m["alias"]), ("repo", short_repo(m["repo"])), ("size", gb(m["size_bytes"])),
                             ("state", state_label(m)), ("tps", f"{tps:.0f}" if tps else "-"),
                             ("used", ago(m.get("last_used"), now))]:
                table.update_cell(m["alias"], col, val)

    def render_activity(self) -> None:
        q = (self.status or {}).get("queue")
        title = "Activity"
        if q:
            title += f" · in flight {q['in_flight']} · queued {q['queued']}"
        if self.busy:
            title += f" · {self.busy}"
        self.query_one("#activity-title", Static).update(title)
        table = self.query_one("#activity", DataTable)
        table.clear()
        for r in reversed((self.status or {}).get("requests", [])):
            def n(v, fmt="{}"):
                return "-" if v is None else fmt.format(v)
            table.add_row(time.strftime("%H:%M:%S", time.localtime(r["time"])), r["alias"],
                          n(r["prompt_tokens"]), n(r["cached_tokens"]), n(r["completion_tokens"]),
                          n(r["ttft_s"], "{:.2f}"), n(r["decode_tps"], "{:.0f}"), r["finish_reason"])

    def render_log(self, logs: dict | None) -> None:
        log = self.query_one("#log", Log)
        if logs is not None and self.status:
            pid = self.status["gateway"]["pid"]
            if pid != self.gateway_pid:  # new or restarted gateway: start over
                log.clear()
                self.gateway_pid, self._file_tail = pid, []
            lines = logs.get("lines", [])
            if lines:
                log.write_lines([short_line(x["line"]) for x in lines])
                self.log_seq = lines[-1]["seq"]
            title = "Log (gateway)"
        elif self.local and self.pid_info is None and self.state != "running":
            title = "Log (no gateway)"
            self._show_file_tail(log, control.log_path(self.client.port))
        elif self.local and self.pid_info is not None:
            title = f"Log ({tilde(self.pid_info.log)})"
            self._show_file_tail(log, self.pid_info.log)
        else:
            title = "Log (unavailable)"
        self.query_one("#log-title", Static).update(title)

    def _show_file_tail(self, log: Log, path) -> None:
        lines = control.tail(path, 30)
        if lines != self._file_tail:
            log.clear()
            log.write_lines([short_line(x) for x in lines])
            log.scroll_end(animate=False)
            self._file_tail = lines
        self.gateway_pid = None

    # --- helpers ---

    def say(self, text: str, error: bool = False) -> None:
        self.query_one("#message", Static).update(Text(text, style="red" if error else ""))

    def selected(self) -> dict | None:
        if not self.status:
            self.say(f"gateway is {self.state}; no models to act on", error=True)
            return None
        table = self.query_one("#models", DataTable)
        if table.row_count == 0:
            return None
        alias = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
        return next((m for m in self.status["models"] if m["alias"] == alias), None)

    def confirm(self, question: str, then) -> None:
        def done(yes: bool | None) -> None:
            if yes:
                then()
            else:
                self.say("cancelled")
        self.push_screen(ConfirmScreen(question), done)

    def run_call(self, label: str, fn, ok, show=None) -> None:
        """Run blocking fn() in a thread; report `ok(result)` or the error; refresh."""
        if self.busy:
            self.say(f"busy: {self.busy}", error=True)
            return

        async def go():
            self.busy = label
            self.say(label + " …")
            self.render_activity()
            try:
                result = await asyncio.to_thread(fn)
            except Exception as e:  # noqa: BLE001 - report it in the UI, never crash
                self.say(f"{label} failed: {getattr(e, 'message', None) or e}", error=True)
            else:
                self.say(ok(result))
                if show:
                    self.push_screen(TextScreen(show(result)))
            finally:
                self.busy = None
            await self.refresh_now()
        self.run_worker(go(), group="action", exit_on_error=False)

    # --- actions ---

    def action_start(self) -> None:
        if not self.local:
            self.say("start is only available for a gateway on this machine", error=True)
            return
        if self.state == "running":
            self.say("a gateway is already running at " + self.client.base_url, error=True)
            return
        if self.state in ("starting", "error"):
            self.say(f"gateway is {self.state}; not starting another", error=True)
            return
        if not self.config:
            self.say("no config file to start a gateway with (--config)", error=True)
            return
        port, host = self.client.port, self.client.host
        self.run_call(f"starting gateway on {host}:{port}",
                      lambda: control.start(self.config, port, "127.0.0.1" if host == "localhost" else host),
                      lambda info: f"started gateway pid {info.pid}; logs to {tilde(info.log)}")

    def action_stop(self) -> None:
        if not self.local:
            self.say("stop is only available for a gateway on this machine", error=True)
            return
        info = self.pid_info
        if info is None:
            if self.state == "running":
                self.say("cannot stop: this gateway was not started by mlxgw-tui "
                         f"(no pid file at {control.pid_path(self.client.port)})", error=True)
            else:
                self.say("no gateway started by mlxgw-tui is running", error=True)
            return
        q = (self.status or {}).get("queue") or {}
        busy = q.get("in_flight", 0) + q.get("queued", 0)
        warn = f"\n{busy} request(s) in flight or queued will be cut off." if busy else ""
        self.confirm(f"Stop the gateway (pid {info.pid}) at {self.client.base_url}?{warn}",
                     lambda: self.run_call(f"stopping gateway pid {info.pid}",
                                           lambda: control.stop(self.client.port),
                                           lambda i: f"stopped gateway pid {i.pid}"))

    def action_download(self) -> None:
        if (m := self.selected()) is None:
            return
        if m["state"] != "not downloaded":
            self.say(f"{m['alias']} is {m['state']}; nothing to download", error=True)
            return
        where = "this machine" if self.local else self.client.host
        size = gb(m["size_bytes"]) if m["size_bytes"] else "unknown size"
        self.confirm(f"Download {m['alias']} ({m['repo']}, {size}) to the Hugging Face cache on {where}?",
                     lambda: self.run_call(f"starting download of {m['alias']}",
                                           lambda: self.client.download(m["alias"]),
                                           lambda r: f"downloading {m['alias']} (c cancels)"))

    def action_cancel_download(self) -> None:
        if (m := self.selected()) is None:
            return
        if (m.get("download") or {}).get("state") != "downloading":
            self.say(f"{m['alias']} is not downloading", error=True)
            return
        self.run_call(f"cancelling download of {m['alias']}", lambda: self.client.cancel_download(m["alias"]),
                      lambda r: f"cancelled download of {m['alias']}")

    def _evicts(self, m: dict) -> list[str]:
        """Aliases that loading m would push out (approximation of the gateway's LRU)."""
        mach = self.status["machine"]
        loaded = sorted((x for x in self.status["models"] if x["state"] == "loaded"),
                        key=lambda x: x.get("last_used") or 0)
        free, out = mach["budget_bytes"] - mach["resident_bytes"], []
        for x in loaded:
            if free >= (m["size_bytes"] or 0):
                break
            free += x["size_bytes"] or 0
            out.append(x["alias"])
        return out

    def action_load(self) -> None:
        if (m := self.selected()) is None:
            return
        if m["state"] == "loaded":
            self.say(f"{m['alias']} is already loaded")
            return
        if m["state"] != "on disk":
            self.say(f"{m['alias']} is {m['state']}; download it first", error=True)
            return

        def go():
            self.run_call(f"loading {m['alias']}", lambda: self.client.load(m["alias"]),
                          lambda r: f"loaded {m['alias']} in {r.get('load_s', 0):.1f}s")
        if evicts := self._evicts(m):
            self.confirm(f"Loading {m['alias']} ({gb(m['size_bytes'])}) will unload {', '.join(evicts)} "
                         "to stay within the memory budget. Continue?", go)
        else:
            go()

    def action_unload(self) -> None:
        if (m := self.selected()) is None:
            return
        if m["state"] != "loaded":
            self.say(f"{m['alias']} is not loaded", error=True)
            return
        self.confirm(f"Unload {m['alias']} ({gb(m['size_bytes'])})? Its prompt caches are dropped and the next "
                     "request reloads it.",
                     lambda: self.run_call(f"unloading {m['alias']}", lambda: self.client.unload(m["alias"]),
                                           lambda r: f"unloaded {m['alias']}"))

    def action_bench(self) -> None:
        if (m := self.selected()) is None:
            return
        if m["state"] not in ("loaded", "on disk"):
            self.say(f"{m['alias']} is {m['state']}; benchmark needs it on disk", error=True)
            return

        def go():
            self.run_call(f"benchmarking {m['alias']} ({BENCH_RUNS} runs)",
                          lambda: run_benchmark(self.client, m["alias"]),
                          lambda r: "decode {} tok/s for {}".format(
                              bench._cell(r["summary"]["decode_tps"], ".1f"), m["alias"]),
                          show=format_benchmark)
        if m["state"] == "on disk":
            evicts = self._evicts(m)
            extra = f" and unload {', '.join(evicts)}" if evicts else ""
            self.confirm(f"Benchmarking {m['alias']} loads it ({gb(m['size_bytes'])}){extra}. Continue?", go)
        else:
            go()


def screen_text(app: App) -> str:
    """Plain-text rendering of the current screen (like export_screenshot, as text)."""
    import io

    from rich.console import Console
    width, height = app.size
    console = Console(width=width, height=height, file=io.StringIO(), record=True,
                      color_system=None, legacy_windows=False, safe_box=False)
    console.print(app.screen._compositor.render_update(full=True, screen_stack=app._background_screens))
    return console.export_text()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlxgw-tui", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.environ.get("MLXGW_CONFIG", "config.toml"),
                    help="config used to find the local port and to start a local gateway")
    ap.add_argument("--base-url", default=os.environ.get("MLXGW_URL"),
                    help="gateway to manage (default http://127.0.0.1:<port from config>)")
    ap.add_argument("--api-key", default=os.environ.get("MLXGW_API_KEY"))
    args = ap.parse_args(argv)

    config = args.config if os.path.isfile(args.config) else None
    base_url, api_key = args.base_url, args.api_key
    if config and (not base_url or not api_key):
        try:
            s = load_settings(config)
            base_url = base_url or f"http://127.0.0.1:{s.port}"
            api_key = api_key or s.api_key
        except Exception as e:  # noqa: BLE001 - a bad local config should not block remote use
            print(f"mlxgw-tui: ignoring {config}: {e}", file=sys.stderr)
    client = GatewayClient(base_url or "http://127.0.0.1:8080", api_key)
    app = GatewayTUI(client, config=os.path.abspath(config) if config else None)
    app.run()
    if app.state == "running":
        where = f"pid {app.pid_info.pid}, log {app.pid_info.log}" if app.pid_info else "not started by this tool"
        print(f"gateway still running at {client.base_url} ({where}). "
              "To stop it: mlxgw-tui, then x.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
