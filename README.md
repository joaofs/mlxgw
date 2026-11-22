# mlxgw

One Python process that serves MLX models (via mlx-lm, in-process) behind an
OpenAI-compatible HTTP API. No plugins; an optional terminal UI manages it.

Endpoints: `POST /v1/chat/completions` (streaming and non-streaming), `GET /v1/models`,
and a small `/admin` API used by the terminal UI (below).

## Install

Requires Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

**mlx-lm is pinned to an unreleased build.** The `inkling` alias (Inkling-2-Mini) uses
the `inkling2_loop` architecture, which no mlx-lm release supports yet. `pyproject.toml`
pins mlx-lm to the head commit of the open pull request
[ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086). That build is
mlx-lm `main` from after 0.32.2 plus the PR, so it reports the version
`0.1.dev1008+ge2b5bbd3b`. Once the PR is merged and released, switch back to the PyPI
release. See [docs/status.md](docs/status.md#inkling-2-mini).

## Run

```sh
uv run mlxgw                     # uses ./config.toml, binds 127.0.0.1:8080
uv run mlxgw --config /path/config.toml --port 9000
```

Models are downloaded from Hugging Face on first use, so the first request to an
alias waits for the download. The coder models are 17-45 GB. To download ahead of time:
`uv run hf download <repo>`.

Point any OpenAI client at `http://127.0.0.1:8080/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
client.chat.completions.create(model="coder", messages=[{"role": "user", "content": "hi"}])
```

## Config

Everything is in `config.toml`. Environment variables override it:

| env | meaning |
|---|---|
| `MLXGW_CONFIG` | path to the config file |
| `MLXGW_HOST`, `MLXGW_PORT` | bind address (default `127.0.0.1:8080`) |
| `MLXGW_API_KEY` | bearer token; required when the host is not loopback |
| `MLXGW_PROFILE` | force a profile, e.g. `96gb` |
| `MLXGW_MEMORY_FRACTION` | share of RAM for resident model weights (default 0.70) |
| `MLXGW_MEMORY_BUDGET_GB` | absolute budget, overrides the fraction |

`--host` and `--port` on the command line override both.

**Auth.** On loopback (`127.0.0.1`, `::1`, `localhost`) no token is needed. On any other
address the server refuses to start without an API key, and every request must send
`Authorization: Bearer <key>`. If a key is set on loopback, it is enforced there too.

## Per-machine profiles

At startup the gateway reads `hw.memsize` and `machdep.cpu.brand_string` via `sysctl`,
picks the profile with the largest `min_memory_gb` that the machine satisfies, computes
the model memory budget, and logs the result:

```
hardware: Apple M-series, 48 GB RAM -> profile '48gb', model memory budget 33.6 GB
```

Each profile maps the same aliases (`coder`, `fast`, `tiny`, `inkling`) to a repo sized
for that machine, so `coder` is Qwen3-Coder-30B-A3B 4-bit on a 48 GB Mac and
Qwen3-Coder-Next 4-bit on a 96 GB Mac. `inkling` is Inkling-2-Mini 4-bit in both. These
are starting defaults; edit them to suit.

**Model lifecycle.** A model loads on first use and stays resident. Before a load, the
size of its `*.safetensors` files is compared with the budget. Least-recently-used models
are evicted until the new one fits, and a model larger than the whole budget is refused
with a 503 error. The budget covers weights only; KV caches come on top.

**Prompt cache.** For each loaded model the gateway keeps the KV caches of the last
`prompt_cache_slots` conversations. A new request reuses the longest matching token
prefix, so turn N of a chat only processes the new messages.
`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens were reused.

## Responses

Standard OpenAI shapes, plus an `x_gateway` object with timings (on the final chunk
when streaming): `queue_s`, `load_s` (includes any download), `ttft_s` (request
received to first token), `prefill_s`, `prefill_tokens`, `prompt_tps`, `decode_tps`,
`peak_memory_gb`, `model`, `profile`.

Supported request fields: `model`, `messages` (text only), `stream`,
`stream_options.include_usage`, `temperature`, `top_p`, `max_tokens`,
`max_completion_tokens`, `stop`, `seed`, `n=1`, `tools`, `tool_choice` (`auto` or
`none`). Recognised but unimplemented OpenAI fields (`response_format`, `logprobs`,
`n>1`, penalties, `logit_bias`, ...) return 400 unless set to their no-op value. Unknown
fields also return 400, as they do on OpenAI.

**Tool calls.** Function tools are rendered by the model's chat template, and the
model's output is parsed with the tool parser mlx-lm picks for that template. Responses
use the OpenAI shape (`tool_calls`, `finish_reason: "tool_calls"`); when streaming, each
call arrives whole in one `delta.tool_calls` chunk. Tool results go back as
`role: "tool"` messages, and the prompt cache carries over between those turns. A model
without tool support gets a 400. Details in [docs/api.md](docs/api.md#tool-calls).

## Use with OpenCode

[OpenCode](https://opencode.ai) 2.x takes a custom OpenAI-compatible provider in
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "mlxgw": {
      "name": "mlxgw (local MLX)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "coder": {
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 65536, "output": 16384 }
        },
        "fast": {
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 65536, "output": 16384 }
        }
      }
    }
  }
}
```

```sh
opencode -m mlxgw/coder           # or: opencode run -m mlxgw/fast "..."
```

- Add `"apiKey": "{env:MLXGW_API_KEY}"` under `settings` when the gateway has a token.
- `input` is text only, because the gateway accepts no images.
- The models support 262,144 tokens of context. The 65,536 limit keeps the KV cache
  to about 6-10 GB per conversation on a 48 GB Mac (Qwen3-4B uses 144 KB per
  token). OpenCode compacts the conversation when it nears the limit.
- Use a provider ID that is not in OpenCode's built-in catalog. For example, `llmgateway`
  is the ID of a hosted provider there; an entry with that ID inherits that provider's
  model list (only `coder` and `fast` would work), and each session first asks one of the
  catalog models for a title, gets a 404, and falls back to the session model.

## Benchmark

One command, on each machine:

```sh
uv run mlxgw-bench                              # every alias already downloaded
uv run mlxgw-bench --alias coder --alias fast   # named aliases (downloads if needed)
uv run mlxgw-bench --json results-$(hostname -s).json
```

If a gateway answers at `--base-url` (default `http://127.0.0.1:<port from config>`),
the benchmark uses it and leaves it running. Otherwise it starts one on a free loopback
port with `--config` and stops it when done, including after errors and Ctrl-C.

With no `--alias`, only aliases whose weights are already on disk are run. The others
are listed with their size and the command that includes them. Nothing is downloaded
unless you name the alias.

For each alias: one warm-up request loads the model (its load time is reported
separately), then `--runs` (default 3) measured runs. Each run sends a fresh prompt
(`--prompt-words`, default 500, with a unique nonce so nothing comes from the prompt
cache), generates up to `--max-tokens` (default 256), then sends a follow-up turn of
the same conversation, which is served from the prompt cache.

The table shows median (min-max) per alias:

| column | meaning |
|---|---|
| decode tok/s | generation speed after the first token (server-measured); the headline number |
| prefill tok/s | prompt processing speed on the fresh prompt |
| cold TTFT s | time to first token, fresh prompt (client-measured) |
| warm TTFT s | time to first token on the follow-up turn |
| cached tok | prompt tokens of the follow-up turn served from the cache |
| peak GB | highest MLX memory seen (weights + KV cache + scratch) |

`--json PATH` writes every run plus machine info (chip, RAM, profile, budget, mlx-lm
version), so you can compare results from different Macs. `--verbose` prints per-run
lines. Sizes and the budget are in GiB.

## Terminal UI

```sh
uv sync --extra tui
uv run mlxgw-tui                       # the gateway on this machine (port from config.toml)
uv run mlxgw-tui --base-url http://<host>:8080 --api-key ...   # another machine
```

`--base-url` / `--api-key` default to `MLXGW_URL` / `MLXGW_API_KEY`. One screen: gateway
state, machine, profile and a memory bar (resident weights vs budget); the models with
size, state (not downloaded, downloading n%, on disk, loaded), last decode tok/s and
last use; recent requests with in-flight/queued counts; and the gateway's log tail.

| key | action |
|---|---|
| `s` / `x` | start / stop the local gateway |
| `d` / `c` | download the selected model on the gateway machine / cancel the download |
| `l` / `u` | load / unload the selected model |
| `b` | benchmark the selected model (same measurement as `mlxgw-bench`, 3 runs) |
| `q` | quit; the gateway keeps running |

Stop, unload, download and anything that loads a model over the budget ask first.

**Local lifecycle.** `s` starts `mlxgw` detached from the terminal, writes
`~/.local/state/mlxgw/gateway-<port>.pid` and logs to `gateway-<port>.log` next to
it (`MLXGW_STATE_DIR` or `XDG_STATE_HOME` move it). Quitting leaves it running; run
`mlxgw-tui` again to reattach. `x` only stops a gateway named by that pid file whose
command line is an mlxgw server on that port; a gateway started any other way is
left alone, and the UI says so. On a remote gateway `s` and `x` are disabled.

**Admin API** (same bearer token as `/v1`): `GET /admin/status`, `GET /admin/logs?after=<seq>`,
`POST /admin/models/<alias>/load|unload|download|download/cancel`. A download runs in a
child process on the gateway machine, so generation continues and cancel is immediate;
partial files stay in the Hugging Face cache and a later download resumes them. While
an alias is downloading, chat requests and loads for it get `409 model_downloading`.
`load` never downloads; use `download` first.

## Tests

```sh
uv run pytest
```

The tests use a fake engine and a fake downloader; they need no model download. The
TUI tests need the dev group (which includes Textual).
