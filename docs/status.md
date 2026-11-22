# Status

## Example measurements

On a 48 GB Apple Silicon Mac. Your numbers depend on the chip, its GPU core count and
memory bandwidth; run `mlxgw-bench` to get your own.

| Alias | Model | Decode tok/s | Prompt tok/s | First token, cold | First token, cached |
|---|---|---:|---:|---:|---:|
| `fast` | Qwen3-4B 4-bit | 101.8 | 2,908 | 0.198 s | 0.050 s |
| `tiny` | Qwen2.5-0.5B 4-bit | 468.8 | 20,877 | 0.029 s | 0.015 s |

Medians of 3 runs from `mlxgw-bench`, with a prompt of about 500 words and 256
output tokens. Decode speed moves with other activity on the machine: `tiny` varied by about 20%
across sessions.

`coder` has not been downloaded or measured. To measure it:

```sh
uv run mlxgw-bench --alias coder
```

## Limits

- **Budget counts weights only.** See [models-and-memory.md](models-and-memory.md).
- **One generation at a time.** Concurrent clients queue. There is no batching, and the
  queue has no length limit.
- **First use of an alias downloads the model inside that request.** Download ahead of
  time from the TUI or as shown in the project README.
- **Unknown request fields get a 400.** A client that sends `top_k`, for example, is
  rejected.
- **Reasoning text is not separated.** A thinking model's reasoning appears in the
  content.
- **Only text content parts are accepted.**
- **Tool calls cannot be forced.** `tool_choice` is `auto` or `none`; `required` and a
  named function get a 400.
- **An unparseable tool call is dropped.** If a small model writes invalid JSON inside
  `<tool_call>`, the call is logged and dropped, and the response may then be empty with
  `finish_reason: "stop"`. The client gets no error to retry on.
- **Tool calls stream whole.** Each call arrives as one chunk once it is complete, not
  argument by argument.
- **A benchmark started from the TUI cannot be cancelled** and blocks other TUI actions
  while it runs.
- **Peak memory in the benchmark is for the whole process,** so it includes every
  resident model.

## Inkling-2-Mini

The `inkling` alias serves `mlx-community/Inkling-2-Mini-4bit` (17.3 GB, 16.1 GiB of
weights), an MLX conversion of `thinkingmachines/Inkling-2-Mini` (Apache-2.0).

**Pinned mlx-lm build.** No mlx-lm release supports the `inkling2_loop` architecture.
`pyproject.toml` pins mlx-lm to commit `e2b5bbd` of the open pull request
`ml-explore/mlx-lm#2086`. The commit is mlx-lm `main` from after 0.32.2 plus that PR, and
reports the version `0.1.dev1008+ge2b5bbd3b`. It also brings unreleased changes to shared
mlx-lm code, for example a rewritten `TextStateMachine`, which the gateway's tool-call
splitter now uses. Switch back to the PyPI release once the PR is merged and released,
and re-run the tests and the checks below.

Example measurements on a 48 GB Apple Silicon Mac:

| Measurement | Value |
|---|---:|
| Load time | 4-6 s |
| Decode tok/s (bench median, min-max) | 38.2 (31.7-38.3) |
| Prompt tok/s, about 620-token prompt | 912 (696-914) |
| First token, cold | 0.691 s |
| First token, follow-up turn | 0.939 s, nothing cached |
| Peak memory, whole process | 20.1 GB |

From `mlxgw-bench --alias inkling`, 3 runs, 256 output tokens. The first request after a
load is slower: 7.6 s of prefill for a 27-token prompt.

What works and what does not:

- **Chat works**, streaming and not.
- **No tool calls.** The chat template marks calls with `<|content_invoke_tool_json|>`,
  and mlx-lm has no tool parser for it. A request with `tools` gets
  `400 tools_not_supported` (checked live). Do not list `inkling` as a tool model in
  OpenCode.
- **Reasoning and template markup appear in the content.** The template asks for
  thinking (effort 0.9 by default), and the reply includes it with the template's
  marker tokens, for example:
  `<|message_model|><|content_thinking|>The user asks: ...<|end_message|><|message_model|><|content_text|>red, yellow, blue<|end_message|>`.
  A client that sends this content back on the next turn sends the markup too.
- **No prompt cache reuse between turns.** The model's cache cannot be trimmed, so the
  gateway reuses a cache only when its tokens are a full prefix of the next request. The
  cache holds the generated thinking, which the next turn's rendered prompt does not
  contain, so `cached_tokens` is 0 on turn 2 (checked with a two-turn chat and in the
  benchmark). Every turn processes the whole conversation again.
- **Memory.** On a 48 GB Mac (budget 33.6 GiB), `inkling` and `coder` (16.0 GiB) fit
  together at 32.1 GiB, which leaves about 16 GiB of RAM for macOS and KV caches.
  Loading `fast` as well evicts the least recently used model.

## Tool calls with OpenCode

Run on a 48 GB Apple Silicon Mac with `fast` (Qwen3-4B) and OpenCode 2.3.2 (`opencode run`), in a
throwaway directory. All three tasks completed: read a file and report a value (glob,
read), create a file with given content (write), and change a value then run a script
(read, edit, shell). Follow-up turns served about 8,650 of 8,700 prompt tokens from the
cache. Round trips through the `openai` SDK, with one call and with two calls in one
response, streaming and not, also passed.

## Not yet verified

- **Tool calls with `coder`.** Qwen3-Coder uses mlx-lm's `qwen3_coder` (XML) parser,
  not the `json_tools` parser tested live. `coder` has not been downloaded.
- **Nothing has run on a 96 GB machine.** The `96gb` profile is tested with simulated
  hardware values only, and none of its large models has been loaded.
- **A real multi-GB download.** Progress reporting and cancelling mid-transfer are
  covered by tests with a fake downloader.
- **Remote TUI use** from a second machine.
- **Stopping the gateway while a generation is running.**
- **Ctrl-C during a benchmark** in a real terminal; the cleanup path has a unit test.
- **Eviction against the real budget.** Tested with a fake engine only.
- **Auth on a real non-loopback bind.** Tested in-process only.
