# API

Any client or SDK that accepts a custom base URL works unchanged:

```
base_url = "http://127.0.0.1:8080/v1"
api_key  = "<token>"     # checked only when a token is configured
model    = "coder"       # alias, resolved per machine
```

## OpenAI-compatible endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, streaming or not |
| `GET /v1/models` | Aliases this machine serves |

Supported request fields: `model`, `messages`, `stream`,
`stream_options.include_usage`, `temperature`, `top_p`, `max_tokens`,
`max_completion_tokens`, `stop` (string or list), `seed`, `tools`, `tool_choice`.

Behaviour to know:

- Streaming uses server-sent events in the OpenAI chunk shape and ends with
  `data: [DONE]`.
- Responses include `usage`, with cached prompt tokens under `prompt_tokens_details`.
- Gateway timings (time to first token, tokens per second, load time, peak memory) are
  in an extra `x_gateway` field that standard clients ignore.
- Errors use the OpenAI error shape: `{"error": {"message", "type", "code", "param"}}`.
- Unknown fields get a 400. Recognised but unsupported fields (`response_format`, `n`
  above 1, `logprobs`, `reasoning_effort` other than `"none"`, ...) are accepted only at
  their no-op value.
- If both are sent, `max_completion_tokens` wins over `max_tokens`.
- `top_p = 0` means greedy decoding. Default `temperature` and `top_p` are 1.0.

## Tool calls

- `tools`: function tools only (`{"type": "function", "function": {name, description,
  parameters}}`). `function.strict` is accepted and ignored; other unknown keys get a 400.
- `tool_choice`: `"auto"` (the default) or `"none"`. With `"none"` the tools are not shown
  to the model and its output is not parsed. `"required"` and a named function get a 400:
  forcing a call needs constrained decoding.
- Messages can be `role: "tool"` with `tool_call_id`, and assistant messages can carry
  `tool_calls` (with `arguments` as a JSON string). Content can be a string or a list of
  text parts. Legacy `functions` / `function_call` are not supported.
- Tools go into the model's chat template. The tool-call markers and parser are the
  ones mlx-lm picks for that template (`json_tools` for Qwen2.5/Qwen3, `qwen3_coder` for
  Qwen3-Coder). A model for which mlx-lm has none gets a 400 `tools_not_supported`; when
  streaming, this arrives as an error event, because the model must be loaded to know.
- Responses: `message.tool_calls` with `id`, `type`, `function.name` and
  `function.arguments` (a JSON string), `content` null when there is no text, and
  `finish_reason: "tool_calls"`. When streaming, each call is one `delta.tool_calls`
  chunk with its `index`, sent once its end marker is generated. Markup never reaches
  `content`.
- A call that does not parse (invalid JSON, cut off by `max_tokens`) is dropped and
  logged as a warning. `stop` strings apply to the text, not to call arguments.

## Admin endpoints

Used by `mlxgw-tui`. Not part of the OpenAI contract.

| Endpoint | Purpose |
|---|---|
| `GET /admin/status` | Machine, budget, resident memory, queue, model states, last 50 requests |
| `GET /admin/logs?after=&limit=` | Last 500 log lines, with a sequence number for tailing |
| `POST /admin/models/{alias}/load` | Make a model resident now |
| `POST /admin/models/{alias}/unload` | Free a model |
| `POST /admin/models/{alias}/download` | Start a background download |
| `POST /admin/models/{alias}/download/cancel` | Cancel that download |

Conflict responses (409):

| Code | Meaning |
|---|---|
| `model_downloading` | Chat or load for a model that is mid-download |
| `already_downloading` | A second download of the same model |
| `already_downloaded` | Download of a model that is on disk |
| `not_downloaded` | Load of a model that is not on disk |
| `not_downloading` | Cancel when no download is running |

## Auth

Both prefixes share one bearer-token check.

- On loopback with no token configured, no token is needed.
- If a token is configured, it is enforced on loopback too.
- The gateway refuses to start on a non-loopback address without a token.

## Not in the contract yet

Structured output (`response_format`), `POST /v1/responses`, and embeddings.
