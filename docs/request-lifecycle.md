# Request lifecycle

## Life of a chat request

1. **Validate.** The body is checked against the supported OpenAI fields. Unknown
   fields, and unsupported ones such as `response_format`, get a 400 in the OpenAI error
   shape. Tool definitions, tool messages and assistant tool calls are normalised to the
   shape chat templates expect (call arguments as objects).
2. **Authorize and resolve.** A bearer token is required whenever the gateway listens
   beyond loopback. The model name must be an alias from this machine's profile. A model
   that is mid-download gets a 409.
3. **Queue.** The job joins the worker's FIFO queue. The event loop stays free.
4. **Get the model.** If it is resident, it moves to the front of the LRU order. If
   not, the worker evicts least recently used models until the new one fits the budget,
   then loads it. A model larger than the whole budget is refused with a 503.
5. **Find a cache.** The chat template turns the messages, and the tools if any, into
   tokens. The cache store returns the entry with the longest matching token prefix, so
   only the new tokens are processed.
6. **Generate.** Tokens stream from mlx-lm to the client as SSE chunks. With tools, the
   text first passes the tool-call splitter, which cuts out the call markup and parses
   each call; the stop-string matcher then applies to the remaining text. A client
   disconnect sets a cancel flag that the loop checks on every token and during prompt
   processing.
7. **Finish.** The last chunk carries usage and an `x_gateway` field with timings. The
   cache goes back to the store with the generated tokens added, and the request is
   recorded for the TUI.

## Prompt cache reuse

In chat and agent loops most of each prompt repeats. Keeping the KV cache for the shared
prefix is the biggest win for time to first token.

Example with the 0.5B test model on a 48 GB Apple Silicon Mac:

```
after turn 1, the cache entry holds 47 tokens

  [ 34 prompt tokens                ][ 13 generated ]

turn 2, the request is 61 prompt tokens

  [ 47 reused from cache                            ][ 14 new    ]
  0                                                 47           61
```

The second turn processed 14 tokens and produced its first token in 0.03 s.

Tool-call turns reuse the cache the same way. The chat template renders an earlier
call back into exactly the tokens the model generated for it, so the next request
extends the stored conversation completely. Example with `fast` (Qwen3-4B) driven by
OpenCode, whose system prompt and 13 tools come to about 8,650 tokens:

| Turn | Prompt tokens | Cached | First token |
|---|---:|---:|---:|
| 1, cold | 8,648 | 0 | 11.5 s |
| 2, after a tool result | 8,748 | 8,668 | 0.38 s |
| 3, after a tool result | 8,808 | 8,768 | 0.18 s |

Rules of the cache store (`PromptCacheStore` in `engine.py`):

- **Hand over, do not copy.** When a request fully extends a stored conversation, the
  cache is passed to the generation without copying. That is the normal multi-turn case.
- **Copy on a partial match.** When a request shares only part of a stored conversation,
  it gets a trimmed copy, and the original stays for its own conversation.
- **Bounded.** Each loaded model keeps the caches of its last 4 conversations
  (`prompt_cache_slots` in the config).
- **Models that cannot trim.** Some hybrid models cannot roll a cache back. They reuse a
  cache only on a full prefix match.
- **Checked before keeping.** After generation the cache's token offset must equal
  prompt plus generated tokens, or the cache is dropped.
- **Dropped on cancel during prompt processing**, because its state is partial.

## Stop strings and tool-call markup

Text that could be the start of a stop string is held back until it is disambiguated,
so a stop string is never partially streamed.

Tool-call markers are handled the same way, by mlx-lm's `TextStateMachine`: text that
could be the start of `<tool_call>` is held back, and text between the markers is
collected, not streamed. When the end marker arrives, the call is parsed with the
model's mlx-lm parser and sent as one `delta.tool_calls` chunk. Whitespace between and
after calls is dropped. A call still open when generation ends is parsed then, and
dropped with a warning if it does not parse.

## Cancellation

- **Streaming:** when the client disconnects, Starlette cancels the response and the
  cancel flag is set.
- **Non-streaming:** a watcher task waits for the disconnect message and sets the flag.
- **Queued:** a job whose client left while it was waiting is skipped.
