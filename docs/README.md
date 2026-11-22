# mlxgw design

How the gateway is built. For install and usage, see the [project README](../README.md).

mlxgw is a local gateway that serves MLX models through an OpenAI-compatible API.
It keeps models resident within a memory budget, reuses the prompt cache between turns,
and picks its models from the machine it starts on.

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | Processes, threads, and what each module does |
| [request-lifecycle.md](request-lifecycle.md) | The path of a chat request, and prompt-cache reuse |
| [models-and-memory.md](models-and-memory.md) | Machine profiles, the memory budget, model states |
| [api.md](api.md) | The `/v1` and `/admin` endpoints |
| [status.md](status.md) | Measured speeds, limits, and what is not yet verified |

At a glance:

- 2,668 lines of Python in 8 modules
- 3 core dependencies: mlx-lm, starlette, uvicorn (Textual is an optional extra for the TUI)
- 185 tests passing
- Built against mlx-lm 0.32.2
