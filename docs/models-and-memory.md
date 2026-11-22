# Models and memory

## Machine profiles

At startup the gateway reads RAM and chip from `sysctl`, picks the profile with the
largest `min_memory_gb` that fits, and sets the memory budget to 70% of RAM. The chosen
profile is logged.

| | `48gb` profile | `96gb` profile |
|---|---|---|
| Selected when RAM is | below 80 GB | 80 GB or more |
| Budget for weights | 33.6 GB | 67.2 GB |
| `coder` | Qwen3-Coder-30B-A3B 4-bit | Qwen3-Coder-Next 4-bit |
| `fast` | Qwen3-4B 4-bit | Qwen3-30B-A3B 4-bit |
| `tiny` | Qwen2.5-0.5B 4-bit | Qwen2.5-0.5B 4-bit |
| `inkling` | Inkling-2-Mini 4-bit | Inkling-2-Mini 4-bit |

Clients use the alias. The same request runs a larger model on a bigger Mac without any
client change. The model choices are starting defaults in `config.toml`; adjust them
there.

## Overrides

Environment variables win over the config file, and both win over detection.

| Variable | Effect |
|---|---|
| `MLXGW_PROFILE` | Force a profile instead of picking by RAM |
| `MLXGW_MEMORY_FRACTION` | Share of RAM used as the budget (default 0.70) |
| `MLXGW_MEMORY_BUDGET_GB` | Absolute budget; overrides the fraction |
| `MLXGW_HOST`, `MLXGW_PORT` | Bind address |
| `MLXGW_API_KEY` | Bearer token; required off loopback |
| `MLXGW_CONFIG` | Path to the config file |

## Model states

```
not downloaded --download--> downloading --done--> on disk
on disk --load or first request--> loaded
loaded --unload or LRU eviction--> on disk
```

- A chat request for a model that is not on disk downloads it inside that request.
- An explicit load from the TUI or admin API refuses a model that is not on disk.
- A chat or load request for a model that is mid-download gets a 409.

## Residency and eviction

- Models load on first use and stay resident.
- Size is estimated from the `.safetensors` weight files on disk.
- When a new model would exceed the budget, the least recently used models are evicted
  until it fits.
- A model larger than the whole budget is refused, and nothing is evicted for it.

## What the budget does not count

The budget covers weights only. Prompt caches (up to 4 conversations per model) and
working memory during generation come on top. Lower `memory_fraction` if memory
pressure shows up.

Which defaults can be resident together:

| Profile | Budget | Fits together | Does not fit |
|---|---:|---|---|
| `48gb` | 33.6 GB | `coder` + `inkling` (32.1 GB), with about 16 GB of RAM left for macOS, KV caches and scratch | all four (34.5 GB): loading the last one evicts the least recently used |
| `96gb` | 67.2 GB | `coder` + `fast` + `tiny` (58.0 GB) | adding `inkling` (74.1 GB): the least recently used model is evicted |

Sizes here are weights on disk in GiB, the unit the gateway uses for the budget.
`inkling` is 16.1 GiB (17.3 GB).
