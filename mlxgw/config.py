"""Configuration: TOML file + environment overrides + hardware profile."""

from __future__ import annotations

import logging
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("mlxgw")

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
GB = 1024**3


@dataclass(frozen=True)
class Hardware:
    chip: str
    memory_bytes: int

    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / GB


@dataclass
class Settings:
    host: str
    port: int
    api_key: str | None
    profile: str
    aliases: dict[str, str]  # alias -> Hugging Face repo or local path
    memory_budget_bytes: int
    prompt_cache_slots: int
    default_max_tokens: int
    hardware: Hardware

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, check=True
    ).stdout.strip()


def detect_hardware() -> Hardware:
    return Hardware(chip=_sysctl("machdep.cpu.brand_string"), memory_bytes=int(_sysctl("hw.memsize")))


def select_profile(profiles: dict, memory_bytes: int) -> str:
    """Pick the profile with the largest `min_memory_gb` that this machine satisfies."""
    mem_gb = memory_bytes / GB
    eligible = [
        (float(p.get("min_memory_gb", 0)), name)
        for name, p in profiles.items()
        if float(p.get("min_memory_gb", 0)) <= mem_gb
    ]
    if not eligible:
        raise ValueError(f"no profile fits {mem_gb:.0f} GB of memory")
    return max(eligible)[1]


def load_settings(
    path: str | Path,
    env: dict[str, str] | None = None,
    hardware: Hardware | None = None,
    host: str | None = None,
    port: int | None = None,
) -> Settings:
    """Load settings. Precedence: explicit args > environment > TOML file > defaults."""
    env = os.environ if env is None else env
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    server = raw.get("server", {})
    profiles = raw.get("profiles", {})
    if not profiles:
        raise ValueError(f"{path}: no [profiles.*] tables defined")

    hardware = hardware or detect_hardware()

    profile = env.get("MLXGW_PROFILE") or server.get("profile") or select_profile(profiles, hardware.memory_bytes)
    if profile not in profiles:
        raise ValueError(f"profile {profile!r} not in config (have: {', '.join(profiles)})")

    budget_gb = env.get("MLXGW_MEMORY_BUDGET_GB") or server.get("memory_budget_gb")
    if budget_gb is not None:
        budget = int(float(budget_gb) * GB)
    else:
        fraction = float(env.get("MLXGW_MEMORY_FRACTION") or server.get("memory_fraction", 0.70))
        if not 0 < fraction <= 1:
            raise ValueError(f"memory_fraction must be in (0, 1], got {fraction}")
        budget = int(hardware.memory_bytes * fraction)

    return Settings(
        host=host or env.get("MLXGW_HOST") or server.get("host", "127.0.0.1"),
        port=int(port or env.get("MLXGW_PORT") or server.get("port", 8080)),
        api_key=env.get("MLXGW_API_KEY") or server.get("api_key") or None,
        profile=profile,
        aliases=dict(profiles[profile].get("models", {})),
        memory_budget_bytes=budget,
        prompt_cache_slots=int(server.get("prompt_cache_slots", 4)),
        default_max_tokens=int(server.get("default_max_tokens", 4096)),
        hardware=hardware,
    )


def log_profile(s: Settings) -> None:
    log.info(
        "hardware: %s, %.0f GB RAM -> profile %r, model memory budget %.1f GB",
        s.hardware.chip, s.hardware.memory_gb, s.profile, s.memory_budget_bytes / GB,
    )
    for alias, repo in s.aliases.items():
        log.info("  alias %-8s -> %s", alias, repo)
