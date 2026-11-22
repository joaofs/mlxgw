from pathlib import Path

import pytest

from mlxgw.config import GB, Hardware, load_settings, select_profile

CONFIG = Path(__file__).parent.parent / "config.toml"
MAC_48 = Hardware("Apple M-series", 48 * GB)
MAC_96 = Hardware("Apple M-series", 96 * GB)


def settings(hw, env=None, **kw):
    return load_settings(CONFIG, env=env or {}, hardware=hw, **kw)


def test_profile_selection_by_memory():
    profiles = {"48gb": {"min_memory_gb": 0}, "96gb": {"min_memory_gb": 80}}
    assert select_profile(profiles, 48 * GB) == "48gb"
    assert select_profile(profiles, 64 * GB) == "48gb"
    assert select_profile(profiles, 96 * GB) == "96gb"
    assert select_profile(profiles, 192 * GB) == "96gb"
    with pytest.raises(ValueError):
        select_profile({"big": {"min_memory_gb": 64}}, 48 * GB)


def test_48gb_machine_uses_48gb_profile_and_budget():
    s = settings(MAC_48)
    assert s.profile == "48gb"
    assert s.memory_budget_bytes == int(48 * GB * 0.70)
    assert s.aliases["coder"] == "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"


def test_96gb_machine_resolves_same_aliases_to_bigger_models():
    small, big = settings(MAC_48), settings(MAC_96)
    assert big.profile == "96gb"
    assert big.memory_budget_bytes == int(96 * GB * 0.70)
    assert set(small.aliases) == set(big.aliases)
    assert big.aliases["coder"] != small.aliases["coder"]
    assert big.aliases["fast"] != small.aliases["fast"]


def test_env_overrides_win():
    s = settings(MAC_48, env={"MLXGW_PROFILE": "96gb", "MLXGW_MEMORY_FRACTION": "0.5",
                              "MLXGW_HOST": "0.0.0.0", "MLXGW_PORT": "9000", "MLXGW_API_KEY": "k"})
    assert (s.profile, s.memory_budget_bytes, s.host, s.port, s.api_key) == \
        ("96gb", int(48 * GB * 0.5), "0.0.0.0", 9000, "k")
    assert not s.is_loopback
    assert settings(MAC_48, env={"MLXGW_MEMORY_BUDGET_GB": "20"}).memory_budget_bytes == 20 * GB


def test_explicit_args_beat_env():
    s = settings(MAC_48, env={"MLXGW_PORT": "9000"}, port=7000, host="::1")
    assert (s.port, s.host, s.is_loopback) == (7000, "::1", True)


def test_defaults_are_loopback_8080():
    s = settings(MAC_48)
    assert (s.host, s.port, s.is_loopback, s.api_key) == ("127.0.0.1", 8080, True, None)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        settings(MAC_48, env={"MLXGW_PROFILE": "nope"})
