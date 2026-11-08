from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_manifest() -> dict:
    return yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))


def test_manifest_declares_platform_and_tools():
    m = load_manifest()
    assert m["kind"] == "platform"
    assert m["name"] == "housemate"
    assert m["requires_hermes"] == ">=0.23.0"
    assert isinstance(m["version"], str)
    assert m["provides_tools"] == [
        "housemate_rooms",
        "housemate_announce",
        "housemate_timer_set",
        "housemate_alarm_set",
        "housemate_timer_list",
        "housemate_timer_cancel",
        "housemate_show",
    ]
    assert all(name.startswith("housemate_") for name in m["provides_tools"])
    assert (ROOT / "tools.py").is_file(), "provides_tools needs tools.py with register_tools(ctx)"


def test_pyproject_dependencies_have_upper_bounds():
    import tomllib

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert deps, "dependencies must be declared"
    for dep in deps:
        assert ">=" in dep and "<" in dep, f"{dep} needs a floor and an upper bound"


def test_manifest_declares_every_env_the_plugin_reads():
    """Rule 6 (review L3/M6): every env var Housemate reads is in optional_env."""
    import re

    declared = {e["name"] for e in load_manifest().get("optional_env", [])}
    used: set[str] = set()
    for path in [ROOT / "__init__.py", ROOT / "adapter.py", *(ROOT / "core").glob("*.py")]:
        used |= set(re.findall(r"[\"'](HOUSEMATE_[A-Z_]+)[\"']", path.read_text(encoding="utf-8")))
    assert used <= declared, f"undeclared: {sorted(used - declared)}"
    assert "HOUSEMATE_ALLOW_ALL_DEVICES" in declared
