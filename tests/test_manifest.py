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
    assert isinstance(m["provides_tools"], list)


def test_pyproject_dependencies_have_upper_bounds():
    import tomllib

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert deps, "dependencies must be declared"
    for dep in deps:
        assert ">=" in dep and "<" in dep, f"{dep} needs a floor and an upper bound"
