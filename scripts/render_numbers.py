"""Fill the numbers in README.md from results/*.json.

Every measured number in the docs sits between markers, so it is never typed by hand::

    Answers start <!-- num:latency_p50 -->2.9 s<!-- /num --> after you stop talking.

``python scripts/render_numbers.py`` rewrites the text between each pair from ``results/``;
``--check`` only reports whether the file is up to date (exit 1 if not). Unknown keys are
listed and left as they are.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"(<!-- num:(?P<key>[a-z0-9_]+) -->)(?P<old>.*?)(<!-- /num -->)", re.S)


def _get(data: dict, path: str) -> Any:
    for part in path.split("."):
        if isinstance(data, list):
            data = data[int(part)]
        else:
            data = data[part]
    return data


# key -> (results file, path inside it, formatter)
KEYS: dict[str, tuple[str, str, Callable[[Any], str]]] = {
    "latency_p50": ("latency.json", "p50_s", lambda v: f"{v:.1f} s"),
    "latency_p95": ("latency.json", "p95_s", lambda v: f"{v:.1f} s"),
    "latency_runs": ("latency.json", "warm_runs", lambda v: f"{v}"),
    "wake_recall": ("wake.json", "positives.clean.recall", lambda v: f"{v * 100:.0f} %"),
    "wake_recall_n": ("wake.json", "positives.clean.n", lambda v: f"{v}"),
    "wake_recall_noisy": ("wake.json", "positives.speech_10db.recall", lambda v: f"{v * 100:.0f} %"),
    "wake_fa_per_hour": ("wake.json", "negatives.fa_per_hour", lambda v: f"{v:.2f}"),
    "wake_false_accepts": ("wake.json", "negatives.false_accepts", lambda v: f"{v}"),
    "wake_negative_hours": ("wake.json", "negatives.hours", lambda v: f"{v:.1f}"),
    "tests_passed": ("tests.json", "runs.0.passed", lambda v: f"{v}"),
    "hermes_version": ("latency.json", "environment.hermes_version", lambda v: f"{v}"),
}


def load_numbers(results_dir: Path | str) -> dict[str, str]:
    """Every key whose results file and field exist, formatted for prose."""
    results_dir = Path(results_dir)
    files: dict[str, dict] = {}
    numbers: dict[str, str] = {}
    for key, (name, path, fmt) in KEYS.items():
        if name not in files:
            try:
                files[name] = json.loads((results_dir / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                files[name] = {}
        try:
            value = _get(files[name], path)
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if value is not None:
            numbers[key] = fmt(value)
    return numbers


def render(text: str, numbers: dict[str, str]) -> tuple[str, list[str]]:
    """``text`` with every known marker filled in, and the keys that had no number."""
    missing: list[str] = []

    def fill(m: re.Match) -> str:
        key = m.group("key")
        if key not in numbers:
            missing.append(key)
            return m.group(0)
        return f"{m.group(1)}{numbers[key]}{m.group(4)}"

    return MARKER.sub(fill, text), missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill <!-- num:key --> markers from results/*.json")
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--file", action="append", help="file to render (default README.md); repeatable")
    parser.add_argument("--check", action="store_true", help="only check; exit 1 if a file is out of date")
    args = parser.parse_args(argv)
    numbers = load_numbers(args.results)
    stale = False
    for name in args.file or [str(ROOT / "README.md")]:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        out, missing = render(text, numbers)
        for key in missing:
            print(f"{path.name}: no number for '{key}'", file=sys.stderr)
        if out != text:
            if args.check:
                print(f"{path.name}: numbers are out of date", file=sys.stderr)
                stale = True
            else:
                path.write_bytes(out.encode("utf-8"))
                print(f"{path.name}: numbers updated")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
