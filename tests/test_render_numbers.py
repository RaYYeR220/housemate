import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_numbers  # noqa: E402


def write_results(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "latency.json").write_text(json.dumps({"warm_runs": 20, "p50_s": 2.94, "p95_s": 3.61}), encoding="utf-8")
    (d / "wake.json").write_text(json.dumps({
        "positives": {"clean": {"n": 50, "detected": 48, "recall": 0.96}},
        "negatives": {"hours": 9.21, "false_accepts": 1, "fa_per_hour": 0.1086}}), encoding="utf-8")
    (d / "tests.json").write_text(json.dumps({"runs": [{"python": "3.14.9", "passed": 340, "skipped": 0, "failed": 0}]}),
                                  encoding="utf-8")


def test_render_replaces_markers(tmp_path):
    write_results(tmp_path / "results")
    text = (
        "Answers start <!-- num:latency_p50 -->?<!-- /num --> after you stop talking "
        "(p95 <!-- num:latency_p95 -->old<!-- /num -->, <!-- num:latency_runs --><!-- /num --> runs).\n"
        "Wake recall <!-- num:wake_recall -->x<!-- /num -->, "
        "<!-- num:wake_fa_per_hour -->y<!-- /num --> false wakes per hour of speech "
        "(<!-- num:wake_negative_hours -->z<!-- /num --> h). Tests: <!-- num:tests_passed -->0<!-- /num -->.\n"
    )
    numbers = render_numbers.load_numbers(tmp_path / "results")
    out, missing = render_numbers.render(text, numbers)
    assert missing == []
    assert "<!-- num:latency_p50 -->2.9 s<!-- /num -->" in out
    assert "<!-- num:latency_p95 -->3.6 s<!-- /num -->" in out
    assert "<!-- num:latency_runs -->20<!-- /num -->" in out
    assert "<!-- num:wake_recall -->96 %<!-- /num -->" in out
    assert "<!-- num:wake_fa_per_hour -->0.11<!-- /num -->" in out
    assert "<!-- num:wake_negative_hours -->9.2<!-- /num -->" in out
    assert "<!-- num:tests_passed -->340<!-- /num -->" in out
    # rendering again changes nothing
    assert render_numbers.render(out, numbers) == (out, [])


def test_render_reports_unknown_keys_and_leaves_them(tmp_path):
    out, missing = render_numbers.render("a <!-- num:nope -->keep<!-- /num --> b", {})
    assert out == "a <!-- num:nope -->keep<!-- /num --> b" and missing == ["nope"]


def test_main_check_mode_fails_on_stale_readme(tmp_path):
    write_results(tmp_path / "results")
    readme = tmp_path / "README.md"
    readme.write_text("p50 <!-- num:latency_p50 -->9.9 s<!-- /num -->\n", encoding="utf-8")
    args = ["--results", str(tmp_path / "results"), "--file", str(readme)]
    assert render_numbers.main(args + ["--check"]) == 1
    assert render_numbers.main(args) == 0
    assert "2.9 s" in readme.read_text(encoding="utf-8")
    assert render_numbers.main(args + ["--check"]) == 0
