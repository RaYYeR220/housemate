"""Thin wrapper around openWakeWord's official openwakeword/train.py.

It does three things and nothing else:
  1. resolves the recipe config (train_config.yaml) into an effective config for train.py:
     `custom_negative_repeat` (our key) repeats each curated near-miss phrase N times, because
     train.py feeds the phrase list to Piper in order, one clip per entry, so a phrase listed once
     yields one clip among ~25k adversarial negatives;
  2. `extra_positives` stage (our key `extra_positives` in the config): voices additional positive
     clips with the same Piper generator and the same `generate_samples()` function train.py uses,
     but with unmixed speakers (`slerp_weights: [1.0]` walks all 904 LibriTTS-R speakers; train.py's
     own call mixes every clip 50/50 with one of the first few dozen speakers) and with punctuation
     variants for prosody. The clips go into the same positive_train/positive_test folders, prefixed
     `x_`, before train.py's generate step tops the folders up;
  3. runs train.py with the requested stages (generate_clips, augment_clips, train_model).

  python /recipe/oww_train.py --config /recipe/train_config.yaml \
      extra_positives generate_clips augment_clips train_model
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

TRAIN_PY = "/opt/openwakeword/openwakeword/train.py"
STAGES = ["extra_positives", "generate_clips", "augment_clips", "train_model"]


def extra_positives(cfg: dict) -> None:
    spec = cfg.get("extra_positives")
    if not spec:
        print("extra_positives: not configured, skipping", flush=True)
        return
    sys.path.insert(0, os.path.abspath(cfg["piper_sample_generator_path"]))
    from generate_samples import generate_samples  # piper-sample-generator v2.0.0, as train.py does

    base = Path(cfg["output_dir"]) / cfg["model_name"]
    for split, n in (("positive_train", spec["n_train"]), ("positive_test", spec["n_test"])):
        out = base / split
        out.mkdir(parents=True, exist_ok=True)
        have = sum(1 for f in out.iterdir() if f.name.startswith("x_"))
        if have >= 0.95 * n:
            print(f"extra_positives: {split} already has {have} x_ clips", flush=True)
            continue
        generate_samples(
            text=spec["texts"], max_samples=n - have, batch_size=cfg["tts_batch_size"],
            slerp_weights=spec.get("slerp_weights", [1.0]),
            noise_scales=spec.get("noise_scales", [0.98]), noise_scale_ws=spec.get("noise_scale_ws", [0.98]),
            length_scales=spec.get("length_scales", [0.75, 1.0, 1.25]),
            output_dir=str(out), auto_reduce_batch_size=True,
            file_names=[f"x_{uuid.uuid4().hex}.wav" for _ in range(n)],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/recipe/train_config.yaml", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="recompute features in augment_clips")
    ap.add_argument("stages", nargs="+", choices=STAGES)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    rep = int(cfg.pop("custom_negative_repeat", 1))
    phrases = list(cfg.get("custom_negative_phrases") or [])
    cfg["custom_negative_phrases"] = phrases * rep  # interleaved: p1 p2 ... pk p1 p2 ...

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if "extra_positives" in a.stages:
        extra_positives(cfg)
    stages = [s for s in a.stages if s != "extra_positives"]
    if not stages:
        return

    eff = out / "effective_config.yaml"
    eff.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "extra_positives"}, sort_keys=False))
    cmd = [sys.executable, TRAIN_PY, "--training_config", str(eff)] + [f"--{s}" for s in stages]
    if a.overwrite:
        cmd.append("--overwrite")
    print("+", " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
