"""Thin wrapper around openWakeWord's official openwakeword/train.py.

It does four things and nothing else:
  1. resolves the recipe config (train_config.yaml) into an effective config for train.py:
     `custom_negative_repeat` (our key) repeats each curated near-miss phrase N times, because
     train.py feeds the phrase list to Piper in order, one clip per entry, so a phrase listed once
     yields one clip among ~25k adversarial negatives;
  2. `extra_positives` / `extra_negatives` stages (our config keys of the same names): voice additional
     clips with the same Piper generator and the same `generate_samples()` function train.py uses,
     but with unmixed speakers (`slerp_weights: [1.0]` walks all 904 LibriTTS-R speakers; train.py's
     own call mixes every clip 50/50 with one of the first few dozen speakers). Positives get
     punctuation variants for prosody; negatives are the near-misses that fired in evaluation. Clips go
     into the same positive_/negative_ train/test folders (prefixes `x_` / `xn_`), before train.py's
     generate step tops the folders up;
  3. `import_clips` stage (config key `import_clips`): copies externally made clips (e.g. the Edge TTS
     training voices from make_eval_corpus.py --split train) into those folders, `copies` times each
     (prefix `c<k>_`), so each copy gets its own random augmentation; `holdout_fraction` of the source
     clips go to the matching *_test folder instead (checkpoint selection);
  4. runs train.py with the requested stages (generate_clips, augment_clips, train_model).

  python /recipe/oww_train.py --config /recipe/train_config.yaml \
      extra_positives extra_negatives import_clips generate_clips augment_clips train_model
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

TRAIN_PY = "/opt/openwakeword/openwakeword/train.py"
STAGES = ["extra_positives", "extra_negatives", "import_clips", "generate_clips", "augment_clips", "train_model"]


def extra_clips(cfg: dict, key: str, kind: str, prefix: str) -> None:
    spec = cfg.get(key)
    if not spec:
        print(f"{key}: not configured, skipping", flush=True)
        return
    sys.path.insert(0, os.path.abspath(cfg["piper_sample_generator_path"]))
    from generate_samples import generate_samples  # piper-sample-generator v2.0.0, as train.py does

    base = Path(cfg["output_dir"]) / cfg["model_name"]
    for split, n in ((f"{kind}_train", spec["n_train"]), (f"{kind}_test", spec["n_test"])):
        out = base / split
        out.mkdir(parents=True, exist_ok=True)
        have = sum(1 for f in out.iterdir() if f.name.startswith(prefix))
        if have >= 0.95 * n:
            print(f"{key}: {split} already has {have} {prefix} clips", flush=True)
            continue
        generate_samples(
            text=spec["texts"], max_samples=n - have,
            batch_size=cfg["tts_batch_size"] if kind == "positive" else max(1, cfg["tts_batch_size"] // 7),
            slerp_weights=spec.get("slerp_weights", [1.0]),
            noise_scales=spec.get("noise_scales", [0.98]), noise_scale_ws=spec.get("noise_scale_ws", [0.98]),
            length_scales=spec.get("length_scales", [0.75, 1.0, 1.25]),
            output_dir=str(out), auto_reduce_batch_size=True,
            file_names=[f"{prefix}{uuid.uuid4().hex}.wav" for _ in range(n)],
        )


def import_clips(cfg: dict) -> None:
    base = Path(cfg["output_dir"]) / cfg["model_name"]
    for spec in cfg.get("import_clips") or []:
        src, kind = Path(spec["src"]), spec["kind"]  # kind: positive | negative
        copies, hold = int(spec.get("copies", 1)), float(spec.get("holdout_fraction", 0.0))
        files = sorted(src.glob("*.wav"))
        n_train = n_test = 0
        for f in files:
            to_test = int(hashlib.sha1(f.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < hold
            dests = [(base / f"{kind}_test" / f"c0_{f.name}")] if to_test else \
                [(base / f"{kind}_train" / f"c{k}_{f.name}") for k in range(copies)]
            for d in dests:
                d.parent.mkdir(parents=True, exist_ok=True)
                if not d.exists():
                    shutil.copyfile(f, d)
            n_test += to_test
            n_train += 0 if to_test else copies
        print(f"import_clips: {src} -> {kind}_train {n_train} files ({copies} copies), {kind}_test {n_test}",
              flush=True)


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
        extra_clips(cfg, "extra_positives", "positive", "x_")
    if "extra_negatives" in a.stages:
        extra_clips(cfg, "extra_negatives", "negative", "xn_")
    if "import_clips" in a.stages:
        import_clips(cfg)
    stages = [s for s in a.stages if s in ("generate_clips", "augment_clips", "train_model")]
    if not stages:
        return

    eff = out / "effective_config.yaml"
    ours = ("extra_positives", "extra_negatives", "import_clips")
    eff.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k not in ours}, sort_keys=False))
    cmd = [sys.executable, TRAIN_PY, "--training_config", str(eff)] + [f"--{s}" for s in stages]
    if a.overwrite:
        cmd.append("--overwrite")
    print("+", " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
