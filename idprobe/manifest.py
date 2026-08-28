"""Record exactly how a set of results was produced, so they can be reused later.

Results outlive the session that made them. Without a manifest, a `results/`
directory is a set of numbers with no record of the sample sizes, GRIDE scales,
or package versions behind them -- and this project has already been bitten once
by activations from two different extractions being mixed silently.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "dadapy", "numpy", "sklearn", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "unavailable"
    return out


def write_manifest(cfg, timestamp: str, extra: dict | None = None) -> Path:
    """Write results/<model>/run_manifest.json describing this run."""
    rdir = cfg.results_dir
    rdir.mkdir(parents=True, exist_ok=True)

    acts = {}
    root = Path("activations") / cfg.model_tag
    if root.exists():
        for d in sorted(root.glob("*")):
            f = d / "train.npy"
            if f.exists():
                acts[d.name] = list(np.load(f, mmap_mode="r").shape)

    scales = {}
    t = rdir / "table_c1_scales.csv"
    if t.exists():
        import pandas as pd
        df = pd.read_csv(t)
        scales = dict(zip(df.tag, df.k.astype(int)))

    manifest = {
        "model_id": cfg.model_id,
        "written_at": timestamp,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "gride_scales": scales,
        "activation_shapes": acts,
        "versions": _versions(),
        **(extra or {}),
    }
    path = rdir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


__all__ = ["write_manifest"]
