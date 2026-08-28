"""Regression tests for correcting a run's reference corpus without redoing task work."""
from __future__ import annotations

import numpy as np

import run
from idprobe import config
from idprobe.config import Config
from idprobe.intrinsic_dim import empty_sweep


def test_requested_corpus_tags_excludes_stale_corpora(tmp_path, monkeypatch):
    cfg = Config(model_id="example/model")
    cfg.corpora = ("bookcorpus",)
    monkeypatch.setattr(config, "ACTS", tmp_path)

    for tag in ("_corpus_bookcorpus_sane", "_corpus_bookcorpus_shuffled",
                "_corpus_pile_sane"):
        path = tmp_path / cfg.model_tag / tag / "train.npy"
        path.parent.mkdir(parents=True)
        path.touch()

    assert run._requested_corpus_tags(cfg, no_shuffled=False) == [
        "_corpus_bookcorpus_sane", "_corpus_bookcorpus_shuffled"
    ]
    assert run._requested_corpus_tags(cfg, no_shuffled=True) == [
        "_corpus_bookcorpus_sane"
    ]


def test_cached_task_sweep_round_trips_json_values():
    sweep = empty_sweep(16)
    curves = {
        f"task/{layer}": {
            "k": sweep["k"].tolist(),
            "id": (sweep["id"] + layer).tolist(),
            "err": sweep["err"].tolist(),
            "r": sweep["r"].tolist(),
            "n_unique": layer + 10,
        }
        for layer in range(3)
    }

    recovered = run._cached_task_sweep(curves, "task", n_layers=3, range_max=16)

    assert recovered is not None
    assert len(recovered) == 3
    assert np.array_equal(recovered[0]["k"], sweep["k"])
    assert recovered[2]["n_unique"] == 12


def test_cached_task_sweep_rejects_incompatible_grid():
    curves = {"task/0": {"k": [1, 2], "id": [1, 1], "err": [0, 0],
                          "r": [0, 0], "n_unique": 10}}
    assert run._cached_task_sweep(curves, "task", n_layers=1, range_max=16) is None
