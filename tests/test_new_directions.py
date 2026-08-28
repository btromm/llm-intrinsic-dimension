"""Tests for the NEXT_STEPS directions 1 and 2.

Each test builds a case whose right answer is known by construction, so a failure
points at the code rather than at the data -- which matters most for the conditional-ID
guards, where a plausible-looking number is the failure mode rather than a crash.

    conda run -n id python -m pytest tests/test_new_directions.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from idprobe.conditional_id import conditional_id, conditional_id_null, id_at_scale, residual_id
from idprobe.stats import classify_layer


# ------------------------------------------------------- Direction 1: ID at scale
def test_id_at_scale_matches_the_full_sweep():
    """A narrow sweep ending at k must give the same estimate as a wide one at k.

    This is the whole basis for conditional ID: it is what lets a few-hundred-point
    class be estimated at the same k as the pooled set, which the wide sweep would
    refuse outright.
    """
    from idprobe.intrinsic_dim import id_scaling

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 40))
    full = id_scaling(X, 512, n_points=None, seed=0)
    for k in (2, 4, 16, 64):
        expected = full["id"][list(full["k"]).index(k)]
        assert id_at_scale(X, k, seed=0) == pytest.approx(expected, rel=1e-9)


def test_id_at_scale_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        id_at_scale(np.random.default_rng(0).normal(size=(100, 5)), 3)


def test_id_at_scale_recovers_a_known_dimension():
    """A 5-d plane embedded in 50-d should read as ~5 dimensions."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(3000, 5))
    X = latent @ rng.normal(size=(5, 50))
    assert id_at_scale(X, 16, seed=0) == pytest.approx(5.0, abs=0.6)


# --------------------------------------------------- Direction 1: conditional ID
def test_conditional_id_matches_sample_size():
    """Every estimate must be taken at the smallest class's size, never at n."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(900, 20))
    y = np.array([0] * 600 + [1] * 300)
    out = conditional_id(X, y, k=4, n_boot=2, seed=0)
    assert out["n_matched"] == 300
    assert out["n_classes"] == 2


def test_conditional_id_returns_nan_when_classes_are_too_small():
    """Fewer than 2k+1 points cannot support a 2k-th neighbour; say so, don't guess."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 10))
    y = np.arange(60) % 6              # 10 per class, far below 2k = 128
    out = conditional_id(X, y, k=64, n_boot=2, seed=0)
    assert np.isnan(out["delta_id"])


def test_label_shuffle_null_is_centred_on_zero():
    """The null's mean is analytically 0 once n is matched, so this checks the matching.

    With classes of equal size, a permuted class is just a uniform random subset of
    the pooled set -- the same distribution the pooled estimate is drawn from. A null
    mean that drifts away from 0 means the sample-size matching is broken, which is
    exactly the failure that would make DeltaID a bias artefact.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 15))    # no class structure at all
    y = np.arange(1200) % 2
    null = conditional_id_null(X, y, k=4, n_perm=12, n_boot=3, seed=0)
    sem = null["delta_null_sd"] / np.sqrt(null["n_perm"])
    assert abs(null["delta_null_mean"]) < 4 * sem


def _binned_latent(n=3000, d=30, n_latent=4, n_bins=20, seed=0):
    """A continuous coordinate binned into classes -- sentence_length's structure."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    latent = rng.normal(size=(n, n_latent))
    X = latent @ basis[:n_latent]
    cuts = np.quantile(latent[:, -1], np.linspace(0, 1, n_bins + 1)[1:-1])
    return X, np.digitize(latent[:, -1], cuts)


def test_conditional_id_detects_a_binned_continuous_coordinate():
    """The positive control: conditioning on a finely-binned coordinate costs ~1 dim.

    A 4-d cloud whose last coordinate is binned into 20 classes. Conditioning on the
    bin pins that coordinate, leaving ~3 dimensions, so DeltaID ~ 1 and sits well
    clear of the label-shuffle null.

    This is the ONLY geometry in which the plan's "removes about one dimension"
    intuition holds. See the two tests below for the cases where it does not.
    """
    X, y = _binned_latent(n_bins=20)
    obs = conditional_id(X, y, k=4, n_boot=3, seed=0)
    null = conditional_id_null(X, y, k=4, n_perm=10, n_boot=3, seed=0)
    z = (obs["delta_id"] - null["delta_null_mean"]) / null["delta_null_sd"]
    assert obs["delta_id"] == pytest.approx(0.9, abs=0.4)
    assert z > 3


def test_delta_id_scales_with_the_number_of_bins_not_only_the_structure():
    """The cardinality confound, on data whose underlying structure is IDENTICAL.

    Same latent cloud, same encoded coordinate, only the bin count changes -- and
    DeltaID rises with it. Any cross-task comparison of DeltaID that does not match
    class counts is partly reading this.
    """
    deltas = [conditional_id(*_binned_latent(n_bins=c), k=4, n_boot=3, seed=0)["delta_id"]
              for c in (5, 20)]
    assert deltas[0] < deltas[1] - 0.2


def test_conditional_id_is_blind_to_separation_beyond_the_neighbourhood_scale():
    """A binary label on well-separated clusters gives DeltaID ~ 0.

    Trivially separable -- any linear probe would score ~100% -- yet DeltaID reports
    nothing, because GRIDE at scale k never looks past its own neighbourhood. This is
    why a near-zero DeltaID on the binary tasks is not evidence of disentanglement.
    """
    rng = np.random.default_rng(0)
    n, d = 800, 30
    a = rng.normal(size=(n, 3)) @ rng.normal(size=(3, d))
    b = rng.normal(size=(n, 3)) @ rng.normal(size=(3, d)) + 40.0
    X = np.vstack([a, b])
    y = np.array([0] * n + [1] * n)
    assert conditional_id(X, y, k=4, n_boot=3, seed=0)["delta_id"] == pytest.approx(0.0, abs=0.3)


def test_residual_id_drops_about_one_dimension_for_one_real_direction():
    """Deleting a direction the data actually occupies should cost about 1 dimension."""
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(25, 25)))[0][:6]   # 6 orthonormal directions
    X = rng.normal(size=(2500, 6)) @ basis
    out = residual_id(X, basis[:1], k=4, n_boot=3, seed=0)
    assert out["id_drop"] == pytest.approx(1.0, abs=0.6)


def test_residual_id_removes_a_span_not_a_row_count():
    """A rank-deficient direction set must cost its RANK, not its row count.

    This is the realistic case, not a corner one: a probe's centred weight matrix has
    C rows spanning only C-1 dimensions, so a projector that trusted the row count
    would always delete one dimension too many.
    """
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(25, 25)))[0][:6]
    X = rng.normal(size=(2500, 6)) @ basis
    duplicated = np.vstack([basis[0], basis[0], 2.0 * basis[0]])   # rank 1, three rows
    out = residual_id(X, duplicated, k=4, n_boot=3, seed=0)
    assert out["id_drop"] == pytest.approx(1.0, abs=0.6)


# --------------------------------------------------------- Direction 2: verdicts
def _mc(n01: int, n10: int, n: int, gap: float) -> dict:
    return {"n_mlp_only": n01, "n_linear_only": n10, "n_total": n, "acc_gap": gap}


def test_monotonic_encoding_requires_an_ordinal_probe_that_closes_the_gap():
    significant = _mc(n01=300, n10=100, n=10_000, gap=0.02)
    # Ordinal vs MLP: almost no discordance, so the MLP's remaining edge is tiny.
    ordinal_closes = _mc(n01=52, n10=50, n=10_000, gap=0.0002)
    assert classify_layer(significant, q_value=0.001) == "nonlinear_advantage"
    assert classify_layer(significant, q_value=0.001,
                          mc_ordinal=ordinal_closes) == "monotonic_encoding"


def test_ordinal_probe_that_does_not_close_the_gap_leaves_the_verdict_alone():
    significant = _mc(n01=300, n10=100, n=10_000, gap=0.02)
    ordinal_fails = _mc(n01=400, n10=100, n=10_000, gap=0.03)
    assert classify_layer(significant, q_value=0.001,
                          mc_ordinal=ordinal_fails) == "nonlinear_advantage"


def test_existing_verdicts_are_unchanged_without_an_ordinal_probe():
    """The new argument must not perturb the verdicts already in the results."""
    tight = _mc(n01=50, n10=50, n=100_000, gap=0.0)
    assert classify_layer(tight, q_value=0.9) == "linear_sufficient"
    assert classify_layer(_mc(20, 15, 200, 0.025), q_value=0.9) == "inconclusive"
