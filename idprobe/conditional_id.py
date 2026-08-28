"""Class-conditional intrinsic dimension: entanglement measured in ID's own units.

The rest of the pipeline correlates a *global* statistic (manifold dimension over a
whole dataset) with a *per-feature* readout (one probe direction). Those live at
different levels of description, which is why the correlations are weak and need a
shift-permutation null to be interpretable at all. This module closes that gap by
measuring the feature's footprint as a dimension:

    DeltaID = ID(X) - mean_c ID(X | y = c)

If a feature is disentangled and linearly encoded it occupies roughly one direction,
so conditioning on it should remove about one dimension. If it is superposed across
many directions, conditioning collapses the manifold more diffusely and DeltaID is
larger. A feature the layer does not encode at all gives DeltaID ~ 0.

Two things make this measurement honest, and it is worthless without both:

1. **Matched sample size.** ID(X | y=c) sees ~n/C points while ID(X) sees n, and GRIDE
   is biased *downward* at small n. A naive DeltaID is therefore mostly a sample-size
   artefact. Every estimate here is taken at exactly `n_match` points.
2. **A label-shuffle null.** Permuting labels preserves the class sizes and the
   estimator bias but destroys the structure, so it calibrates what DeltaID looks like
   when there is nothing to find. We report DeltaID - DeltaID_shuffled, never raw
   DeltaID.

Note that (2) is also a self-check on (1). Once n is matched, a shuffled "class" is
just a uniform random subset of the pooled set, so the null's mean is analytically
zero. A `delta_null_mean` that drifts away from zero means the matching is broken,
not that a real effect appeared.

Pooled and conditional estimates always use the SAME k. Different k here would
reintroduce exactly the scale confound that `intrinsic_dim.scale_group` exists to
remove -- the difference would then partly measure the change in neighbourhood size.

WHAT DeltaID DOES NOT MEASURE
-----------------------------
It is not a separability measure, and reading it as one will mislead. GRIDE at scale
k looks at a neighbourhood of roughly 2k points, so if two classes are separated by
more than that neighbourhood's radius, every point's neighbours already belong to its
own class, conditioning changes nothing locally, and DeltaID is ~0 -- for perfectly
separated classes that any linear probe would nail. (Verified in the test suite: two
3-d clusters offset by 40 units give DeltaID = -0.03 at k=4.)

DeltaID answers a narrower question: does the feature consume a dimension WITHIN the
local neighbourhood structure? A large DeltaID means conditioning collapses the
manifold the estimator can actually see. A DeltaID near 0 is therefore ambiguous
between "the layer does not encode this" and "it encodes it at a scale coarser than
k", and only the probe accuracy at that layer distinguishes the two.

DeltaID IS NOT COMPARABLE ACROSS TASKS WITH DIFFERENT CLASS COUNTS
------------------------------------------------------------------
This is the sharpest limitation, and it constrains what the measurement can be used
to argue. A label can only remove the dimension it actually varies along, so the
attainable DeltaID is set by the label's cardinality (measured, in the test suite, on
synthetic data with a known answer):

    binary label, two clean clusters      DeltaID -> 0.0   (0.56 only when the
                                          classes heavily overlap, i.e. when the
                                          feature is HARDEST to decode)
    continuous coordinate in  5 bins      DeltaID ~ 0.36
    continuous coordinate in 10 bins      DeltaID ~ 0.52
    continuous coordinate in 20 bins      DeltaID ~ 0.88

So a binary task is structurally pinned near 0 whether or not its feature is
superposed, and a finely-binned task can approach 1 for purely geometric reasons.
Comparing `bigram_shift` (2 classes) against `sentence_length` (6) therefore compares
class counts at least as much as entanglement -- and would "confirm" a prediction of
"flat for the binary tasks, larger for sentence_length" no matter what the model does.

The label-shuffle null does NOT fix this. It holds cardinality fixed WITHIN a task,
which is what makes each task's own z-score valid; the confound lives in the
cross-task comparison. To compare tasks, match the class count first (e.g. threshold
sentence_length to two classes) and compare like with like.
"""
from __future__ import annotations

import numpy as np

from .intrinsic_dim import id_scaling


def id_at_scale(X: np.ndarray, k: int, seed: int = 0, n_jobs: int | None = 1) -> float:
    """GRIDE's ID estimate at one neighbourhood size `k`.

    Implemented by asking `id_scaling` for a sweep that *ends* at k (range_max = 2k)
    and taking its last entry. GRIDE reads each scale off the k-th and 2k-th
    neighbour distances independently, so this is identical to indexing k out of the
    full range_max=512 sweep -- but the sweep's "too few points" guard then trips at
    n <= 2k instead of n <= 512.

    That distinction is what makes this module possible at all: conditioning on a
    6-class label leaves a few hundred points per class, which the wide sweep would
    refuse outright while the estimate at k=2 is perfectly well defined.
    """
    if k < 1 or (k & (k - 1)):
        raise ValueError(f"k must be a power of two (GRIDE sweeps k=1,2,4,...); got {k}")
    # n_jobs=1 by default: these are many small estimates on matched-n subsets, where
    # DADApy's default of one worker per core is pure spawn overhead, and the stage is
    # meant to be parallelised across TASKS instead.
    return float(id_scaling(X, range_max=2 * k, n_points=None, seed=seed,
                            n_jobs=n_jobs)["id"][-1])


def _subsample(rng: np.random.Generator, pool: np.ndarray, n: int) -> np.ndarray:
    return rng.choice(pool, size=n, replace=False)


def conditional_id(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_match: int | None = None,
    n_boot: int = 5,
    seed: int = 0,
) -> dict:
    """ID pooled vs within-class at matched sample size.

    Returns {'id_pooled', 'id_per_class', 'id_within_mean', 'delta_id', 'n_matched',
    'n_classes', 'k'}.

    `n_match` defaults to the smallest class count, which is the largest size every
    class can supply without replacement. The pooled estimate is drawn to the same
    size -- that is the whole point, and taking it at the full n instead would make
    DeltaID a measure of how much data each estimate saw.

    `n_boot` independent subsamples are averaged. A single draw of a few hundred
    points is a noisy ID estimate, and the bootstrap spread is the only thing that
    distinguishes "conditioning removed a dimension" from "this draw ran low".
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    # Labels are written before the activation memmap is filled, so a task caught
    # mid-re-extraction has a new label file beside an old, shorter activation file.
    # Fancy-indexing would raise something obscure several frames down; say it here.
    if len(X) != len(y):
        raise ValueError(
            f"{len(X)} activation rows but {len(y)} labels. These come from different "
            f"extractions -- most likely `run.py extract` is still running for this "
            f"task, or was interrupted. Re-extract before estimating conditional ID."
        )
    classes = np.unique(y)
    by_class = {int(c): np.flatnonzero(y == c) for c in classes}

    smallest = min(len(v) for v in by_class.values())
    n_match = smallest if n_match is None else min(int(n_match), smallest)

    # 2k+1 points are needed for a 2k-th neighbour to exist at all. Returning NaN
    # beats returning a number GRIDE derived from a truncated neighbour list.
    if n_match <= 2 * k:
        return {"id_pooled": float("nan"), "id_per_class": {}, "id_within_mean": float("nan"),
                "delta_id": float("nan"), "n_matched": int(n_match),
                "n_classes": int(len(classes)), "k": int(k)}

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(X))
    pooled_runs: list[float] = []
    class_runs: dict[int, list[float]] = {c: [] for c in by_class}

    for b in range(n_boot):
        pooled_runs.append(id_at_scale(X[_subsample(rng, all_idx, n_match)], k, seed=seed + b))
        for c, idx in by_class.items():
            class_runs[c].append(id_at_scale(X[_subsample(rng, idx, n_match)], k, seed=seed + b))

    with np.errstate(invalid="ignore"):
        id_pooled = float(np.nanmean(pooled_runs))
        id_per_class = {c: float(np.nanmean(v)) for c, v in class_runs.items()}
        id_within_mean = float(np.nanmean(list(id_per_class.values())))
        # Spread of the pooled estimate across subsamples. This is the honest
        # power indicator: GRIDE at k=64 on 333 points has an sd of ~2 ID units,
        # which swamps any DeltaID worth reporting, and without this number a
        # near-zero z-score is indistinguishable from a real absence of effect.
        pooled_sd = float(np.nanstd(pooled_runs, ddof=1)) if n_boot > 1 else float("nan")
        # Spread ACROSS classes, not across subsamples: large values mean the
        # classes sit on manifolds of genuinely different dimension, so a single
        # mean_c ID summarises them poorly.
        class_sd = float(np.nanstd(list(id_per_class.values()), ddof=1)) \
            if len(id_per_class) > 1 else float("nan")

    return {
        "id_pooled": id_pooled,
        "id_pooled_sd": pooled_sd,
        "id_per_class": id_per_class,
        "id_within_mean": id_within_mean,
        "id_class_sd": class_sd,
        "delta_id": id_pooled - id_within_mean,
        "n_matched": int(n_match),
        "n_classes": int(len(classes)),
        "k": int(k),
    }


def conditional_id_null(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_perm: int = 20,
    seed: int = 0,
    **kw,
) -> dict:
    """The estimator-bias null: the same DeltaID with the labels permuted.

    Each permutation keeps the class sizes exactly and therefore inherits exactly the
    same small-n bias, while carrying no real structure. The spread across
    permutations is what turns a raw DeltaID into a z-score.

    Returns {'delta_null_mean', 'delta_null_sd', 'deltas', 'n_perm'}.
    """
    y = np.asarray(y)
    deltas = []
    for p in range(n_perm):
        # A separate stream per permutation, offset well clear of the observed run's
        # seeds so a permutation never reuses the observed subsample draws.
        perm_rng = np.random.default_rng([seed, p + 1])
        yp = perm_rng.permutation(y)
        deltas.append(conditional_id(X, yp, k, seed=seed + 1000 * (p + 1), **kw)["delta_id"])

    d = np.asarray(deltas, dtype=float)
    finite = d[np.isfinite(d)]
    return {
        "delta_null_mean": float(finite.mean()) if len(finite) else float("nan"),
        # ddof=1: this is a sample of permutations standing in for a distribution,
        # not the population itself.
        "delta_null_sd": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
        "deltas": d.tolist(),
        "n_perm": int(len(finite)),
    }


def residual_id(
    X: np.ndarray,
    directions: np.ndarray,
    k: int,
    n_match: int | None = None,
    n_boot: int = 5,
    seed: int = 0,
) -> dict:
    """ID before and after projecting out the trained probe direction(s).

    The complementary reading of the same question. If a feature really is one linear
    direction, deleting that direction should cost about one dimension. If deleting it
    costs much more, the probe direction was entangled with others and the model was
    not storing the feature in a private subspace.

    `directions` is [n_dirs, d]; its span is removed by an SVD projector truncated at
    numerical rank. Rank truncation rather than a plain QR because a probe's centred
    weight matrix is rank-deficient BY CONSTRUCTION -- centring C class rows leaves
    rank C-1 -- and QR would hand back C orthonormal columns regardless, the last of
    them pure rounding noise, deleting one more dimension than the probe actually
    uses and inflating the measured cost.

    Both estimates are taken in whatever space the caller passes in. Feed this the
    SAME standardised features the probe was trained on: the direction is only
    meaningful there, and per-feature standardisation is not a global rescaling, so
    GRIDE's scale-invariance does not paper over a mismatch.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.atleast_2d(np.asarray(directions, dtype=np.float64))

    U, sv, _ = np.linalg.svd(W.T, full_matrices=False)
    tol = sv.max() * max(W.shape) * np.finfo(float).eps if sv.size else 0.0
    rank = int((sv > tol).sum())
    Q = U[:, :rank]                            # [d, rank] orthonormal basis of the span
    X_res = X - (X @ Q) @ Q.T

    n_match = len(X) if n_match is None else min(int(n_match), len(X))
    if n_match <= 2 * k:
        return {"id_full": float("nan"), "id_residual": float("nan"),
                "id_drop": float("nan"), "n_dirs": int(len(W)), "n_matched": int(n_match)}

    rng = np.random.default_rng(seed)
    full, residual = [], []
    for b in range(n_boot):
        idx = _subsample(rng, np.arange(len(X)), n_match)
        full.append(id_at_scale(X[idx], k, seed=seed + b))
        residual.append(id_at_scale(X_res[idx], k, seed=seed + b))

    with np.errstate(invalid="ignore"):
        id_full, id_res = float(np.nanmean(full)), float(np.nanmean(residual))
    return {
        "id_full": id_full,
        "id_residual": id_res,
        "id_drop": id_full - id_res,
        "n_dirs": int(len(W)),
        "n_matched": int(n_match),
    }


__all__ = ["id_at_scale", "conditional_id", "conditional_id_null", "residual_id"]
