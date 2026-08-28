"""Intrinsic-dimension estimation via DADApy.

Cheng et al. (2025) estimate ID with GRIDE (Denti et al. 2022), the estimator
implemented in DADApy by the same group. We call it directly rather than
reimplementing it, so the numbers are comparable to the paper by construction.

GRIDE works from the ratio of the 2k-th to the k-th nearest-neighbour distance.
k sets the *scale*: small k probes fine local structure and is noise-sensitive,
large k averages over a wider neighbourhood and washes out curvature.
`return_id_scaling_gride` sweeps k = 1, 2, 4, ... range_max/2 in one pass.

Because the estimator depends only on distance *ratios*, it is invariant to any
global rescaling of the representations -- do not standardise before calling it.
"""
from __future__ import annotations

import numpy as np
from dadapy import Data


def id_scaling(
    X: np.ndarray,
    range_max: int = 512,
    n_points: int | None = 10_000,
    seed: int = 0,
    n_jobs: int | None = None,
) -> dict[str, np.ndarray]:
    """Run GRIDE across scales. Returns {'k','id','err','r','n_unique'}.

    Duplicate rows are dropped first: identical points give a zero k-th neighbour
    distance, making the ratio undefined. This is not a corner case -- at layer 0
    of a task dataset it is the norm. Layer 0 is the raw token embedding of the
    LAST token, and SentEval sentences overwhelmingly end in the same punctuation,
    so a 2000-sentence split can collapse to ~5 distinct vectors.

    `n_jobs` caps DADApy's internal parallelism (it otherwise grabs every core);
    leave it None for a single-process sweep, set it to 1 when running one process
    per task.

    The returned grid is ALWAYS `log2(range_max)` scales wide. A layer with too
    few unique points to support the sweep returns NaN rather than a silently
    truncated grid measured at different scales from its neighbours -- that
    truncation corrupts any cross-layer comparison built on top of it.
    """
    n_scales = int(np.log2(range_max))
    ks = 2 ** np.arange(n_scales)
    nan = np.full(n_scales, np.nan)

    X = np.unique(np.asarray(X, dtype=np.float64), axis=0)
    n_unique = len(X)
    rng = np.random.default_rng(seed)
    if n_points is not None and len(X) > n_points:
        X = X[rng.permutation(len(X))[:n_points]]

    if len(X) <= range_max:
        return {"k": ks, "id": nan, "err": nan, "r": nan, "n_unique": n_unique}

    # DADApy hardcodes n_jobs to the machine's core count, which is right for one
    # sweep over a big matrix and badly wrong when many processes each run many
    # small sweeps: 4 concurrent callers then spawn 40 workers on 10 cores and
    # everything slows to a crawl. `n_jobs=None` keeps DADApy's default.
    data = Data(X, verbose=False) if n_jobs is None else Data(X, verbose=False, n_jobs=n_jobs)
    ids, errs, rs = data.return_id_scaling_gride(range_max=range_max)
    pad = lambda a: np.pad(np.asarray(a, float), (0, max(0, n_scales - len(a))),
                           constant_values=np.nan)[:n_scales]
    return {"k": ks, "id": pad(ids), "err": pad(errs), "r": pad(rs), "n_unique": n_unique}


def empty_sweep(range_max: int = 512) -> dict[str, np.ndarray]:
    """A NaN-filled sweep with the correct grid, for layers we deliberately skip.

    Keeping a same-shaped placeholder means positions in a profile still equal
    layer numbers, so every downstream consumer can index by layer without
    tracking an offset.
    """
    n = int(np.log2(range_max))
    nan = np.full(n, np.nan)
    return {"k": 2 ** np.arange(n), "id": nan, "err": nan, "r": nan, "n_unique": 0}


def select_plateau(scaling: dict[str, np.ndarray], window: int = 3) -> tuple[int, float]:
    """Locate the plateau in one layer's scale sweep (Cheng et al. Appendix C).

    Their reasoning: noise dominates small scales, while curvature and density
    variation dominate large ones, so the trustworthy estimate sits in an
    intermediate region where ID is stable across neighbourhood sizes. They pick
    that region by eye; we pick the sliding window of `window` consecutive scales
    with the smallest relative spread, which is the same judgement made numerically.

    Returns (centre index of the flattest window, its relative spread). A spread
    near 0 means a genuine plateau; a large spread means the curve never flattens
    and the ID estimate is scale-dependent at every scale.
    """
    ids = np.asarray(scaling["id"], float)
    if len(ids) < window:
        return len(ids) // 2, float("inf")
    best, best_spread = len(ids) // 2, float("inf")
    for i in range(len(ids) - window + 1):
        w = ids[i : i + window]
        if not np.all(np.isfinite(w)) or w.mean() <= 0:
            continue
        spread = (w.max() - w.min()) / w.mean()
        if spread < best_spread:
            best, best_spread = i + window // 2, spread
    return best, best_spread


def choose_common_scale(
    curves: list[dict[str, np.ndarray]], tol: float = 0.10
) -> tuple[int, float, bool]:
    """Pick ONE scale index for every layer of a group (Cheng et al. App. C).

    "For simplicity, per model-corpus combination, we choose one scale for all
    layers." This matters: comparing ID across layers is only meaningful if every
    layer is measured at the same neighbourhood size, and selecting a per-layer
    plateau silently compares estimates taken at different k, producing a sawtooth
    that is an artefact of the selection rather than a property of the model.

    Returns (index, median spread, plateau_found). The scale always comes from
    the sweep: when enough layers plateau (relative spread < `tol`) it is their
    median plateau, and otherwise it is the median flattest window across layers.
    There is deliberately no constant fallback -- a hard-coded k would silently
    override what the data says, and the caller pins the result in scales.json
    so the choice is made once per (model, corpus) rather than re-derived.
    """
    usable = [c for c in curves if np.any(np.isfinite(c["id"]))]
    if not usable:
        return 0, float("inf"), False
    picks = [select_plateau(c) for c in usable]
    spreads = [sp for _, sp in picks]
    median_spread = float(np.median([sp for sp in spreads if np.isfinite(sp)] or [np.inf]))

    good = [i for i, spread in picks if spread < tol]
    if len(good) >= max(2, len(picks) // 2):
        return int(np.median(good)), median_spread, True
    return int(np.median([i for i, _ in picks])), median_spread, False


def scale_group(tag: str, reference_corpus: str = "bookcorpus") -> str:
    """Which scale group a tag belongs to -- i.e. whose k it is measured at.

    Cheng et al. fix one scale per (model, corpus), so only corpora define a
    scale; each gets its own group and keeps its own k.

    Probing tasks define none. Conneau et al. built all five probing sets from
    the Toronto Book Corpus, so they are five views of one corpus rather than
    five corpora, and they borrow `reference_corpus`'s k. That keeps the count
    at one k per (model, corpus) and, more importantly, puts task ID on the same
    scale as the corpus baseline it gets compared against -- comparing them at
    different k would measure the scale difference, not the representations.

    Sane and shuffled stay separate on both sides: a shuffled task variant
    borrows the shuffled corpus's k, matching the paper's pairing, so a
    word-order control is never placed on the same axis as sane text.
    """
    if tag.startswith("_"):
        return tag
    variant = "shuffled" if tag.endswith("__shuffled") else "sane"
    return f"_corpus_{reference_corpus}_{variant}"


def scale_robustness(
    curves: list[dict[str, np.ndarray]], idx: int
) -> dict[str, np.ndarray]:
    """ID profiles at half, chosen, and double the selected scale (Figure C.2).

    Cheng et al.'s robustness check: the *shape* of the layerwise ID curve should
    survive moving one logarithmic step in either direction.
    """
    out = {}
    for label, j in (("half", idx - 1), ("chosen", idx), ("double", idx + 1)):
        if 0 <= j < len(curves[0]["id"]):
            out[label] = np.array([c["id"][j] for c in curves])
            out[f"{label}_k"] = int(curves[0]["k"][j])
    return out


def find_peak(profile: np.ndarray, layers: np.ndarray | None = None) -> dict[str, float | int]:
    """Delimit the high-ID phase, following Cheng et al. section 3.4.

    The peak's END is the first inflection point after the maximum (where the
    discrete second difference changes sign); its START is the last layer before
    the maximum whose ID is still >= the ID at that end point.
    """
    profile = np.asarray(profile, dtype=float)
    if not np.any(np.isfinite(profile)):
        return {"peak_layer": -1, "peak_value": float("nan"),
                "peak_start": -1, "peak_end": -1, "peak_layer_rel": float("nan")}
    peak = int(np.nanargmax(profile))

    second = np.diff(profile, n=2)  # second[i] is the curvature at layer i+1
    end = len(profile) - 1
    for i in range(peak, len(second)):
        if np.sign(second[i]) != np.sign(second[i - 1] if i > peak else second[i]):
            end = i + 1
            break

    thresh = profile[end]
    start = peak
    for i in range(peak, -1, -1):
        if profile[i] >= thresh:
            start = i
        else:
            break

    # `layers` maps positions back to real layer numbers; without it, positions
    # ARE the layer numbers. Needed because profiles start at layer 1 when the
    # embedding layer is skipped.
    lay = np.arange(len(profile)) if layers is None else np.asarray(layers)
    return {
        "peak_layer": int(lay[peak]),
        "peak_value": float(profile[peak]),
        "peak_start": int(lay[start]),
        "peak_end": int(lay[end]),
        "peak_layer_rel": float((lay[peak] - lay[0]) / max(lay[-1] - lay[0], 1)),
    }


__all__ = ["id_scaling", "empty_sweep", "select_plateau", "choose_common_scale", "scale_group",
           "scale_robustness", "find_peak"]
