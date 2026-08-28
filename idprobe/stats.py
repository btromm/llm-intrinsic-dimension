"""Significance testing for the ID <-> probe-accuracy relationship.

Two distinct questions, two different tests:

1. "Is the linear probe as good as the MLP?"  ->  paired test on the SAME test
   examples. McNemar's test is the right tool: it conditions on the discordant
   pairs (examples one probe gets right and the other wrong) and ignores the
   examples both handle identically, which is exactly the information that
   distinguishes the two classifiers.

2. "Does ID track probe accuracy across layers?"  ->  correlation across layers.
   The catch is that layer-wise curves are strongly autocorrelated: adjacent
   layers are near-duplicates, so the effective sample size is far below the
   layer count and a textbook Spearman p-value is badly anti-conservative.
   `layer_correlation` therefore uses a circular-shift permutation null, which
   preserves each curve's autocorrelation while destroying their alignment.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired comparison of two classifiers on identical test examples.

    `correct_a`/`correct_b` are per-example boolean arrays (probe A = linear,
    probe B = MLP by convention). Returns the discordant counts, the p-value,
    and the signed accuracy gap b - a.
    """
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if a.shape != b.shape:
        raise ValueError(f"prediction arrays misaligned: {a.shape} vs {b.shape}")

    n01 = int((~a & b).sum())   # A wrong, B right  -> favours the MLP
    n10 = int((a & ~b).sum())   # A right, B wrong  -> favours the linear probe
    table = [[int((a & b).sum()), n10], [n01, int((~a & ~b).sum())]]

    # Exact binomial when discordants are few; chi-square with continuity
    # correction otherwise (the exact test gets slow at large counts).
    exact = (n01 + n10) < 25
    res = mcnemar(table, exact=exact, correction=not exact)

    return {
        "n_mlp_only": n01,
        "n_linear_only": n10,
        "n_discordant": n01 + n10,
        "n_total": int(len(a)),
        "acc_linear": float(a.mean()),
        "acc_mlp": float(b.mean()),
        "acc_gap": float(b.mean() - a.mean()),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "test": "exact" if exact else "chi2",
    }


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """BH-FDR correction. Returns (rejected, q_values) aligned to the input order.

    We run one McNemar test per (task, layer) cell -- on the order of 200 tests --
    so an uncorrected alpha would manufacture roughly ten spurious "the MLP wins
    here" layers. FDR rather than Bonferroni because these tests are positively
    correlated across adjacent layers and we want power, not family-wise certainty.

    Note `rejected` is direction-agnostic: McNemar is two-sided, so a rejected
    cell may be one where the LINEAR probe won. Callers that want "the MLP is
    better here" must also check the sign of acc_gap.
    """
    reject, q, _, _ = multipletests(np.asarray(pvals, dtype=float),
                                    alpha=alpha, method="fdr_bh")
    return reject, q


def layer_correlation(id_profile, acc_profile, n_perm: int = 10_000, seed: int = 0,
                      layers=None) -> dict:
    """Spearman correlation between the ID and accuracy curves, with a shift-null p.

    The circular-shift null keeps both curves intact (autocorrelation and all)
    and only breaks their relative alignment, so the p-value answers "are these
    two curves aligned more than chance?" rather than "are these 37 points
    independent samples?", which they are emphatically not.
    """
    x = np.asarray(id_profile, float)
    y = np.asarray(acc_profile, float)
    ok = np.isfinite(x) & np.isfinite(y)
    # Track real layer numbers through the mask: dropping the skipped embedding
    # layer shifts positions by one, so reporting argmax positions as layer
    # numbers would be off by exactly that many.
    lay = np.arange(len(x)) if layers is None else np.asarray(layers)
    x, y, lay = x[ok], y[ok], lay[ok]

    rho = stats.spearmanr(x, y).statistic

    # Enumerate every non-identity circular shift when there are few enough of
    # them (there are only n-1, and n is the layer count). Sampling with
    # replacement from so small a set would duplicate some shifts and omit
    # others for no benefit.
    all_shifts = np.arange(1, len(x))
    if len(all_shifts) <= n_perm:
        shifts = all_shifts
    else:
        shifts = np.random.default_rng(seed).choice(all_shifts, size=n_perm, replace=False)

    null = np.array([stats.spearmanr(x, np.roll(y, int(s))).statistic for s in shifts])
    # Add-one correction: a permutation test can never justify p = 0, and the
    # attainable floor is 1/(1+n_shifts).
    p = float((1 + int((np.abs(null) >= abs(rho)).sum())) / (1 + len(null)))

    return {
        "spearman_rho": float(rho),
        "p_perm": p,
        "pearson_r": float(stats.pearsonr(x, y).statistic),
        "n_layers": int(len(x)),
        "argmax_id": int(lay[np.argmax(x)]),
        "argmax_acc": int(lay[np.argmax(y)]),
        "argmax_offset": int(lay[np.argmax(y)] - lay[np.argmax(x)]),
    }


def peak_alignment_test(id_profiles: dict, acc_profiles: dict) -> dict:
    """Descriptive summary of where each task's ID and accuracy peaks land.

    This is the crux of the hypothesis: a peak shared by every task would be
    consistent with Cheng et al.'s single abstraction phase but would NOT
    support "task X peaks later than task Y".

    NOT a hypothesis test -- it reports observed spreads, not a p-value.
    Testing "peaks differ across tasks" needs a noise model for how much a
    single task's peak layer moves under resampling, which a handful of tasks
    cannot supply. `uniform_null_spread` is given only as calibration: the
    population SD of peaks placed uniformly at random over the layer stack,
    sqrt((L^2-1)/12). An observed spread far below it is consistent with a
    shared phase; nothing here licenses a significance claim either way.
    """
    tasks = sorted(set(id_profiles) & set(acc_profiles))
    if not tasks:
        return {"tasks": [], "id_peak_layers": [], "acc_peak_layers": [],
                "id_peak_spread": float("nan"), "acc_peak_spread": float("nan"),
                "uniform_null_spread": float("nan"), "peak_offset_mean": float("nan"),
                "peak_offset_corr": float("nan")}

    id_peaks = np.array([int(np.nanargmax(id_profiles[t])) for t in tasks])
    acc_peaks = np.array([int(np.nanargmax(acc_profiles[t])) for t in tasks])
    n_layers = len(id_profiles[tasks[0]])
    return {
        "tasks": tasks,
        "id_peak_layers": id_peaks.tolist(),
        "acc_peak_layers": acc_peaks.tolist(),
        "id_peak_spread": float(id_peaks.std()),
        "acc_peak_spread": float(acc_peaks.std()),
        "uniform_null_spread": float(np.sqrt((n_layers**2 - 1) / 12)),
        "peak_offset_mean": float((acc_peaks - id_peaks).mean()),
        "peak_offset_corr": float(stats.spearmanr(id_peaks, acc_peaks).statistic)
        if len(tasks) > 2 else float("nan"),
    }


def gap_confidence_interval(mc: dict, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided CI for the paired accuracy gap (acc_mlp - acc_linear).

    Uses the standard paired-proportions variance, which counts only the
    discordant pairs: concordant examples contribute nothing to the difference,
    so treating the two accuracies as independent would overstate the variance.
    """
    n01, n10, n = mc["n_mlp_only"], mc["n_linear_only"], mc["n_total"]
    gap = (n01 - n10) / n
    var = ((n01 + n10) - (n01 - n10) ** 2 / n) / n**2
    half = stats.norm.ppf(1 - alpha / 2) * np.sqrt(max(var, 0.0))
    return gap - half, gap + half


def classify_layer(mc: dict, q_value: float, alpha: float = 0.05, margin: float = 0.01,
                   mc_ordinal: dict | None = None) -> str:
    """Verdict on linear separability at one layer.

    Returns "nonlinear_advantage" | "monotonic_encoding" | "linear_sufficient"
    | "inconclusive".

    `mc_ordinal` is an optional McNemar comparing an ORDINAL probe (a) against the
    MLP (b), i.e. mcnemar_test(correct_ordinal, correct_mlp). Supplying it splits the
    "nonlinear_advantage" verdict in two, which matters because the two halves make
    materially different claims:

      nonlinear_advantage  the MLP wins and an ordered readout does not recover it,
                           so the feature really is not linearly decodable.
      monotonic_encoding   the MLP wins over a linear softmax, but an ordinal probe
                           closes the gap. The quantity is encoded monotonically
                           along a direction and the *binned classifier* was the
                           limitation -- this says something about readout geometry,
                           not about entanglement.

    Note the asymmetry with `linear_sufficient` below: promoting to
    "monotonic_encoding" also requires an equivalence argument (the MLP's remaining
    edge over the ordinal probe must be bounded below `margin`), not merely a
    non-significant test.

    A non-significant McNemar does NOT establish linear separability -- absence
    of evidence is not evidence of absence, and an underpowered layer produces
    exactly that. So "linear_sufficient" requires an equivalence argument: the
    upper confidence bound on the MLP's advantage must fall below `margin`,
    i.e. we can rule out a gain large enough to be scientifically interesting.
    Layers meeting neither bar are honestly "inconclusive".

    `margin` is in absolute accuracy points (default 0.01 = 1 point). Consider
    scaling it to task headroom if you care more about relative error reduction:
    one point at 55% accuracy means something different than one point at 95%.
    """
    if q_value <= alpha and mc["acc_gap"] > 0:
        if mc_ordinal is not None:
            # acc_gap here is acc_mlp - acc_ordinal; bounding it below `margin`
            # rules out the MLP retaining an interesting edge over the ordered
            # readout, which is what "the ordering was the missing piece" means.
            _, upper_ord = gap_confidence_interval(mc_ordinal, alpha)
            if upper_ord < margin:
                return "monotonic_encoding"
        return "nonlinear_advantage"
    _, upper = gap_confidence_interval(mc, alpha)
    if upper < margin:
        return "linear_sufficient"
    return "inconclusive"


__all__ = [
    "mcnemar_test", "benjamini_hochberg", "layer_correlation",
    "peak_alignment_test", "classify_layer", "gap_confidence_interval",
]
