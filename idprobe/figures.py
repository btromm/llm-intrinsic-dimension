"""Reproductions of specific figures from Cheng et al. (ICLR 2025).

Kept separate from `plots.py` (our own diagnostics) so it is always clear which
figures are theirs and which are this project's extensions.

Tag scheme produced by `run.py extract`:
    "<task>"                     task, unshuffled
    "<task>__shuffled"           task, word-shuffled control  (Fig I.2)
    "_corpus_<corpus>_<mode>"    ID corpus, mode in {sane, shuffled}
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TASKS, Config

SURFACE = [t for t, g in TASKS.items() if g == "surface"]
DEEP = [t for t, g in TASKS.items() if g != "surface"]
CORPUS_COLOUR = {"bookcorpus": "#2e7d4f"}


def _corpus_rows(ids: pd.DataFrame) -> pd.DataFrame:
    c = ids[ids.tag.str.startswith("_corpus_")].copy()
    parts = c.tag.str.split("_", n=3, expand=True)      # ['', 'corpus', name, mode]
    c["corpus"], c["mode"] = parts[2], parts[3]
    return c


# --------------------------------------------------------------------- Fig C.1
def figure_c1(cfg: Config, tag: str | None = None) -> "plt.Figure":
    """Scale analysis: ID vs GRIDE k, one line per layer, plateau band highlighted.

    The paper's Figure C.1. Three regimes are visible left to right: noise at
    small k, a plateau at intermediate k where the estimate is trustworthy, and
    decline at large k where curvature and density variation dominate.
    """
    R = cfg.results_dir
    curves = json.loads((R / "id_scaling_curves.json").read_text())
    table = pd.read_csv(R / "table_c1_scales.csv").set_index("tag")
    tag = tag or next((t for t in table.index if t.startswith("_corpus_")
                       and t.endswith("_sane")), table.index[0])

    keys = sorted((k for k in curves if k.rsplit("/", 1)[0] == tag),
                  key=lambda k: int(k.rsplit("/", 1)[1]))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, key in enumerate(keys):
        c = curves[key]
        ax.plot(c["k"], c["id"], color=cmap(i / max(len(keys) - 1, 1)), lw=1.3,
                marker="o", ms=3)

    chosen = int(table.loc[tag, "k"])
    ax.axvspan(chosen / 2, chosen * 2, color="gold", alpha=0.22, zorder=0,
               label=f"plateau region (chosen k={chosen})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GRIDE scale k (nearest-neighbour rank)")
    ax.set_ylabel("estimated ID")
    ax.set_title(f"Figure C.1 — scale analysis, {tag}", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                 norm=plt.Normalize(0, len(keys) - 1)), ax=ax, label="layer")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------- Fig 1
def figure_1(cfg: Config) -> "plt.Figure":
    """ID over layers: original (solid) vs word-shuffled (dashed), overlaid.

    Cheng et al. put these in adjacent panels (Figure 1 left and centre); drawing
    them on shared axes makes the quantity actually being claimed -- the gap
    between the two in the middle layers -- directly legible instead of requiring
    the reader to compare across panels.

    Each (corpus, mode) is plotted at its own Table C.1 scale, which is what the
    paper does: plateau selection picks the k at which the estimate is stable, so
    two plateau values estimate the same quantity even when read at different k.
    The chosen k is shown in the legend so that stays visible rather than implicit.
    """
    c = _corpus_rows(pd.read_csv(cfg.results_dir / "id_profiles.csv"))
    scales = {}
    tab = cfg.results_dir / "table_c1_scales.csv"
    if tab.exists():
        t = pd.read_csv(tab).set_index("tag")
        scales = {i: int(t.loc[i, "k"]) for i in t.index if i.startswith("_corpus_")}

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    style = {"sane": ("-", "o", 1.9), "shuffled": ("--", "s", 1.5)}
    for (corpus, mode), g in c.groupby(["corpus", "mode"]):
        g = g.sort_values("layer")
        ls, mk, lw = style.get(mode, ("-", "o", 1.7))
        k = scales.get(f"_corpus_{corpus}_{mode}")
        label = f"{corpus} — {'original' if mode == 'sane' else 'word-shuffled'}"
        if k:
            label += f"  (k={k})"
        ax.plot(g.layer, g["id"], ls, marker=mk, ms=3.4, lw=lw,
                color=CORPUS_COLOUR.get(corpus), alpha=1.0 if mode == "sane" else 0.75,
                label=label)

    ax.set_xlabel("layer")
    ax.set_ylabel("intrinsic dimension")
    ax.set_title(f"Figure 1 — ID across layers ({cfg.model_tag})\n"
                 "solid: original text,  dashed: word-shuffled",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- Fig 5 and I.1
def figure_5(cfg: Config, rows=("surface", "deep")) -> "plt.Figure":
    """Probing accuracy with the ID profile overlaid, split by task family.

    The paper's Figure 5 (and I.1, which is the same panel for further models).
    Row (a): surface-form tasks, whose probe accuracy *decreases* through the ID
    peak. Row (b): syntactic and semantic tasks, which attain their maximum
    within it. Accuracy is mean +/- 2 SD over seeds, matching their presentation.
    """
    R = cfg.results_dir
    probes = pd.read_csv(R / "probe_results.csv")
    probes = probes[~probes.task.str.endswith("__shuffled")]
    ids = _corpus_rows(pd.read_csv(R / "id_profiles.csv"))
    ref = ids[(ids["mode"] == "sane")].groupby("layer")["id"].mean()

    groups = {"surface": [t for t in SURFACE if t in set(probes.task)],
              "deep": [t for t in DEEP if t in set(probes.task)]}
    rows = [r for r in rows if groups[r]]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 4.0 * len(rows)), squeeze=False)

    for ax, row in zip(axes[:, 0], rows):
        for task in groups[row]:
            g = probes[(probes.task == task) & (probes.kind == "linear")] \
                .groupby("layer")["test_acc"]
            m, sd = g.mean(), g.std().fillna(0)
            ax.plot(m.index, m.values, marker="o", ms=3.5, lw=1.8, label=task)
            ax.fill_between(m.index, m - 2 * sd, m + 2 * sd, alpha=0.18)
        ax.set_ylabel("probe accuracy")
        ax.set_xlabel("layer")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title({"surface": "(a) surface-form tasks",
                      "deep": "(b) syntactic and semantic tasks"}[row],
                     fontsize=11, fontweight="bold")
        ax2 = ax.twinx()
        ax2.plot(ref.index, ref.values, color="#c1442e", ls="--", lw=1.6, alpha=0.75)
        ax2.set_ylabel("ID (corpus mean)", color="#c1442e")
        ax2.tick_params(axis="y", labelcolor="#c1442e")

    fig.suptitle(f"Figure 5 — probing vs ID ({cfg.model_tag})", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- Fig I.2
def figure_i2(cfg: Config) -> "plt.Figure":
    """Probe accuracy on real (solid) vs word-shuffled (dashed) sentences.

    The paper's Figure I.2. Shuffled performance sitting at chance is what lets
    accuracy stand in for *selectivity* (Hewitt & Liang 2019): a probe that only
    succeeds when linguistic structure is intact is reading the model, not
    memorising the task from the labels.
    """
    probes = pd.read_csv(cfg.results_dir / "probe_results.csv")
    real = probes[~probes.task.str.endswith("__shuffled")]
    tasks = sorted(real.task.unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cmap = plt.get_cmap("tab10")

    for i, task in enumerate(tasks):
        col = cmap(i % 10)
        for suffix, style, lab in ((" ", "-", task), ("__shuffled", "--", None)):
            name = task if suffix == " " else f"{task}__shuffled"
            g = probes[(probes.task == name) & (probes.kind == "linear")] \
                .groupby("layer")["test_acc"].mean()
            if len(g):
                ax.plot(g.index, g.values, style, color=col, lw=1.7, label=lab)
    ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy")
    ax.set_title("Figure I.2 — solid: real sentences,  dashed: word-shuffled",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- Fig H.1
def figure_h1(matrix: np.ndarray, name_a: str, name_b: str,
              peak_a=None, peak_b=None) -> "plt.Figure":
    """Cross-model layer similarity by linear CKA, with ID-peak bands marked."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="linear CKA")
    if peak_a:
        ax.axhspan(peak_a[0], peak_a[1], color="cyan", alpha=0.16)
    if peak_b:
        ax.axvspan(peak_b[0], peak_b[1], color="cyan", alpha=0.16)
    ax.set_ylabel(f"{name_a} layer"); ax.set_xlabel(f"{name_b} layer")
    ax.set_title("Figure H.1 — cross-model linear CKA\n(bands = high-ID phase)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------- Direction 1: conditional ID
def figure_conditional_id(cfg: Config) -> "plt.Figure":
    """DeltaID across layers per task, with the label-shuffle null band shaded.

    Same visual grammar as Figure 1: layers on x, one line per task, the null drawn
    as a band rather than a line so the eye reads "outside the band" as the claim.

    The band is +/-2 SD of the permutation null, NOT a confidence interval on
    DeltaID. It answers "how big does DeltaID get when the labels mean nothing?",
    which is the only reference point that makes a raw DeltaID interpretable given
    GRIDE's sample-size bias.
    """
    df = pd.read_csv(cfg.results_dir / "conditional_id.csv")
    tasks = sorted(df.tag.unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cmap = plt.get_cmap("tab10")

    for i, task in enumerate(tasks):
        g = df[df.tag == task].sort_values("layer")
        col = cmap(i % 10)
        ax.plot(g.layer, g.delta_corrected, marker="o", ms=3.6, lw=1.8, color=col,
                label=f"{task} (k={int(g.k.iloc[0])}, n={int(g.n_matched.iloc[0])})")
        ax.fill_between(g.layer, -2 * g.delta_null_sd, 2 * g.delta_null_sd,
                        color=col, alpha=0.10, lw=0)

    ax.axhline(0, color="#444444", lw=1.0, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ID  =  ID(X) $-$ mean$_c$ ID(X | y=c)")
    ax.set_title(f"Class-conditional ID ({cfg.model_tag})\n"
                 "shaded: +/-2 SD of the label-shuffle null",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def figure_magnitude(cfg: Config) -> "plt.Figure":
    """Direction 2: do the cheaper explanations account for the MLP's advantage?

    Top row puts the four classification readouts on one axis per task, so "the MLP
    wins" can be read directly against "an ordered readout wins too". The bottom row
    is a SINGLE panel carrying the quantities that do not live on an accuracy scale --
    regression R^2 against the raw magnitude, and Spearman(||h||, magnitude) -- because
    a task can be almost perfectly encoded as a continuous quantity while every binned
    classifier looks mediocre, and that gap IS the finding.

    Only tasks in MAGNITUDE_TASKS have a continuous target, so the bottom panel spans
    the full width rather than leaving an empty axis under every other task: an empty
    pair of axes invites the reader to see a flat line at zero, a measured null, where
    there is simply no measurement.
    """
    df = pd.read_csv(cfg.results_dir / "magnitude.csv")
    tasks = sorted(df.task.unique())
    has_cont = [t for t in tasks
                if "regression_r2" in df and df[df.task == t]["regression_r2"].notna().any()]

    nrows = 2 if has_cont else 1
    fig = plt.figure(figsize=(4.6 * len(tasks), 4.2 * nrows))
    gs = fig.add_gridspec(nrows, len(tasks), hspace=0.32, wspace=0.26)

    series = [("acc_linear", "linear", "#1f4e79"), ("acc_mlp", "MLP", "#e8a33d"),
              ("acc_ordinal", "ordinal", "#2e7d4f"),
              ("acc_norm_mlp", r"$\|h\|$ only (MLP)", "#999999")]
    for j, task in enumerate(tasks):
        g = df[df.task == task].sort_values("layer")
        ax = fig.add_subplot(gs[0, j])
        for col, lab, c in series:
            if col in g:
                ax.plot(g.layer, g[col], marker="o", ms=3.2, lw=1.7, color=c, label=lab)
        # Chance level makes "0.62 is good" legible without knowing the class count.
        if "n_classes" in g:
            pass
        ax.set_xlabel("layer")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{task}", fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="best")

    if has_cont:
        ax2 = fig.add_subplot(gs[1, :])
        for t in has_cont:
            g = df[df.task == t].sort_values("layer")
            ax2.plot(g.layer, g.regression_r2, marker="s", ms=4.2, lw=2.0, color="#c1442e",
                     label=f"{t}: regression $R^2$ vs raw magnitude")
            if "spearman_norm_target" in g:
                ax2.plot(g.layer, g.spearman_norm_target.abs(), marker="^", ms=4.0, lw=1.7,
                         color="#6a3d9a", label=rf"{t}: |Spearman($\|h\|$, magnitude)|")
            if "acc_ordinal" in g:
                ax2.plot(g.layer, g.acc_ordinal, marker="o", ms=3.4, lw=1.5, ls="--",
                         color="#2e7d4f", alpha=0.8,
                         label=f"{t}: ordinal probe accuracy (for contrast)")
        ax2.axhline(0, color="#444444", lw=1.0, ls=":")
        ax2.set_xlabel("layer")
        ax2.set_ylabel("continuous-readout score")
        ax2.set_title("Continuous readouts — near-perfect $R^2$ despite mediocre binned accuracy",
                      fontweight="bold", fontsize=11)
        ax2.legend(fontsize=9, loc="best")

    fig.suptitle(f"Magnitude vs feature ({cfg.model_tag})", fontweight="bold", fontsize=13)
    return fig


__all__ = ["figure_c1", "figure_1", "figure_5", "figure_i2", "figure_h1",
           "figure_conditional_id", "figure_magnitude"]
