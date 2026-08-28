"""Figures: ID profiles against probe accuracy, and the linear-vs-MLP contrast."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ACTS, Config

ID_COLOUR = "#c1442e"
LIN_COLOUR = "#1f4e79"
MLP_COLOUR = "#e8a33d"
VERDICT_MARKER = {
    "nonlinear_advantage": ("v", "#c1442e", "MLP better"),
    "linear_sufficient": ("o", "#2e7d4f", "linear sufficient"),
    "inconclusive": (".", "#999999", "inconclusive"),
}


def _chance_level(cfg: Config, task: str) -> float | None:
    """Majority-class rate on the test split, the honest baseline for accuracy."""
    p = cfg.act_path(task, "test").with_name("test_labels.npy")
    if not p.exists():
        return None
    y = np.load(p)
    return float(np.bincount(y).max() / len(y))


def figure_id_vs_accuracy(cfg: Config, tasks: list[str]) -> "plt.Figure":
    RESULTS = cfg.results_dir
    """ID profile and linear-probe accuracy on twin axes, one panel per task.

    This is the paper's headline view: if the hypothesis holds, accuracy should
    crest inside the shaded high-ID phase.
    """
    ids = pd.read_csv(RESULTS / "id_profiles.csv")
    peaks = pd.read_csv(RESULTS / "id_peaks.csv").set_index("tag")
    probes = pd.read_csv(RESULTS / "probe_results.csv")

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        idp = ids[ids.tag == task].sort_values("layer")
        ax.plot(idp.layer, idp["id"], color=ID_COLOUR, lw=2, marker="o", ms=4, label="ID (GRIDE)")
        if "id_err" in idp:
            ax.fill_between(idp.layer, idp["id"] - idp.id_err, idp["id"] + idp.id_err,
                            color=ID_COLOUR, alpha=0.15)
        if task in peaks.index:
            r = peaks.loc[task]
            ax.axvspan(r.peak_start, r.peak_end, color=ID_COLOUR, alpha=0.08)
            ax.axvline(r.peak_layer, color=ID_COLOUR, ls=":", lw=1.2)
        ax.set_xlabel("layer")
        ax.set_ylabel("intrinsic dimension", color=ID_COLOUR)
        ax.tick_params(axis="y", labelcolor=ID_COLOUR)

        ax2 = ax.twinx()
        g = probes[(probes.task == task) & (probes.kind == "linear")].groupby("layer")["test_acc"]
        mean, sd = g.mean(), g.std().fillna(0)
        ax2.plot(mean.index, mean.values, color=LIN_COLOUR, lw=2, marker="s", ms=4,
                 label="linear probe")
        ax2.fill_between(mean.index, mean - 2 * sd, mean + 2 * sd, color=LIN_COLOUR, alpha=0.18)
        chance = _chance_level(cfg, task)
        if chance:
            ax2.axhline(chance, color="grey", ls="--", lw=1)
            ax2.text(0.02, chance, "chance", transform=ax2.get_yaxis_transform(),
                     ha="left", va="bottom", fontsize=8, color="grey")
        ax2.set_ylabel("linear probe accuracy", color=LIN_COLOUR)
        ax2.tick_params(axis="y", labelcolor=LIN_COLOUR)

        k = idp["k"].iloc[0] if "k" in idp and len(idp) else "?"
        ax.set_title(f"{task}   (GRIDE k={k})", fontsize=11, fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Intrinsic dimension vs. linear-probe accuracy (shaded = high-ID phase)",
                 fontsize=12)
    fig.tight_layout()
    return fig


def figure_linear_vs_mlp(RESULTS, tasks: list[str]) -> "plt.Figure":
    """Linear and MLP accuracy per layer, annotated with the per-layer verdict."""
    comp = pd.read_csv(RESULTS / "linear_vs_mlp.csv")
    probes = pd.read_csv(RESULTS / "probe_results.csv")

    fig, axes = plt.subplots(2, len(tasks), figsize=(6 * len(tasks), 6.4),
                             squeeze=False, height_ratios=[2.4, 1])
    for col, task in enumerate(tasks):
        ax, axg = axes[0][col], axes[1][col]
        means = {}
        for kind, colour, label in [("linear", LIN_COLOUR, "linear"),
                                    ("mlp", MLP_COLOUR, "MLP (1x200)")]:
            g = probes[(probes.task == task) & (probes.kind == kind)].groupby("layer")["test_acc"]
            m, sd = g.mean(), g.std().fillna(0)
            means[kind] = m
            ax.plot(m.index, m.values, color=colour, lw=2, marker="o", ms=4, label=label)
            ax.fill_between(m.index, m - 2 * sd, m + 2 * sd, color=colour, alpha=0.18)
        ax.set_ylabel("test accuracy")
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

        d = comp[comp.task == task].sort_values("layer")
        # Bar heights come from the SAME seed means as the curves above, so the
        # two halves of the panel agree; linear_vs_mlp.csv reports the
        # best-validation seed instead, which would visibly disagree with them.
        gap = (means["mlp"] - means["linear"]).reindex(d.layer).to_numpy()
        axg.axhline(0, color="grey", lw=1)
        axg.bar(d.layer, gap, color=[VERDICT_MARKER.get(v, (".", "#999", ""))[1]
                                     for v in d.verdict], width=0.7)
        for (_, r), gv in zip(d.iterrows(), gap):
            if r.verdict == "nonlinear_advantage":
                axg.text(r.layer, gv, "*", ha="center",
                         va="bottom" if gv >= 0 else "top", fontsize=11)
        axg.set_xlabel("layer")
        axg.set_ylabel("MLP - linear\n(mean over seeds)")

        seen = {v for v in d.verdict}
        axg.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=lab)
                            for v, (_, c, lab) in VERDICT_MARKER.items() if v in seen],
                   fontsize=7, loc="best", ncol=len(seen))

    fig.suptitle("Does a nonlinearity help?  (* = significant after BH-FDR)", fontsize=12)
    fig.tight_layout()
    return fig


def figure_id_scaling(RESULTS, tasks: list[str], max_panels: int = 2) -> "plt.Figure":
    """GRIDE scale sweeps per layer -- the plateau diagnostic.

    A trustworthy ID estimate is flat over some range of k. Curves that never
    flatten mean the estimate is scale-dependent and should not be interpreted.
    """
    curves = json.loads((RESULTS / "id_scaling_curves.json").read_text())
    tasks = tasks[:max_panels]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.6 * len(tasks), 4), squeeze=False)

    for ax, task in zip(axes[0], tasks):
        keys = sorted((k for k in curves if k.startswith(f"{task}/")),
                      key=lambda k: int(k.split("/")[1]))
        cmap = plt.get_cmap("viridis")
        for i, key in enumerate(keys):
            c = curves[key]
            ax.plot(c["k"], c["id"], color=cmap(i / max(len(keys) - 1, 1)), lw=1.4, marker="o", ms=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("GRIDE scale k (neighbour rank)")
        ax.set_ylabel("estimated ID")
        ax.set_title(f"{task} — scale analysis", fontsize=11, fontweight="bold")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(keys) - 1))
        fig.colorbar(sm, ax=ax, label="layer")

    fig.suptitle("ID scale sweeps: a trustworthy estimate has a plateau", fontsize=12)
    fig.tight_layout()
    return fig


def make_all(cfg: Config, tasks: list[str] | None = None) -> list:
    RESULTS = cfg.results_dir
    probes = pd.read_csv(RESULTS / "probe_results.csv")
    tasks = tasks or sorted(probes.task.unique())
    out = []
    for name, fig in [
        ("fig1_id_vs_accuracy", figure_id_vs_accuracy(cfg, tasks)),
        ("fig2_linear_vs_mlp", figure_linear_vs_mlp(RESULTS, tasks)),
        ("fig3_id_scaling", figure_id_scaling(RESULTS, tasks)),
    ]:
        path = RESULTS / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)
        print(f"wrote {path}")
    return out


__all__ = ["make_all", "figure_id_vs_accuracy", "figure_linear_vs_mlp", "figure_id_scaling"]
