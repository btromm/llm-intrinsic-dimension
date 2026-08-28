"""Aggregate ID profiles and probe results into the paper's headline tables/figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TASKS, Config
from .stats import (
    benjamini_hochberg, classify_layer, layer_correlation, mcnemar_test, peak_alignment_test,
)


def _best_seed_correct(correct: dict, task: str, layer: int, kind: str,
                       probes: pd.DataFrame) -> np.ndarray:
    """Predictions from the seed with the best VALIDATION accuracy.

    Selecting on validation and reporting on test keeps the McNemar comparison
    clean; picking the best *test* seed would bias the very gap we are testing.
    """
    sub = probes[(probes.task == task) & (probes.layer == layer) & (probes.kind == kind)]
    seed = int(sub.loc[sub.val_acc.idxmax(), "seed"])
    return correct[f"{task}/{layer}/{kind}/{seed}"]


def run_analysis(cfg: Config) -> dict:
    RESULTS = cfg.results_dir
    probes = pd.read_csv(RESULTS / "probe_results.csv")
    ids = pd.read_csv(RESULTS / "id_profiles.csv")
    correct = dict(np.load(RESULTS / "probe_correct.npz"))

    # ---- 1. Linear vs MLP, per (task, layer) -------------------------------
    # The __shuffled variants are the Figure I.2 control, not a probing result:
    # they answer "does the probe need real linguistic structure?", not "is this
    # feature linearly encoded?". Keep them out of the linear-vs-MLP comparison
    # and out of the FDR family, or they dilute the correction with ~150 tests
    # whose answer is known in advance.
    real = probes[~probes.task.str.endswith("__shuffled")]
    rows = []
    for (task, layer), _ in real.groupby(["task", "layer"]):
        mc = mcnemar_test(
            _best_seed_correct(correct, task, layer, "linear", real),
            _best_seed_correct(correct, task, layer, "mlp", real),
        )
        rows.append({"task": task, "group": TASKS[task], "layer": layer, **mc})
    comp = pd.DataFrame(rows)
    comp["reject"], comp["q_value"] = benjamini_hochberg(comp["p_value"])
    # McNemar is two-sided, so `reject` alone does not say WHICH probe won.
    # Split it by the sign of the gap before reporting anything directional.
    comp["mlp_better"] = comp["reject"] & (comp["acc_gap"] > 0)
    comp["linear_better"] = comp["reject"] & (comp["acc_gap"] < 0)

    comp["verdict"] = [classify_layer(dict(r), r["q_value"]) for _, r in comp.iterrows()]

    comp.to_csv(RESULTS / "linear_vs_mlp.csv", index=False)

    # ---- 2. ID <-> accuracy correlation, per task --------------------------
    acc = (real[real.kind == "linear"]
           .groupby(["task", "layer"])["test_acc"].mean().reset_index())
    id_profiles, acc_profiles, corr_rows = {}, {}, []

    # Reference corpus for the "corpus ID vs task accuracy" rows. Tags are
    # "_corpus_<name>_<mode>"; the old bare "_corpus" name no longer exists, and
    # matching it silently produced zero corpus rows rather than an error.
    corpus_tags = sorted(t for t in ids.tag.unique()
                         if t.startswith("_corpus_") and t.endswith("_sane"))

    for task in sorted(acc.task.unique()):
        gi = ids[ids.tag == task].sort_values("layer")
        ga = acc[acc.task == task].sort_values("layer")
        idp, accp = gi["id"].to_numpy(), ga["test_acc"].to_numpy()
        lay = gi["layer"].to_numpy()          # real layer numbers, not positions
        if len(idp) != len(accp):
            print(f"note: {task} has {len(idp)} ID layers vs {len(accp)} probe layers; skipping")
            continue
        id_profiles[task], acc_profiles[task] = idp, accp
        corr_rows.append({"task": task, "group": TASKS[task], "id_source": "per_task",
                          **layer_correlation(idp, accp, layers=lay)})

        for ct in corpus_tags:
            gc = ids[ids.tag == ct].sort_values("layer")
            corpus = gc["id"].to_numpy()
            if len(corpus) == len(accp):
                corr_rows.append({"task": task, "group": TASKS[task], "id_source": ct,
                                  **layer_correlation(corpus, accp,
                                                      layers=gc["layer"].to_numpy())})

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(RESULTS / "id_accuracy_correlation.csv", index=False)

    # ---- 2b. Selectivity: real minus shuffled (Figure I.2) -----------------
    sel = []
    for task in sorted(real.task.unique()):
        r = real[(real.task == task) & (real.kind == "linear")].groupby("layer")["test_acc"].mean()
        sh = probes[(probes.task == f"{task}__shuffled") & (probes.kind == "linear")] \
            .groupby("layer")["test_acc"].mean()
        if not len(sh):
            continue
        sel.append({"task": task, "group": TASKS[task],
                    "real_max": float(r.max()), "real_argmax": int(r.idxmax()),
                    "shuffled_max": float(sh.max()),
                    "selectivity": float(r.max() - sh.max())})
    if sel:
        pd.DataFrame(sel).to_csv(RESULTS / "selectivity.csv", index=False)

    # ---- 3. Do ID peaks discriminate across tasks? -------------------------
    align = peak_alignment_test(id_profiles, acc_profiles)
    pd.DataFrame([align]).to_csv(RESULTS / "peak_alignment.csv", index=False)

    _summarise(comp, corr, align)
    return {"comparison": comp, "correlation": corr, "alignment": align}


def _summarise(comp: pd.DataFrame, corr: pd.DataFrame, align: dict) -> None:
    print("\n=== linear vs MLP (BH-FDR corrected) ===")
    for task, g in comp.groupby("task"):
        n_mlp = int(g["mlp_better"].sum())
        n_lin = int(g["linear_better"].sum())
        print(f"{task:24s} MLP better at {n_mlp}/{len(g)} layers, "
              f"linear better at {n_lin}/{len(g)} | "
              f"max gap {g.acc_gap.max():+.4f} @ layer {int(g.loc[g.acc_gap.idxmax(),'layer'])}")

    print("\n=== ID vs linear-probe accuracy across layers ===")
    for _, r in corr.iterrows():
        print(f"{r.task:24s} [{r.id_source:8s}] rho={r.spearman_rho:+.3f} "
              f"p_perm={r.p_perm:.4f} | ID peak L{r.argmax_id} vs acc peak L{r.argmax_acc} "
              f"(offset {r.argmax_offset:+d})")

    print("\n=== peak discrimination across tasks ===")
    print(f"ID peak layers:  {dict(zip(align['tasks'], align['id_peak_layers']))}")
    print(f"acc peak layers: {dict(zip(align['tasks'], align['acc_peak_layers']))}")
    print(f"ID peak spread {align['id_peak_spread']:.2f} layers vs "
          f"{align['uniform_null_spread']:.2f} if peaks were placed at random "
          f"(descriptive only -- not a significance test)")


__all__ = ["run_analysis"]
