#!/usr/bin/env python
"""Pipeline entry point.

    python run.py check                 # environment + model reachability
    python run.py extract               # activations for all tasks + ID corpus
    python run.py id                    # per-layer ID profiles (per-task and corpus)
    python run.py probe                 # linear + MLP probes at every layer
    python run.py analyze               # McNemar, FDR, ID<->accuracy correlations
    python run.py plots                 # figures -> results/*.png
    python run.py robustness            # is each ID peak stable across scale?   [Step 0b]
    python run.py conditional-id        # DeltaID = ID(X) - mean_c ID(X|y=c)   [Direction 1]
    python run.py magnitude             # ordinal / regression / norm readouts [Direction 2]

Every stage is resumable: it skips outputs that already exist unless --force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from idprobe import data as D
from idprobe import extract as E
from idprobe import intrinsic_dim as ID
from idprobe.config import ACTS, MAGNITUDE_TASKS, SCALES, TASKS, Config
from idprobe.probes import prepare_data, train_probe

SPLITS = ("train", "val", "test")


def stage_check(cfg: Config, args) -> None:
    from transformers import AutoConfig

    dev = E.resolve_device(cfg.device)
    print(f"torch {torch.__version__} | device={dev} | dtype={E.resolve_dtype(cfg.dtype, dev)}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu: {p.name}, {p.total_memory / 1e9:.0f} GB")
    mc = AutoConfig.from_pretrained(cfg.model_id)
    print(f"model: {cfg.model_id} | layers={mc.num_hidden_layers} | d_model={mc.hidden_size}")
    cfg.report_disk(mc.num_hidden_layers, mc.hidden_size)
    import dadapy  # noqa: F401
    print(f"dadapy: OK (GRIDE) | numpy {np.__version__} -- dadapy requires numpy<2")


def stage_extract(cfg: Config, args) -> None:
    """Extract activations for every (task, mode) and every (corpus, mode).

    Tag scheme, used consistently by every later stage:
      task, sane       -> "<task>"
      task, shuffled   -> "<task>__shuffled"      (Figure I.2 control)
      corpus           -> "_corpus_<corpus>_<mode>"
    """
    tok, model, dev = E.load_model(cfg)
    cfg.report_disk(model.config.num_hidden_layers, model.config.hidden_size)
    modes = ("sane",) if args.no_shuffled else cfg.modes

    for task in (() if args.no_tasks else cfg.tasks):
        splits = D.prepare_task(task, cfg)
        print(f"[{task}] {D.label_counts(splits)}")
        for mode in modes:
            tag = task if mode == "sane" else f"{task}__shuffled"
            for split in SPLITS:
                path = cfg.act_path(tag, split)
                if path.exists() and not args.force:
                    print(f"  skip {tag}/{split} (exists)")
                    continue
                df = splits[split]
                if mode == "shuffled":
                    df = D.shuffle_task_sentences(df, seed=cfg.id_seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path.with_name(f"{split}_labels.npy"), df["y"].to_numpy(np.int64))
                E.extract(df["sentence"].tolist(), tok, model, dev, path,
                          batch_size=cfg.batch_size, max_length=cfg.max_length)

    if args.no_corpus:
        return
    for corpus in cfg.corpora:
        for mode in modes:
            tag = f"_corpus_{corpus}_{mode}"
            path = cfg.act_path(tag, "train")
            if path.exists() and not args.force:
                print(f"  skip {tag} (exists)")
                continue
            texts = D.prepare_corpus(cfg, corpus=corpus, mode=mode)
            texts = E.filter_by_length(texts, tok, cfg.corpus_seq_len)
            print(f"[{tag}] {len(texts)} sequences of exactly {cfg.corpus_seq_len} tokens")
            E.extract(texts, tok, model, dev, path,
                      batch_size=cfg.batch_size, exact_len=cfg.corpus_seq_len)


def _load_scales() -> dict:
    """Pinned GRIDE scales, keyed model_tag -> scale group -> k."""
    return json.loads(SCALES.read_text()) if SCALES.exists() else {}


def _save_scales(d: dict) -> None:
    SCALES.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def stage_id(cfg: Config, args) -> None:
    rdir = cfg.results_dir
    rdir.mkdir(parents=True, exist_ok=True)

    # Corpora are any "_"-prefixed activation dir, so a shuffled-word-order
    # control dropped in beside "_corpus" is picked up without config churn.
    corpora = sorted(d.name for d in (ACTS / cfg.model_tag).glob("_*") if d.is_dir())

    # Pass 1: sweep every scale for every layer of every tag.
    sweeps: dict[str, list[dict]] = {}
    for tag in [*cfg.tasks, *corpora]:
        path = cfg.act_path(tag, "train")
        if not path.exists():
            print(f"[{tag}] no activations, skipping")
            continue
        # Skipped layers are never estimated at all. GRIDE on the embedding
        # layer is not merely unreported but meaningless: on task data its final
        # tokens collapse to a handful of distinct vectors, and on corpora it
        # measures the spread of the token vocabulary rather than anything the
        # network computed.
        n_layers = E.n_layers_of(path)
        sweeps[tag] = [
            ID.empty_sweep(cfg.id_range_max) if l in cfg.id_skip_layers
            else ID.id_scaling(E.load_layer(path, l), cfg.id_range_max,
                               cfg.id_n_points, cfg.id_seed)
            for l in range(n_layers)
        ]
        skipped = sorted(set(cfg.id_skip_layers) & set(range(n_layers)))
        print(f"[{tag}] swept {n_layers - len(skipped)}/{n_layers} layers "
              f"(skipped {skipped})")

    if not sweeps:
        print("no activations found for any tag -- run `python run.py extract` first")
        return

    # Pass 2: fix ONE scale per corpus (Cheng et al. App. C). Only corpora
    # define a scale; probing tasks inherit their reference corpus's k, so the
    # count stays at one k per (model, corpus). See ID.scale_group.
    groups: dict[str, list[str]] = {}
    for tag in sweeps:
        groups.setdefault(ID.scale_group(tag, cfg.reference_corpus), []).append(tag)

    pinned = _load_scales()
    model_scales = pinned.setdefault(cfg.model_tag, {})

    chosen: dict[str, tuple[int, int, bool, str]] = {}
    for gname, tags in groups.items():
        # Layers we never report are also kept out of the scale choice: the
        # embedding layer has too few distinct points to plateau anywhere, and
        # letting it vote would drag the median toward a degenerate scale.
        # Only the corpus itself votes on its own scale; tasks in the group are
        # inheriting it, and letting their sweeps shift k would reintroduce the
        # task-derived scale this grouping exists to remove.
        definers = [t for t in tags if t == gname]
        pooled = [c for t in definers for l, c in enumerate(sweeps[t])
                  if l not in cfg.id_skip_layers]
        grid = np.asarray((pooled or [sweeps[tags[0]][0]])[0]["k"])

        if gname in model_scales and not args.rechoose:
            k = int(model_scales[gname])
            if k not in set(grid.tolist()):
                raise SystemExit(
                    f"scales.json pins {cfg.model_tag}/{gname} to k={k}, which is not "
                    f"on this sweep's grid {grid.tolist()}. Fix scales.json or rerun "
                    f"with --rechoose."
                )
            idx = int(np.flatnonzero(grid == k)[0])
            found = True
            spread = float(np.median([ID.select_plateau(c)[1] for c in pooled])) if pooled else float("nan")
            note = f"pinned in {SCALES.name}"
        elif not pooled:
            raise SystemExit(
                f"tags {tags} need the scale of corpus '{gname}', which has no "
                f"activations. Extract it (`python run.py extract`), point "
                f"Config.reference_corpus / --reference-corpus at a corpus you "
                f"have, or pin {cfg.model_tag}/{gname} in {SCALES.name} by hand."
            )
        else:
            idx, spread, found = ID.choose_common_scale(pooled)
            k = int(grid[idx])
            model_scales[gname] = k
            note = ("plateau, now pinned" if found
                    else "NO PLATEAU -- flattest window, now pinned")

        inherits = [t for t in tags if t != gname]
        print(f"[{gname}] scale k={k} ({note}, median flatness {spread:.3f})"
              + (f" -- also used by {', '.join(inherits)}" if inherits else ""))
        for t in tags:
            chosen[t] = (idx, k, found, gname)

    _save_scales(pinned)
    print(f"scales -> {SCALES}")

    rows, curves = [], {}
    scale_table, robustness = [], []
    for tag, sw in sweeps.items():
        idx, k, found, gname = chosen[tag]
        spreads = [ID.select_plateau(c)[1] for c in sw]
        # What this tag WOULD have picked alone: the gap between the two is how
        # much sharing a scale across the group cost this particular dataset.
        alone_idx, _, alone_found = ID.choose_common_scale(
            [c for l, c in enumerate(sw) if l not in cfg.id_skip_layers])
        scale_table.append({
            "tag": tag, "group": gname, "k": k, "plateau_found": found,
            "k_if_alone": int(sw[0]["k"][alone_idx]), "plateau_found_alone": alone_found,
            "median_spread": float(np.median(spreads)),
            "n_layers_plateaued": int(sum(sp < 0.10 for sp in spreads)),
        })
        rob = ID.scale_robustness(sw, idx)
        for label in ("half", "chosen", "double"):
            if label in rob:
                for layer, v in enumerate(rob[label]):
                    robustness.append({"tag": tag, "layer": layer, "which": label,
                                       "k": rob[f"{label}_k"], "id": float(v)})

        for layer, sc in enumerate(sw):
            skip = layer in cfg.id_skip_layers
            rows.append({
                "tag": tag, "group": gname, "layer": layer, "k": k, "plateau_found": found,
                "id": float("nan") if skip else float(sc["id"][idx]),
                "id_err": float("nan") if skip else float(sc["err"][idx]),
                "n_unique": int(sc["n_unique"]),
            })
            curves[f"{tag}/{layer}"] = {
                kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in sc.items()
            }
            if skip:
                print(f"[{tag}] layer {layer:2d}  ID= (skipped)"
                      f"  (n_unique={sc['n_unique']})")
            else:
                print(f"[{tag}] layer {layer:2d}  ID={sc['id'][idx]:6.2f}"
                      f"  (n_unique={sc['n_unique']})")

    df = pd.DataFrame(rows)
    df.to_csv(rdir / "id_profiles.csv", index=False)
    (rdir / "id_scaling_curves.json").write_text(json.dumps(curves, indent=1))

    pd.DataFrame(scale_table).to_csv(rdir / "table_c1_scales.csv", index=False)
    pd.DataFrame(robustness).to_csv(rdir / "scale_robustness.csv", index=False)

    peaks = [{"tag": t, **ID.find_peak(g.sort_values("layer")["id"].to_numpy())}
             for t, g in df.groupby("tag")]
    pd.DataFrame(peaks).to_csv(rdir / "id_peaks.csv", index=False)
    print(f"\nwrote {rdir/'id_profiles.csv'} and id_peaks.csv")


def stage_probe(cfg: Config, args) -> None:
    rdir = cfg.results_dir
    rdir.mkdir(parents=True, exist_ok=True)
    dev = E.resolve_device(cfg.device)
    rows, correct = [], {}

    # Probe the shuffled variants too -- they are the Figure I.2 selectivity
    # control, and must be trained under exactly the same protocol to count.
    task_tags = []
    for t in cfg.tasks:
        task_tags.append(t)
        if not args.no_shuffled and cfg.act_path(f"{t}__shuffled", "train").exists():
            task_tags.append(f"{t}__shuffled")

    for task in task_tags:
        paths = {s: cfg.act_path(task, s) for s in SPLITS}
        if not all(p.exists() for p in paths.values()):
            print(f"[{task}] missing activations, skipping")
            continue
        y = {s: np.load(paths[s].with_name(f"{s}_labels.npy")) for s in SPLITS}

        for layer in range(E.n_layers_of(paths["train"])):
            X = {s: E.load_layer(paths[s], layer) for s in SPLITS}
            data = prepare_data(
                X["train"], y["train"], X["val"], y["val"], X["test"], y["test"], dev
            )
            for kind in ("linear", "mlp"):
                for seed in cfg.probe_seeds:
                    r = train_probe(
                        kind, data,
                        seed=seed, epochs=cfg.probe_epochs, lr=cfg.probe_lr,
                        weight_decay=cfg.probe_weight_decay, batch_size=cfg.probe_batch_size,
                        patience=cfg.patience, hidden=cfg.mlp_hidden,
                        activation=cfg.mlp_activation,
                    )
                    rows.append({
                        "task": task, "group": TASKS[task.replace("__shuffled", "")],
                        "layer": layer,
                        "kind": kind, "seed": seed,
                        "test_acc": r.test_acc, "val_acc": r.val_acc,
                    })
                    correct[f"{task}/{layer}/{kind}/{seed}"] = r.correct
            accs = {k: np.mean([x["test_acc"] for x in rows
                                if x["layer"] == layer and x["task"] == task and x["kind"] == k])
                    for k in ("linear", "mlp")}
            print(f"[{task}] layer {layer:2d}  linear={accs['linear']:.4f}  mlp={accs['mlp']:.4f}")

    pd.DataFrame(rows).to_csv(rdir / "probe_results.csv", index=False)
    np.savez_compressed(rdir / "probe_correct.npz", **correct)
    print(f"\nwrote {rdir/'probe_results.csv'} and probe_correct.npz")


def _merge_rows(path: "Path", rows: list[dict], key: str) -> pd.DataFrame:
    """Write `rows`, keeping any existing rows for tasks this run did not touch.

    These stages are naturally run one task at a time -- only `sentence_length` is a
    magnitude task, and conditional ID skips tasks whose classes are too small -- so a
    plain overwrite would quietly delete the other tasks' results every time. Rows for
    tasks present in `rows` are replaced wholesale, which keeps a re-run authoritative
    for what it recomputed without resurrecting stale numbers for it.

    Read-modify-write is not atomic, and these stages are worth running one task per
    process in parallel -- so without a lock two tasks finishing together would both
    read the pre-existing file and the second write would silently drop the first
    task's rows. The lock is a directory: mkdir is atomic on POSIX and, unlike a
    lockfile plus O_EXCL, leaves nothing to clean up if the interpreter is killed
    between create and unlink.
    """
    import os
    import time

    lock = path.with_suffix(".lock")
    for attempt in range(600):
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            time.sleep(0.5)
    else:
        print(f"warning: gave up waiting for {lock}; writing without it")

    try:
        fresh = pd.DataFrame(rows)
        if path.exists():
            old = pd.read_csv(path)
            if key in old.columns:
                kept = old[~old[key].isin(fresh[key].unique())]
                fresh = pd.concat([kept, fresh], ignore_index=True)
        fresh = fresh.sort_values([key, "layer"]).reset_index(drop=True)
        # Write-then-rename so a reader never sees a half-written table.
        tmp = path.with_suffix(".csv.partial")
        fresh.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass
    return fresh


def _resolve_task_k(cfg: Config, tag: str, override: int | None) -> tuple[int, str]:
    """The GRIDE scale a conditional-ID estimate must use, and where it came from.

    Conditioning changes the point set but must NOT change the scale: comparing
    ID(X) at one k with ID(X|y=c) at another would measure the change in
    neighbourhood size as much as the effect of conditioning. So this never picks a
    scale of its own -- it recovers the one the `id` stage already used.

    Preference order, most authoritative first:
      1. an explicit --cid-k, for deliberate sensitivity checks
      2. table_c1_scales.csv, which records the k each tag was actually measured at
      3. scales.json, the pinned scale for the tag's group (see ID.scale_group)
    """
    if override is not None:
        return int(override), "cli"

    tab = cfg.results_dir / "table_c1_scales.csv"
    if tab.exists():
        t = pd.read_csv(tab).set_index("tag")
        if tag in t.index:
            return int(t.loc[tag, "k"]), "table_c1_scales.csv"

    group = ID.scale_group(tag, cfg.reference_corpus)
    pinned = _load_scales().get(cfg.model_tag, {})
    if group in pinned:
        return int(pinned[group]), f"scales.json[{group}]"

    raise SystemExit(
        f"no GRIDE scale known for '{tag}'. Run `python run.py id` for "
        f"{cfg.model_tag} first, pin {cfg.model_tag}/{group} in {SCALES.name}, "
        f"or pass --cid-k."
    )


def stage_conditional_id(cfg: Config, args) -> None:
    """Direction 1: class-conditional ID, DeltaID = ID(X) - mean_c ID(X|y=c).

    Entanglement measured in ID's own units rather than inferred by correlating a
    dataset-level statistic against a direction-level readout. See
    idprobe/conditional_id.py for why the matched-n and label-shuffle guards are
    not optional.
    """
    from idprobe.conditional_id import conditional_id, conditional_id_null, residual_id

    rdir = cfg.results_dir
    rdir.mkdir(parents=True, exist_ok=True)
    dev = E.resolve_device(cfg.device)
    rows = []

    for task in cfg.tasks:
        path = cfg.act_path(task, "train")
        if not path.exists():
            print(f"[{task}] no activations, skipping")
            continue
        y = np.load(path.with_name("train_labels.npy"))
        n_rows = int(np.load(path, mmap_mode="r").shape[1])
        if n_rows != len(y):
            print(f"[{task}] SKIP: {n_rows} activation rows vs {len(y)} labels -- this "
                  f"task is mid-extraction or was interrupted; re-extract it first")
            continue
        k, k_src = _resolve_task_k(cfg, task, args.cid_k)
        smallest = int(np.bincount(y).min())

        # word_content has 1000 classes and ~25 examples each even at full scale;
        # there is no k at which a 25-point cloud has a meaningful dimension. Skip
        # on the arithmetic rather than by name, so the reason travels with the run.
        if smallest <= 2 * k:
            print(f"[{task}] SKIP: smallest class has {smallest} examples, "
                  f"need > 2k = {2 * k} for a {k}-neighbour estimate")
            continue

        n_match = smallest if args.cid_n_match is None else min(args.cid_n_match, smallest)
        n_layers = E.n_layers_of(path)
        layers = [l for l in range(n_layers) if l not in cfg.id_skip_layers][:: args.cid_stride]
        print(f"[{task}] k={k} (from {k_src}), {len(np.unique(y))} classes, "
              f"n_match={n_match} (smallest class {smallest}), {len(layers)} layers")

        for layer in layers:
            X = E.load_layer(path, layer).astype(np.float64)
            obs = conditional_id(X, y, k, n_match=n_match,
                                 n_boot=args.cid_boot, seed=cfg.id_seed)
            null = conditional_id_null(X, y, k, n_match=n_match, n_perm=args.cid_perm,
                                       n_boot=args.cid_boot, seed=cfg.id_seed)
            sd = null["delta_null_sd"]
            z = ((obs["delta_id"] - null["delta_null_mean"]) / sd
                 if sd and np.isfinite(sd) and sd > 0 else float("nan"))

            row = {
                "tag": task, "group": TASKS[task], "layer": layer, "k": k, "k_source": k_src,
                "n_matched": obs["n_matched"], "n_classes": obs["n_classes"],
                "id_pooled": obs["id_pooled"], "id_pooled_sd": obs["id_pooled_sd"],
                "id_within_mean": obs["id_within_mean"], "id_class_sd": obs["id_class_sd"],
                "delta_id": obs["delta_id"],
                "delta_null_mean": null["delta_null_mean"],
                "delta_null_sd": sd, "n_perm": null["n_perm"],
                "delta_corrected": obs["delta_id"] - null["delta_null_mean"], "z": z,
            }
            # Per-class ID is not just an intermediate. For a magnitude task the
            # classes turn out to sit on manifolds of visibly different dimension
            # (short sentences are far lower-dimensional than long ones), which the
            # mean over classes hides completely.
            row.update({f"id_class{c}": v for c, v in sorted(obs["id_per_class"].items())})

            if not args.no_residual:
                row.update(_residual_row(cfg, task, layer, X, y, k, dev, args, residual_id))
            rows.append(row)
            print(f"[{task}] layer {layer:2d}  ID={obs['id_pooled']:6.2f}  "
                  f"within={obs['id_within_mean']:6.2f}  dID={obs['delta_id']:+6.3f}  "
                  f"null={null['delta_null_mean']:+.3f}+/-{sd:.3f}  z={z:+5.2f}")

    if not rows:
        print("no tasks produced a conditional-ID estimate")
        return
    out = rdir / "conditional_id.csv"
    print(f"\nwrote {out} ({len(_merge_rows(out, rows, 'tag'))} rows)")


def _residual_row(cfg: Config, task: str, layer: int, X: np.ndarray, y: np.ndarray,
                  k: int, dev, args, residual_id) -> dict:
    """ID cost of deleting the probe's own direction (the complementary variant).

    If a feature is one linear direction, removing it should cost about one
    dimension; a much larger drop means the probe direction was entangled with
    others rather than sitting in a private subspace.

    Everything happens in the probe's STANDARDISED feature space. Per-feature
    standardisation is not a global rescaling, so GRIDE's scale-invariance does not
    excuse measuring the direction in one space and the manifold in another.
    """
    paths = {s: cfg.act_path(task, s) for s in SPLITS}
    if not all(p.exists() for p in paths.values()):
        return {}
    ys = {s: np.load(paths[s].with_name(f"{s}_labels.npy")) for s in SPLITS}
    Xtr = E.load_layer(paths["train"], layer)
    data = prepare_data(Xtr, ys["train"],
                        E.load_layer(paths["val"], layer), ys["val"],
                        E.load_layer(paths["test"], layer), ys["test"], dev)
    res = train_probe("linear", data, seed=cfg.probe_seeds[0], epochs=cfg.probe_epochs,
                      lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay,
                      batch_size=cfg.probe_batch_size, patience=cfg.patience,
                      hidden=cfg.mlp_hidden, activation=cfg.mlp_activation)

    # Softmax is invariant to adding one vector to every class row, so the raw [C, d]
    # weight matrix has a redundant direction that carries no decision information.
    # Centring the rows leaves the rank-(C-1) contrast subspace that actually
    # separates the classes -- projecting out the uncentred matrix would delete one
    # extra, arbitrary dimension and inflate the measured cost.
    W = res.direction - res.direction.mean(0, keepdims=True)
    Xs = (Xtr - Xtr.mean(0, keepdims=True)) / (Xtr.std(0, keepdims=True) + 1e-6)
    n_match = int(np.bincount(y).min())
    if args.cid_n_match is not None:
        n_match = min(args.cid_n_match, n_match)
    r = residual_id(Xs.astype(np.float64), W, k,
                    n_match=n_match, n_boot=args.cid_boot, seed=cfg.id_seed)
    return {"probe_acc": res.test_acc, "n_dirs": r["n_dirs"],
            "id_full_std": r["id_full"], "id_residual": r["id_residual"],
            "id_drop": r["id_drop"]}


def _raw_magnitude(cfg: Config, task: str) -> dict[str, np.ndarray] | None:
    """The continuous quantity a magnitude task's labels are bins of, per split.

    Recovered by regenerating the splits: `prepare_task` is deterministic given
    `data_seed`, and `extract` wrote activations in exactly that row order, so the
    regenerated sentences line up with the stored activations one-for-one.

    That assumption is load-bearing and cheap to check, so it IS checked -- against
    the labels saved beside the activations. A silent misalignment here would attach
    every sentence's length to some other sentence's representation and produce a
    confidently wrong R^2 rather than an error.
    """
    fn = MAGNITUDE_TASKS.get(task)
    if fn is None:
        return None
    splits = D.prepare_task(task, cfg)
    out = {}
    for s in SPLITS:
        saved = np.load(cfg.act_path(task, s).with_name(f"{s}_labels.npy"))
        regen = splits[s]["y"].to_numpy()
        if saved.shape != regen.shape or not (saved == regen).all():
            raise SystemExit(
                f"[{task}/{s}] regenerated split does not match the stored labels "
                f"({regen.shape[0]} vs {saved.shape[0]} rows). The activations were "
                f"extracted at different --n-train/--n-val/--n-test than this run "
                f"uses; pass the extraction-time values so the rows line up."
            )
        out[s] = splits[s]["sentence"].map(fn).to_numpy(np.float64)
    return out


def stage_magnitude(cfg: Config, args) -> None:
    """Direction 2: is the task a magnitude rather than a feature?

    `sentence_length` is the one task where the MLP beats the linear probe almost
    everywhere -- which points the WRONG way for a naive superposition story, since
    that predicts abstract features are the entangled ones. The likely explanation is
    not entanglement at all: a quantity encoded monotonically in norm or scale is
    badly served by a linear softmax over bins and trivially fixed by one hidden
    layer. These four readouts separate the two explanations.
    """
    from scipy import stats as sstats

    from idprobe.probes import norm_baseline, train_regression_probe
    from idprobe.stats import benjamini_hochberg, classify_layer, mcnemar_test

    rdir = cfg.results_dir
    rdir.mkdir(parents=True, exist_ok=True)
    dev = E.resolve_device(cfg.device)
    rows, mcs = [], []

    for task in cfg.tasks:
        paths = {s: cfg.act_path(task, s) for s in SPLITS}
        if not all(p.exists() for p in paths.values()):
            print(f"[{task}] missing activations, skipping")
            continue
        y = {s: np.load(paths[s].with_name(f"{s}_labels.npy")) for s in SPLITS}
        raw = _raw_magnitude(cfg, task)
        if raw is None:
            print(f"[{task}] not a magnitude task (see MAGNITUDE_TASKS); "
                  f"ordinal and norm readouts only, no regression target")

        n_layers = E.n_layers_of(paths["train"])
        layers = list(range(n_layers))[:: args.mag_stride]
        print(f"[{task}] {len(layers)} layers, {len(np.unique(y['train']))} classes")

        for layer in layers:
            X = {s: E.load_layer(paths[s], layer) for s in SPLITS}
            kw = dict(epochs=args.mag_epochs, lr=cfg.probe_lr,
                      weight_decay=cfg.probe_weight_decay,
                      batch_size=args.mag_batch, patience=args.mag_patience,
                      hidden=cfg.mlp_hidden, activation=cfg.mlp_activation)
            data = prepare_data(X["train"], y["train"], X["val"], y["val"],
                                X["test"], y["test"], dev)

            row = {"task": task, "group": TASKS[task], "layer": layer,
                   "epochs": args.mag_epochs, "batch_size": args.mag_batch}
            res = {kind: [train_probe(kind, data, seed=sd, **kw) for sd in cfg.probe_seeds]
                   for kind in ("linear", "mlp", "ordinal")}
            for kind, rs in res.items():
                row[f"acc_{kind}"] = float(np.mean([r.test_acc for r in rs]))

            # Paired comparisons on identical test examples, using the seed chosen on
            # VALIDATION accuracy -- selecting on test would bias the very gap being
            # tested. Both McNemars come from probes trained under THIS stage's
            # protocol; comparing a converged ordinal probe against the `probe`
            # stage's undertrained MLP would measure the protocol, not the readout.
            best = {k: max(rs, key=lambda r: r.val_acc) for k, rs in res.items()}
            mc_lin = mcnemar_test(best["linear"].correct, best["mlp"].correct)
            mc_ord = mcnemar_test(best["ordinal"].correct, best["mlp"].correct)
            row.update({"p_linear_vs_mlp": mc_lin["p_value"],
                        "gap_linear_vs_mlp": mc_lin["acc_gap"],
                        "gap_ordinal_vs_mlp": mc_ord["acc_gap"]})
            mcs.append((mc_lin, mc_ord))
            nb = norm_baseline(X["train"], y["train"], X["val"], y["val"],
                               X["test"], y["test"], dev, seed=cfg.probe_seeds[0], **kw)
            row.update({f"acc_{k}": v.test_acc for k, v in nb.items()})

            # Share of the linear->MLP gap that each cheaper explanation accounts
            # for. Near 1 means the gap was never evidence of entanglement.
            gap = row["acc_mlp"] - row["acc_linear"]
            row["mlp_gap"] = gap
            row["ordinal_closes"] = ((row["acc_ordinal"] - row["acc_linear"]) / gap
                                     if abs(gap) > 1e-9 else float("nan"))

            if raw is not None:
                rr = train_regression_probe(X["train"], raw["train"], X["val"], raw["val"],
                                            X["test"], raw["test"], dev,
                                            seed=cfg.probe_seeds[0], **kw)
                norms = np.linalg.norm(X["test"].astype(np.float64), axis=1)
                row["regression_r2"] = rr.test_acc
                row["spearman_norm_target"] = float(
                    sstats.spearmanr(norms, raw["test"]).statistic)
            rows.append(row)
            print(f"[{task}] layer {layer:2d}  lin={row['acc_linear']:.4f} "
                  f"mlp={row['acc_mlp']:.4f} ord={row['acc_ordinal']:.4f} "
                  f"norm={row['acc_norm_mlp']:.4f}"
                  + (f"  R2={row['regression_r2']:+.3f}"
                     f"  rho(|h|,len)={row['spearman_norm_target']:+.3f}"
                     if raw is not None else ""))

    if not rows:
        print("no tasks produced magnitude results")
        return

    # FDR across every (task, layer) cell tested here, for the same reason the
    # analyze stage corrects: one McNemar per layer is ~30 tests per task, and an
    # uncorrected alpha would manufacture a handful of spurious wins.
    _, q = benjamini_hochberg([m[0]["p_value"] for m in mcs])
    for row, (mc_lin, mc_ord), qv in zip(rows, mcs, q):
        row["q_linear_vs_mlp"] = float(qv)
        # mc_ord compares the ordinal probe against the MLP, so a verdict of
        # monotonic_encoding means the MLP's win over the LINEAR probe survived FDR
        # but it retains no interesting edge over an ORDERED readout -- the missing
        # ingredient was the ordering, not a nonlinearity.
        row["verdict"] = classify_layer(mc_lin, qv, mc_ordinal=mc_ord)

    counts = pd.Series([r["verdict"] for r in rows]).value_counts().to_dict()
    print(f"\nverdicts: {counts}")
    out = rdir / "magnitude.csv"
    print(f"\nwrote {out} ({len(_merge_rows(out, rows, 'task'))} rows)")


def stage_robustness(cfg: Config, args) -> None:
    """Step 0b: is each task's ID peak stable under a one-step change of scale?

    The question is NOT "where is the peak" but "does the peak survive halving and
    doubling k" -- Cheng et al.'s Figure C.2 criterion. A peak that moves under a
    one-step scale change is an artefact of the scale choice, not a finding, so this
    table is what licenses (or refuses) any per-task peak claim.

    Reads what `run.py id` already wrote; computes nothing new.
    """
    rdir = cfg.results_dir
    rob_path, tab_path = rdir / "scale_robustness.csv", rdir / "table_c1_scales.csv"
    if not rob_path.exists():
        raise SystemExit(f"{rob_path} not found -- run `python run.py id` first")

    rob = pd.read_csv(rob_path)
    tab = pd.read_csv(tab_path).set_index("tag") if tab_path.exists() else pd.DataFrame()

    rows = []
    for tag, g in rob.groupby("tag"):
        peaks, ks = {}, {}
        for which, gw in g.groupby("which"):
            gw = gw.sort_values("layer")
            vals = gw["id"].to_numpy(float)
            if np.any(np.isfinite(vals)):
                peaks[which] = int(gw["layer"].to_numpy()[np.nanargmax(vals)])
                ks[which] = int(gw["k"].iloc[0])
        found = [peaks[w] for w in ("half", "chosen", "double") if w in peaks]
        row = {
            "tag": tag,
            "peak_half": peaks.get("half", -1), "k_half": ks.get("half", -1),
            "peak_chosen": peaks.get("chosen", -1), "k_chosen": ks.get("chosen", -1),
            "peak_double": peaks.get("double", -1), "k_double": ks.get("double", -1),
            # The C.2 verdict: how far the peak travels across a factor-of-4 span of k.
            "peak_range": int(max(found) - min(found)) if found else -1,
            "stable": bool(found and (max(found) - min(found)) <= 1),
        }
        if tag in tab.index:
            row.update({
                "k_if_alone": int(tab.loc[tag, "k_if_alone"]),
                # How much sharing the corpus's scale cost this dataset. A large
                # ratio means the borrowed k is nowhere near where this task would
                # have measured itself.
                "borrowed_k_ratio": float(tab.loc[tag, "k"]) / float(tab.loc[tag, "k_if_alone"]),
                "median_spread": float(tab.loc[tag, "median_spread"]),
                # <0.10 is the plateau criterion used in choose_common_scale; without
                # a plateau the ID estimate is scale-dependent at every scale.
                "has_plateau": bool(float(tab.loc[tag, "median_spread"]) < 0.10),
            })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)
    out = rdir / "peak_robustness.csv"
    df.to_csv(out, index=False)

    print(f"\n=== Step 0b: peak stability across scale ({cfg.model_tag}) ===")
    for _, r in df.iterrows():
        extra = (f" | k_alone={r.k_if_alone:>3} (borrowed {r.borrowed_k_ratio:>5.1f}x)"
                 f" spread={r.median_spread:.3f}{'' if r.has_plateau else ' NO PLATEAU'}"
                 if "k_if_alone" in df.columns and pd.notna(r.get("k_if_alone")) else "")
        print(f"{r.tag:26s} peak @ k/2={r.peak_half:>3} k={r.peak_chosen:>3} "
              f"2k={r.peak_double:>3} | range {r.peak_range:>2} "
              f"{'STABLE' if r.stable else 'MOVES'}{extra}")
    n_stable = int(df.stable.sum())
    print(f"\n{n_stable}/{len(df)} tags keep their peak within one layer across "
          f"a 4x span of k. A peak that moves is not a finding.")
    print(f"wrote {out}")


def stage_analyze(cfg: Config, args) -> None:
    from idprobe.analyze import run_analysis

    run_analysis(cfg)


def stage_figures(cfg: Config, args) -> None:
    """Reproductions of Cheng et al.'s figures: C.1, 1, 5/I.1, I.2, H.1."""
    from idprobe import figures as F
    from idprobe.cka import cka_matrix

    rdir = cfg.results_dir
    made = []
    for name, fn in [("cheng_C1_scale_analysis", lambda: F.figure_c1(cfg)),
                     ("cheng_fig1_id_layers", lambda: F.figure_1(cfg)),
                     ("cheng_fig5_probing_vs_id", lambda: F.figure_5(cfg)),
                     ("cheng_I2_shuffled_control", lambda: F.figure_i2(cfg)),
                     ("dir1_conditional_id", lambda: F.figure_conditional_id(cfg)),
                     ("dir2_magnitude", lambda: F.figure_magnitude(cfg))]:
        try:
            fig = fn()
            path = rdir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            made.append(path); print(f"wrote {path}")
        except Exception as e:
            print(f"SKIP {name}: {type(e).__name__}: {e}")

    # H.1 needs a second model. CKA is computed on TASK activations: the ID
    # corpora are filtered to an exact token count, and two tokenizers keep
    # different subsets, so corpus rows would not correspond to the same inputs.
    other = args.compare_model
    if other:
        oc = Config(model_id=other)
        task = args.cka_task
        pa, pb = cfg.act_path(task, "train"), oc.act_path(task, "train")
        if pa.exists() and pb.exists():
            na, nb = E.n_layers_of(pa), E.n_layers_of(pb)
            M = cka_matrix(lambda l: E.load_layer(pa, l), na,
                           lambda l: E.load_layer(pb, l), nb)
            np.save(rdir / "cka_matrix.npy", M)
            peaks = pd.read_csv(rdir / "id_peaks.csv").set_index("tag")
            pk_a = (int(peaks.loc[task, "peak_start"]), int(peaks.loc[task, "peak_end"])) \
                if task in peaks.index else None
            fig = F.figure_h1(M, cfg.model_tag, oc.model_tag, peak_a=pk_a)
            path = rdir / "cheng_H1_cross_model_cka.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            made.append(path); print(f"wrote {path}")
        else:
            print(f"SKIP H.1: need {task} activations for both models")
    return made


def stage_plots(cfg: Config, args) -> None:
    from idprobe.plots import make_all

    make_all(cfg, list(cfg.tasks))


STAGES = {
    "check": stage_check, "extract": stage_extract,
    "id": stage_id, "probe": stage_probe, "analyze": stage_analyze,
    "plots": stage_plots, "figures": stage_figures,
    "robustness": stage_robustness,
    "conditional-id": stage_conditional_id, "magnitude": stage_magnitude,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--model", default=None, help="HF model id (default: Qwen/Qwen3-8B-Base)")
    ap.add_argument("--tasks", nargs="+", default=None, choices=list(TASKS))
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-corpus", type=int, default=None,
                    help="sequences for the Pile ID baseline (default 10000)")
    ap.add_argument("--compare-model", default=None,
                    help="second model for the H.1 cross-model CKA figure")
    ap.add_argument("--cka-task", default="bigram_shift",
                    help="task whose activations align the two models for CKA")
    ap.add_argument("--corpora", nargs="+", default=None,
                    help="ID corpora to use (default: pile wikitext bookcorpus)")
    ap.add_argument("--no-shuffled", action="store_true",
                    help="skip the word-shuffled control (Fig 1 centre, Fig I.2)")
    ap.add_argument("--no-corpus", action="store_true",
                    help="skip the Pile ID baseline (per-task ID only)")
    ap.add_argument("--no-tasks", action="store_true",
                    help="skip the probing tasks (corpus ID only). The untrained "
                         "control only needs the corpus profile, and the corpus is "
                         "10k sequences of 20 tokens against 40k of up to 128")
    ap.add_argument("--reference-corpus", default=None,
                    help="corpus whose GRIDE scale the probing tasks borrow")
    ap.add_argument("--rechoose", action="store_true",
                    help="re-derive GRIDE scales from the sweep and repin scales.json")
    # Direction 1 -- class-conditional ID
    ap.add_argument("--cid-k", type=int, default=None,
                    help="override the GRIDE scale for conditional ID (default: the "
                         "k the `id` stage used, from table_c1_scales.csv/scales.json)")
    ap.add_argument("--cid-boot", type=int, default=5,
                    help="matched-n subsamples averaged per estimate")
    ap.add_argument("--cid-perm", type=int, default=20,
                    help="label permutations forming the estimator-bias null")
    ap.add_argument("--cid-n-match", type=int, default=None,
                    help="cap the matched sample size per class (default: the smallest "
                         "class). GRIDE is O(n^2), so at n_train=25000 the uncapped "
                         "12500/class costs ~7s per estimate and hundreds per layer; "
                         "ID is also strongly n-dependent, so this must be held FIXED "
                         "across any runs whose DeltaID you intend to compare")
    ap.add_argument("--cid-stride", type=int, default=1,
                    help="take every Nth layer (a cheap preview of the profile)")
    ap.add_argument("--no-residual", action="store_true",
                    help="skip the probe-direction residual ID (saves one probe/layer)")
    # Direction 2 -- magnitude tests
    ap.add_argument("--mag-stride", type=int, default=1,
                    help="take every Nth layer in the magnitude stage")
    # The magnitude readouts get their own optimisation budget, because the readouts
    # do NOT converge at the same rate and comparing them before they have converged
    # measures convergence rate rather than encoding. Measured on 1.7B layer 11,
    # sentence_length (n=25000): at 100 epochs the MLP is already flat (0.7940) while
    # linear and ordinal are still climbing, and by 200 epochs they reach 0.7614 and
    # 0.8171. The defaults below are sized for that scale.
    #
    # Epoch count is NOT scale-free: what matters is gradient steps, which is
    # epochs * n_train / batch. At n=2000 these defaults give only ~800 steps and
    # undertrain badly -- there the ordinal probe needs ~400 epochs at batch 128 to
    # stop losing to the linear probe. Raise --mag-epochs when n_train is small.
    ap.add_argument("--mag-epochs", type=int, default=200)
    ap.add_argument("--mag-batch", type=int, default=512)
    ap.add_argument("--mag-patience", type=int, default=25)
    ap.add_argument("--random-init", action="store_true",
                    help="Step 0c: untrained-weights control. Same architecture and "
                         "tokenizer, random weights; writes to a separate -randominit tag")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.random_init = args.random_init
    for attr, val in [
        ("model_id", args.model), ("n_train", args.n_train), ("n_val", args.n_val),
        ("n_test", args.n_test), ("batch_size", args.batch_size), ("device", args.device),
        ("n_corpus", args.n_corpus), ("reference_corpus", args.reference_corpus),
    ]:
        if val is not None:
            setattr(cfg, attr, val)
    if args.tasks:
        cfg.tasks = tuple(args.tasks)
    if args.corpora:
        cfg.corpora = tuple(args.corpora)
    if args.seeds:
        cfg.probe_seeds = tuple(args.seeds)

    STAGES[args.stage](cfg, args)


if __name__ == "__main__":
    main()
