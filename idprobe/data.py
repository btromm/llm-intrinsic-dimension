"""Fetch and prepare the SentEval probing tasks plus the corpus ID baseline.

SentEval probing files are TSV with columns: split ("tr"/"va"/"te"), label, sentence.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

from .config import CORPORA, DATA, SENTEVAL_URL, TASKS, Config

SPLIT_MAP = {"tr": "train", "va": "val", "te": "test"}


def download_task(task: str, force: bool = False) -> "pd.DataFrame":
    """Download one SentEval probing file and return it as a DataFrame."""
    raw = DATA / "raw" / f"{task}.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if force or not raw.exists():
        # Download to a temp path and rename on success: writing straight to
        # `raw` would leave an interrupted download looking like a valid cached
        # file, and the `raw.exists()` guard would never re-fetch it.
        tmp = raw.with_suffix(".txt.part")
        urllib.request.urlretrieve(SENTEVAL_URL.format(task=task), tmp)
        os.replace(tmp, raw)
    df = pd.read_csv(
        raw, sep="\t", header=None, names=["split", "label", "sentence"],
        quoting=3, on_bad_lines="skip",
    )
    df["split"] = df["split"].map(SPLIT_MAP)
    return df.dropna(subset=["split", "label", "sentence"])


def prepare_task(task: str, cfg: Config, force: bool = False) -> dict[str, "pd.DataFrame"]:
    """Download, subsample, and integer-encode labels for one task.

    Subsampling is stratified so that class balance survives; this matters for
    word_content, which has 1000 classes with only ~100 examples each.
    """
    df = download_task(task, force=force)
    classes = sorted(df["label"].unique())
    code = {c: i for i, c in enumerate(classes)}
    df["y"] = df["label"].map(code)

    caps = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}
    out = {}
    for split, cap in caps.items():
        sub = df[df["split"] == split]
        if len(sub) > cap:
            sub = _stratified_sample(sub, cap, seed=cfg.data_seed)
        out[split] = sub.reset_index(drop=True)
    return out


def _stratified_sample(df: "pd.DataFrame", n: int, seed: int) -> "pd.DataFrame":
    """Sample n rows keeping class proportions, via positional indices.

    Uses groupby().indices rather than groupby().apply(): as of pandas 3.0 the
    grouping column is excluded from the frame handed to apply, which silently
    drops the label column we are stratifying on.
    """
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    picks = []
    for _, pos in df.groupby("y").indices.items():
        take = min(len(pos), max(1, round(len(pos) * frac)))
        picks.append(rng.choice(pos, size=take, replace=False))
    sel = np.concatenate(picks)
    if len(sel) > n:
        sel = rng.choice(sel, size=n, replace=False)
    return df.iloc[np.sort(sel)]


def shuffle_words(text: str, rng: "np.random.Generator") -> str:
    """Randomly permute word order within a text.

    Cheng et al.'s central control: feeding *shuffled* corpora should flatten the
    ID profile, because the high-ID phase is supposed to reflect genuine
    linguistic processing rather than any property of the token distribution.
    Shuffling words preserves the unigram distribution exactly and destroys only
    syntax and compositional meaning, which is what makes it a sharp control --
    any ID peak that survives cannot be about linguistic structure.
    """
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def prepare_corpus(cfg: Config, corpus: str = "bookcorpus", mode: str = "sane") -> list[str]:
    """Cheng et al.'s ID corpora: n_corpus documents truncated to corpus_seq_len tokens.

    Truncation to an exact token count happens in the extractor; here we only
    select documents long enough to cover the budget, and optionally shuffle.
    """
    from datasets import load_dataset

    repo, config_name = CORPORA[corpus]
    args = (repo, config_name) if config_name else (repo,)
    print(f"[corpus:{corpus}] opening {repo} in streaming mode "
          "(Hugging Face may download metadata)...", flush=True)
    try:
        ds = load_dataset(*args, split="train", streaming=True)
        print(f"[corpus:{corpus}] streaming dataset ready", flush=True)
    except RuntimeError as exc:
        # `datasets` constructs a shared torch scalar for streaming iteration.
        # Some macOS PyTorch builds ship a torch_shm_manager that cannot start;
        # an in-memory Dataset has identical row order and avoids shared memory.
        if "torch_shm_manager" not in str(exc):
            raise
        print("[corpus:bookcorpus] streaming unavailable (torch_shm_manager); "
              "loading the non-streaming dataset instead. Download/preparation "
              "progress should appear below.", flush=True)
        ds = load_dataset(*args, split="train", streaming=False)
        print(f"[corpus:{corpus}] non-streaming dataset ready ({len(ds)} rows)", flush=True)

    want = cfg.n_corpus * 3          # oversample; many documents are too short
    print(f"[corpus:{corpus}] scanning until {want} documents have at least "
          f"{cfg.corpus_seq_len} whitespace-separated tokens...", flush=True)
    texts = []
    for row in ds:
        t = row.get("text", "")
        if isinstance(t, str) and len(t.split()) >= cfg.corpus_seq_len:
            texts.append(t)
            if len(texts) % 5000 == 0:
                print(f"[corpus:{corpus}] collected {len(texts)}/{want} documents",
                      flush=True)
        if len(texts) >= want:
            break

    print(f"[corpus:{corpus}] selecting {cfg.n_corpus} documents with seed "
          f"{cfg.id_seed} ({mode} mode)", flush=True)
    rng = np.random.default_rng(cfg.id_seed)
    idx = rng.permutation(len(texts))[: cfg.n_corpus]
    out = [texts[i] for i in idx]
    if mode == "shuffled":
        out = [shuffle_words(t, rng) for t in out]
    return out


def shuffle_task_sentences(df: "pd.DataFrame", seed: int = 0) -> "pd.DataFrame":
    """Word-shuffled copy of a probing split, labels untouched (Figure I.2).

    Probes trained on these must sit at chance. Cheng et al. use this to argue
    that reported accuracy is equivalent to *selectivity* (Hewitt & Liang 2019):
    if a probe can only succeed when the linguistic structure is intact, the
    probe is reading the model rather than memorising the task.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["sentence"] = [shuffle_words(s, rng) for s in out["sentence"]]
    return out


def label_counts(splits: dict[str, "pd.DataFrame"]) -> str:
    return " ".join(f"{k}={len(v)}" for k, v in splits.items())


__all__ = ["download_task", "prepare_task", "prepare_corpus", "shuffle_words",
           "shuffle_task_sentences", "label_counts", "TASKS", "CORPORA"]
