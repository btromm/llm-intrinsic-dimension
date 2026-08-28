"""Extract per-layer last-token hidden states from a causal LM.

Cheng et al. represent a sequence by its LAST token's hidden state at each layer:
under causal attention that is the only position guaranteed to have attended to
the whole sequence. Mean-pooling would mix positions with different context windows.

Output layout is a float16 memmap of shape [n_layers + 1, n_sequences, d_model].
Layer 0 is the embedding output, so index l is "after l transformer blocks".
Layer-major ordering is deliberate: every downstream consumer (ID estimation,
probe training) works one layer at a time, so this keeps those reads contiguous.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config

# float16 tops out here; bfloat16 activations can exceed it and would become inf.
FP16_MAX = 65504.0


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if device.type == "mps" and spec == "bfloat16":
        return torch.float16  # bf16 support on MPS is patchy
    return getattr(torch, spec)


def load_model(cfg: Config):
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if cfg.random_init:
        # The untrained control (Step 0c). Same architecture, same tokenizer, same
        # extraction path -- only the weights differ, which is what makes it a
        # control rather than a different experiment.
        from transformers import AutoConfig

        torch.manual_seed(cfg.random_init_seed)
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(cfg.model_id), dtype=dtype)
        print(f"[extract] RANDOM INIT: {cfg.model_id} architecture, untrained weights "
              f"(seed {cfg.random_init_seed}) -> tag {cfg.model_tag}")
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)
    model.to(device).eval()
    return tok, model, device


def filter_by_length(sentences: list[str], tok, exact_len: int) -> list[str]:
    """Keep only sentences with >= exact_len tokens, for fixed-length ID corpora.

    Done before allocating the memmap so its row count is known up front and
    stays aligned with the caller's labels.
    """
    keep = []
    for s in sentences:
        if len(tok(s, truncation=True, max_length=exact_len + 1)["input_ids"]) >= exact_len:
            keep.append(s)
    return keep


@torch.no_grad()
def extract(
    sentences: list[str],
    tok,
    model,
    device: torch.device,
    out_path: Path,
    batch_size: int = 64,
    max_length: int = 128,
    exact_len: int | None = None,
) -> Path:
    """Write last-token hidden states for `sentences` to `out_path` as a memmap.

    `exact_len` truncates every sequence to exactly that many tokens,
    reproducing Cheng et al.'s fixed-length ID corpora. Callers should run
    `filter_by_length` first so nothing is silently dropped here.
    """
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if exact_len is not None:
        max_length = exact_len

    # Write to a temp path and rename on success. open_memmap allocates the full
    # file up front, so writing straight to out_path would leave an interrupted
    # run looking complete-but-zero-filled to the next run's existence check.
    tmp_path = out_path.with_name(out_path.name + ".partial")
    mm = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float16, shape=(n_layers + 1, len(sentences), d_model)
    )

    n_clipped = 0
    for start in tqdm(range(0, len(sentences), batch_size), desc=out_path.stem):
        batch = sentences[start : start + batch_size]
        enc = tok(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        lengths = enc["attention_mask"].sum(dim=1)
        # [B, 1, D] index selecting each sequence's final real token.
        gather_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, d_model)

        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple, len n_layers+1
        b = len(batch)
        for layer, h in enumerate(hs):
            picked = h.gather(1, gather_idx).squeeze(1)  # [B, D]
            arr = picked.float().cpu().numpy()
            over = np.abs(arr) > FP16_MAX
            if over.any():
                # Outlier features in large LMs can exceed the float16 range.
                # Clip rather than let the cast produce inf, which would poison
                # every downstream distance and standardisation.
                n_clipped += int(over.sum())
                np.clip(arr, -FP16_MAX, FP16_MAX, out=arr)
            mm[layer, start : start + b, :] = arr.astype(np.float16)

    mm.flush()
    del mm  # close the memmap before renaming
    os.replace(tmp_path, out_path)
    if n_clipped:
        print(f"  warning: clipped {n_clipped} activation values to +/-{FP16_MAX} "
              f"(float16 range); check for outlier features")
    return out_path


def load_layer(path: Path, layer: int) -> np.ndarray:
    """Read one layer's [n, d] matrix as float32 (probes and ID both want float32)."""
    mm = np.load(path, mmap_mode="r")
    return np.asarray(mm[layer], dtype=np.float32)


def n_layers_of(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


__all__ = [
    "load_model", "extract", "load_layer", "n_layers_of",
    "filter_by_length", "resolve_device", "resolve_dtype",
]
