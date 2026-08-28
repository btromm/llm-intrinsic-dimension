#!/usr/bin/env python
"""Check, on the machine that will actually run the pipeline, that a long job
would not die three hours in on something knowable in thirty seconds.

Checks, in order: the env (numpy<2 for DADApy, torch, transformers), the GPU,
every model config (which also proves the weights are reachable or cached), the
SentEval task files, streaming access to each ID corpus, and the disk the run
will need against the disk that is free.

Exit status is 0 only if every REQUIRED check passed, so it is safe to put at
the head of a Slurm dependency chain.

    python slurm/preflight.py --models Qwen/Qwen3-8B-Base [--corpora bookcorpus ...]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, bool, str]] = []   # (name, ok, required, detail)


def check(name: str, required: bool = True):
    """Run a check, record PASS/FAIL/WARN, never raise."""
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, required, detail))
        except Exception as e:                     # noqa: BLE001
            RESULTS.append((name, False, required, f"{type(e).__name__}: {e}"))
            if "--traceback" in sys.argv:
                traceback.print_exc()
        tag = "PASS" if RESULTS[-1][1] else ("FAIL" if required else "WARN")
        print(f"[{tag}] {name}: {RESULTS[-1][3]}", flush=True)
        return fn
    return deco


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--corpora", nargs="+", default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--no-shuffled", action="store_true")
    ap.add_argument("--skip-corpus-stream", action="store_true",
                    help="skip the network test for the ID corpora")
    ap.add_argument("--no-gpu", action="store_true",
                    help="downgrade the GPU check to a warning (for a login node)")
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args()

    from idprobe.config import TASKS, Config

    cfg = Config()
    for attr, val in [("n_train", args.n_train), ("n_val", args.n_val),
                      ("n_test", args.n_test)]:
        if val is not None:
            setattr(cfg, attr, val)
    tasks = tuple(args.tasks) if args.tasks else tuple(TASKS)
    corpora = tuple(args.corpora) if args.corpora else cfg.corpora
    n_modes = 1 if args.no_shuffled else 2

    print(f"preflight for {len(args.models)} model(s): {' '.join(args.models)}")
    print(f"tasks: {' '.join(tasks)}")
    print(f"corpora: {' '.join(corpora)} x {n_modes} mode(s)")
    print("-" * 70)

    @check("python packages")
    def _pkgs():
        import numpy, torch, transformers            # noqa: PLC0415
        import dadapy, pandas, scipy, sklearn, statsmodels   # noqa: F401,PLC0415
        if not numpy.__version__.startswith("1."):
            raise RuntimeError(
                f"numpy {numpy.__version__} but DADApy needs numpy<2 -- "
                f"this env was built wrong, rebuild with slurm/setup.sh")
        return (f"torch {torch.__version__}, transformers {transformers.__version__}, "
                f"numpy {numpy.__version__}")

    @check("gpu visible", required=not args.no_gpu)
    def _gpu():
        import torch                                  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- the job did "
                               "not get a GPU (check GPU_REQUEST in config.sh)")
        p = torch.cuda.get_device_properties(0)
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        _ = (x @ x).float().sum().item()              # prove bf16 matmul works
        return f"{p.name}, {p.total_memory / 1e9:.0f} GB, bfloat16 matmul OK"

    total_bytes = 0
    for model_id in args.models:
        @check(f"model config: {model_id}")
        def _model(model_id=model_id):
            nonlocal total_bytes
            from transformers import AutoConfig, AutoTokenizer   # noqa: PLC0415
            mc = AutoConfig.from_pretrained(model_id)
            AutoTokenizer.from_pretrained(model_id)
            n_layers, d = mc.num_hidden_layers, mc.hidden_size
            per_seq = (n_layers + 1) * d * 2
            n_task_seq = (cfg.n_train + cfg.n_val + cfg.n_test) * len(tasks) * n_modes
            n_corp_seq = cfg.n_corpus * len(corpora) * n_modes
            need = per_seq * (n_task_seq + n_corp_seq)
            total_bytes += need
            return (f"{n_layers} layers, d_model={d} -> {per_seq / 1e6:.2f} MB/seq, "
                    f"{need / 1e9:.0f} GB of activations")

    @check("SentEval task files")
    def _senteval():
        from idprobe import data as D                 # noqa: PLC0415
        got = []
        for t in tasks:
            df = D.download_task(t)
            got.append(f"{t}={len(df)}")
        return " ".join(got)

    if not args.skip_corpus_stream:
        @check("ID corpora reachable (streaming)")
        def _corpora():
            from datasets import load_dataset          # noqa: PLC0415

            from idprobe.config import CORPORA         # noqa: PLC0415
            got = []
            for c in corpora:
                repo, name = CORPORA[c]
                ds = (load_dataset(repo, name, split="train", streaming=True)
                      if name else load_dataset(repo, split="train", streaming=True))
                row = next(iter(ds))
                got.append(f"{c}({len(row.get('text', ''))} chars)")
            return " ".join(got)

    @check("activations directory writable")
    def _acts():
        from idprobe.config import ACTS               # noqa: PLC0415
        ACTS.mkdir(parents=True, exist_ok=True)
        probe = ACTS / ".preflight"
        probe.write_bytes(b"ok")
        probe.unlink()
        real = ACTS.resolve()
        if real == ACTS and not ACTS.is_symlink():
            note = " (NOT a symlink -- activations will land inside the repo; " \
                   "run slurm/setup.sh to point them at scratch)"
        else:
            note = ""
        return f"{ACTS} -> {real}{note}"

    @check("disk headroom")
    def _disk():
        from idprobe.config import ACTS               # noqa: PLC0415
        free = shutil.disk_usage(ACTS).free
        msg = (f"need ~{total_bytes / 1e9:.0f} GB for all models, "
               f"{free / 1e9:.0f} GB free")
        if free < total_bytes:
            raise RuntimeError(
                msg + " -- extract will fail partway. Either free space, cut "
                      "--n-train, set RUN_SHUFFLED=0, or run the models one at a "
                      "time and delete each model's activations after its "
                      "analyze job (slurm/cleanup_activations.sh).")
        return msg

    print("-" * 70)
    failed = [n for n, ok, req, _ in RESULTS if not ok and req]
    warned = [n for n, ok, req, _ in RESULTS if not ok and not req]
    if warned:
        print(f"{len(warned)} warning(s): {', '.join(warned)}")
    if failed:
        print(f"PREFLIGHT FAILED: {', '.join(failed)}")
        return 1
    print(f"PREFLIGHT OK ({len(RESULTS)} checks) -- "
          f"projected activation storage {total_bytes / 1e9:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
