# Running this pipeline on a Slurm cluster

Everything here is one experiment: extract per-layer hidden states from a Qwen3
model, estimate the intrinsic dimension (ID) of each layer, train linear and MLP
probes at each layer, and test whether the ID peak predicts where probes succeed.
The project README explains the science; this file is only about getting it to
run on a scheduler.

Three Qwen3 base models by default — 1.7B, 4B, 8B — one GPU at a time, no
distributed training, no multi-node anything. The largest job is a single H100
holding an 8B model in bfloat16 (~16 GB of weights).

---

## Quick start

```bash
git clone <this repo> && cd llm-intrinsic-dimension

$EDITOR slurm/config.sh     # 1. set ACCOUNT / PARTITION_GPU / GPU_REQUEST / scratch
bash slurm/setup.sh         # 2. build the conda env, prefetch weights + data  (login node)
bash slurm/submit_all.sh    # 3. submit the whole graph
```

`bash slurm/submit_all.sh -n` first prints exactly what it would submit without
submitting anything.

If you want the simplest possible thing instead — one model, one job, no
dependency graph:

```bash
sbatch --export=ALL,PIPE_REPO=$PWD,PIPE_MODEL=Qwen/Qwen3-8B-Base \
       -A <account> -p <gpu-partition> slurm/single_model.sbatch
```

---

## The three things that must be right in `config.sh`

| variable | what it is | how to find it |
|---|---|---|
| `ACCOUNT` | the allocation to bill | `sacctmgr show assoc user=$USER format=account,partition` |
| `PARTITION_GPU` | partition with the H100s | `sinfo -s` |
| `GPU_REQUEST` | how this site asks for a GPU | `--gres=gpu:1` at most sites; some want `--gpus=1` or `--gres=gpu:h100:1` |

Also check `SCRATCH_BASE` — it defaults to `$SCRATCH`, then `$WORK`, then
`~/scratch`. Activations must not land in `$HOME`; `setup.sh` symlinks
`activations/` to whatever you set.

Everything else has a working default. Anything left empty is simply not passed
to `sbatch`, so the site default applies.

---

## What gets submitted

```
preflight ──> extract[i] ─┬─> id[i] ─┬─> analyze[i]
                          │          └─> directions[i]
                          └─> probe[i] ──> analyze[i]
          ──> control[i]
```

One array task per model (`i` = index into `MODELS`). Stages are chained with
`aftercorr`, which pairs array task `i` of one job to array task `i` of the
next — so the 1.7B ID job starts as soon as 1.7B extraction finishes rather than
waiting for the 8B. `id` (CPU-only) and `probe` (GPU) run concurrently.

| job | script | GPU | default walltime | what it does |
|---|---|---|---|---|
| preflight | `00_preflight.sbatch` | yes | 30 min | proves the env, GPU, weights, data and disk are all there. Everything else depends on it with `afterok`, so a broken setup fails in minutes instead of hours. Ends with the tiny-model smoke test from the project README. |
| extract | `01_extract.sbatch` | yes | 12 h | last-token hidden states for every (task, mode) and (corpus, mode) |
| id | `02_id.sbatch` | **no** | 24 h | GRIDE ID profile per layer, scale selection, `scales.json` |
| probe | `03_probe.sbatch` | yes | 24 h | linear + MLP probes, every layer × task × seed, plus the shuffled control |
| analyze | `04_analyze.sbatch` | no | 2 h | peak robustness, McNemar + FDR, correlations, figures |
| directions | `05_directions.sbatch` | yes | 24 h | `conditional-id` and `magnitude` (docs/NEXT_STEPS.md). Optional: `RUN_DIRECTIONS=0`. |
| control | `06_random_init.sbatch` | yes | 6 h | Step 0c: the same architecture with untrained weights. Cheap, and the cheapest falsification the project has. Optional: `RUN_RANDOM_INIT=0`. |

The `id` array is throttled to one task at a time (`%1`). Every model writes to
the same `scales.json` and the stage rewrites that file wholesale, so two models
finishing together would clobber each other's pinned GRIDE scale.

---

## Disk

Activation storage is `n_sequences × (n_layers + 1) × d_model × 2` bytes, and it
dominates everything else the run writes. At default sizes (25k/5k/10k per task,
5 tasks, plus the word-shuffled control, plus the bookcorpus baseline in
both modes, 10k sequences each):

| model | layers × d_model | per sequence | total activations |
|---|---|---|---|
| Qwen3-1.7B-Base | 28 × 2048 | 0.12 MB | ~50 GB |
| Qwen3-4B-Base | 36 × 2560 | 0.19 MB | ~80 GB |
| Qwen3-8B-Base | 36 × 4096 | 0.30 MB | ~127 GB |
| **all three** | | | **~260 GB** |

These are estimates from each model's published config; the preflight job prints
the exact projection for your settings, compares it against the free space on
your scratch, and fails if it does not fit. Three ways to shrink it:

* `RUN_SHUFFLED=0` — drops the word-shuffled control and halves the task
  activations. It is a real control (Cheng et al.'s Figure I.2), so this is a
  scientific cut, not a free one.
* Lower `N_TRAIN`. **Leave `N_TEST` at 10000** — the linear-vs-MLP equivalence
  test has no power below that, and every layer comes back `inconclusive`.
  The project README explains the arithmetic.
* Run models one at a time and reclaim as you go:
  `bash slurm/cleanup_activations.sh Qwen3-1.7B-Base` (asks for confirmation,
  and refuses while that model's results are incomplete).

Results (`results/<model-tag>/`) are small — CSVs, figures, and one npz of
per-example probe predictions.

---

## Rough runtimes

Unmeasured guesses for Qwen3-8B on one H100 at default sizes, to size walltimes
by — not benchmarks. Smaller models scale down roughly with parameter count, and
Slurm bills what you use, not what you ask for.

| stage | expected | note |
|---|---|---|
| extract | 1–3 h | GPU-bound at first, then I/O-bound writing ~127 GB |
| id | 2–4 h | CPU-bound. GRIDE is O(n²) over 10k points per layer per tag |
| probe | 3–6 h | ~2200 small probe fits |
| analyze | minutes | reads what is already on disk |
| directions | 4–10 h | dominated by `magnitude`; `conditional-id` is capped by `CID_N_MATCH` |
| control | < 1 h | corpus only, 10k sequences of 20 tokens |

---

## Monitoring

```bash
squeue -u $USER                        # what is queued and running
squeue -u $USER --start                # when Slurm thinks they will start
tail -f logs/idp-extract-*_0.out       # follow the first array task's log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES
```

Every job log opens with a banner naming the host, the GPU, the model, the
CPU/memory allocation and the free disk, and closes with the elapsed time and
(where `seff` exists) the efficiency report. That is usually enough to decide
whether a resource number in `config.sh` needs changing.

---

## When something goes wrong

**`conda env 'id' not found`** — run `bash slurm/setup.sh`. If conda itself is
not on `PATH` in batch jobs, set `CONDA_SH` (and/or `MODULES`) in `config.sh`.

**`numpy 2.x but dadapy needs numpy<2`** — the env was built from `base` or was
polluted afterwards. Rebuild: `conda env remove -n id && bash slurm/setup.sh`.
This is the one dependency constraint the project cannot work around; DADApy is
the reference GRIDE implementation and it pins `numpy<2`.

**`torch.cuda.is_available() is False`** — the job did not get a GPU. Check
`GPU_REQUEST` against what this site expects, and that `PARTITION_GPU` is a GPU
partition.

**CUDA out of memory** in `extract` — lower that model's entry in
`BATCH_SIZES`. Nothing else in the pipeline is GPU-memory-hungry.

**Jobs stuck in `DependencyNeverSatisfied`** — an upstream job failed. They are
submitted with `--kill-on-invalid-dep=yes`, so they should be cancelled
automatically; if any survive, `scancel` them, fix the cause, and resubmit the
remaining stages with `bash slurm/submit_all.sh -s id,probe,analyze --no-preflight`.

**A stage says `no activations, skipping`** — extraction for that model or tag
did not finish. Re-run `extract`; it skips what already exists.

**Compute nodes have no internet.** `setup.sh` downloads the model weights and
the SentEval files on the login node, so those are fine offline. The ID
*corpus* (`bookcorpus`) is read with the streaming API,
which does need network at extraction time — the preflight job tests exactly
this on a real compute node. If it fails there, either set a proxy through
`EXTRA_ENV_SETUP` in `config.sh`, or drop the corpus baseline and pin the GRIDE
scale by hand (`--no-corpus` plus an entry in `scales.json`; see the `id` stage's
error message, which spells out the options).

### If a job hits the walltime

| stage | resumable? | what to do |
|---|---|---|
| extract | **yes** | each (tag, split) is written to `.partial` and renamed on success, and existing outputs are skipped. Just resubmit. |
| id | no | writes everything at the end. Raise `TIME_ID`. With one corpus the sweep is already down to 5 task tags plus 2 corpus tags, and the only lever left is fewer tasks — which overwrites `id_profiles.csv`, so prefer the walltime. |
| probe | no | writes `probe_results.csv` once, at the end, **and overwrites it**. Raise `TIME_PROBE` first. If your site caps walltime too low, run it in task-sized pieces (`--tasks sentence_length word_content`), move `probe_results.csv` and `probe_correct.npz` aside after each piece, and concatenate the CSVs / merge the npz keys before running `analyze` — the analysis reads both by `task/layer/kind/seed` key, so a straight concatenation is enough. Cutting `--seeds 1` (from three) is the cheaper fix. |
| analyze | n/a | reads what is on disk; re-run freely |
| directions | partial | both stages merge per task into their CSV under a lock, so completed tasks survive. Use `CID_STRIDE`/`MAG_STRIDE` to preview every other layer. |
| control | yes | same `.partial` rename as extract |

---

## Outputs

Per model, in `results/<model-tag>/`:

| file | contents |
|---|---|
| `id_profiles.csv` | ID per (tag, layer) at the selected GRIDE scale |
| `id_scaling_curves.json` | the full scale sweep — worth eyeballing |
| `id_peaks.csv`, `peak_robustness.csv` | peak layer, and whether it survives a 4× span of k |
| `probe_results.csv` | accuracy per task × layer × probe × seed |
| `linear_vs_mlp.csv` | McNemar + BH-corrected q per (task, layer) |
| `id_accuracy_correlation.csv`, `peak_alignment.csv` | the headline tests |
| `conditional_id.csv`, `magnitude.csv` | the two directions, if enabled |
| `cheng_*.png`, `dir*.png` | figure reproductions |

`scales.json` in the repo root is also updated: it pins one GRIDE scale per
(model, corpus) so later stages and later runs measure at the same k. It is a
tracked file, so `git status` will show it as modified after the `id` stage —
that is expected, and worth committing alongside the results.

---

## Files

| file | |
|---|---|
| `config.sh` | the only file you should need to edit |
| `setup.sh` | run once on a login node: env, scratch symlinks, prefetch |
| `submit_all.sh` | submits the graph; `-n` for a dry run, `-h` for options |
| `00_…`–`06_…sbatch` | one stage each; readable `#SBATCH` headers |
| `single_model.sbatch` | the whole pipeline for one model in one job |
| `common.sh` | shared env setup and argument building (sourced, not run) |
| `preflight.py` | the checks stage 0 runs |
| `cleanup_activations.sh` | reclaim scratch after a model's analysis is done |
