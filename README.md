# Intrinsic dimensionality vs. feature entanglement in LLM representations

Does the intrinsic-dimension (ID) peak of a layer's representations predict where
linear probes succeed — and does that peak *discriminate* between linguistic tasks?
And where a linear probe works, is the feature genuinely disentangled, or is
linearity merely a sufficient approximation?

## Design

| | |
|---|---|
| **Model** | `Qwen/Qwen3-8B-Base` (36 layers, d=4096) |
| **Tasks** | 5 SentEval probing tasks (Conneau et al. 2018), the set Cheng et al. keep |
| **Representation** | last-token hidden state at each layer (0 = embeddings, 36 = final) |
| **ID** | GRIDE (Denti et al. 2022) via [DADApy](https://github.com/sissa-data-science/DADApy) |
| **Probes** | linear and 1×200 MLP, identical training protocol |
| **Tests** | McNemar (linear vs MLP) + BH-FDR; shift-permuted Spearman (ID vs accuracy) |

Tasks and their grouping, following Cheng et al. §I.1:

| task | group | classes |
|---|---|---|
| `sentence_length` | surface | 6 |
| `word_content` | surface | 1000 |
| `bigram_shift` | syntax | 2 |
| `coordination_inversion` | semantics | 2 |
| `odd_man_out` | semantics | 2 |

They drop `past_present`/`subj_number`/`obj_number` (ceiling effects) and
`top_constituents`/`tree_depth` (dependent on automated parses). We follow that.

## Setup

DADApy pins `numpy<2`, so this project gets its own env — installing it alongside
numpy 2.x silently downgrades numpy for everything else sharing that env.

```bash
conda env create -f environment.yml && conda activate id
pip install torch --index-url https://download.pytorch.org/whl/cu124  # match the H100's CUDA
python run.py check
```

Everything below assumes the `id` env is active (or prefix with `conda run -n id`).

`check` prints the resolved device, dtype, layer count, and — importantly — the
disk the run will need.

## Pipeline

```bash
python run.py extract    # last-token hidden states -> activations/  (the expensive stage)
python run.py id         # GRIDE ID profile per layer, per task + Pile baseline
python run.py probe      # linear + MLP probe at every layer x task x seed
python run.py analyze    # McNemar, FDR, ID<->accuracy correlations -> results/
```

The prerequisites and the two further directions (see `docs/NEXT_STEPS.md`) add
their own stages:

```bash
python run.py robustness       # Step 0b: is each ID peak stable across a 4x span of k?
python run.py conditional-id   # Direction 1: DeltaID = ID(X) - mean_c ID(X|y=c)
python run.py magnitude        # Direction 2: ordinal / regression / norm readouts
```

**Step 0c, the untrained control.** `--random-init` keeps the architecture, tokenizer
and extraction path and swaps only the weights, writing to a separate
`<model>-randominit` tag so it can never overwrite the trained run. Cheng et al.
report the ID peak is absent in untrained models, which makes this the cheapest
falsification available — if a peak still appears, it belongs to the architecture or
the extraction rather than to anything learned.

```bash
python run.py extract --model Qwen/Qwen3-1.7B-Base --random-init --no-shuffled
python run.py id      --model Qwen/Qwen3-1.7B-Base --random-init --reference-corpus pile
```

Every stage is resumable — it skips existing outputs unless you pass `--force`.

**Validate the wiring first.** A few minutes end-to-end, and probes should sit at
chance because the model is randomly initialised (a useful label-leakage check):

```bash
python run.py extract --model hf-internal-testing/tiny-random-LlamaForCausalLM --tasks bigram_shift --n-train 400 --n-val 100 --n-test 200 --device cpu --no-corpus
```

### Scale and disk

Activation storage is `n_seq × (n_layers+1) × d_model × 2` bytes. For Qwen3-8B
that is **0.30 MB per sequence**:

| per-task n (train/val/test) | per task | 5 tasks |
|---|---|---|
| 25k/5k/10k (default) | 12 GB | **61 GB** |
| 100k/10k/10k (full SentEval) | 36 GB | 180 GB |

Override with `--n-train/--n-val/--n-test`. The default is already well past
where ID estimates and probe accuracies saturate; go to full scale only if you
need the extra power for the McNemar tests.

## Running it on a cluster

`slurm/` holds a complete Slurm submission for the Qwen3 ladder (0.6B, 1.7B, 4B,
8B) on a single H100: one job per stage, chained by dependencies, one array task
per model. Edit `slurm/config.sh` (account, partition, scratch), then

```bash
bash slurm/setup.sh         # login node: build the env, prefetch weights + data
bash slurm/submit_all.sh    # submit; -n prints the plan without submitting
```

`slurm/README.md` covers resources, disk, what is resumable, and the failure
modes worth knowing before a 24-hour job starts. A preflight job checks the env,
GPU, weights, data and free space on a compute node before anything long is
queued behind it.

## Walkthrough notebook

`notebooks/walkthrough.ipynb` explains what each stage does and why, with runnable cells against
whatever results are already on disk. It covers the ID estimator and its scale trap, the last-token
representation choice, the linear-vs-MLP design, the equivalence-testing logic, and a claim-by-claim
comparison against Cheng et al.

```bash
conda run -n id jupyter lab notebooks/walkthrough.ipynb
```

## Design decisions, and where they depart from Cheng et al.

**Per-task ID, plus the corpus baseline.** Cheng et al. compute one ID profile
per *corpus*. Your hypothesis — that different tasks peak at different layers —
is only testable if ID is estimated from each task's own representations, so
that is the primary measurement. `run.py id` additionally computes their
Pile-10k baseline (10k sequences × 20 tokens) so you can show whether per-task
peaks actually depart from the generic profile, or merely echo it.
`peak_alignment.csv` reports the spread of per-task peak layers: **a spread near
zero would mean one shared abstraction phase, and would not support the
hypothesis** — worth knowing before you interpret any correlation.

**Linear probes are primary; the MLP is the control.** Cheng et al. probe with
an MLP throughout. Inverting that turns the comparison into a disentanglement
test: if a 200-unit hidden layer buys nothing over a linear readout, the feature
is plausibly linearly encoded in the sense of the LRH.

**Three splits, not two.** Cheng et al. report the best *validation* accuracy
over 5 seeds. That is optimistically biased. Here validation drives early
stopping and seed selection only, and everything reported comes from a held-out
test split — which is also what makes the McNemar comparison legitimate.

**Identical training protocol across architectures.** Same optimiser, LR, weight
decay, batch size, patience, seeds, and standardisation. Only the architecture
differs, so a gap cannot be attributed to a luckier training run.

**Activation dtype.** Stored as float16, loaded as float32. Cheng et al. do not
specify; fp16 halves 61 GB to a manageable footprint and is far below the noise
floor of an ID estimate.

**MLP activation.** Defaults to `relu`; Cheng et al. used `logistic`. Set
`Config.mlp_activation = "logistic"` for a closer replication. ReLU is the
stronger control — a nonlinearity that fails to help is more convincing evidence
of disentanglement when it is a nonlinearity that usually helps.

### Two statistical traps this code handles explicitly

1. **Layer curves are autocorrelated.** Adjacent layers are near-duplicates, so
   the effective sample size is far below 37 and a textbook Spearman p-value is
   badly anti-conservative. `layer_correlation` uses a circular-shift null that
   preserves each curve's autocorrelation and destroys only their alignment.
2. **~200 McNemar tests** (task × layer) would manufacture ~10 spurious "the MLP
   wins here" layers at α=0.05. `benjamini_hochberg` corrects across all cells.

## Reading the linear-vs-MLP verdict

`stats.classify_layer` labels each layer `nonlinear_advantage` /
`monotonic_encoding` / `linear_sufficient` / `inconclusive`.

`monotonic_encoding` splits the `nonlinear_advantage` verdict in two, and is only
reachable when an ordinal probe is supplied (`mc_ordinal`, which the `magnitude` stage
provides). It means the MLP beat the *linear* probe but retains no interesting edge
over an *ordered* readout — the quantity is encoded monotonically along a direction
and the binned softmax was the limitation, which is a claim about readout geometry
rather than about entanglement. Promotion requires the same equivalence argument as
`linear_sufficient`: the MLP's remaining edge over the ordinal probe must be bounded
below `margin`, not merely non-significant.

The asymmetry matters: **a non-significant McNemar does not establish linear
separability.** Absence of evidence is not evidence of absence, and an
underpowered layer produces exactly that result. So `linear_sufficient` requires
an equivalence argument — the upper confidence bound on the MLP's advantage must
fall below `margin` (default 1 accuracy point), ruling out a gain large enough
to be interesting. Layers clearing neither bar are `inconclusive`, which is the
honest label rather than a failure.

**This is why `n_test` defaults to 10000.** For probes that agree on ~93% of
examples (typical for a linear and an MLP probe over identical features), the CI
on the gap is roughly ±0.5 points at n=10k, ±0.75 at n=5k. Below ~10k you cannot
establish 1-point equivalence at all, and every layer comes back `inconclusive`
no matter how similar the two probes actually are. Shrink the train split if you
need to save time; leave the test split alone.

Adjust `margin` in `classify_layer` if absolute accuracy points are the wrong
scale for your claim — one point at 55% accuracy means something quite different
from one point at 95%.

## Outputs

| file | contents |
|---|---|
| `results/id_profiles.csv` | ID per (tag, layer) at the selected scale |
| `results/id_scaling_curves.json` | full GRIDE scale sweep — inspect these |
| `results/id_peaks.csv` | peak layer, start/end of the high-ID phase |
| `results/probe_results.csv` | accuracy per task × layer × probe × seed |
| `results/linear_vs_mlp.csv` | McNemar + BH q-values per (task, layer) |
| `results/id_accuracy_correlation.csv` | ID↔accuracy, per-task and corpus ID |
| `results/peak_alignment.csv` | do peak layers discriminate across tasks? |
| `results/peak_robustness.csv` | peak layer at k/2, k, 2k; borrowed-vs-own k; plateau flag |
| `results/conditional_id.csv` | DeltaID per (task, layer) with its label-shuffle null |
| `results/magnitude.csv` | linear/MLP/ordinal/norm accuracy, regression R², Spearman(‖h‖, target) |

Always eyeball `id_scaling_curves.json`. `select_plateau` automates a criterion
Cheng et al. apply by eye; if a layer has no plateau, its ID is not trustworthy
at any scale and the automated pick silently returns the largest.

## Three caveats on the new directions

**ΔID cannot be compared across tasks with different class counts.** A label only
removes the dimension it varies along, so the attainable ΔID is set by cardinality,
not only by entanglement. On synthetic data with a known answer: a binary label on
separated clusters gives ΔID → 0 (it reaches 0.56 only when the classes overlap so
heavily that the feature is *hardest* to decode), while one continuous coordinate
gives 0.36 at 5 bins and 0.88 at 20. So "flat for the binary tasks, larger for
`sentence_length`" is close to a tautology of class count. The label-shuffle null
fixes cardinality *within* a task, which makes each task's own z-score valid; it does
nothing for the cross-task comparison. Match class counts before comparing.

**ΔID is not a separability measure.** GRIDE at scale k sees a neighbourhood of ~2k
points. Classes separated by more than that give ΔID ≈ 0 even when a linear probe
scores ~100%, so a near-zero ΔID is ambiguous between "not encoded" and "encoded at a
coarser scale than k". Read it alongside the probe accuracy, never alone.

**Readouts converge at different rates, so compare them only once converged.** The
MLP converges fastest, and a budget too small for the others makes it look like it
found nonlinear structure. On 1.7B layer 11, `sentence_length` (n=25000): at 100
epochs the MLP is already flat at 0.7940 while linear and ordinal are still climbing;
by 200 epochs they reach 0.7614 and **0.8171**. The `magnitude` stage therefore uses
its own budget (`--mag-epochs/--mag-batch`, default 200/512) and records it in the CSV.

Epoch count is not scale-free — what matters is gradient steps (`epochs * n_train /
batch`). At n=25000 the defaults give ~9800 steps and the ordinal probe already beats
the MLP under the pipeline's own `probe` protocol. At n=2000 the same settings give
~800 and undertrain badly, flipping the ordinal probe from winning to losing. Raise
`--mag-epochs` when `n_train` is small.

The `probe` stage is deliberately left untouched, and the linear-vs-MLP *gap* it
reports is robust to convergence — what more training changes is the ordinal readout,
and therefore the explanation of the gap, not the gap itself.

## References

- Cheng, Doimo, Kervadec, Macocco, Yu, Laio, Baroni (2025). *Emergence of a
  High-Dimensional Abstraction Phase in Language Transformers*. ICLR.
- Conneau, Kruszewski, Lample, Barrault, Baroni (2018). *What you can cram into a
  single $&!#* vector*. ACL.
- Denti, Doimo, Laio, Mira (2022). *The generalized ratios intrinsic dimension
  estimator*. Scientific Reports.
- Facco, d'Errico, Rodriguez, Laio (2017). *Estimating the intrinsic dimension of
  datasets by a minimal neighborhood information*. Scientific Reports.
