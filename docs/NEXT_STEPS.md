# Implementation plan — beyond the ID/probe correlation

Three directions, in recommended order. Direction 1 is the one that changes what the
project *measures*; 2 explains its most robust result; 3 is the ambitious extension
that stops this being a replication.

Nothing here assumes the per-task peak result is settled either way — Step 0 re-tests it.

---

## Step 0 — Prerequisites (do first, ~1 evening)

**0a. Fix 1.7B.** Its `bigram_shift/train.npy` was silently overwritten with a
10000-row extraction while every other task used 2000, so its ID (sample-size
dependent) and probes (5x the training data) are not comparable to its siblings.
The file is now correct; the stages that consumed it are not.

```bash
python run.py id     --model Qwen/Qwen3-1.7B-Base --reference-corpus pile --rechoose
python run.py probe  --model Qwen/Qwen3-1.7B-Base --device cpu --seeds 1 2
python run.py analyze --model Qwen/Qwen3-1.7B-Base
python run.py figures --model Qwen/Qwen3-1.7B-Base --compare-model Qwen/Qwen3-0.6B-Base
```

**0b. Re-test peak robustness under the new k-sweep mechanics.** The question is not
"where is the peak" but "is the peak stable across scale". Report, per task:

| quantity | why |
|---|---|
| `peak_layer` at chosen k, k/2, k*2 | already in `scale_robustness.csv` — the C.2 criterion |
| `k_if_alone` vs borrowed k | how much the shared scale distorts this task |
| `median_spread` | whether a plateau exists at all (<0.10) |

**A peak that moves under a one-step scale change is not a finding.** Under the old
mechanics, four of five 0.6B tasks moved between k=2 and k=64. If the new sweep gives
plateaus for the task datasets (1.7B's `bigram_shift` already plateaued alone at
spread 0.057), the split may well be real — but it has to be demonstrated, not assumed.

**0c. Add an untrained-model control.** Cheng et al. report the ID peak is absent in
untrained models; we never ran it. A randomly-initialised Qwen3-0.6B costs one
extraction and falsifies the whole pipeline if a peak still appears.

---

## Direction 1 — Class-conditional ID  *(recommended first)*

### The problem it solves

The pipeline currently correlates a **global** statistic (manifold dimension over a
whole dataset) with a **per-feature** readout (one probe direction). Those live at
different levels of description, which is why the correlations are weak, why they need
a shift-permutation null to be interpretable at all, and why the per-task peaks are
fragile: we are asking a dataset-level quantity to explain a direction-level phenomenon.

Class-conditional ID measures entanglement *in the same units as ID*, turning it from a
probe proxy into a geometric quantity.

### The measurement

For a layer's representations `X` with labels `y`:

```
DeltaID = ID(X)  -  mean_c ID(X | y = c)
```

Reasoning: if a feature is disentangled and linearly encoded, it occupies roughly one
direction, and conditioning on it should remove about one dimension. If it is superposed
across many directions, conditioning collapses the manifold more diffusely and `DeltaID`
is larger. A feature the layer does not encode at all gives `DeltaID ~ 0`.

### The critical control (this experiment fails without it)

`ID(X | y = c)` is computed on ~n/C points while `ID(X)` uses n. **GRIDE is
sample-size biased downward**, so a naive `DeltaID` is mostly a sample-size artifact.
Two mandatory guards:

1. **Match n.** Subsample the pooled set to the size of the smallest class before
   estimating `ID(X)`. Average over several subsamples to control variance.
2. **Label-shuffle null.** Recompute `DeltaID` with permuted labels. This has the same
   class sizes and the same estimator bias but no real structure, so it gives the null
   distribution. Report `DeltaID - DeltaID_shuffled`, not raw `DeltaID`.

Use the **same k** for pooled and conditional estimates. Different k here would
reintroduce exactly the scale confound that broke the per-task peak comparison.

### Scope

| task | classes | n per class @ n=25k | viable? |
|---|---|---|---|
| `bigram_shift`, `coordination_inversion`, `odd_man_out` | 2 | 12500 | yes |
| `sentence_length` | 5 | 5000 | yes |
| `word_content` | 1000 | 25 | **no** — exclude |

### Implementation

New module `idprobe/conditional_id.py`:

```python
def conditional_id(X, y, k, n_match=None, n_boot=5, seed=0) -> dict:
    """ID pooled vs within-class at matched sample size.
    Returns {'id_pooled', 'id_per_class', 'delta_id', 'n_matched'}."""

def conditional_id_null(X, y, k, n_perm=20, **kw) -> dict:
    """Same, with permuted labels — the estimator-bias null."""
```

- New stage `run.py conditional-id`, writing `results/<model>/conditional_id.csv`
  with columns `tag, layer, k, id_pooled, id_within_mean, delta_id, delta_null_mean,
  delta_null_sd, z`.
- New figure: `DeltaID` across layers per task, with the null band shaded — same visual
  grammar as the existing Fig 1.

### The payoff

Two predictions worth stating before running, so the result can disconfirm something:

- If the LRH story holds, `DeltaID` should be **small and flat** for the three
  syntactic/semantic tasks (one direction removed) and **larger** for
  `sentence_length` (the one task where the MLP wins).
- `DeltaID` should peak at or near the layer where the probe peaks. If it does, you
  have a *direction-level* version of the ID/probe link that does not depend on
  correlating two curves across autocorrelated layers.

### Complementary variant (cheap, same module)

ID of the **residual** after projecting out the trained probe direction. If a feature
is one linear direction, removing it should drop ID by ~1; if removing it drops ID by
much more, the probe direction was entangled with others.

---

## Direction 2 — Is `sentence_length` a magnitude, not a feature?

### Why this matters

`sentence_length` is the only task where the MLP beats the linear probe — 28/29 layers
at 0.6B, 24/29 at 1.7B, the most robust result in the project. But it points the
*wrong way* for a naive superposition story, which predicts abstract features are the
entangled ones and surface features are easy.

The likely explanation is not superposition at all: **length is a magnitude, not a
direction.** A quantity encoded monotonically in norm or scale is poorly served by a
linear softmax over binned classes and trivially fixed by one hidden layer. If that is
right, "MLP wins" here says something about the *readout geometry*, not about
entanglement — and the same caution applies to any magnitude-like probe.

### Experiments (all reuse existing activations; ~half a day)

| test | prediction if magnitude hypothesis is right |
|---|---|
| Probe on `‖h‖` alone (1 feature) | recovers a large share of the MLP's accuracy |
| Linear **regression** on raw length, report R² | high R² despite poor classification accuracy |
| Ordinal probe (ordered logit / linear + isotonic) | closes most of the linear-MLP gap |
| Spearman(`‖h‖`, length) per layer | large, peaking where the probe peaks |

If the gap closes under an ordinal or regression readout, report `sentence_length` as
*monotonically encoded* rather than *entangled* — a materially different claim, and one
that strengthens rather than weakens the disentanglement result for the other four tasks.

### Implementation

- Extend `idprobe/probes.py` with `train_regression_probe` and `train_ordinal_probe`,
  keeping the identical-protocol discipline (same optimiser, schedule, seeds,
  standardisation) that makes the linear-vs-MLP comparison legitimate.
- Add `norm_baseline(X, y)` — probe on `‖h‖` and on a few scalar summaries.
- Extend `stats.classify_layer` verdicts with `monotonic_encoding` when an ordinal
  probe closes the gap that the linear probe could not.

---

## Direction 3 — SAEs: actually measuring superposition

> **STATUS: deferred, not implemented.** Cut on 2026-08-28 for time. Nothing in
> `idprobe/` or `run.py` implements this section — there is no `sae` stage, no
> all-token extraction path, and `dictionary_learning` is not a dependency.
> Directions 1 and 2 and Step 0 are implemented and run.

### The gap

The RQ is framed around "entanglement/superposition", but nothing in the pipeline
measures superposition. ID measures the dimension of the data manifold. Superposition
is a claim about **N features represented in d < N dimensions under sparsity**. These
are different quantities and can move independently — a layer could have low ID and
heavy superposition, or high ID and none.

A sparse autoencoder gives the missing quantity directly.

### The experiment

Train an SAE per layer (start with every 4th layer), then relate to the ID profile:

| SAE quantity | relation to test |
|---|---|
| alive features (fraction firing above threshold) | does it peak where ID peaks? |
| L0 (mean active features per token) | is the ID peak an *active-feature* peak? |
| FVU / reconstruction error | is the peak layer harder to reconstruct sparsely? |
| probe-direction sparsity in SAE basis | are the linearly-decodable tasks the monosemantic ones? |

**The headline claim if it works:** the high-ID abstraction phase is where the model
represents the most features simultaneously — connecting Cheng et al.'s geometry to
the mechanistic-interpretability account of superposition. That is unclaimed territory.

### Practical constraints (read before committing)

- **Use a reference implementation** — `sae_lens` is the standard; `dictionary_learning`
  (Marks et al.) is the lighter alternative. Do not hand-roll the training loop.
- **Token budget is the binding constraint.** SAEs need millions of activations, not
  the 3000 sequences we extract for ID. This is H100-only work and needs a streaming
  activation pipeline rather than the current memmap-everything approach.
- **Extraction changes.** SAEs want *all token positions*, not the last-token
  representation the rest of the pipeline uses. That is a second extraction path, not
  a reuse of what is on disk — budget for it explicitly.
- Start with one model, one corpus, a subset of layers, and a small dictionary
  (8-16x expansion). Establish the ID/L0 relationship before scaling anything.

---

## Sequencing

| step | effort | hardware | depends on |
|---|---|---|---|
| 0a fix 1.7B | 1 hr | local | — |
| 0b peak robustness | 1 hr | local | 0a |
| 0c untrained control | 1 hr | local | — |
| **1. class-conditional ID** | **1-2 days** | local, then H100 | 0a |
| 2. magnitude tests | 0.5 day | local | — |
| 3. SAE bridge | 1-2 weeks | H100 only | new extraction path |

Directions 1 and 2 both run on activations already on disk, so they can be developed
locally at n=2000 and re-run at scale later. Direction 3 needs its own extraction
pipeline and should not start until 1 has produced a result worth extending.

## Open decisions

1. **Which model carries the headline result?** Qwen3-8B on the H100, or a size sweep
   (0.6B / 1.7B / 8B) that doubles as Cheng et al.'s Figure D.1?
2. **Is `word_content` worth rescuing?** It needs ~50k training examples to be
   meaningful at 1000 classes. Otherwise drop it and report four tasks.
3. **How many corpora?** Fig 1 and C.3 want three; we have Pile only. WikiText and
   BookCorpus are wired up and cheap — `--corpora pile wikitext bookcorpus`.
