# SentEval probing tasks used in this project

The pipeline probes **five** SentEval probing tasks (Conneau et al. 2018,
*What you can cram into a single vector*), the subset retained by Cheng et al.
(2025) §I.1. The canonical list lives in `idprobe/config.py` (`TASKS`); this
file explains what each task actually asks a probe to do.

Files are fetched from SentEval's own repo (`data/probing/<task>.txt`) rather
than reimplemented, so numbers stay comparable to the published papers.

## File format

Every probing file is TSV with three columns and no header:

```
split <TAB> label <TAB> sentence
tr	I	A week she 'd been with the man , just a week , and she had survived barely the ups and downs .
```

* `split` ∈ `tr` / `va` / `te` → mapped to `train` / `val` / `test` in `data.py`.
* `label` is a **string** — `I`/`O`, `C`/`O`, a bin index, or the target word.
  `prepare_task` encodes it via `sorted(unique)`, so integer class ids are
  alphabetical, not semantic.
* Sentences are pre-tokenised (whitespace-separated, punctuation split off) and
  drawn from the Toronto Books Corpus.
* Full size is 100k train / 10k val / 10k test per task; this project
  stratified-subsamples to 25k/5k/10k by default (`Config.n_train` etc.).

## The five tasks

### 1. `sentence_length` — group: **surface**, 6 classes

Predict which length bin the sentence falls into, from the last-token hidden
state alone. Bins (in whitespace tokens, verified against the downloaded file):

| label | token range |
|---|---|
| `0` | 5–8 |
| `1` | 9–12 |
| `2` | 13–16 |
| `3` | 17–20 |
| `4` | 21–25 |
| `5` | 26–28 |

Balanced: 19,998 sentences per bin. Pure surface information — no linguistic
knowledge required, only a running count.

### 2. `word_content` — group: **surface**, 1000 classes

1000 mid-frequency target words were chosen; each sentence contains **exactly
one** of them. The probe must recover *which* one. This is a 1000-way
classification with only ~120 examples per class in the full file.

Labels are the target words themselves (`mattered`, `worn`, `goodbye`, …).

This is why `data.py::_stratified_sample` exists: naive subsampling to 25k
would leave many of the 1000 classes with a handful of training examples and
turn class imbalance into the dominant effect.

### 3. `bigram_shift` — group: **syntax**, 2 classes

Two adjacent words in the sentence have (or have not) been swapped. The probe
detects the inversion.

| label | meaning |
|---|---|
| `I` | **I**ntact — original word order |
| `O` | **O**ff / inverted — one adjacent pair swapped |

Balanced: 60,000 each. Requires sensitivity to local word order, i.e. actual
syntactic structure rather than bag-of-words content.

Example (`O`, "to heart" ← "heart to"):
> `He poured out his to heart God , and after a few minutes the need left him …`

### 4. `coordination_inversion` — group: **semantics**, 2 classes

Sentences consist of two coordinate clauses. In half the cases the order of the
two clauses was inverted. The probe must decide whether the original ordering
was preserved.

| label | meaning |
|---|---|
| `O` | **O**riginal clause order |
| `I` | **I**nverted |

Balanced: 60,003 each. Both versions are grammatical, so the decision rests on
discourse/semantic plausibility, not syntax.

### 5. `odd_man_out` — group: **semantics**, 2 classes

Also called **SOMO** (Semantic Odd Man Out). A randomly chosen noun or verb was
replaced by another word with comparable corpus bigram frequency, so the local
n-gram statistics look normal but the sentence stops making sense.

| label | meaning |
|---|---|
| `O` | **O**riginal sentence |
| `C` | **C**hanged — one noun/verb replaced |

Example (`C`):
> `Gideon brought his phone to his ear and resonated with Bev at HQ .`

The hardest of the five; even strong sentence encoders sit close to chance in
the original paper.

## Grouping and why these five

```python
TASKS = {
    "sentence_length":        "surface",
    "word_content":           "surface",
    "bigram_shift":           "syntax",
    "coordination_inversion": "semantics",
    "odd_man_out":            "semantics",
}
```

The groups form the abstraction ladder that the study's independent variable
rides on: surface properties are countable from the token string, syntax needs
word-order sensitivity, semantics needs plausibility judgments.

SentEval ships ten probing tasks. Following Cheng et al., five are **dropped**:

* `past_present`, `subj_number`, `obj_number` — ceiling effects; nearly every
  model saturates them, so they carry no signal.
* `top_constituents`, `tree_depth` — labels come from automated parses, so the
  probe partly measures agreement with a parser rather than the LM.

## How the pipeline consumes them

1. `data.py::download_task` fetches the TSV (atomic write via a `.part` temp
   file, so an interrupted download is never mistaken for a cached one).
2. `prepare_task` integer-encodes labels and stratified-subsamples each split.
3. `extract.py` encodes each sentence and stores the **last-token** hidden state
   at every layer (0 = embeddings … 36 = final for Qwen3-8B-Base) —
   under causal attention that is the only position that has attended to the
   whole sequence.
4. `probes.py` trains a linear probe and a 1×200 MLP on each layer's
   activations under an identical protocol; test accuracy is the task score.

A separate, unlabelled bookcorpus sample (`rojagtap/bookcorpus`, 10k sequences
truncated to 20 tokens) provides the intrinsic-dimension baseline — it is not a
probing task. It is the same corpus these probing sets were built from, so its
ID profile is the one theirs can be compared against directly, and it is the
corpus whose GRIDE scale they borrow.

## Sanity check

Running the pipeline with `--model hf-internal-testing/tiny-random-LlamaForCausalLM`
should put every probe at chance level per task:

| task | chance |
|---|---|
| `sentence_length` | ~16.7% (1/6) |
| `word_content` | ~0.1% (1/1000) |
| `bigram_shift` | 50% |
| `coordination_inversion` | 50% |
| `odd_man_out` | 50% |

Anything meaningfully above chance on a random-weight model indicates label
leakage in the data pipeline.

## References

* Conneau, Kruszewski, Lample, Barrault, Baroni (2018). *What you can cram into
  a single vector: Probing sentence embeddings for linguistic properties.* ACL.
* Cheng et al. (2025), §I.1 — task subset and grouping followed here.
* SentEval data: https://github.com/facebookresearch/SentEval/tree/main/data/probing
