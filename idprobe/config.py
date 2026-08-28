"""Central configuration. Every script reads from here; override on the CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ACTS = ROOT / "activations"
RESULTS = ROOT / "results"
SCALES = ROOT / "scales.json"   # chosen GRIDE scale per (model, corpus)

# The five SentEval probing tasks Cheng et al. (2025) keep, with their groupings.
# They drop past_present/subj_number/obj_number (ceiling effects) and
# top_constituents/tree_depth (rely on automated parses).
TASKS: dict[str, str] = {
    "sentence_length": "surface",
    "word_content": "surface",
    "bigram_shift": "syntax",
    "coordination_inversion": "semantics",
    "odd_man_out": "semantics",
}

# Tasks whose label is a BINNING of a measurable scalar magnitude of the input,
# mapped to the function recovering that magnitude from the raw sentence.
#
# This distinction drives Direction 2. `sentence_length` is the only SentEval task
# here whose classes are ordered cuts through a continuous quantity (its six bins are
# word counts 5-8, 9-12, 13-16, 17-20, 21-25, 26-28), which means a softmax over
# unordered classes is the wrong instrument for it and an MLP can appear to find
# "nonlinear structure" when it has only supplied the missing ordering. Regression
# and ordinal readouts only carry their intended meaning for tasks listed here.
MAGNITUDE_TASKS: dict[str, "callable"] = {
    "sentence_length": lambda s: len(s.split()),
}

# The canonical `bookcorpus` ships a loading script, which `datasets` >=3 refuses.
# SentEval's probing sets were drawn from BookCorpus, so it is the sole matched
# baseline and the sole source of the shared GRIDE scale.
CORPORA: dict[str, tuple[str, str | None]] = {
    "bookcorpus": ("rojagtap/bookcorpus", None),
}

SENTEVAL_URL = (
    "https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/{task}.txt"
)


@dataclass
class Config:
    model_id: str = "Qwen/Qwen3-1.7B-Base"

    # Scale. SentEval ships 100k/10k/10k per task. Storage for the activation
    # tensor is n_sequences * (n_layers + 1) * d_model * 2 bytes; for Qwen3-1.7B
    # (36 layers, d=4096) that is ~300 KB per sequence, i.e. ~7.6 GB at n=25_000.
    # Config.report_disk() prints the real number before anything is written.
    # Seed for choosing WHICH sentences go in each split. Kept separate from
    # id_seed so that varying the ID subsample does not also reshuffle the
    # probe training data and confound the two.
    data_seed: int = 0

    n_train: int = 25_000
    n_val: int = 5_000
    n_test: int = 10_000

    # Cheng et al. build ID corpora from 10k sequences truncated to 20 tokens.
    n_corpus: int = 10_000
    corpus_seq_len: int = 20
    # Cheng et al. average ID over three corpora; Table C.1 lists a separate
    # GRIDE scale for the "sane" and "shuffled" mode of each. We keep only
    # bookcorpus, because Conneau et al. built all five probing sets from the
    # Toronto Book Corpus: it is the corpus the tasks are drawn FROM, making it
    # both the matched baseline and the source of the shared GRIDE scale.
    corpora: tuple[str, ...] = ("bookcorpus",)

    # Which corpus's GRIDE scale the probing tasks borrow. Conneau et al. built
    # all five probing sets from the Toronto Book Corpus, so bookcorpus is the
    # matched source: task ID and the bookcorpus baseline then sit on the same
    # scale and can be compared directly. Tasks never define a scale of their
    # own, which keeps k to exactly one per (model, corpus) as in Cheng et al.
    reference_corpus: str = "bookcorpus"
    modes: tuple[str, ...] = ("sane", "shuffled")

    max_length: int = 128
    batch_size: int = 64
    dtype: str = "bfloat16"  # bfloat16 on H100; float32 on CPU
    device: str = "auto"

    # Probes
    probe_seeds: tuple[int, ...] = (1, 2, 3)
    probe_epochs: int = 100
    probe_lr: float = 1e-3
    probe_weight_decay: float = 1e-4
    probe_batch_size: int = 512
    patience: int = 10
    mlp_hidden: int = 200          # matches Cheng et al.
    mlp_activation: str = "relu"   # Cheng et al. used "logistic"; see README

    # Intrinsic dimension (GRIDE via DADApy)
    # Cheng et al. App. C: sweep k in powers of 2, pick ONE scale per (model,
    # corpus) at the plateau. Their Table C.1 chooses k in 8..256, usually 32.
    # range_max=512 therefore sweeps k = 1,2,...,256, covering their whole range.
    id_range_max: int = 512        # GRIDE sweeps k = 1,2,4,...,range_max/2
    # Layers excluded from ID entirely. Layer 0 is the embedding of the final
    # token, so on a probing task it takes only a handful of distinct values
    # (most sentences end in the same punctuation) -- its ID is meaningless.
    id_skip_layers: tuple[int, ...] = (0,)
    id_n_points: int = 10_000      # subsample for the ID estimate
    id_seed: int = 0

    tasks: tuple[str, ...] = field(default_factory=lambda: tuple(TASKS))

    # Step 0c: build the model from its config with random weights instead of loading
    # the trained checkpoint. Cheng et al. report the ID peak is absent in untrained
    # models, which makes this the pipeline's cheapest falsification: if a peak still
    # appears here, the peak is a property of the architecture or of the extraction,
    # not of anything the model learned.
    random_init: bool = False
    random_init_seed: int = 0

    @property
    def model_tag(self) -> str:
        """Directory name for this model's activations and results.

        The untrained control gets its OWN tag so it can never overwrite, or be
        confused with, the trained model's outputs -- they differ only in weights and
        would otherwise land in exactly the same paths.
        """
        tag = self.model_id.split("/")[-1]
        return f"{tag}-randominit" if self.random_init else tag

    @property
    def results_dir(self) -> Path:
        """Results are scoped by model so runs of different LMs never clobber."""
        return RESULTS / self.model_tag

    def act_path(self, tag: str, split: str) -> Path:
        return ACTS / self.model_tag / tag / f"{split}.npy"

    def report_disk(self, n_layers: int, d_model: int) -> None:
        per_seq = (n_layers + 1) * d_model * 2 / 1e9
        total = per_seq * (self.n_train + self.n_val + self.n_test) * len(self.tasks)
        print(
            f"[disk] {per_seq * 1e3:.1f} MB/sequence -> "
            f"{total:.1f} GB for {len(self.tasks)} tasks "
            f"({self.n_train}/{self.n_val}/{self.n_test} per task)"
        )
