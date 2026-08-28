#!/usr/bin/env bash
# =============================================================================
#  config.sh -- THE ONE FILE YOU NEED TO EDIT.
#
#  Every job script and submit_all.sh reads this. Nothing else is site-specific,
#  so once these values are right the whole pipeline should run untouched.
#
#  Anything left empty is simply not passed to sbatch / run.py, so the cluster
#  default (or the pipeline default) applies.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Slurm accounting and queues.  `sacctmgr show assoc user=$USER` lists the
#    accounts/partitions you may use; `sinfo -s` lists the partitions that exist.
# -----------------------------------------------------------------------------
ACCOUNT=""                  # sbatch -A. Empty = site does not use accounts.
PARTITION_GPU="gpu"         # partition holding the H100 nodes
PARTITION_CPU=""            # CPU-only partition; empty = reuse PARTITION_GPU
QOS=""                      # optional, e.g. "normal" / "high"
CONSTRAINT=""               # optional node feature, e.g. "h100"

# How this site asks for one GPU. Pick the line your cluster uses:
GPU_REQUEST="--gres=gpu:1"
# GPU_REQUEST="--gres=gpu:h100:1"
# GPU_REQUEST="--gpus=1"
# GPU_REQUEST="--gpus-per-node=h100:1"

MAIL_USER=""                # optional; empty = no mail
MAIL_TYPE="END,FAIL"

# -----------------------------------------------------------------------------
# 2. Software environment.
#    The pipeline needs the conda env named below (built by setup.sh). It pins
#    numpy<2 because DADApy requires it -- do not run this from `base`.
# -----------------------------------------------------------------------------
CONDA_ENV="id"
CONDA_SH=""                 # path to etc/profile.d/conda.sh; empty = auto-detect
MODULES=()                  # modules to load first, e.g. MODULES=(cuda/12.4 anaconda3)

# Free-form shell run after the env is active, before python. Use it for a
# proxy, a license server, NCCL flags, whatever your site needs. Example:
#   EXTRA_ENV_SETUP='export HTTPS_PROXY=http://proxy.example.edu:3128'
EXTRA_ENV_SETUP=''

# -----------------------------------------------------------------------------
# 3. Storage.
#    Activations are BIG (~127 GB for Qwen3-8B alone, ~260 GB for the three-model
#    ladder at default sizes). They must live on scratch, not in $HOME.
#    setup.sh symlinks <repo>/activations -> $ACTIVATIONS_DIR.
#    `run.py check` (and 00_preflight) print the exact projection before anything
#    is written.
# -----------------------------------------------------------------------------
SCRATCH_BASE="${SCRATCH:-${WORK:-$HOME/scratch}}"
ACTIVATIONS_DIR="$SCRATCH_BASE/id-activations"
HF_CACHE_DIR="$SCRATCH_BASE/hf-cache"     # becomes HF_HOME (models + datasets)

# -----------------------------------------------------------------------------
# 4. What to run.
#    MODELS and BATCH_SIZES are positional pairs: BATCH_SIZES[i] is used for
#    MODELS[i]. Drop a line from both arrays to skip that model.
#    Lower a batch size if a job dies with CUDA out-of-memory.
# -----------------------------------------------------------------------------
MODELS=(
  "Qwen/Qwen3-1.7B-Base"
  "Qwen/Qwen3-4B-Base"
  "Qwen/Qwen3-8B-Base"
)
BATCH_SIZES=(256 128 128)
DEFAULT_BATCH=128           # used for any model not listed above

# Dataset sizes. Empty = the pipeline default (25000/5000/10000 per task).
# Shrink N_TRAIN to save time and disk; LEAVE N_TEST AT 10000 -- the
# linear-vs-MLP equivalence test has no power below that (see README.md).
N_TRAIN=""
N_VAL=""
N_TEST=""

# Empty = the pipeline default, which is bookcorpus alone: the corpus Conneau
# et al. built the probing sets from, hence both the baseline the tasks compare
# against and the corpus whose GRIDE scale they borrow. Adding corpora
# (CORPORA="bookcorpus pile wikitext") asks a different question -- whether the
# ID profile is domain-specific -- at one extraction plus two more `id` tags each.
CORPORA=""
REFERENCE_CORPUS=""         # empty = pipeline default (bookcorpus)

RUN_SHUFFLED=1              # word-shuffled control (Fig I.2). 0 halves extract time and disk.
RUN_DIRECTIONS=1            # conditional-id + magnitude stages (docs/NEXT_STEPS.md)
RUN_RANDOM_INIT=1           # Step 0c untrained-weights control (cheap, high value)

# Optional second model for the cross-model CKA figure (Cheng et al. H.1).
# Its activations must already exist, so name a model that is also in MODELS.
FIGURES_COMPARE_MODEL=""
CKA_TASK="bigram_shift"

# --- Direction 1: conditional ID ---------------------------------------------
# GRIDE is O(n^2) and this stage runs ~315 estimates per layer, so the matched
# sample size drives the cost quadratically: uncapped (12500/class at n_train
# 25000) is ~37 min per LAYER; 2000/class is ~1 min. ID is also strongly
# n-dependent, so this value must stay FIXED across every model you intend to
# compare -- change it once, re-run everything.
CID_N_MATCH=2000
CID_STRIDE=1                # 2 = every other layer, a cheaper preview
CID_BOOT=5
CID_PERM=20

# --- Direction 2: magnitude readouts -----------------------------------------
MAG_STRIDE=1
MAG_EPOCHS=200              # these readouts need their own budget; see README.md
MAG_BATCH=512
MAG_TASKS=""                # empty = all tasks; "sentence_length" is much cheaper

# -----------------------------------------------------------------------------
# 5. Resources per stage.
#    Sized for the largest model in the ladder; smaller ones just finish early
#    (Slurm bills what you use, not what you ask for -- only queue priority is
#    affected). If your site's maximum walltime is lower than a value here, cut
#    it and read the "If a job hits the walltime" section of README.md first.
# -----------------------------------------------------------------------------
TIME_PREFLIGHT="00:30:00";  CPUS_PREFLIGHT=4;   MEM_PREFLIGHT="16G"
TIME_EXTRACT="12:00:00";    CPUS_EXTRACT=8;     MEM_EXTRACT="64G"
TIME_ID="24:00:00";         CPUS_ID=32;         MEM_ID="96G"
TIME_PROBE="24:00:00";      CPUS_PROBE=8;       MEM_PROBE="64G"
TIME_ANALYZE="02:00:00";    CPUS_ANALYZE=8;     MEM_ANALYZE="32G"
TIME_DIRECTIONS="24:00:00"; CPUS_DIRECTIONS=16; MEM_DIRECTIONS="64G"
TIME_CONTROL="06:00:00";    CPUS_CONTROL=16;    MEM_CONTROL="64G"
TIME_SINGLE="48:00:00";     CPUS_SINGLE=16;     MEM_SINGLE="96G"

# How many models may occupy GPUs at once. 2 is polite on a shared cluster;
# raise it if you have the allocation.
MAX_CONCURRENT=2
