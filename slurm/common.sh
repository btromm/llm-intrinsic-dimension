#!/usr/bin/env bash
# =============================================================================
#  common.sh -- sourced by every job script. Not meant to be run directly.
#
#  Sets up: repo paths, the conda env, cache/thread environment variables, a
#  banner describing the allocation, and helpers for building run.py arguments.
# =============================================================================
set -euo pipefail

# This file always lives at its real path (unlike a batch script, which Slurm
# copies to a spool directory), so it can locate the repo reliably.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$REPO/slurm"
LOG_DIR="$REPO/logs"

if [[ ! -f "$REPO/run.py" ]]; then
  echo "ERROR: $REPO does not look like the llm-intrinsic-dimension checkout" >&2
  exit 1
fi

# shellcheck source=config.sh
source "$SLURM_DIR/config.sh"

JOB_START=$(date +%s)
JOB_LABEL="${SLURM_JOB_NAME:-local}(${SLURM_JOB_ID:-none})"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
activate_env() {
  # `module` and conda's activation scripts both reference unset variables, so
  # they must not run under `set -u`.
  set +u
  if [[ ${#MODULES[@]} -gt 0 ]]; then
    for m in "${MODULES[@]}"; do
      log "module load $m"
      module load "$m"
    done
  fi

  local hook="$CONDA_SH"
  if [[ -z "$hook" ]]; then
    if command -v conda >/dev/null 2>&1; then
      hook="$(conda info --base 2>/dev/null || true)/etc/profile.d/conda.sh"
    else
      for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /usr/local/anaconda3; do
        [[ -f "$c/etc/profile.d/conda.sh" ]] && { hook="$c/etc/profile.d/conda.sh"; break; }
      done
    fi
  fi
  [[ -f "$hook" ]] || die "cannot find conda.sh. Set CONDA_SH in slurm/config.sh (or load your conda module in MODULES)."

  # shellcheck disable=SC1090
  source "$hook"
  conda activate "$CONDA_ENV" \
    || die "conda env '$CONDA_ENV' not found. Run: bash slurm/setup.sh"
  set -u

  # DADApy pins numpy<2; the whole point of a separate env is that this holds.
  python - <<'PY' || die "environment check failed -- rebuild with slurm/setup.sh"
import sys, numpy
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__}: dadapy needs numpy<2"
import dadapy, torch, transformers  # noqa: F401
print(f"env OK: python {sys.version.split()[0]}, numpy {numpy.__version__}, "
      f"torch {torch.__version__}, transformers {transformers.__version__}")
PY

  export PYTHONUNBUFFERED=1
  export HF_HOME="$HF_CACHE_DIR"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # Keep BLAS/joblib inside the CPU allocation. Without this they size
  # themselves from the node's core count and thrash against other jobs --
  # which matters most in the `id` stage, where GRIDE is BLAS-bound.
  local nthreads="${SLURM_CPUS_PER_TASK:-4}"
  export OMP_NUM_THREADS="$nthreads"
  export MKL_NUM_THREADS="$nthreads"
  export OPENBLAS_NUM_THREADS="$nthreads"
  export NUMEXPR_NUM_THREADS="$nthreads"
  export LOKY_MAX_CPU_COUNT="$nthreads"

  if [[ -n "$EXTRA_ENV_SETUP" ]]; then
    log "EXTRA_ENV_SETUP"
    eval "$EXTRA_ENV_SETUP"
  fi

  mkdir -p "$LOG_DIR"
  cd "$REPO"
}

# -----------------------------------------------------------------------------
# Which model does this array task handle?
#   submit_all.sh writes the selected models to a file and exports
#   PIPE_MODEL_FILE; a hand-submitted job falls back to MODELS in config.sh.
# -----------------------------------------------------------------------------
pipeline_models() {
  if [[ -n "${PIPE_MODEL:-}" ]]; then
    printf '%s\n' "$PIPE_MODEL"
  elif [[ -n "${PIPE_MODEL_FILE:-}" && -f "${PIPE_MODEL_FILE}" ]]; then
    grep -v '^[[:space:]]*$' "$PIPE_MODEL_FILE"
  else
    printf '%s\n' "${MODELS[@]}"
  fi
}

select_model() {
  local -a all
  mapfile -t all < <(pipeline_models)
  local idx="${SLURM_ARRAY_TASK_ID:-0}"
  [[ ${#all[@]} -gt $idx ]] \
    || die "array index $idx but only ${#all[@]} model(s) selected: ${all[*]}"
  MODEL="${all[$idx]}"
  MODEL_TAG="${MODEL##*/}"
  BATCH="$(batch_size_for "$MODEL")"
  export MODEL MODEL_TAG BATCH
}

batch_size_for() {
  local want="$1" i
  for i in "${!MODELS[@]}"; do
    if [[ "${MODELS[$i]}" == "$want" ]]; then
      echo "${BATCH_SIZES[$i]:-$DEFAULT_BATCH}"; return 0
    fi
  done
  echo "$DEFAULT_BATCH"
}

# -----------------------------------------------------------------------------
# run.py argument builders. Every stage shares the model/scale flags, so they
# are assembled once here -- a mismatch between stages (different --n-train,
# say) silently invalidates the run.
# -----------------------------------------------------------------------------
common_args() {
  local -a a=(--model "$MODEL")
  [[ -n "$N_TRAIN" ]] && a+=(--n-train "$N_TRAIN")
  [[ -n "$N_VAL"   ]] && a+=(--n-val   "$N_VAL")
  [[ -n "$N_TEST"  ]] && a+=(--n-test  "$N_TEST")
  [[ -n "$REFERENCE_CORPUS" ]] && a+=(--reference-corpus "$REFERENCE_CORPUS")
  if [[ -n "$CORPORA" ]]; then
    # shellcheck disable=SC2206
    local -a c=($CORPORA)
    a+=(--corpora "${c[@]}")
  fi
  [[ "$RUN_SHUFFLED" == "1" ]] || a+=(--no-shuffled)
  printf '%s\n' "${a[@]}"
}

# Usage: mapfile -t ARGS < <(common_args); python run.py extract "${ARGS[@]}"
run_stage() {
  local stage="$1"; shift
  log "=== run.py $stage $* ==="
  local t0 rc
  t0=$(date +%s)
  set +e
  python "$REPO/run.py" "$stage" "$@"
  rc=$?
  set -e
  local dt=$(( $(date +%s) - t0 ))
  if [[ $rc -ne 0 ]]; then
    log "=== run.py $stage FAILED (exit $rc) after $((dt / 60))m ==="
    return $rc
  fi
  log "=== run.py $stage done in $((dt / 60))m$((dt % 60))s ==="
}

# -----------------------------------------------------------------------------
# Banner + exit summary
# -----------------------------------------------------------------------------
banner() {
  echo "============================================================"
  echo " job      : ${SLURM_JOB_NAME:-?}  id=${SLURM_JOB_ID:-?}" \
       "${SLURM_ARRAY_TASK_ID:+array task=$SLURM_ARRAY_TASK_ID}"
  echo " host     : $(hostname)"
  echo " started  : $(date)"
  echo " repo     : $REPO"
  echo " model    : ${MODEL:-n/a}   batch=${BATCH:-n/a}"
  echo " cpus     : ${SLURM_CPUS_PER_TASK:-?}   mem=${SLURM_MEM_PER_NODE:-?}MB"
  echo " conda    : $CONDA_ENV"
  echo " HF_HOME  : ${HF_HOME:-unset}"
  echo " acts     : $(readlink -f "$REPO/activations" 2>/dev/null || echo "$REPO/activations")"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null | sed 's/^/ gpu      : /' || true
  fi
  { df -h "$(readlink -f "$REPO/activations" 2>/dev/null || echo "$REPO")" 2>/dev/null \
    | tail -1 | awk '{print " disk     : "$4" free of "$2" on "$6}'; } || true
  echo "============================================================"
}

on_exit() {
  local rc=$?
  local dt=$(( $(date +%s) - JOB_START ))
  echo "------------------------------------------------------------"
  if [[ $rc -eq 0 ]]; then
    echo "FINISHED ok   $JOB_LABEL   elapsed $((dt / 3600))h$(( (dt % 3600) / 60 ))m"
  else
    echo "FAILED rc=$rc $JOB_LABEL   elapsed $((dt / 3600))h$(( (dt % 3600) / 60 ))m"
    echo "Jobs downstream of this one were submitted with --kill-on-invalid-dep,"
    echo "so they should be cancelled automatically. Check with: squeue -u \$USER"
  fi
  [[ -n "${SLURM_JOB_ID:-}" ]] && command -v seff >/dev/null 2>&1 \
    && seff "${SLURM_JOB_ID}" 2>/dev/null | sed 's/^/  /' || true
  echo "------------------------------------------------------------"
  return $rc
}
