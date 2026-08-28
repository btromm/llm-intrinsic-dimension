#!/usr/bin/env bash
# =============================================================================
#  setup.sh -- run ONCE, on a LOGIN node, before the first submission.
#
#      bash slurm/setup.sh
#
#  It does four things:
#    1. builds the `id` conda env (python 3.11, numpy<2, dadapy, torch for CUDA)
#    2. points activations/ at scratch, and the HF cache at scratch
#    3. downloads the model weights, the SentEval task files, and warms the
#       datasets cache -- so no compute job depends on the login node's internet
#    4. runs the preflight checks in login-node mode
#
#  Options:
#    --skip-env        env already exists and is correct
#    --skip-download   do not fetch model weights / data
#    --no-weights      fetch data but not the (large) model checkpoints
#    --torch-index URL wheel index for torch (default: CUDA 12.4 build)
#    --update          update an existing env from environment.yml instead of
#                      creating it
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$REPO/slurm"
cd "$REPO"                 # the prefetch snippets import idprobe from here
source "$SLURM_DIR/config.sh"

SKIP_ENV=0; SKIP_DOWNLOAD=0; NO_WEIGHTS=0; UPDATE=0
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-env)      SKIP_ENV=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --no-weights)    NO_WEIGHTS=1; shift ;;
    --update)        UPDATE=1; shift ;;
    --torch-index)   TORCH_INDEX="$2"; shift 2 ;;
    -h|--help)       awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------------------
say "1/4  conda environment '$CONDA_ENV'"
# ---------------------------------------------------------------------------
set +u          # `module` and conda's activate scripts trip over `set -u`
if [[ ${#MODULES[@]} -gt 0 ]]; then
  for m in "${MODULES[@]}"; do echo "module load $m"; module load "$m"; done
fi

hook="$CONDA_SH"
if [[ -z "$hook" ]] && command -v conda >/dev/null 2>&1; then
  hook="$(conda info --base 2>/dev/null || true)/etc/profile.d/conda.sh"
fi
[[ -f "$hook" ]] || {
  echo "ERROR: no conda.sh found. Install miniconda, or set CONDA_SH (and/or" >&2
  echo "       MODULES) in slurm/config.sh." >&2; exit 1; }
# shellcheck disable=SC1090
source "$hook"

if [[ $SKIP_ENV -eq 0 ]]; then
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    if [[ $UPDATE -eq 1 ]]; then
      echo "updating existing env"
      conda env update -n "$CONDA_ENV" -f "$REPO/environment.yml" --prune
    else
      echo "env '$CONDA_ENV' already exists (pass --update to refresh it)"
    fi
  else
    conda env create -n "$CONDA_ENV" -f "$REPO/environment.yml"
  fi
fi

conda activate "$CONDA_ENV"
set -u

if [[ $SKIP_ENV -eq 0 ]]; then
  # environment.yml pulls torch from PyPI; replace it with the build matching
  # the cluster's CUDA runtime, as the project README prescribes.
  echo "installing torch from $TORCH_INDEX"
  pip install --upgrade torch --index-url "$TORCH_INDEX"
fi

python - <<'PY'
import numpy, torch, transformers, dadapy   # noqa: F401
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__}: dadapy needs numpy<2"
print(f"  numpy {numpy.__version__}  torch {torch.__version__} "
      f"(cuda build: {torch.version.cuda})  transformers {transformers.__version__}")
PY

# ---------------------------------------------------------------------------
say "2/4  storage"
# ---------------------------------------------------------------------------
mkdir -p "$ACTIVATIONS_DIR" "$HF_CACHE_DIR" "$REPO/logs"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

link_scratch() {                      # link_scratch <name> <target>
  local link="$REPO/$1" target="$2"
  if [[ -L "$link" ]]; then
    echo "  $link -> $(readlink -f "$link")"
  elif [[ -d "$link" ]]; then
    if [[ -n "$(ls -A "$link" 2>/dev/null)" ]]; then
      echo "  WARNING: $link is a real directory and not empty; leaving it alone."
      echo "           Activations will consume quota HERE, not on scratch."
      echo "           Move it aside and re-run setup if that is not what you want."
    else
      rmdir "$link"; ln -s "$target" "$link"; echo "  $link -> $target"
    fi
  else
    ln -s "$target" "$link"; echo "  $link -> $target"
  fi
}
link_scratch activations "$ACTIVATIONS_DIR"
echo "  HF_HOME  = $HF_CACHE_DIR"
df -h "$ACTIVATIONS_DIR" | tail -1 | awk '{print "  free: "$4" of "$2" on "$6}'

# ---------------------------------------------------------------------------
say "3/4  prefetch (so compute nodes never need the internet for weights/data)"
# ---------------------------------------------------------------------------
if [[ $SKIP_DOWNLOAD -eq 1 ]]; then
  echo "  skipped (--skip-download)"
else
  if [[ $NO_WEIGHTS -eq 0 ]]; then
    for m in "${MODELS[@]}"; do
      echo "  $m"
      python - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    sys.argv[1],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
)
print(f"    -> {p}")
PY
    done
  else
    echo "  model weights skipped (--no-weights)"
  fi

  # The preflight job runs a smoke test on a tiny model; fetch it here too, so
  # that test is not the one thing still needing the internet on a compute node.
  echo "  hf-internal-testing/tiny-random-LlamaForCausalLM (smoke test)"
  python - <<'SMOKE'
from huggingface_hub import snapshot_download
snapshot_download("hf-internal-testing/tiny-random-LlamaForCausalLM")
SMOKE

  echo "  SentEval probing files"
  python - <<'PY'
import sys
sys.path.insert(0, ".")
from idprobe import data as D
from idprobe.config import TASKS
for t in TASKS:
    print(f"    {t}: {len(D.download_task(t))} rows")
PY

  echo "  ID corpora (streaming probe + cache warm)"
  python - "${CORPORA:-}" <<'PY'
import sys
sys.path.insert(0, ".")
from datasets import load_dataset
from idprobe.config import CORPORA
from idprobe.config import Config
names = sys.argv[1].split() if len(sys.argv) > 1 and sys.argv[1] else list(Config().corpora)
for c in names:
    repo, cfg_name = CORPORA[c]
    ds = (load_dataset(repo, cfg_name, split="train", streaming=True)
          if cfg_name else load_dataset(repo, split="train", streaming=True))
    row = next(iter(ds))
    print(f"    {c}: OK ({len(row.get('text', ''))} chars in the first row)")
PY
fi

# ---------------------------------------------------------------------------
say "4/4  preflight (login-node mode: no GPU expected here)"
# ---------------------------------------------------------------------------
cd "$REPO"
PRE=(--models "${MODELS[@]}" --no-gpu)
[[ -n "$N_TRAIN" ]] && PRE+=(--n-train "$N_TRAIN")
[[ -n "$N_VAL"   ]] && PRE+=(--n-val   "$N_VAL")
[[ -n "$N_TEST"  ]] && PRE+=(--n-test  "$N_TEST")
if [[ -n "$CORPORA" ]]; then
  # shellcheck disable=SC2206
  C=($CORPORA); PRE+=(--corpora "${C[@]}")
fi
[[ "$RUN_SHUFFLED" == "1" ]] || PRE+=(--no-shuffled)
python "$SLURM_DIR/preflight.py" "${PRE[@]}"

cat <<DONE

=== setup complete ===
Next:
  bash slurm/submit_all.sh -n     # see what would be submitted
  bash slurm/submit_all.sh        # submit it
DONE
