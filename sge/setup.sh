#!/usr/bin/env bash
# =============================================================================
#  Run ONCE on a Myriad login node, before submitting anything.
#
#      bash sge/setup.sh
#
#  Builds the venv with uv, points activations/ at Scratch, and downloads the
#  model + data. Compute nodes have no internet, so everything must be cached
#  here first.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/Scratch/id-venv"
MODEL="Qwen/Qwen3-1.7B-Base"

export UV_CACHE_DIR="$HOME/Scratch/uv-cache"
export HF_HOME="$HOME/Scratch/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

cd "$REPO"

# --- 1. uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- 2. venv -----------------------------------------------------------------
# 3.11, not newer: dadapy's Cython wheels stop being available above it.
echo "== venv at $VENV"
uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"

# torch first, from the CUDA 12.4 index. The wheel bundles its own CUDA runtime,
# so no `module load cuda` is needed on the GPU node -- just the driver.
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

python -c "import numpy, dadapy, torch; assert numpy.__version__[0] == '1'; \
print(f'numpy {numpy.__version__}  torch {torch.__version__}')"

# --- 3. storage --------------------------------------------------------------
# Activations for 1.7B are ~30GB. Home and Scratch share one 1TB quota on
# Myriad, so check headroom with `lquota` if this is not your only run.
mkdir -p "$HOME/Scratch/id-activations" "$HF_HOME" "$REPO/logs"
[[ -L "$REPO/activations" ]] || { rmdir "$REPO/activations" 2>/dev/null || true; \
  ln -s "$HOME/Scratch/id-activations" "$REPO/activations"; }
echo "== activations -> $(readlink -f "$REPO/activations")"

# --- 4. prefetch -------------------------------------------------------------
echo "== downloading $MODEL"
python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1],
      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"]))
PY

echo "== downloading SentEval probing files"
python - <<'PY'
import sys; sys.path.insert(0, ".")
from idprobe import data as D
from idprobe.config import TASKS
for t in TASKS:
    print(f"  {t}: {len(D.download_task(t))} rows")
PY

echo "== warming the bookcorpus cache"
python - <<'PY'
import sys; sys.path.insert(0, ".")
from datasets import load_dataset
from idprobe.config import CORPORA
repo, cfg = CORPORA["bookcorpus"]           # rojagtap/bookcorpus, not the bare name
ds = load_dataset(repo, cfg, split="train", streaming=True) if cfg else \
     load_dataset(repo, split="train", streaming=True)
print(f"  ok ({len(next(iter(ds))['text'])} chars in the first row)")
PY

echo
echo "done. Now:  qsub sge/run_qwen1.7b.sh"
