#!/bin/bash
# =============================================================================
#  Stage 2 -- intrinsic dimension.  CPU ONLY (GRIDE is a BLAS/neighbour search,
#  not a GPU workload). One array task per model.
#
#  Two things worth knowing before you change the resources:
#
#  * NOT resumable. The stage sweeps every scale of every layer of every tag and
#    writes id_profiles.csv at the end, so a walltime kill loses the whole job.
#    Give it room: ~2-4 h for Qwen3-8B (5 task tags plus the bookcorpus
#    baseline in both modes, ~37 layers each).
#  * The array is throttled to ONE task at a time (%1, set in submit_all.sh).
#    All models share scales.json, and the stage rewrites that file wholesale,
#    so two models finishing together would clobber each other's pinned scale.
# =============================================================================
#SBATCH --job-name=idp-id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

for _d in "${PIPE_REPO:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")"; do
  if [[ -n "$_d" && -f "$_d/slurm/common.sh" ]]; then source "$_d/slurm/common.sh"; break; fi
done
if [[ -z "${REPO:-}" ]]; then
  echo "ERROR: cannot find slurm/common.sh. Submit with slurm/submit_all.sh, or add" >&2
  echo "       --export=ALL,PIPE_REPO=/path/to/llm-intrinsic-dimension to sbatch." >&2
  exit 1
fi
trap on_exit EXIT
activate_env
select_model
banner

mapfile -t ARGS < <(common_args)
run_stage id "${ARGS[@]}"

log "GRIDE scales now pinned in scales.json:"
python - "$MODEL_TAG" <<'PY'
import json, sys
from pathlib import Path
p = Path("scales.json")
d = json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(d.get(sys.argv[1], {}), indent=2, sort_keys=True))
PY
