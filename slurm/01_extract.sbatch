#!/bin/bash
# =============================================================================
#  Stage 1 -- extract.  GPU. One array task per model.
#
#  Writes last-token hidden states for every (task, mode) and (corpus, mode) to
#  activations/<model-tag>/. This is the expensive stage and the only one that
#  needs the model weights at all.
#
#  Resumable: each (tag, split) is written to a .partial file and renamed on
#  success, and an existing output is skipped. If this job is killed (walltime,
#  preemption, node failure), just resubmit it -- it picks up where it stopped.
# =============================================================================
#SBATCH --job-name=idp-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
run_stage extract "${ARGS[@]}" --batch-size "$BATCH"

log "activations for $MODEL_TAG:"
du -sh "$REPO/activations/$MODEL_TAG" 2>/dev/null || true
