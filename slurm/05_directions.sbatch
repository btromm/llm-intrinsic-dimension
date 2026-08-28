#!/bin/bash
# =============================================================================
#  Stage 5 -- the two directions from docs/NEXT_STEPS.md.  GPU. Optional
#  (RUN_DIRECTIONS in config.sh). One array task per model.
#
#    conditional-id : DeltaID = ID(X) - mean_c ID(X|y=c), with its matched-n and
#                     label-shuffle guards. Needs stage 2, because it must reuse
#                     the GRIDE scale that stage pinned -- it never picks its own.
#    magnitude      : linear / MLP / ordinal / norm readouts, to separate "the
#                     feature is entangled" from "the label is a binned quantity".
#
#  Cost warning: GRIDE is O(n^2) and conditional-id runs ~315 estimates per
#  layer, so CID_N_MATCH drives the runtime quadratically. At the uncapped
#  12500/class it is ~37 min PER LAYER; config.sh caps it at 2000 (~1 min).
#  Whatever you choose must stay fixed across models, or their DeltaID values
#  are not comparable.
#
#  Both stages merge into their CSVs per task under a lock, so re-running one
#  task does not delete another's rows.
# =============================================================================
#SBATCH --job-name=idp-directions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
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

CID=(--cid-boot "$CID_BOOT" --cid-perm "$CID_PERM" --cid-stride "$CID_STRIDE")
[[ -n "$CID_N_MATCH" ]] && CID+=(--cid-n-match "$CID_N_MATCH")
run_stage conditional-id "${ARGS[@]}" "${CID[@]}"

MAG=(--mag-stride "$MAG_STRIDE" --mag-epochs "$MAG_EPOCHS" --mag-batch "$MAG_BATCH")
if [[ -n "$MAG_TASKS" ]]; then
  # shellcheck disable=SC2206
  MT=($MAG_TASKS)
  MAG+=(--tasks "${MT[@]}")
fi
run_stage magnitude "${ARGS[@]}" "${MAG[@]}"
