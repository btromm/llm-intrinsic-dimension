#!/bin/bash
# =============================================================================
#  The whole pipeline for ONE model, in ONE job. GPU held throughout.
#
#  Use this when you want the simplest possible thing -- a first run on one
#  model, or a cluster where job dependencies are awkward. submit_all.sh is
#  better for the full ladder: it runs models concurrently and does not hold a
#  GPU during the CPU-only ID stage.
#
#      sbatch --export=ALL,PIPE_REPO=$PWD,PIPE_MODEL=Qwen/Qwen3-8B-Base \
#             -A <account> -p <gpu-partition> slurm/single_model.sbatch
#
#  Without PIPE_MODEL it runs MODELS[0] from config.sh.
# =============================================================================
#SBATCH --job-name=idp-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

for _d in "${PIPE_REPO:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")"; do
  if [[ -n "$_d" && -f "$_d/slurm/common.sh" ]]; then source "$_d/slurm/common.sh"; break; fi
done
if [[ -z "${REPO:-}" ]]; then
  echo "ERROR: cannot find slurm/common.sh. Add" >&2
  echo "       --export=ALL,PIPE_REPO=/path/to/llm-intrinsic-dimension to sbatch." >&2
  exit 1
fi
trap on_exit EXIT
activate_env
select_model          # no array here, so this resolves to PIPE_MODEL or MODELS[0]
banner

mapfile -t ARGS < <(common_args)

run_stage extract "${ARGS[@]}" --batch-size "$BATCH"
run_stage id      "${ARGS[@]}"
run_stage probe   "${ARGS[@]}"
run_stage robustness "${ARGS[@]}"
run_stage analyze    "${ARGS[@]}"
run_stage figures    "${ARGS[@]}"
run_stage plots      "${ARGS[@]}"

if [[ "$RUN_DIRECTIONS" == "1" ]]; then
  CID=(--cid-boot "$CID_BOOT" --cid-perm "$CID_PERM" --cid-stride "$CID_STRIDE")
  [[ -n "$CID_N_MATCH" ]] && CID+=(--cid-n-match "$CID_N_MATCH")
  run_stage conditional-id "${ARGS[@]}" "${CID[@]}"

  MAG=(--mag-stride "$MAG_STRIDE" --mag-epochs "$MAG_EPOCHS" --mag-batch "$MAG_BATCH")
  if [[ -n "$MAG_TASKS" ]]; then
    # shellcheck disable=SC2206
    MT=($MAG_TASKS); MAG+=(--tasks "${MT[@]}")
  fi
  run_stage magnitude "${ARGS[@]}" "${MAG[@]}"
  run_stage figures   "${ARGS[@]}"      # redraw now that the direction CSVs exist
fi

log "results:"
ls -lh "$REPO/results/$MODEL_TAG" 2>/dev/null || true
