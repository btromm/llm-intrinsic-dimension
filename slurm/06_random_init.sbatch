#!/bin/bash
# =============================================================================
#  Stage 6 -- Step 0c, the untrained control.  GPU. Optional (RUN_RANDOM_INIT).
#  One array task per model.
#
#  Same architecture, same tokenizer, same extraction path, random weights.
#  Cheng et al. report the ID peak is absent in untrained models, so this is the
#  cheapest falsification the project has: if a peak still appears here, it
#  belongs to the architecture or to the extraction rather than to anything the
#  model learned.
#
#  Deliberately cheap: the corpus baseline only (10k sequences of 20 tokens), no
#  probing tasks, no shuffled control. Outputs land under the separate
#  <model>-randominit tag, so they can never overwrite the trained run.
#
#  Independent of stages 1-5 -- only the preflight has to pass first.
# =============================================================================
#SBATCH --job-name=idp-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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

run_stage extract --model "$MODEL" --random-init --no-tasks --no-shuffled \
  --batch-size "$BATCH"

run_stage id --model "$MODEL" --random-init

log "compare results/${MODEL_TAG}-randominit/id_profiles.csv against"
log "        results/${MODEL_TAG}/id_profiles.csv -- the peak should be absent here."
