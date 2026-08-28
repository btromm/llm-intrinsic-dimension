#!/bin/bash
# =============================================================================
#  Stage 3 -- probes.  GPU. One array task per model.
#
#  Trains a linear and a 1x200 MLP probe at every layer x task x seed, plus the
#  word-shuffled control tags. For Qwen3-8B that is ~2200 probe fits; on an H100
#  they are small and the stage runs in roughly 3-6 h.
#
#  NOT resumable, and it OVERWRITES probe_results.csv: the stage collects every
#  row in memory and writes once at the end. So it must finish inside one job,
#  and it must cover every task in one process -- see "If a job hits the
#  walltime" in slurm/README.md before splitting it with --tasks.
#
#  Independent of stage 2, so this runs concurrently with the ID job.
# =============================================================================
#SBATCH --job-name=idp-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
run_stage probe "${ARGS[@]}"
