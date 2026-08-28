#!/bin/bash
# =============================================================================
#  Stage 4 -- analysis and figures.  CPU. One array task per model.
#
#  Needs BOTH stage 2 (id) and stage 3 (probe) to have finished for this model.
#  Reads what is already on disk and computes nothing expensive:
#    robustness -> peak_robustness.csv   (is each ID peak stable across scale?)
#    analyze    -> linear_vs_mlp.csv, id_accuracy_correlation.csv, peak_alignment.csv
#    figures    -> the Cheng et al. figure reproductions
#    plots      -> the project's own summary plots
#
#  `figures` prints "SKIP <name>" for anything whose inputs are missing (the
#  two NEXT_STEPS figures, if RUN_DIRECTIONS=0) rather than failing.
# =============================================================================
#SBATCH --job-name=idp-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
run_stage robustness "${ARGS[@]}"
run_stage analyze    "${ARGS[@]}"

FIG_ARGS=("${ARGS[@]}")
if [[ -n "$FIGURES_COMPARE_MODEL" && "$FIGURES_COMPARE_MODEL" != "$MODEL" ]]; then
  other_tag="${FIGURES_COMPARE_MODEL##*/}"
  if [[ -d "$REPO/activations/$other_tag" ]]; then
    FIG_ARGS+=(--compare-model "$FIGURES_COMPARE_MODEL" --cka-task "$CKA_TASK")
  else
    log "no activations for $other_tag; skipping the cross-model CKA figure"
  fi
fi
run_stage figures "${FIG_ARGS[@]}"
run_stage plots   "${ARGS[@]}"

log "results for $MODEL_TAG:"
ls -lh "$REPO/results/$MODEL_TAG" 2>/dev/null || true
