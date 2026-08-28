#!/bin/bash
# =============================================================================
#  Stage 0 -- preflight. One short GPU job that proves, on a real compute node,
#  that the long jobs behind it can succeed: env, GPU, model weights, SentEval
#  files, corpus streaming, and disk headroom.
#
#  Everything else in the chain depends on this with afterok, so a failure here
#  stops the run before it burns an allocation.
# =============================================================================
#SBATCH --job-name=idp-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Slurm copies this script to a spool directory, so $0 cannot locate the repo.
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
banner

mapfile -t MODELS_TO_CHECK < <(pipeline_models)

ARGS=(--models "${MODELS_TO_CHECK[@]}")
[[ -n "$N_TRAIN" ]] && ARGS+=(--n-train "$N_TRAIN")
[[ -n "$N_VAL"   ]] && ARGS+=(--n-val   "$N_VAL")
[[ -n "$N_TEST"  ]] && ARGS+=(--n-test  "$N_TEST")
if [[ -n "$CORPORA" ]]; then
  # shellcheck disable=SC2206
  CORP=($CORPORA)
  ARGS+=(--corpora "${CORP[@]}")
fi
[[ "$RUN_SHUFFLED" == "1" ]] || ARGS+=(--no-shuffled)

log "checking: ${MODELS_TO_CHECK[*]}"
python "$SLURM_DIR/preflight.py" "${ARGS[@]}"

# The tiny-model smoke test the project README prescribes: a full extract on a
# 2-layer random model. Probes on it must land at chance, which is the pipeline's
# own label-leakage check. Costs about a minute and exercises the real code path.
log "smoke test on hf-internal-testing/tiny-random-LlamaForCausalLM"
python "$REPO/run.py" extract \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --tasks bigram_shift --n-train 400 --n-val 100 --n-test 200 \
  --device cpu --no-corpus --no-shuffled --force
rm -rf "$REPO/activations/tiny-random-LlamaForCausalLM"
log "smoke test OK (activations removed)"
