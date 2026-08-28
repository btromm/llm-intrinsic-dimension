#!/usr/bin/env bash
# =============================================================================
#  Delete one model's activations once its analysis is done.
#
#      bash slurm/cleanup_activations.sh Qwen3-8B-Base
#      bash slurm/cleanup_activations.sh --list
#
#  Activations are the only large output (~140 GB for Qwen3-8B) and nothing in
#  results/ needs them again -- but re-creating them means re-running the
#  extract stage, so this asks before deleting and refuses unless the model's
#  results actually exist.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTS="$REPO/activations"

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

if [[ "$1" == "--list" ]]; then
  echo "activation directories under $(readlink -f "$ACTS" 2>/dev/null || echo "$ACTS"):"
  du -sh "$ACTS"/*/ 2>/dev/null || echo "  (none)"
  exit 0
fi

TAG="${1##*/}"                     # accept either Qwen/Qwen3-8B-Base or Qwen3-8B-Base
DIR="$ACTS/$TAG"
RES="$REPO/results/$TAG"

[[ -d "$DIR" ]] || { echo "no such directory: $DIR" >&2; exit 1; }

missing=()
for f in id_profiles.csv probe_results.csv; do
  [[ -f "$RES/$f" ]] || missing+=("$f")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "REFUSING: results/$TAG is missing ${missing[*]}." >&2
  echo "Those stages read the activations, so deleting now would mean" >&2
  echo "re-running extract. Finish the analysis first." >&2
  exit 1
fi

SIZE="$(du -sh "$DIR" | cut -f1)"
echo "About to delete $DIR ($SIZE)."
echo "results/$TAG is complete, so this only costs a re-extraction if you need"
echo "the raw hidden states again."
read -r -p "Type the model tag to confirm: " confirm
if [[ "$confirm" != "$TAG" ]]; then
  echo "not confirmed; nothing deleted"; exit 1
fi
rm -rf "$DIR"
echo "deleted $DIR"
