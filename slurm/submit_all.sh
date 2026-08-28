#!/usr/bin/env bash
# =============================================================================
#  submit_all.sh -- submit the whole pipeline as one dependency chain.
#  Run this from a LOGIN node after editing config.sh and running setup.sh.
#
#      bash slurm/submit_all.sh                 # everything in config.sh
#      bash slurm/submit_all.sh -n              # dry run: print, submit nothing
#      bash slurm/submit_all.sh -m Qwen/Qwen3-8B-Base
#      bash slurm/submit_all.sh -s id,probe,analyze --no-preflight
#
#  The graph it builds (one array task per model, index i = model i):
#
#      preflight ──> extract[i] ─┬─> id[i] ─┬─> analyze[i]
#                                │          └─> directions[i]
#                                └─> probe[i] ──> analyze[i]
#                ──> control[i]
#
#  Stages linked with aftercorr, so 1.7B's ID job starts as soon as 1.7B's
#  extraction is done -- it does not wait for 8B. id[] and probe[] run at the
#  same time (one is CPU-only, the other GPU).
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$REPO/slurm"
LOG_DIR="$REPO/logs"
source "$SLURM_DIR/config.sh"

DRY=0
NO_PREFLIGHT=0
STAGES=""
MODELS_OVERRIDE=""

usage() {
  # everything from line 2 up to the first non-comment line
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  cat <<'USAGE'

Options:
  -m, --models "A B"   space-separated model ids, overriding MODELS in config.sh
  -s, --stages LIST    comma-separated subset of:
                       preflight,extract,id,probe,analyze,directions,control
      --no-preflight   drop the preflight job (use once it has already passed)
  -n, --dry-run        print the sbatch commands without submitting
  -h, --help           this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--models)  MODELS_OVERRIDE="$2"; shift 2 ;;
    -s|--stages)  STAGES="$2"; shift 2 ;;
    --no-preflight) NO_PREFLIGHT=1; shift ;;
    -n|--dry-run) DRY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# --- sanity checks the scheduler would otherwise report hours later ----------
[[ -f "$REPO/run.py" ]] || { echo "ERROR: $REPO/run.py not found" >&2; exit 1; }
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found -- run this on a login node of the cluster." >&2
  exit 1
fi
case "$REPO$LOG_DIR" in
  *,*|*\ *) echo "ERROR: the repo path contains a comma or a space, which sbatch" >&2
            echo "       --export cannot carry. Move the checkout somewhere plainer." >&2
            exit 1 ;;
esac
if [[ -z "$PARTITION_GPU" ]]; then
  echo "WARNING: PARTITION_GPU is empty in config.sh; jobs go to the cluster default." >&2
fi

mkdir -p "$LOG_DIR"

# --- which models --------------------------------------------------------------
if [[ -n "$MODELS_OVERRIDE" ]]; then
  # shellcheck disable=SC2206
  SELECTED=($MODELS_OVERRIDE)
else
  SELECTED=("${MODELS[@]}")
fi
N=${#SELECTED[@]}
[[ $N -gt 0 ]] || { echo "ERROR: no models selected" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
MODEL_FILE="$LOG_DIR/models-$STAMP.txt"
printf '%s\n' "${SELECTED[@]}" > "$MODEL_FILE"

# --- which stages --------------------------------------------------------------
want() { [[ ",$STAGE_LIST," == *",$1,"* ]]; }

if [[ -n "$STAGES" ]]; then
  STAGE_LIST="$STAGES"
else
  STAGE_LIST="preflight,extract,id,probe,analyze"
  [[ "$RUN_DIRECTIONS"  == "1" ]] && STAGE_LIST="$STAGE_LIST,directions"
  [[ "$RUN_RANDOM_INIT" == "1" ]] && STAGE_LIST="$STAGE_LIST,control"
fi

# Normalise the list rather than editing the string: a substring removal of
# "preflight," misses `-s preflight` and `-s extract,preflight`, where the name
# is not followed by a comma, and would silently submit the job anyway.
# Trimming also lets `-s "extract, id"` work -- `want` matches list elements
# exactly, so a stray space would silently drop that stage.
_stages=""
IFS=',' read -ra _parts <<< "$STAGE_LIST"
for _s in "${_parts[@]+"${_parts[@]}"}"; do
  _s="${_s#"${_s%%[![:space:]]*}"}"      # strip leading space
  _s="${_s%"${_s##*[![:space:]]}"}"      # strip trailing space
  [[ -z "$_s" ]] && continue
  [[ $NO_PREFLIGHT -eq 1 && "$_s" == "preflight" ]] && continue
  _stages="${_stages:+$_stages,}$_s"
done
STAGE_LIST="$_stages"
[[ -n "$STAGE_LIST" ]] || { echo "ERROR: no stages selected" >&2; exit 1; }

ARRAY_GPU="0-$((N - 1))%$MAX_CONCURRENT"
ARRAY_CPU="0-$((N - 1))%$MAX_CONCURRENT"
ARRAY_ID="0-$((N - 1))%1"          # see the header of 02_id.sbatch

echo "repo    : $REPO"
echo "models  : ${SELECTED[*]}"
echo "stages  : $STAGE_LIST"
echo "logs    : $LOG_DIR"
echo

# --- submission helper ---------------------------------------------------------
# submit_stage <label> <script> <time> <cpus> <mem> <gpu 0|1> <array|-> <dep|->
submit_stage() {
  local label="$1" script="$2" time="$3" cpus="$4" mem="$5" gpu="$6" array="$7" dep="$8"

  local part="$PARTITION_CPU"
  if [[ "$gpu" == "1" || -z "$part" ]]; then part="$PARTITION_GPU"; fi

  local -a cmd=(sbatch --parsable
                --export="ALL,PIPE_REPO=$REPO,PIPE_MODEL_FILE=$MODEL_FILE"
                --job-name "idp-$label"
                --time "$time" --cpus-per-task "$cpus" --mem "$mem")
  if [[ -n "$part"       ]]; then cmd+=(--partition "$part"); fi
  if [[ -n "$ACCOUNT"    ]]; then cmd+=(--account "$ACCOUNT"); fi
  if [[ -n "$QOS"        ]]; then cmd+=(--qos "$QOS"); fi
  if [[ -n "$CONSTRAINT" ]]; then cmd+=(--constraint "$CONSTRAINT"); fi
  if [[ -n "$MAIL_USER"  ]]; then cmd+=(--mail-user "$MAIL_USER" --mail-type "$MAIL_TYPE"); fi
  # shellcheck disable=SC2206
  if [[ "$gpu" == "1"    ]]; then cmd+=($GPU_REQUEST); fi
  if [[ "$array" != "-"  ]]; then
    cmd+=(--array "$array" --output "$LOG_DIR/idp-$label-%A_%a.out")
  else
    cmd+=(--output "$LOG_DIR/idp-$label-%j.out")
  fi
  # --kill-on-invalid-dep stops a job whose dependency failed from sitting in the
  # queue forever as DependencyNeverSatisfied.
  if [[ "$dep" != "-"    ]]; then cmd+=(--dependency "$dep" --kill-on-invalid-dep=yes); fi
  cmd+=("$SLURM_DIR/$script")

  if [[ $DRY -eq 1 ]]; then
    # The command goes to stderr: stdout is captured by the caller as a job id.
    { printf '  %q' "${cmd[@]}"; echo; } >&2
    # Deterministic fake ids (a subshell increment would not survive), so the
    # dependency graph printed below is still readable.
    local -A FAKE=([preflight]=1001 [extract]=1002 [id]=1003 [probe]=1004
                   [analyze]=1005 [directions]=1006 [control]=1007)
    echo "${FAKE[$label]:-1099}"
    return 0
  fi
  # --parsable prints "jobid" (or "jobid;cluster" on a federated cluster).
  local out
  out="$("${cmd[@]}")"
  echo "${out%%;*}"
}

declare -A JID
note() { printf '%-12s %-12s %s\n' "$1" "$2" "$3"; }
printf '%-12s %-12s %s\n' "STAGE" "JOB ID" "DEPENDS ON"

if want preflight; then
  JID[preflight]="$(submit_stage preflight 00_preflight.sbatch \
      "$TIME_PREFLIGHT" "$CPUS_PREFLIGHT" "$MEM_PREFLIGHT" 1 - -)"
  note preflight "${JID[preflight]}" "-"
fi

pre_dep="-"
if [[ -n "${JID[preflight]:-}" ]]; then pre_dep="afterok:${JID[preflight]}"; fi

if want extract; then
  JID[extract]="$(submit_stage extract 01_extract.sbatch \
      "$TIME_EXTRACT" "$CPUS_EXTRACT" "$MEM_EXTRACT" 1 "$ARRAY_GPU" "$pre_dep")"
  note extract "${JID[extract]}" "$pre_dep"
fi

ex_dep="$pre_dep"
if [[ -n "${JID[extract]:-}" ]]; then ex_dep="aftercorr:${JID[extract]}"; fi

if want id; then
  JID[id]="$(submit_stage id 02_id.sbatch \
      "$TIME_ID" "$CPUS_ID" "$MEM_ID" 0 "$ARRAY_ID" "$ex_dep")"
  note id "${JID[id]}" "$ex_dep"
fi
if want probe; then
  JID[probe]="$(submit_stage probe 03_probe.sbatch \
      "$TIME_PROBE" "$CPUS_PROBE" "$MEM_PROBE" 1 "$ARRAY_GPU" "$ex_dep")"
  note probe "${JID[probe]}" "$ex_dep"
fi

# analyze needs both id and probe for the same model, hence two aftercorr terms
# (comma-separated dependencies are ANDed).
an_dep=""
if [[ -n "${JID[id]:-}"    ]]; then an_dep="aftercorr:${JID[id]}"; fi
if [[ -n "${JID[probe]:-}" ]]; then an_dep="${an_dep:+$an_dep,}aftercorr:${JID[probe]}"; fi
[[ -n "$an_dep" ]] || an_dep="$ex_dep"

if want analyze; then
  JID[analyze]="$(submit_stage analyze 04_analyze.sbatch \
      "$TIME_ANALYZE" "$CPUS_ANALYZE" "$MEM_ANALYZE" 0 "$ARRAY_CPU" "$an_dep")"
  note analyze "${JID[analyze]}" "$an_dep"
fi

if want directions; then
  di_dep="$ex_dep"
  if [[ -n "${JID[id]:-}" ]]; then di_dep="aftercorr:${JID[id]}"; fi
  JID[directions]="$(submit_stage directions 05_directions.sbatch \
      "$TIME_DIRECTIONS" "$CPUS_DIRECTIONS" "$MEM_DIRECTIONS" 1 "$ARRAY_GPU" "$di_dep")"
  note directions "${JID[directions]}" "$di_dep"
fi

if want control; then
  JID[control]="$(submit_stage control 06_random_init.sbatch \
      "$TIME_CONTROL" "$CPUS_CONTROL" "$MEM_CONTROL" 1 "$ARRAY_GPU" "$pre_dep")"
  note control "${JID[control]}" "$pre_dep"
fi

echo
if [[ $DRY -eq 1 ]]; then
  echo "dry run -- nothing was submitted (job ids above are fake)."
  exit 0
fi

cat <<SUMMARY
model list for this submission: $MODEL_FILE
(array index i runs line i+1 of that file)

  watch the queue : squeue -u \$USER
  follow a log    : tail -f $LOG_DIR/idp-extract-*_0.out
  why is it idle  : squeue -u \$USER --start ; scontrol show job <jobid>
  cancel it all   : scancel ${JID[*]}

Results land in results/<model-tag>/ ; activations in
$(readlink -f "$REPO/activations" 2>/dev/null || echo "$REPO/activations")
SUMMARY
