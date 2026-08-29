#!/bin/bash -l
# =============================================================================
#  The whole pipeline for Qwen3-1.7B-Base, in one Grid Engine job.
#
#      qsub sge/run_qwen1.7b.sh
#
#  Run sge/setup.sh on a login node first.
# =============================================================================

# 48h is Myriad's cap for a multi-slot job. The probe stage is not resumable
# and rewrites its CSV at the end, so a walltime kill loses it -- ask for the max.
#$ -l h_rt=48:0:0

# *** mem is PER SLOT, not per job ***  8 x 4G = 32G total.
# 4.4G is the per-slot ceiling on Myriad's GPU nodes, so more RAM means more slots.
#$ -pe smp 8
#$ -l mem=4G

# Local disk for $TMPDIR. Activations go to Scratch, so this only needs to cover
# scratch files from HF/datasets.
#$ -l tmpfs=15G

# One GPU, pinned to an L node (A100 40GB). Without `-ac allow=L` you may land
# on an E/F node whose V100s are a much tighter fit.
#$ -l gpu=1
#$ -ac allow=L

#$ -N idp-1.7b

# EDIT THIS: absolute path to your checkout. Grid Engine takes the whole rest of
# the line as the argument, so never put a trailing comment on a #$ line.
#$ -wd /home/YOUR_UCL_ID/Code/id

# Logs land in <wd>/logs/ as idp-1.7b.o<jobid>; -j y merges stderr into stdout.
#$ -o logs/
#$ -j y

set -euo pipefail

MODEL="Qwen/Qwen3-1.7B-Base"
BATCH=128                # A100 40GB. Halve it if you hit CUDA out-of-memory.

source "$HOME/Scratch/id-venv/bin/activate"

export HF_HOME="$HOME/Scratch/hf-cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Compute nodes have no internet. setup.sh cached everything; fail fast on a
# miss rather than hanging on a connection timeout.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Keep BLAS inside the allocation -- GRIDE in the `id` stage is BLAS-bound, and
# by default it would size its thread pool from the node's 36 cores, not our 8.
export OMP_NUM_THREADS="$NSLOTS"
export MKL_NUM_THREADS="$NSLOTS"
export OPENBLAS_NUM_THREADS="$NSLOTS"

echo "host $(hostname)  slots $NSLOTS  started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python run.py extract    --model "$MODEL" --batch-size "$BATCH"
python run.py id         --model "$MODEL"
python run.py probe      --model "$MODEL"
python run.py robustness --model "$MODEL"
python run.py analyze    --model "$MODEL"
python run.py figures    --model "$MODEL"
python run.py plots      --model "$MODEL"

# To also run the NEXT_STEPS directions, add:
#   python run.py conditional-id --model "$MODEL" --cid-n-match 2000
#   python run.py magnitude      --model "$MODEL"

echo "finished $(date)"
ls -lh results/Qwen3-1.7B-Base
