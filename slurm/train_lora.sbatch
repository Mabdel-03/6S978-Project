#!/bin/bash
#SBATCH --job-name=dnah-sft
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#
# Train + merge one Stage-3 condition. Identical hyperparameters across conditions.
#   sbatch slurm/train_lora.sbatch <student> <condition: coding|rlm|answer> <port>
#   e.g. sbatch slurm/train_lora.sbatch bioreason-1.7b rlm 8012
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
ROOT="${DNAH_ROOT:-$(sed -n 's/^root: *//p' configs/paths.yaml | sed 's/ *#.*//')}"
export HF_HOME="$ROOT/hf_cache" DNAH_ROOT="$ROOT"
STUDENT="$1"; COND="$2"; PORT="$3"
PY="$ROOT/envs/serve/bin/python"
DATA="$ROOT/sft_data/$STUDENT/$COND.jsonl"
ADAPTER="$ROOT/ckpts/adapters/$STUDENT-$COND"
"$PY" -m dnaharness.sft.train_lora --base "$STUDENT" --data "$DATA" --out "$ADAPTER" ${TRAIN_ARGS:-}
"$PY" -m dnaharness.sft.merge --base "$STUDENT" --adapter "$ADAPTER" --name "$STUDENT-sft-$COND" --port "$PORT"
