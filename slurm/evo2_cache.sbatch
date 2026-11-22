#!/bin/bash
#SBATCH --job-name=dnah-evo2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#
# One-time Evo2-1B embedding cache for every DNA window referenced by generated tasks.
#   sbatch slurm/evo2_cache.sbatch            (defaults to all _meta dirs under <datasets>)
# Hopper (H200; H100 is the same sm_90 arch) requested for parity with the paper's hardware; with TE
# blocked, A100/L40S also work.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
ROOT="${DNAH_ROOT:-$(sed -n 's/^root: *//p' configs/paths.yaml | sed 's/ *#.*//')}"
export HF_HOME="$ROOT/hf_cache"
module load apptainer
unset APPTAINER_BINDPATH APPTAINER_BIND SINGULARITY_BINDPATH SINGULARITY_BIND
META_DIRS=("$@")
[ ${#META_DIRS[@]} -gt 0 ] || mapfile -t META_DIRS < <(find "$ROOT/datasets" -maxdepth 2 -type d -name _meta)
echo "meta dirs: ${META_DIRS[*]}"
# evo2.sif has its CUDA compat libcuda removed (see containers/evo2.def), but the image's ldconfig cache
# still lists it, and Triton locates libcuda via `ldconfig -p` -> point it at the host driver (--nv).
apptainer exec --nv --containall -B "$REPO:$REPO" -B "$ROOT:$ROOT" --env HF_HOME="$HF_HOME" \
  --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
  --env PYTHONPATH="$REPO/src" "$ROOT/sif/evo2.sif" \
  python -m dnaharness.serving.evo2_cache --meta-dirs "${META_DIRS[@]}" --out "$ROOT/evo2_cache" \
    --bioreason-repo "$REPO/third_party/bio-posttrain/BioReason"
