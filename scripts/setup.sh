#!/bin/bash
# One-time setup. Run on a node with internet (login or a CPU job). Everything large goes under the
# storage root from configs/paths.yaml (override with DNAH_ROOT).
#
#   bash scripts/setup.sh envs        # uv envs: main (Harbor, py3.12) + serve (vLLM 0.21 + gateway + train)
#   bash scripts/setup.sh secrets     # HMAC key for DNA handles + vLLM/gateway keys (chmod 600)
#   bash scripts/setup.sh ckpts       # download pinned checkpoints, patch BioReason rope_theta
#   bash scripts/setup.sh containers  # build bio-sandbox.sif and evo2.sif (apptainer --fakeroot)
#   bash scripts/setup.sh data        # manifest + all Harbor datasets (+ pilot/train/audit tasks)
#   bash scripts/setup.sh data_train_pool  # kegg-dx-train_pool only; not part of data or all
#   bash scripts/setup.sh all
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ROOT="${DNAH_ROOT:-$(sed -n 's/^root: *//p' configs/paths.yaml | sed 's/ *#.*//')}"
export UV_CACHE_DIR="$ROOT/uv_cache" HF_HOME="$ROOT/hf_cache" DNAH_ROOT="$ROOT"
mkdir -p "$ROOT"/{envs,hf_cache,sif,jobs,evo2_cache,ckpts,datasets,logs}
MAIN="$ROOT/envs/main/bin/python"
SERVE="$ROOT/envs/serve/bin/python"

envs() {
  git submodule update --init --recursive
  [ -x "$MAIN" ] || uv venv --python 3.12 "$ROOT/envs/main"
  VIRTUAL_ENV="$ROOT/envs/main" uv pip install -e ".[harbor,analysis,dev]"
  [ -x "$SERVE" ] || uv venv --python 3.12 "$ROOT/envs/serve"
  VIRTUAL_ENV="$ROOT/envs/serve" uv pip install -e ".[serve,train,dev]"
}
secrets() { "$MAIN" -c "from dnaharness import config; config.load_secrets(create=True); print('secrets at', config.path('secrets'))"; }
ckpts() { "$SERVE" -m dnaharness.serving.prepare_ckpt qwen3-1.7b bioreason-1.7b; }
ckpts_teacher() { "$SERVE" -m dnaharness.serving.prepare_ckpt qwen3-32b; }
containers() {
  module load apptainer
  unset APPTAINER_BINDPATH APPTAINER_BIND SINGULARITY_BINDPATH SINGULARITY_BIND
  export APPTAINER_CACHEDIR="$ROOT/apptainer_cache" APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer-$$"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  apptainer build --fakeroot --force "$ROOT/sif/bio-sandbox.sif" containers/bio-sandbox.def
  apptainer build --fakeroot --force "$ROOT/sif/evo2.sif" containers/evo2.def
}
data() {
  "$MAIN" -m dnaharness.data.cohort
  "$MAIN" -m dnaharness.adapters.kegg_dx.adapter --interfaces raw native native_raw \
      --conditions wt no_dna scramble revert_ref swap_pair text_conflict
  "$MAIN" -m dnaharness.adapters.kegg_dx.adapter --role dev --interfaces raw native --conditions wt
  "$MAIN" -m dnaharness.adapters.kegg_dx.adapter --output-dir "$ROOT/datasets/kegg-dx-smoke" --interfaces raw --conditions wt --limit 5
  "$MAIN" -m dnaharness.adapters.harness_tasks --split pilot
  "$MAIN" -m dnaharness.adapters.harness_tasks --split train
  "$MAIN" -m dnaharness.adapters.audit
}
data_train_pool() {
  "$MAIN" -m dnaharness.adapters.kegg_dx.adapter --role train_pool --interfaces raw native \
      --conditions wt no_dna scramble
}

[ $# -gt 0 ] || set -- all
for step in "$@"; do
  case "$step" in
    envs) envs ;; secrets) secrets ;; ckpts) ckpts ;; ckpts_teacher) ckpts_teacher ;;
    containers) containers ;; data) data ;; data_train_pool) data_train_pool ;;
    all) envs; secrets; ckpts; data; containers ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
