#!/bin/bash
#SBATCH --job-name=dnah-run
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#
# Start vLLM (one server per checkpoint) + the LM gateway on this node, then run one or more
# rendered Harbor jobs against them, in order:
#
#   python -m dnaharness.jobs render configs/jobs/audit.yaml
#   python -m dnaharness.jobs render configs/jobs/smoke.yaml
#   sbatch slurm/serve_and_run.sbatch <jobs>/audit <jobs>/smoke
#
# All jobs in one allocation must share the serving spec (models + policy) of the FIRST job.
# A job whose name starts with "audit" is a hard gate: if it fails, nothing after it runs.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
[ $# -ge 1 ] || { echo "usage: sbatch $0 <rendered job dir> [...]"; exit 2; }

ROOT="${DNAH_ROOT:-$(sed -n 's/^root: *//p' configs/paths.yaml | sed 's/ *#.*//')}"
MAIN="$ROOT/envs/main/bin"
SERVE="$ROOT/envs/serve/bin"
export HF_HOME="$ROOT/hf_cache"
export DNAH_ROOT="$ROOT"
export TMPDIR="/tmp/dnah-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"     # node-local (Harbor staging)
# vLLM's default FlashInfer top-k/top-p sampler JIT-compiles with nvcc, which GPU nodes don't have on
# PATH. The gateway sends a per-request seed, for which vLLM uses the PyTorch-native sampler anyway;
# this makes every path (incl. the startup profile run) use it.
export VLLM_USE_FLASHINFER_SAMPLER=0
# On Hopper, vLLM 0.21 runs a DeepGEMM (FP8) kernel warm-up even for bf16 models and dies when the
# deep_gemm package is absent; nothing here uses FP8.
export VLLM_USE_DEEP_GEMM=0
# one context length for vLLM (--max-model-len) and the gateway (clamps max_tokens to the room left)
export DNAH_MAX_MODEL_LEN="${DNAH_MAX_MODEL_LEN:-32768}"

module load apptainer
# CRITICAL: the apptainer module exports APPTAINER_BINDPATH=/orcd,/nfs; `--no-mount bind-paths` does
# not cover it, and it would expose lab storage (HF cache with labels, task dirs) to every sandbox.
unset APPTAINER_BINDPATH APPTAINER_BIND SINGULARITY_BINDPATH SINGULARITY_BIND
# Harbor 0.27 leaks each finished sandbox's whole process tree (starter/fuse/fakeroot/python); the
# reaper (dnaharness/harbor_ext/sandbox_reaper.py) is loaded into every `harbor` process via this
# sitecustomize hook. Orphans left on this node by earlier allocations are reaped up front.
HOOK="$REPO/slurm/pyhooks"
harbor_() { PYTHONPATH="$HOOK${PYTHONPATH:+:$PYTHONPATH}" "$MAIN/harbor" "$@"; }
"$MAIN/python" -m dnaharness.harbor_ext.sandbox_reaper || true

# ---- per-allocation port block. Several allocations can share a node and sandboxes use the HOST
# network, so each allocation claims a node-local slot k (atomic mkdir) and shifts every port
# (gateway + all vLLM ports, via dnaharness.config) by 100*k. Agents and mini-swe read
# DNAH_GATEWAY_URL at run time, so resumed jobs in a later allocation follow the new ports.
PORT_SLOT=""
for k in $(seq 0 9); do
  d="/tmp/dnah-portslot-$k"
  if ! mkdir "$d" 2>/dev/null; then
    owner=$(cat "$d/owner" 2>/dev/null || true)
    # reclaim a slot whose allocation is gone (killed before its cleanup ran)
    if [ -n "$owner" ] && ! squeue -h -j "$owner" 2>/dev/null | grep -q .; then
      rm -rf "$d"; mkdir "$d" 2>/dev/null || continue
    else
      continue
    fi
  fi
  echo "${SLURM_JOB_ID:-$$}" > "$d/owner"
  ports=$(DNAH_PORT_OFFSET=$((100 * k)) "$MAIN/python" -c "from dnaharness import config as c; p=c.paths(); print(p['gateway_port'], *p['vllm_ports'].values())")
  busy=""
  for pt in $ports; do ss -Htln 2>/dev/null | awk '{print $4}' | grep -q -E ":$pt\$" && busy="$pt"; done
  if [ -n "$busy" ]; then rm -rf "$d"; continue; fi        # someone else's service: try the next slot
  PORT_SLOT=$k; break
done
[ -n "$PORT_SLOT" ] || { echo "[serve] no free port slot on $(hostname)"; exit 5; }
export DNAH_PORT_OFFSET=$((100 * PORT_SLOT))

FIRST="$1"
SERVE_JSON="$FIRST/serve.json"
read_json() { "$MAIN/python" -c "import json,sys;d=json.load(open('$SERVE_JSON'));v=d$1;print(' '.join(v) if isinstance(v,list) else v)"; }
MODELS=$(read_json "['models']")
POLICY=$(read_json "['policy']")
META_DIRS=$(read_json "['meta_dirs']")
LOG_MESSAGES=$(read_json "['log_messages']")
VLLM_KEY=$("$MAIN/python" -c "from dnaharness import config;print(config.load_secrets()['vllm_api_key'])")
GW_PORT=$("$MAIN/python" -c "from dnaharness import config;print(config.paths()['gateway_port'])")
export DNAH_GATEWAY_URL="http://127.0.0.1:$GW_PORT"
echo "[serve] port slot $PORT_SLOT on $(hostname): gateway :$GW_PORT"
GWLOG="$ROOT/logs/gateway/$(basename "$FIRST")-${SLURM_JOB_ID:-local}"
mkdir -p "$GWLOG" logs
for J in "$@"; do basename "$J"; done > "$GWLOG/jobs.txt"   # analysis.collect finds this log for every job listed

PIDS=()
copy_gwlogs() {   # one subdir per allocation (a resumed job has several); analysis.collect reads them all
  for J in "$@"; do mkdir -p "$J/gateway_logs/$(basename "$GWLOG")"; cp -r "$GWLOG/." "$J/gateway_logs/$(basename "$GWLOG")/" 2>/dev/null || true; done
}
ALL_JOBS=("$@")
cleanup() {
  copy_gwlogs "${ALL_JOBS[@]}"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  "$MAIN/python" -m dnaharness.harbor_ext.sandbox_reaper >/dev/null 2>&1 || true
  rm -rf "$TMPDIR" "/tmp/dnah-portslot-$PORT_SLOT"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

ALLOC_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
# vLLM 0.21 cannot use MIG slices (it int()-parses CUDA_VISIBLE_DEVICES). A bare --gres=gpu:1 can be
# given one on mixed partitions (e.g. B300 3g.135gb), so always request a full-GPU type.
case "$ALLOC_GPUS" in *MIG-*) echo "[serve] got a MIG slice ($ALLOC_GPUS); request a full GPU type, e.g. --gres=gpu:l40s:1"; exit 4;; esac
N_MODELS=$(echo $MODELS | wc -w)
MEM_FRAC=$("$MAIN/python" -c "print(round(0.85/$N_MODELS, 2))")
for M in $MODELS; do
  PORT=$("$MAIN/python" -c "from dnaharness import config as c;print(c.model_port('$M'))")
  NATIVE=$("$MAIN/python" -c "from dnaharness import config as c;print(int(bool(c.model_cfg('$M').get('native_dna'))))")
  TP=$("$MAIN/python" -c "from dnaharness import config as c;print(c.model_cfg('$M').get('tensor_parallel',1))")
  EXTRA=()
  [ "$NATIVE" = "1" ] && EXTRA+=(--enable-prompt-embeds)
  DEV=$("$MAIN/python" -c "import json;d=json.load(open('$SERVE_JSON')).get('placement',{}).get('$M',{});print(d.get('devices',''))")
  MF=$("$MAIN/python" -c "import json;d=json.load(open('$SERVE_JSON')).get('placement',{}).get('$M',{});print(d.get('mem',''))")
  [ -n "$MF" ] && MEM_FRAC_M="$MF" || MEM_FRAC_M="$MEM_FRAC"
  echo "[serve] vLLM $M on :$PORT (native=$NATIVE tp=$TP mem=$MEM_FRAC_M devices=${DEV:-all})"
  # placement indices are relative to the GPUs Slurm gave this job
  DEV_MAPPED="$ALLOC_GPUS"
  if [ -n "$DEV" ]; then
    DEV_MAPPED=$("$MAIN/python" -c "a='$ALLOC_GPUS'.split(',');print(','.join(a[int(i)] for i in '$DEV'.split(',')))")
  fi
  CUDA_VISIBLE_DEVICES="$DEV_MAPPED" "$SERVE/vllm" serve "$ROOT/ckpts/$M" --served-model-name "$M" --host 127.0.0.1 --port "$PORT" \
      --api-key "$VLLM_KEY" --dtype bfloat16 --max-model-len "$DNAH_MAX_MODEL_LEN" --gpu-memory-utilization "$MEM_FRAC_M" \
      --tensor-parallel-size "$TP" --enable-prefix-caching --seed 0 "${EXTRA[@]}" \
      > "$GWLOG/vllm-$M.log" 2>&1 &
  PIDS+=($!)
  for i in $(seq 1 180); do
    curl -sf -H "Authorization: Bearer $VLLM_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
    kill -0 "${PIDS[-1]}" 2>/dev/null || { echo "vLLM $M died"; tail -50 "$GWLOG/vllm-$M.log"; exit 1; }
    sleep 5
  done
done

GW_ARGS=(--models $MODELS --policy "$POLICY" --log-dir "$GWLOG")
[ -n "$META_DIRS" ] && GW_ARGS+=(--meta-dirs $META_DIRS)
[ "$LOG_MESSAGES" = "True" ] && GW_ARGS+=(--log-messages)
"$SERVE/python" -m dnaharness.serving.gateway "${GW_ARGS[@]}" > "$GWLOG/gateway.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null && break; sleep 2; done
curl -sf "http://127.0.0.1:$GW_PORT/health" || { echo "gateway failed"; tail -50 "$GWLOG/gateway.log"; exit 1; }
echo

# Per-allocation isolation audit on THIS node (hard gate; ~10 s). Disable with DNAH_AUDIT=0.
if [ "${DNAH_AUDIT:-1}" = "1" ] && [[ "$(basename "$FIRST")" != audit* ]]; then
  AJ="$ROOT/jobs/_audits/${SLURM_JOB_ID:-local-$$}"
  # audit task generated for THIS allocation's ports (DNAH_PORT_OFFSET) into its own dataset dir
  DNAH_DATASETS="$AJ/datasets" "$MAIN/python" -m dnaharness.adapters.audit >/dev/null
  DNAH_DATASETS="$AJ/datasets" "$MAIN/python" -m dnaharness.jobs render configs/jobs/audit.yaml --jobs-dir "$AJ" >/dev/null
  harbor_ run -c "$AJ/audit/harbor_config.yaml" > "$AJ/harbor.log" 2>&1 || true
  "$MAIN/python" -m dnaharness.analysis.collect --gate-audit "$AJ/audit" \
    || { echo "[gate] AUDIT FAILED on $(hostname) -- aborting before any trial"; exit 3; }
  echo "[gate] audit passed on $(hostname)"
fi

for JOB in "$@"; do
  NAME=$(basename "$JOB")
  if [ -f "$JOB/lock.json" ]; then
    # an earlier allocation ran (part of) this job: resume it -- only unfinished trials run again.
    # DNAH_RESUME_FILTER="ErrA ErrB": trials that ended with these exception types are removed and re-run.
    # default: every infrastructure failure seen so far (sandbox start/upload/server errors, and trials cut
    # off when an allocation was cancelled). Model outcomes (AgentTimeoutError) and verifier timeouts are
    # kept: re-running them would give those trials a second sample. DNAH_RESUME_FILTER="" disables.
    DNAH_RESUME_FILTER="${DNAH_RESUME_FILTER-EnvironmentStartTimeoutError RuntimeError AddTestsDirError FileNotFoundError ConnectError}"
    FILT=(); for e in ${DNAH_RESUME_FILTER:-}; do FILT+=(-f "$e"); done
    "$MAIN/python" -m dnaharness.harbor_ext.resume_repair "$JOB"     # Harbor-scrubbed dummy keys (old renders)
    echo "[run] resuming harbor job $NAME ${DNAH_RESUME_FILTER:+(re-running: $DNAH_RESUME_FILTER)}"
    harbor_ jobs resume -p "$JOB" "${FILT[@]}" || echo "[run] harbor exited non-zero for $NAME"
  else
    echo "[run] harbor job $NAME"
    harbor_ run -c "$JOB/harbor_config.yaml" || echo "[run] harbor exited non-zero for $NAME"
  fi
  copy_gwlogs "$JOB"
  if [[ "$NAME" == audit* ]]; then
    "$MAIN/python" -m dnaharness.analysis.collect --gate-audit "$JOB" || { echo "[gate] AUDIT FAILED -- aborting"; exit 3; }
  fi
done
echo "[done] gateway logs: $GWLOG"
