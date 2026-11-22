# Experiment grid

New work is a cell under `experiments/`. The frozen baseline specs stay in `configs/jobs/` (see [baselines/README.md](baselines/README.md)). Execution stays on the shared `bio-sandbox.sif` (`sandbox_sif` in `configs/paths.yaml`).

Base checkpoints, served as `qwen3-1.7b` and `bioreason-1.7b` in `configs/models.yaml`. Locked arms in `configs/harness/arms.yaml`: `direct`, `coding_repl`, `rlm`. Exp B controls, used when the checkpoint has native DNA (`native_dna: true`, as on `bioreason-1.7b`): `direct_harness`, `vote`, `coding_repl_tool`.

Prompts stay at `PROMPT_VERSION` `v3` (`src/dnaharness/runtime/prompts.py`) unless a dev-only edit bumps the version.

## Grids

### `harness_opt`

A harness-only cell adds an arm and evaluates it on the locked eval sets for every base model, using the base checkpoints. One Harbor job per base model (`qwen3-1.7b` and `bioreason-1.7b`). The arm key is the name the cell adds; the job serves the base checkpoint.

### `model_opt`

A model-only cell registers a new served name and evaluates it on the locked arms (`direct`, `coding_repl`, `rlm`). When the checkpoint has native DNA, also evaluate the Exp B controls (`direct_harness`, `vote`, `coding_repl_tool`). The arms stay the locked ones; the served name is the new checkpoint.

### `joint`

A joint cell changes both the arm and the checkpoint. Compare it as a joint change against the parent model and parent arm on the same locked eval sets.

## Claim a cell

1. Copy `experiments/_template/` to `experiments/<kind>/<approach>/`. `<kind>` is `harness_opt`, `model_opt`, or `joint`. `<approach>` is a short slug and contains no `__`.
2. Fill `approach.yaml`: owner, kind, parent model, parent arm, and one line each for the task-accuracy claim and the DNA-use claim (intact DNA versus no-DNA and scramble).
3. Fill `job.yaml`. One spec per arm and served model. The job `name` is `<kind>__<approach>__<arm>__<model>` so it cannot collide with `exp_a_shared-*` or `sft_eval_*`. `<arm>` and `<model>` in that name are the cell’s arm key and served name (`qwen3-1.7b`, `bioreason-1.7b`, or the name this cell registers).
4. A `model_opt` or `joint` cell also commits `model.yaml` in that directory: `base` (an existing served name), the train datasets (`harness-train`, `kegg-dx-train_pool`), and `port` of at least 8010 (above the base vLLM ports in `configs/paths.yaml`). Derived checkpoints stay out of `configs/models.yaml`. Registration stays the existing `<ckpts>/registry.yaml` path that `dnaharness.config.models` already merges (`src/dnaharness/config.py`). Leave that loader unchanged. `dnaharness.sft.merge` is the writer that already appends a served name there.
5. The first render smokes the new cell only: `purpose: dev` on `kegg-dx-dev`, or `harness-pilot` only to check that the cell runs. Reported numbers use `purpose: eval` and the locked eval sets (`kegg-dx/...` and/or `harness-pilot/pilot`).

Locked eval sets (`purpose: eval`): the `kegg-dx` cohort, every interface and condition the baseline jobs already name (`raw-*`, `native-*`, `native_raw-*`), and `harness-pilot/pilot`. The dev smoke set is `kegg-dx-dev`.
