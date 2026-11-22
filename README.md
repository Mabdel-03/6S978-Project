# 6.S978 Course Project: Investigating The Impacts of Harness + Model on Biological Reasoning Models' Input Utilization

How can we improve biological reasoning models' utilization of their biological inputs in DNA
related tasks? That is the question this repository is built to answer. We take models that can
consume DNA (or DNA-derived embeddings) alongside pathway text, and ask whether harness design and
post-training change not only disease-prediction accuracy, but whether the model actually uses the
sequence content rather than ignoring it or treating it as a format cue. The evaluation design
follows BioMirage ([arXiv:2610.00898](https://arxiv.org/abs/2610.00898)): the same intact DNA, text
only, and composition-preserving scramble conditions on a fixed cohort of KEGG variant to disease
examples.

In concrete terms, the primary task is: given a short pathway description and a pair of DNA windows
(reference and variant), name the disease the variant contributes to from a closed label list. We
also run a small set of non-biological harness competence tasks so a model must prove it can operate
a REPL agent before we trust harness results on DNA. For worked input and output examples, see
[Task examples](#task-examples).

Runs are [Harbor](https://harborframework.com) jobs on the ORCD Slurm cluster, inside Apptainer sandboxes.

| Resource | Link |
|---|---|
| Claim a cell | Copy [`experiments/_template`](experiments/_template); grid rules in [experiments/README.md](experiments/README.md) |
| Task input and output examples | [Task examples](#task-examples) |
| Finished numbers and Harbor incident log | [docs/STATUS.md](docs/STATUS.md) |
| Metric definitions | [docs/METRICS.md](docs/METRICS.md) |

## Base models

Pinned in [`configs/models.yaml`](configs/models.yaml) (revisions fetched 2026-11-21). The two
1.7B checkpoints are the bases. `qwen3-32b` is the stage-3 teacher only.

| Served name | Checkpoint | Revision | Native DNA | Role |
|---|---|---|---|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | no | base |
| `bioreason-1.7b` | `mims-harvard/bio-posttrain-qwen3-1.7b-dna-sft` | `0fc8d19d6a0fc3046cc3aa7b75e3627c4e2f225f` | yes | base |
| `qwen3-32b` | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | no | teacher only |

BioReason is Qwen3-1.7B with a frozen Evo2-1B encoder and a trained 1920→2048 projection. On the
native interface the gateway turns DNA handles into projected Evo2 `prompt_embeds`. Qwen3-32B is
downloaded separately (`bash scripts/setup.sh ckpts_teacher`, tensor parallel 2).

Leave `configs/models.yaml` as these three rows. A model cell commits its own `model.yaml` (base,
train datasets, port `>= 8010`). Serving loads that checkpoint from `<ckpts>/registry.yaml`, which
[`src/dnaharness/config.py`](src/dnaharness/config.py) already merges.

## Access classes

[`configs/data_access.yaml`](configs/data_access.yaml) is the permission list. When a spec sets
`purpose`, every dataset is checked against that class before a job directory is written. A spec
with no `purpose`, including [`configs/jobs/audit.yaml`](configs/jobs/audit.yaml), renders as it
does today. `purpose: train` and `purpose: dev` cannot name an eval dataset.

| Class | Datasets | Purpose |
|---|---|---|
| `eval` | `kegg-dx` (165-row cohort, every interface and condition), `harness-pilot` | Report-only |
| `dev` | `kegg-dx-dev` | Prompt and harness debugging |
| `train` | `harness-train`, `kegg-dx-train_pool` | The only biological text and DNA post-training may see |
| `plumbing` | `audit`, `kegg-dx-smoke` | Smoke rows are cohort rows; their accuracy is not a result |

The Hugging Face KEGG `train` / `val` / `test` split is BioReason’s history (126 of 165 cohort
rows are `train`). `orig_split` stays a reporting column in
[`src/dnaharness/analysis/metrics.py`](src/dnaharness/analysis/metrics.py). It is not permission
to train.

`scripts/setup.sh data` builds the cohort, dev, smoke, pilot, train, and audit tasks. The train
pool is a separate one-time command (see [Setup](#setup-through-audit-and-smoke)) and writes only
`<datasets>/kegg-dx-train_pool/`.

## Experiment grids

Three directories under `experiments/`. Execution stays on the shared `bio-sandbox.sif`.
[`experiments/baselines/README.md`](experiments/baselines/README.md) points at the frozen specs in
`configs/jobs/` and does not copy them.

| Grid | What the cell changes | How it is compared |
|---|---|---|
| [`harness_opt`](experiments/harness_opt) | A new arm, on the base checkpoints, for every base model, on the locked eval sets | Harness-only |
| [`model_opt`](experiments/model_opt) | A new served name, on the locked arms | Model-only |
| [`joint`](experiments/joint) | An arm and a checkpoint together | As a joint change |

Locked arms are `direct`, `coding_repl`, and `rlm`, plus the Experiment B controls
(`direct_harness`, `vote`, `coding_repl_tool`) when the checkpoint has native DNA.

### Claiming a cell

Copy [`experiments/_template`](experiments/_template) to `experiments/<kind>/<approach>/`.

- `approach.yaml`: owner, kind (`harness_opt`, `model_opt`, or `joint`), parent model, parent arm,
  and one line each for the task-accuracy claim and the DNA-use claim.
- `job.yaml`: an ordinary job spec. Use `purpose: dev` on the first run and `purpose: eval` on the
  locked run. Set `name` to `<kind>__<approach>__<arm>__<model>` so it cannot collide with
  `exp_a_shared-*` or `sft_eval_*`.

The first render is a smoke of the new cell on `kegg-dx-dev` or `harness-pilot` only. Reported
numbers use `purpose: eval` datasets. Prompts stay at `PROMPT_VERSION = v3` in
[`src/dnaharness/runtime/prompts.py`](src/dnaharness/runtime/prompts.py). A dev-only edit bumps
that version.

Submit the copied spec. Leave the baseline YAMLs in `configs/jobs/` unchanged.

## Setup, through audit and smoke

Storage root is [`configs/paths.yaml`](configs/paths.yaml): `/orcd/data/lhtsai/001/mabdel03/6S978`.
Override with `DNAH_ROOT` (`scripts/setup.sh` reads the same file). The paths below are that
default. One-time setup needs a node with internet.

```bash
bash scripts/setup.sh envs secrets ckpts data   # one-time (internet)
bash scripts/setup.sh containers                # apptainer --fakeroot builds (needs internet)
sbatch slurm/evo2_cache.sbatch                  # one-off Evo2 embeddings for every DNA window

M=/orcd/data/lhtsai/001/mabdel03/6S978/envs/main/bin/python
for s in audit smoke; do $M -m dnaharness.jobs render configs/jobs/$s.yaml; done
J=/orcd/data/lhtsai/001/mabdel03/6S978/jobs
sbatch slurm/serve_and_run.sbatch $J/audit $J/smoke              # audit is a HARD gate
```

`audit` is the isolation gate. `smoke` is plumbing on five `kegg-dx-smoke` rows.

The train pool is separate from `scripts/setup.sh data`. Run it once; it writes only
`<datasets>/kegg-dx-train_pool/`:

```bash
$M -m dnaharness.adapters.kegg_dx.adapter --role train_pool --interfaces raw native --conditions wt no_dna scramble
```

## Submit a cell

```bash
$M -m dnaharness.jobs render experiments/<kind>/<approach>/job.yaml
sbatch slurm/serve_and_run.sbatch $J/<name>
```

`<name>` is the spec’s `name`. `--chunk i/n` is the existing chunking (every n-th task, starting
at i) when a cell has to fit the 6 h limit; the directory is then `$J/<name>-c<i>of<n>`. Each
allocation serves the models listed in the first job’s `serve.json`. Cells on different models
need separate allocations.

### Partitions

The scripts default to `mit_normal_gpu`, which caps each user at 32 CPUs and 2 GPUs. Override at
submission. Always name a full GPU type: a bare `gpu:1` can land on a MIG slice, which vLLM rejects.

```bash
sbatch -p ou_bcs_normal --gres=gpu:a100:1 -c 24 --mem=64G slurm/serve_and_run.sbatch $J/<job>   # sweeps (16 GPUs/user, 1 day)
sbatch -p ou_bcs_high   --gres=gpu:h100:1 slurm/serve_and_run.sbatch $J/<job>                   # quick, high priority (1 GPU, 4 h)
sbatch -p pi_manoli     --gres=gpu:a100:1 ...      # or pi_tpoggio: 7-day limit, 8×A100 nodes
sbatch -p ou_bcs_low    --gres=gpu:a100:1 ...      # preemptable overflow (requeue)
```

Every model in this study runs on A100, H100/H200, and L40S. H100 needs `VLLM_USE_DEEP_GEMM=0`, which the launcher sets.

### What each allocation does

`slurm/serve_and_run.sbatch`:

1. **Claims a port slot.** It claims a node-local slot k and shifts every port by 100·k: the gateway on 8000+100k, vLLM on 8001+100k and up. Several allocations can share a node, and sandboxes use the host network. Agents, including mini-swe, find the gateway through `DNAH_GATEWAY_URL` at run time, not through the URL baked into the config.
2. **Runs an isolation audit on its own node**, against its own ports. This is a hard gate.
3. **Runs or resumes each job.** If a job directory already has a `lock.json`, the job is resumed with `harbor jobs resume`, and only unfinished trials run again. Resubmitting a chunk that was killed, preempted, or timed out therefore picks up where it stopped.
4. **Copies the gateway call log** to `<job>/gateway_logs/<allocation>/`.

The Harbor 0.27 sandbox-leak and resume incidents, including what was re-run, are in
[docs/STATUS.md](docs/STATUS.md). `bash scripts/sweep_status.sh` lists every submitted allocation
(from `logs/sweep_jobs.tsv`) with its finished/total trials.

### After a job finishes

```bash
$M -m dnaharness.analysis.collect $J/<job> ... --out results/trials.parquet
$M -m dnaharness.analysis.leak_scan $J/<job> ...
$M -m dnaharness.analysis.report results/trials.parquet --out results/report
```

## Task examples

Every trial is one Harbor task directory. The agent sees only `instruction.md` and
`/task/inputs.json`. Gold labels live under `tests/` and are mounted for the verifier only after the
agent finishes. The required solution format is always the same: write the final answer alone to
`/logs/agent/answer.txt` (for example `echo "creutzfeldt-jakob disease" > /logs/agent/answer.txt`),
then exit.

### KEGG variant to disease (primary eval)

This is the BioMirage genome-dependent cohort (`kegg-dx`, 165 rows). The question text names a
chromosome, pathway edges, and genes, and asks which disease the allele contributes to. DNA is a
2,048-base window centered on the first differing base between reference and variant (bio-mirage
windowing, then clipped to Evo2's length). The correct output is exactly one disease name from the
closed 37-label vocabulary, copied verbatim (no explanation).

Illustrative cohort episode (raw interface, intact DNA). Sequences are shortened here; on disk each
is 2,048 bases:

```json
{
  "question": "Chromosome Number: 20\nNetwork Definition of the pathway: PRNP* -> NOTCH1\nGenes in the pathway: PRNP; prion protein | NOTCH1; notch receptor 1\n\nGiven this context, what is the biological effect of this PRNP allele, specifically what disease does this contribute to?",
  "reference_sequence": "...CAGCCCCATGG...",
  "variant_sequence":   "...CAGCCTCATGG..."
}
```

At the first differing site in this example the reference has `C` and the variant has `T` (context
`CAGCC[C]CATGG` vs `CAGCC[T]CATGG`). The gold label written by the oracle solution (and checked by
the verifier) is:

```text
creutzfeldt-jakob disease
```

Same episode under other interfaces and conditions (question text unchanged unless noted):

| Variant | What `/task/inputs.json` contains | Correct output |
|---|---|---|
| `raw` + `wt` | `question`, `reference_sequence`, `variant_sequence` | same disease name |
| `native` + `wt` | `question`, `dna_handles` (opaque tokens such as `<DNA:6a1baf2eff89b54b>`); no raw bases | same disease name |
| `no_dna` | `question` only | same disease name (text-only baseline) |
| `scramble` | same keys as `wt`, but both windows are shuffled with the same base composition | same disease name (content control) |

If intact DNA is used, accuracy should differ from `no_dna` and from `scramble` in a coherent way.
If the model only needs the text format of the prompt, those conditions look alike. That contrast is
how we measure utilization of the biological input ([docs/METRICS.md](docs/METRICS.md)).

### Harness competence (pilot)

Before trusting a harness on KEGG, we run non-biological, auto-checked tasks
(`harness-pilot`). These use synthetic DNA strings and exact or numeric gold answers. Example
`diff_substring` episode:

```json
{
  "question": "The variant sequence differs from the reference at one place. Answer in the form '<offset>:<reference segment>><variant segment>' ...",
  "reference_sequence": "<2048-ish synthetic bases>",
  "variant_sequence":   "<same string with an insertion of TAAG at index 223>"
}
```

Gold answer for that generated episode (exact string match):

```text
223:->TAAG
```

Meaning: at 0-based offset 223 the reference has an empty segment and the variant inserts `TAAG`.

Other pilot types include inspecting a sequence property (integer answer), combining observations
into a small JSON object, making one `llm_query` child call (RLM only), and stopping within a turn
budget. Stage-3 SFT trajectories are generated on a disjoint `harness-train` distribution, never on
`harness-pilot` or `kegg-dx`.

## Architecture

```
GPU node:  vLLM (one per checkpoint, --api-key)  ->  LM gateway :8000  ->  every LM call
                                                     * one decoding policy, per-trial budgets
                                                     * /child route returns 403 for coding arms
                                                     * DNA handle -> Evo2 -> projection -> prompt_embeds
                                                     * JSONL accounting (text tokens, DNA positions,
                                                       thinking tokens, latency)
           harbor run  ->  N x Apptainer sandboxes (bio-sandbox.sif, CPU)
                            /task/inputs.json only; labels arrive in /tests only after the agent finishes
```

The gateway resolves DNA handles in only two places: the root model's first message and child-call
prompts. Anywhere else (for example, REPL output that echoes a handle), it replaces the handle with
`[DNA handle]`, so the DNA is never re-injected.

### Arms

Defined in [`configs/harness/arms.yaml`](configs/harness/arms.yaml):

- **`rlm`**: the official `rlms==0.1.3` loop. Depth-1 `llm_query` calls go to the *same* checkpoint.
- **`coding_repl`**: the *same* loop, REPL, tools, budgets, and prompt, with sub-calls removed. This is the controlled "generic coding harness".
- **`mini_swe`**: Harbor's built-in off-the-shelf mini-swe-agent, text-based mode, as a check of external validity.
- **`direct`**: a single call, with no loop. For native inputs this is BioReason's exact training format.

The remaining arms are controls that separate recursion from simply getting one well-formatted call:

- **`direct_harness`**: the harness prompt plus DNA in one call.
- **`coding_repl_tool`**: a coding arm with one fixed-format `bioreason_answer()` tool.
- **`vote`**: majority vote, matched to the RLM arm's compute.

### Interfaces

What `/task/inputs.json` contains:

- `raw`: the question plus the reference and variant 2,048-bp windows.
- `native`: the question plus opaque DNA *handles*, with no raw sequence. The gateway turns each handle into projected Evo2 `prompt_embeds`.
- `native_raw`: both.

### Conditions

- `wt`, `no_dna` (text only), `scramble` (both sequences shuffled, same base composition).
- Extended diagnostics: `revert_ref`, `swap_pair` (the donor's complete reference/variant pair, not a variant-only swap), `text_conflict` (DNA from a same-gene donor with a different label).

All conditions use bio-mirage's own window and perturbation code (`third_party/bio-mirage`).

### Decoding

Qwen3 thinking-mode sampling (T=0.6, top_p=0.95, top_k=20), 3 seeds per episode. The gateway enforces this for every arm. Greedy decoding is used only for the reproduction track.

### Frozen baseline

Specs in `configs/jobs/` are the frozen grid. New work does not edit them. Numbers are in
[docs/STATUS.md](docs/STATUS.md).

| Stage | Question | Cells |
|---|---|---|
| **Pilot** | Can each checkpoint operate each harness at all? | {Qwen, BioReason} × {coding_repl, rlm, mini_swe} on non-biological, auto-checked tasks |
| **Exp A** (shared interface) | Does biological post-training or recursion help when everyone gets the same raw DNA strings and deterministic tools? | {Qwen, BioReason} × {direct, coding_repl, rlm, mini_swe} × {wt, no_dna, scramble} |
| **Exp B** (native Evo2) | Does RLM change how BioReason uses its *native* Evo2 representations? | BioReason × {direct, direct_harness, vote@k, coding_repl, coding_repl+bioreason_tool, rlm}; plus `native_raw` (embeddings + raw strings) × {coding_repl, rlm} |
| **Stage 3** (harness SFT) | Does training on harness trajectories help, and is it RLM-specific? | {instructions, k=3 demos, coding-trajectory SFT, RLM-trajectory SFT, task-answer SFT} × {coding_repl, rlm} |
| Reproduction | Is the native path wired correctly? | greedy native BioReason; target ≈0.54 (wt) / ≈0.62 (scramble) |

## Layout

```
configs/      paths.yaml (storage root) · models.yaml (pinned revisions, decoding) · data_access.yaml
              harness/arms.yaml · jobs/*.yaml (frozen baselines)
experiments/  README.md · _template/ · baselines/ · harness_opt/ · model_opt/ · joint/
docs/         STATUS.md (lab notebook) · METRICS.md
containers/   bio-sandbox.def (Harbor sandbox) · evo2.def (one-off embedding cache)
slurm/        serve_and_run.sbatch · evo2_cache.sbatch · train_lora.sbatch
scripts/      setup.sh (envs, secrets, checkpoints, containers, data)
src/dnaharness/
  data/       kegg.py · cohort.py (165-row manifest) · conditions.py · labels.py (bio-mirage metric)
  adapters/   kegg_dx/ (Harbor adapter) · harness_tasks.py (pilot + SFT tasks) · audit.py (isolation gate)
  serving/    gateway.py · native.py (rendering + prompt_embeds) · evo2_cache.py · prepare_ckpt.py
  harbor_ext/ agents.py (RlmAgent, DirectAgent, VoteAgent)
  runtime/    in-sandbox: run_rlm.py · seqtools.py · prompts.py · scoring.py
  analysis/   collect.py (host re-scoring) · metrics.py · report.py · leak_scan.py
  sft/        export.py · demos.py · train_lora.py · merge.py
  jobs.py     experiment spec → Harbor job config + serving spec
third_party/  bio-mirage, bio-posttrain (git submodules, pinned)
```

Large artifacts live under the storage root. They are never stored in the repo.

## Metrics

Task accuracy is overall accuracy, pair accuracy, and minority-label accuracy. DNA use is the
text-only delta (intact minus no DNA), the flip rate of each perturbation against intact DNA, and
DNA positions per trial. The map onto the code is [docs/METRICS.md](docs/METRICS.md).

## Stage 3 (existing pipeline)

Run these only after the pilot shows the 1.7B models can follow the REPL protocol at all.

1. `bash scripts/setup.sh ckpts_teacher`. This downloads the Qwen3-32B teacher, about 65 GB.
2. Render and run `sft_teacher_coding` and `sft_teacher_rlm` (2 GPUs; the root is the teacher, the child is the student).
3. `python -m dnaharness.sft.export --coding … --rlm … --student bioreason-1.7b --out <root>/sft_data/bioreason-1.7b`. This keeps only tasks *both* teachers solved, matches the two sets on completion tokens, and builds the task-answer control.
4. `python -m dnaharness.sft.demos …` for the in-context-demos condition.
5. `sbatch slurm/train_lora.sbatch bioreason-1.7b {coding|rlm|answer} <port>`. This trains a LoRA on attention and MLP only, merges it, checks that the embeddings are unchanged, and registers the result for serving.
6. Render and run `sft_eval`: every condition × both harnesses, plus native direct and a root-only diagnostic.

Counts from the 2026-11-22 run are in [docs/STATUS.md](docs/STATUS.md). Known limitations (sandbox
network, the BioReason prompt shift, the deferred generalization track, Evo2 with Transformer
Engine blocked) are recorded there too.
