# Status log moved from the README.

## Harbor incident notes

**Two Harbor 0.27 workarounds** (found mid-sweep on 2026-11-21):
- **Sandbox leak.** Harbor's Singularity `stop()` leaves every finished sandbox's process tree running (`starter`, the FUSE mounts, `fakeroot`, Python). Within about 3.5 h this drove two shared 8-GPU nodes to a load of about 8,600 with 2 GB free. New sandboxes then failed to start: about 450 trials failed with `EnvironmentStartTimeoutError`.
  - `harbor_ext/sandbox_reaper.py` is loaded into every `harbor` process through `slurm/pyhooks/sitecustomize.py`. It kills each sandbox's tree as soon as its server exits.
  - Each allocation also reaps its own orphans on that node at start and end.
- **Resume refused.** Harbor scrubs the agent's env values from saved trial files, so a dummy `api_key` copied into mini-swe's config was stored as `[REDACTED]`, and `harbor jobs resume` then refused the job.
  - New renders leave the key out of the config.
  - `harbor_ext/resume_repair.py` restores the field in older jobs before each resume.
- **What was re-run.** Infrastructure failures (`EnvironmentStartTimeoutError`, `RuntimeError`, `AddTestsDirError`) are re-run, via `DNAH_RESUME_FILTER`.
- **What was kept.** Agent timeouts and verifier timeouts are kept. Agent timeouts are model outcomes, and answers are rescored on the host anyway.

**The degraded window.** Before the fix, nodes slowed sharply from 23:30 UTC until the cancellation at about 00:21 UTC:

| | Before 23:30 UTC | 23:30 to 00:21 UTC |
|---|---|---|
| Trials finished per half-hour | ~3,000 | ~450 |
| Agent-timeout rate | 0.1% | 4–13% |

Every trial that finished in that window was discarded and re-run, whatever its outcome (`harbor_ext/archive_window.py`; selected by finish time only, so failed trials don't get a second chance). The originals are kept under `jobs/_archive/degraded-20261121/`, with a manifest.

`bash scripts/sweep_status.sh` lists every submitted allocation (from `logs/sweep_jobs.tsv`) with its finished/total trials.

## Verification status

Done (`pytest`: 86 + 22 tests across the main and serve envs):
- **Cohort**: counts reproduce the paper (165/63/14/12/6, split 126/19/20, 98 PRNP, 37 labels).
- **Perturbations**: byte-identical to bio-mirage.
- **Scoring**: the in-sandbox scorer agrees with bio-mirage's metric.
- **Generated tasks**: inputs are whitelisted, no label appears in anything the agent can see, and every task and job config validates against Harbor's schemas.
- **Gateway**: budget limits return 429, coding-arm child calls return 403, sampling is enforced, seeds are unique, and the mini-swe session route works.
- **Driver, end to end** (real `rlms` loop and real gateway): `llm_query` works in the rlm arm and is undefined in the coding arm, with zero child calls.
- **Native `prompt_embeds`**, on the real checkpoint: bit-identical to the BioReason/bio-mirage construction, and changing the variant changes only the variant span.
- **SFT template**: BioReason's template renders text-only conversations identically to Qwen3's official one, which is what SFT training uses.
- **SFT CPU smoke test**: LoRA training and merging, with the embeddings-unchanged check.

Done on the cluster (2026-11-21):
- **Containers**: `bio-sandbox.sif` and `evo2.sif` both build, and the sandbox invariants hold.
- **Plumbing**: real Apptainer sandboxes on CPU give oracle 5/5 correct and nop 0/5 valid.
- **Isolation audit**: passes on a GPU node, all 10 required checks:
  - no `/orcd`, `/nfs`, host home, HF cache or host `/tmp`, and no `/tests` during the agent phase
  - proxied HTTP is blocked
  - vLLM rejects requests without the key, and so does the gateway admin API
  - the gateway is reachable

  Raw-socket egress also failed from inside the sandbox.
- **Evo2 cache**: 679 windows in 53 GPU-seconds on an L40S, with Transformer Engine blocked.
- **Smoke, all 6 arms**: runs end to end through the gateway, and the leak scan is clean.
  - **mini-swe-agent**: sends `X-Session-ID`, the gateway enforces the sampling policy and gives each call a unique seed, and host-side collection joins the session usage.

- **Paper reproduction** (greedy native BioReason, A100, 330 trials, Evo2 cache from an L40S): accuracy matches the paper,
  and DNA input still doesn't help.
  - Intact DNA: **0.539** [0.420, 0.659]; paper ≈0.54.
  - Scrambled DNA: **0.606** [0.494, 0.713]; paper ≈0.62.
  - Scrambled minus intact: +0.067 [−0.050, +0.182].
  - Pair accuracy: 0.14 intact, 0.18 scrambled.
  - Flip rate under scrambling: 0.42.
  - Answers were valid 98–99% of the time, and every call carried 4,096 DNA positions.
  - The intervals are the 14-cluster bootstrap.
- **Bootstrap speed**: `metrics.bootstrap` uses an exact closed form, about 100× faster. It is checked draw for draw against
  explicit resampling (`tests/test_bootstrap_fast.py`).

- **Evo2 linear-probe check** (`results/evo2_probe/summary.md`), within the cohort using bio-mirage's protocol:
  - A probe on the mean-pooled Evo2 embedding scores 0.73 [0.65, 0.81]. On minority-label rows it scores 0.53, against 0.30 for BioReason, which scores 0.54 overall.
  - The text-majority baseline scores 0.69, and the label-permutation null 0.58.
  - The projection preserves the signal (0.74), so the bottleneck is how BioReason's language model uses the embedding.
  - A probe trained outside the cohort is impossible here: only 2–10% of cohort rows have a label that appears anywhere else in KEGG.

Fixes found by running it (all in the code now):
- **vLLM sampler**: vLLM's FlashInfer sampler needs `nvcc`, so the launcher sets `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Evo2 image**:
  - The NGC image's CUDA forward-compat `libcuda` (575) shadows the newer host driver (590) and causes CUDA error 803, so it is removed.
  - Triton then needs `TRITON_LIBCUDA_PATH`.
  - vortex can't be imported at build time, because its Triton autotuners need a GPU.
- **mini-swe endpoint**: with `litellm_textbased`, Harbor refuses an agent-level `base_url`. The endpoint goes in `model.model_kwargs.api_base`.
- **Prompt v2**: prompt v1 showed REPL code without fences, and Qwen3-1.7B copied it verbatim. In smoke-v1, 0 of 10 rlm and coding trials executed a single fenced `repl` block, and every answer came from rlms' out-of-turns fallback call. v2 uses fenced examples. The driver now reports that fallback as `max_iterations` with `fallback_answer=true`, where it used to report `ok`.
- **Direct-arm token cap**: in smoke-v1, 4 of 5 direct raw-DNA answers were cut off mid-thinking at 4,096 tokens. The per-call cap is now 16,384, and the gateway clamps each call to the room left in the context window. The per-trial budget is unchanged.

- **Smoke v2** (prompt v2, 16,384-token cap; 5 Qwen tasks per arm):

  | Arm | Correct | Fallback rate (v1 → v2) |
  |---|---|---|
  | rlm | 0.8 | 1.0 → 0 |
  | coding_repl | 0.6 | 1.0 → 0.2 |
  | direct | 0.6 | — (truncation 0.8 → 0) |
  | mini_swe | 0.0 | — |

- **Harness-competence pilot** (864 trials; `results/pilot_table.csv`). Scored only on task types that apply to each arm:

  | Checkpoint | Arm | Correct | In-protocol | Valid |
  |---|---|---|---|---|
  | Qwen3-1.7B | coding_repl | 0.33 | 0.33 | 0.88 |
  | Qwen3-1.7B | rlm | 0.46 | 0.28 | 0.87 |
  | Qwen3-1.7B | mini_swe | 0.04 | 0.01 | 0.59 |
  | BioReason | coding_repl | 0.07 | 0.06 | 0.16 |
  | BioReason | rlm | 0.03 | 0.00 | 0.19 |
  | BioReason | mini_swe | 0.00 | 0.00 | 0.05 |

  - **Qwen3-1.7B** can operate both REPL harnesses, and the rlm arm makes sub-calls when a task needs one (1.25 per trial).
  - **BioReason fails the competence gate.** It ignores the REPL format and answers "Answer: <variant statement>" every turn: 243 of 254 turns had no fence, and 93% of rlm answers are rlms fallbacks. 41–43% of its turns run away in thinking, and calls past the 900 s client timeout cause 39–52% errors.
  - **mini-swe-agent** never submits within 12 steps, for either checkpoint.
- **Dev set** (20 rows held out from the cohort; 1 seed):
  - BioReason direct: 0.70 native, 0.80 raw.
  - BioReason REPL arms: 0.30–0.50, all from fallback answers.
  - Qwen: only 20–25% of its free-text answers map to one of the 37 labels, which caps its accuracy at 0.10–0.40.

**Experiment A: done** (10,395 trials; report in `results/report_exp_a/`). Raw DNA strings with deterministic tools, prompt v3, sampled decoding, 3 seeds × 165 rows. Accuracy on intact DNA:

| | Direct | Coding REPL | RLM | mini-swe |
|---|---|---|---|---|
| Qwen | 0.42 | 0.28 | 0.27 | 0.03 |
| BioReason | 0.41 | 0.10 | 0.09 | — |

- **No recursion effect:** RLM minus coding REPL is −0.01 [−0.09, 0.04] for Qwen and −0.01 [−0.06, 0.04] for BioReason.
- **The harnesses cost accuracy** compared with one direct call: RLM minus direct is −0.16 for Qwen and −0.32 for BioReason.
- **No configuration shows significant DNA use.** Raw DNA *hurts* Qwen: intact minus text-only is −0.12 [−0.20, −0.03] for direct and −0.07 for the coding REPL. BioReason direct shows +0.05 [−0.01, +0.11].
- **Pair accuracy is at most 0.13 everywhere.**

**Experiment B: done** (11,880 trials; report in `results/report_exp_b/`). BioReason reading native Evo2 embeddings, sampled decoding. Accuracy on intact DNA:

| Arm | Accuracy | 95% CI |
|---|---|---|
| direct (training format) | 0.55 | [0.45, 0.65] |
| vote@21 | 0.64 | [0.54, 0.74] |
| direct_harness | 0.31 | |
| coding_repl | 0.14 | |
| coding_repl + BioReason tool | 0.11 | |
| rlm | 0.13 | |

- **No recursion effect:** RLM minus coding REPL is −0.01 [−0.06, 0.03], and RLM minus coding+tool is +0.02 [−0.02, 0.07].
- **The harness prompt alone costs accuracy:** direct_harness minus direct is −0.24 [−0.36, −0.14].
- **DNA presence helps, DNA content doesn't:**
  - Embeddings versus no DNA: +0.17 [+0.07, +0.29] for direct and +0.30 for vote@21.
  - Intact versus scrambled is about 0 in every arm.
  - This matches the paper: an in-format prompt helps, but the sequence content is not used.
- **Native + raw:** only RLM shows intact minus scrambled +0.04 [+0.00, +0.07]. That is 1 of 16 tests, so it is consistent with chance.
- **Fix:** the coding_repl_tool × no_dna cell was invalid, because the driver refused the tool without DNA. It was fixed 2026-11-22 and is being re-run. The 495 original trials are in `jobs/_archive/tool-nodna-20261122/`.

**Stage 3** (2026-11-22):
- **Teachers:** Qwen3-32B, about 16.8k trials. The coding teacher solves 86% of training tasks; the RLM teacher, with BioReason as its sub-model, solves 75%.
- **SFT data:** each harness uses its own successes, matched on completion tokens (user decision):

  | Set | Samples | Completion tokens | Notes |
  |---|---|---|---|
  | Coding | 4,575 | 4.22M | |
  | RLM | 4,835 | 4.14M | 392 sub-call samples |
  | Task-answer | 1,950 | | |
- **Training:** three LoRAs, trained and merged; final losses 0.44 (coding), 0.43 (RLM), 0.26 (answer).
- **Evaluation:** trimmed, 23,859 trials (user decision). Specs: `configs/jobs/sft_eval_{pilot,kegg,direct}.yaml`. The untrained baselines are reused from Experiments A and B.

- **SFT damaged BioReason's biological skill.** Native-direct check, canonical format, 3 seeds × 165 rows (`results/sft_eval_direct_trials.parquet`):

  | Model | Intact DNA | No DNA | Scrambled | Intact − base (paired 95% CI) |
  |---|---|---|---|---|
  | BioReason base (Exp B) | 0.55 | 0.38 | 0.57 | |
  | Coding SFT | 0.33 | 0.34 | 0.36 | −0.22 [−0.33, −0.10] |
  | RLM SFT | 0.32 | 0.33 | 0.32 | −0.23 [−0.34, −0.12] |
  | Answer SFT | 0.30 | 0.46 | 0.29 | −0.25 [−0.38, −0.12] |

  - **Format survives:** 97–100% of answers stay valid.
  - **The DNA-presence benefit is erased:** base gains +0.17 from intact DNA over no DNA; after SFT, intact ≈ no DNA.
  - **Likely cause:** 300 steps (about 4 epochs) of LoRA (r=64, all attention and MLP layers) on purely non-biological harness tasks, with no biological replay.

## Running now (2026-11-22)

- **Stage 3 evaluation:** `sft_eval_pilot` (1 allocation) and `sft_eval_kegg` (10 chunks) are running. `sft_eval_direct` is done.
  - The direct chunks first failed the per-node audit: it probed only the standard vLLM ports, and the SFT models are served on registered ports ≥ 8010. The audit now also probes registered model ports (`adapters/audit.py`).

## Known limitations

- **Sandbox network.** Sandboxes share the host network, because Harbor's singularity backend requires it.
  - A dead proxy blocks ordinary HTTP clients.
  - Code that deliberately opens raw sockets can still reach the internet. `leak_scan.py` flags this, and the audit reports it.
  - Other sandboxes' exec servers are also reachable over that network. This is mitigated by host-side re-scoring and the vLLM API key, but not eliminated.
- **Harness prompts are out of distribution for BioReason**, which was trained only on `[ref][var][question]`. The `direct_harness` control measures how much this matters.
- **No generalization track yet.** It is deferred until the checkpoint's training provenance has been audited. The manifest already reserves the fields for grouping by biological identity.
- **Evo2 settings.** Evo2 runs with Transformer Engine blocked, because that is how BioReason's projection was trained. Do not "fix" this by enabling FP8.
