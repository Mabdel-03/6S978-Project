# Frozen baselines

The baseline specs stay in `configs/jobs/`. This directory points at them. Compare a new cell to its parent model and parent arm on the same locked eval sets.

| Spec | File | What it fixes |
|---|---|---|
| `exp_a_shared` | [configs/jobs/exp_a_shared.yaml](../../configs/jobs/exp_a_shared.yaml) | Shared raw-DNA interface. `qwen3-1.7b` and `bioreason-1.7b` × `direct`, `coding_repl`, `rlm` (plus `mini_swe` for Qwen) on `kegg-dx` `raw-wt`, `raw-no_dna`, `raw-scramble`. |
| `exp_b_native` | [configs/jobs/exp_b_native.yaml](../../configs/jobs/exp_b_native.yaml) | Native BioReason embeddings. `bioreason-1.7b` × `direct`, `direct_harness`, `vote`, `coding_repl`, `coding_repl_tool`, `rlm` on `native-wt`, `native-no_dna`, `native-scramble`. |
| `exp_b_native_raw` | [configs/jobs/exp_b_native_raw.yaml](../../configs/jobs/exp_b_native_raw.yaml) | Raw DNA and native embeddings together. `bioreason-1.7b` × `coding_repl`, `rlm` on `native_raw-wt`, `native_raw-no_dna`, `native_raw-scramble`. |
| `sft_eval_pilot` | [configs/jobs/sft_eval_pilot.yaml](../../configs/jobs/sft_eval_pilot.yaml) | Stage-3 harness competence on `harness-pilot/pilot`. |
| `sft_eval_kegg` | [configs/jobs/sft_eval_kegg.yaml](../../configs/jobs/sft_eval_kegg.yaml) | Stage-3 on the KEGG cohort, raw and native, intact versus text-only DNA. |
| `sft_eval_direct` | [configs/jobs/sft_eval_direct.yaml](../../configs/jobs/sft_eval_direct.yaml) | Stage-3 native `direct` calls for each SFT checkpoint. |
| `sft_teacher_coding` | [configs/jobs/sft_teacher_coding.yaml](../../configs/jobs/sft_teacher_coding.yaml) | Teacher trajectories, coding harness. Root `qwen3-32b` on `harness-train/train`. |
| `sft_teacher_rlm` | [configs/jobs/sft_teacher_rlm.yaml](../../configs/jobs/sft_teacher_rlm.yaml) | Teacher trajectories, RLM harness. Root `qwen3-32b`, child `bioreason-1.7b`, on `harness-train/train`. |
