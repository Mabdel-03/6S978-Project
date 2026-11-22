# Metrics

Short map onto [`src/dnaharness/analysis/metrics.py`](../src/dnaharness/analysis/metrics.py).
The definitions below are the module docstring, fixed before any sweep. `summary_table` is the
function that puts them on one row per configuration and condition.

## A. Task accuracy

| Reported name | Function | Definition in the docstring |
|---|---|---|
| Accuracy | `accuracy` | Mean over episodes of the seed-averaged correctness. |
| Pair accuracy | `pair_accuracy` | Within each text group, all unordered pairs of rows with different labels and different `pair_id`. A pair scores 1 for a seed if both rows are correct for that seed. Average over seeds, then within the group, then across groups (each group weighs 1). |
| Minority accuracy | `minority_accuracy` | Accuracy on rows whose label is not their text group's majority label (`is_minority_label`). |

`pair_accuracy(d, pair_weighted=True)` is the `pair_accuracy_pw` column: every pair weighs the
same. The default pair accuracy is group-weighted. `group_accuracy` is the same correctness with
each of the 14 text groups weighing 1. `bootstrap` and `paired_bootstrap` return a point and a
percentile interval for these four functions.

## B. Whether DNA changed the answer

| Reported name | Where | Definition in the docstring |
|---|---|---|
| Text-only delta | `summary_table` column `text_only_delta` | `accuracy(wt) - accuracy(no_dna)`, same arm, model, and interface. Attached on the `wt` row when that config has a `no_dna` condition. |
| Flip rate | `flip_rate` | `P(predicted label under c != predicted label under wt)`, matched on row and seed. `summary_table` stores it as `flip_rate_vs_wt` for every condition other than `wt`. Scramble is one of those conditions. |
| DNA positions | `health` | Cost. Mean DNA positions per trial: the `dna_positions` column, emitted as `mean_dna_positions`, next to mean prompt, completion, and reasoning tokens. |

`paired_bootstrap` is how two configurations share one resample: text groups
are drawn with replacement, then `pair_id`s inside each sampled group, and every seed of a sampled
pair stays together. Paired differences use that same draw for both sides. There are 14 clusters,
so the intervals are wide.

`summary_table(..., by_split=True)` repeats each row by `orig_split`. That column is BioReason's
historical Hugging Face split, kept for reporting. Training permission is the access class in
[`configs/data_access.yaml`](../configs/data_access.yaml).
