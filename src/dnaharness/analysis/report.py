"""Results report: main input-use table, split-stratified table, pilot competence, key paired contrasts.

    python -m dnaharness.analysis.report results/trials.parquet --out results/report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dnaharness.analysis import metrics as M

# Pre-registered paired contrasts (A - B), evaluated on identical episodes.
CONTRASTS = [
    # (label, filter A, filter B)
    ("RLM - coding (Qwen, raw)", dict(arm="rlm", model="qwen3-1.7b", interface="raw"),
     dict(arm="coding_repl", model="qwen3-1.7b", interface="raw")),
    ("RLM - coding (BioReason, raw)", dict(arm="rlm", model="bioreason-1.7b", interface="raw"),
     dict(arm="coding_repl", model="bioreason-1.7b", interface="raw")),
    ("BioReason - Qwen (rlm, raw)", dict(arm="rlm", model="bioreason-1.7b", interface="raw"),
     dict(arm="rlm", model="qwen3-1.7b", interface="raw")),
    ("BioReason - Qwen (coding, raw)", dict(arm="coding_repl", model="bioreason-1.7b", interface="raw"),
     dict(arm="coding_repl", model="qwen3-1.7b", interface="raw")),
    ("RLM - coding (BioReason, native)", dict(arm="rlm", model="bioreason-1.7b", interface="native"),
     dict(arm="coding_repl", model="bioreason-1.7b", interface="native")),
    ("RLM - coding+tool (BioReason, native)", dict(arm="rlm", model="bioreason-1.7b", interface="native"),
     dict(arm="coding_repl_tool", model="bioreason-1.7b", interface="native")),
    ("RLM - vote@k (BioReason, native)", dict(arm="rlm", model="bioreason-1.7b", interface="native"),
     dict(arm="vote@21", model="bioreason-1.7b", interface="native")),
    ("direct_harness - direct (BioReason, native)", dict(arm="direct_harness", model="bioreason-1.7b", interface="native"),
     dict(arm="direct", model="bioreason-1.7b", interface="native")),
    ("RLM - direct (Qwen, raw)", dict(arm="rlm", model="qwen3-1.7b", interface="raw"),
     dict(arm="direct", model="qwen3-1.7b", interface="raw")),
    ("RLM - direct (BioReason, raw)", dict(arm="rlm", model="bioreason-1.7b", interface="raw"),
     dict(arm="direct", model="bioreason-1.7b", interface="raw")),
    ("BioReason - Qwen (direct, raw)", dict(arm="direct", model="bioreason-1.7b", interface="raw"),
     dict(arm="direct", model="qwen3-1.7b", interface="raw")),
    ("mini_swe - coding (Qwen, raw)", dict(arm="mini_swe", model="qwen3-1.7b", interface="raw"),
     dict(arm="coding_repl", model="qwen3-1.7b", interface="raw")),
]


def dna_effects(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Within each config: accuracy(wt) - accuracy(no_dna) [text-only delta] and wt - scramble, with the
    paired cluster bootstrap (same rows, same resamples)."""
    out = []
    for key, d in df.groupby(["experiment", "arm", "model", "interface"], dropna=False):
        wt = d[d.condition == "wt"]
        for other, label in (("no_dna", "wt - no_dna (text-only delta)"), ("scramble", "wt - scramble")):
            o = d[d.condition == other]
            common = set(wt.row_id) & set(o.row_id)
            if not common:
                continue
            a, b = wt[wt.row_id.isin(common)], o[o.row_id.isin(common)]
            p, lo, hi = M.paired_bootstrap(a, b, M.accuracy, n=n_boot)
            out.append(dict(zip(["experiment", "arm", "model", "interface"], key), effect=label,
                            diff=p, lo=lo, hi=hi, n_rows=len(common)))
    return pd.DataFrame(out)


def _sel(df, cond, **kv):
    m = df["condition"] == cond
    for k, v in kv.items():
        m &= df[k] == v
    return df[m]


def contrasts(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    out = []
    for label, fa, fb in CONTRASTS:
        for cond in ("wt",):
            a, b = _sel(df, cond, **fa), _sel(df, cond, **fb)
            if not len(a) or not len(b):
                continue
            common = set(a.row_id) & set(b.row_id)
            a, b = a[a.row_id.isin(common)], b[b.row_id.isin(common)]
            for name, fn in (("accuracy", M.accuracy), ("pair_accuracy", M.pair_accuracy)):
                p, lo, hi = M.paired_bootstrap(a, b, fn, n=n_boot)
                out.append({"contrast": label, "condition": cond, "metric": name, "diff": p, "lo": lo, "hi": hi,
                            "n_rows": len(common)})
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--out", default="results/report")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    df = M.with_experiment(pd.concat([pd.read_parquet(t) for t in a.trials], ignore_index=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kegg = df[df.row_id.notna()] if "row_id" in df else df.iloc[0:0]
    parts = ["# Results\n"]
    if len(kegg):
        main_t = M.summary_table(kegg, n_boot=a.n_boot)
        main_t.to_csv(out / "summary.csv", index=False)
        cols = [c for c in ["experiment", "arm", "model", "interface", "condition", "accuracy", "accuracy_lo", "accuracy_hi",
                            "group_accuracy", "pair_accuracy", "pair_accuracy_lo", "pair_accuracy_hi",
                            "minority_accuracy", "text_only_delta", "flip_rate_vs_wt", "valid_rate", "unmapped_rate",
                            "budget_exhausted_rate", "attempted_subcall_rate", "mean_completion_tokens",
                            "mean_dna_positions", "n_trials"] if c in main_t]
        parts += ["## Main table (episode-level CIs: hierarchical cluster bootstrap)\n",
                  main_t[cols].round(3).to_markdown(index=False), "\n"]
        split_t = M.summary_table(kegg, n_boot=0, by_split=True)
        split_t.to_csv(out / "summary_by_split.csv", index=False)
        parts += ["## Stratified by original dataset split (126/165 rows are BioReason SFT-train rows)\n",
                  split_t[[c for c in ["arm", "model", "interface", "condition", "orig_split", "accuracy",
                                       "pair_accuracy", "n_trials"] if c in split_t]].round(3).to_markdown(index=False), "\n"]
        c = contrasts(kegg, a.n_boot)
        if len(c):
            c.to_csv(out / "contrasts.csv", index=False)
            parts += ["## Pre-registered paired contrasts (A - B)\n", c.round(3).to_markdown(index=False), "\n"]
        e = dna_effects(kegg, a.n_boot)
        if len(e):
            e.to_csv(out / "dna_effects.csv", index=False)
            parts += ["## DNA-input effects within each config (paired; positive = DNA helps)\n",
                      e.round(3).to_markdown(index=False), "\n"]
    pil = M.pilot_table(df)
    if len(pil):
        pil.to_csv(out / "pilot.csv", index=False)
        parts += ["## Harness-competence pilot\n", pil.round(3).to_markdown(index=False), "\n"]
    (out / "report.md").write_text("\n".join(parts))
    print(f"-> {out / 'report.md'}")


if __name__ == "__main__":
    main()
