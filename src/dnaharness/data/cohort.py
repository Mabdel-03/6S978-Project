"""The 165-row genome-dependent cohort (BioMirage `sensitivity` stratum) + dev / train-pool splits.

Selection is delegated to bio-mirage `kegg_pairs.build_strata` (no reimplementation): a question-text
group is genome-dependent when it contains >1 distinct (reference, variant) pair AND >1 distinct
disease label. We then attach the identities the analysis needs (text group, DNA pair, gene,
majority/minority label within group) and assert the counts published in the paper (B.3.1).

    python -m dnaharness.data.cohort            # writes the manifest parquet (configs/paths.yaml)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dnaharness import config
from dnaharness.data.kegg import label_vocabulary, load_rows, pair_key, text_key

EXPECTED = {
    "rows": 165, "pairs": 63, "text_groups": 14, "labels": 12, "genes": 6,
    "split": {"train": 126, "test": 19, "val": 20}, "PRNP": 98,
}
DEV_SIZE = 20
DEV_SEED = 20261121


def _bm():
    config.ensure_bio_mirage_on_path()
    from input_use.modalities.dna import interventions, kegg_pairs  # noqa: WPS433
    return kegg_pairs, interventions


def build_manifest(rows=None) -> pd.DataFrame:
    """One row per KEGG example with cohort membership and identity columns."""
    kegg_pairs, interventions = _bm()
    rows = list(rows if rows is not None else load_rows())
    strata = kegg_pairs.build_strata(rows)
    sens = set(strata["sensitivity"]["rows"])
    stab = set(strata["stability"]["rows"])
    donors = kegg_pairs.build_donors(rows)

    by_text: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_text[r["question"]].append(r)

    recs = []
    for r in rows:
        group = by_text[r["question"]]
        counts = Counter(g["answer"] for g in group)
        # Majority label of the text group; ties broken alphabetically (deterministic, pre-registered).
        top = max(counts.values())
        majority = sorted(l for l, c in counts.items() if c == top)[0]
        recs.append({
            "row_id": r["example_id"], "split": r["split"], "index": r["index"],
            "text_group_id": text_key(r["question"]),
            "pair_id": pair_key(r["reference_sequence"], r["variant_sequence"]),
            "gene": (interventions.stem_gene(r["question"]) or "").upper(),
            "label": r["answer"],
            "group_size": len(group),
            "group_n_labels": len(counts),
            "group_n_pairs": len({(g["reference_sequence"], g["variant_sequence"]) for g in group}),
            "group_majority_label": majority,
            "is_minority_label": r["answer"] != majority,
            "stratum": "sensitivity" if r["example_id"] in sens else
                       "stability" if r["example_id"] in stab else "unpaired",
            "in_cohort": r["example_id"] in sens,
            "bm_donor_row_id": (donors.get(r["example_id"]) or {}).get("donor_example_id"),
            "ref_len": len(r["reference_sequence"]), "var_len": len(r["variant_sequence"]),
        })
    df = pd.DataFrame.from_records(recs)
    df["role"] = "other"
    df.loc[df.in_cohort, "role"] = "cohort"
    dev_ids = select_dev(df)
    df.loc[df.row_id.isin(dev_ids), "role"] = "dev"
    # Train pool for optional bio-targeted SFT trajectories: shares neither DNA pair nor text with
    # the cohort (nor with dev).
    blocked_pairs = set(df.loc[df.role.isin(["cohort", "dev"]), "pair_id"])
    blocked_text = set(df.loc[df.role.isin(["cohort", "dev"]), "text_group_id"])
    pool = (df.role == "other") & ~df.pair_id.isin(blocked_pairs) & ~df.text_group_id.isin(blocked_text)
    df.loc[pool, "role"] = "train_pool"
    ds_cfg = config.models()["dataset"]
    df.attrs["dataset"] = f"{ds_cfg['repo']}@{ds_cfg['revision']}"
    # Reserved for the deferred generalization track (grouping by biological identity).
    df["bio_identity"] = df["gene"] + ":" + df["pair_id"]
    return df


def select_dev(df: pd.DataFrame, n: int = DEV_SIZE, seed: int = DEV_SEED) -> List[str]:
    """Prompt-development rows: no shared DNA pair or question text with the cohort; one row per
    pair_id; spread over genes. Deterministic."""
    cohort = df[df.in_cohort]
    cand = df[~df.in_cohort & ~df.pair_id.isin(cohort.pair_id) & ~df.text_group_id.isin(cohort.text_group_id)]
    cand = cand.drop_duplicates("pair_id").drop_duplicates("text_group_id")
    rng = random.Random(seed)
    by_gene: Dict[str, List[str]] = defaultdict(list)
    for rid, g in zip(cand.row_id, cand.gene):
        by_gene[g].append(rid)
    for g in by_gene:
        by_gene[g].sort()
        rng.shuffle(by_gene[g])
    out: List[str] = []
    genes = sorted(by_gene)
    rng.shuffle(genes)
    while len(out) < n and any(by_gene.values()):
        for g in genes:
            if by_gene[g] and len(out) < n:
                out.append(by_gene[g].pop())
    return sorted(out)


def cohort_summary(df: pd.DataFrame) -> dict:
    c = df[df.in_cohort]
    return {
        "rows": len(c), "pairs": c.pair_id.nunique(), "text_groups": c.text_group_id.nunique(),
        "labels": c.label.nunique(), "genes": c.gene.nunique(),
        "split": {k: int(v) for k, v in c.split.value_counts().items()},
        "PRNP": int((c.gene == "PRNP").sum()),
        "minority_rows": int(c.is_minority_label.sum()),
        "dev_rows": int((df.role == "dev").sum()),
        "train_pool_rows": int((df.role == "train_pool").sum()),
        "label_vocabulary": len(label_vocabulary()),
    }


def check_expected(summary: dict) -> None:
    for k, v in EXPECTED.items():
        assert summary[k] == v, f"cohort {k}: got {summary[k]!r}, expected {v!r} (arXiv 2610.00898 B.3.1)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(config.path("manifest")))
    a = ap.parse_args()
    df = build_manifest()
    s = cohort_summary(df)
    check_expected(s)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out.parent / "manifest_summary.json").write_text(json.dumps({**s, "dataset": df.attrs["dataset"]}, indent=2))
    print(json.dumps(s, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
