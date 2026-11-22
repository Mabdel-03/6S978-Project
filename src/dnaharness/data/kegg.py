"""Load wanglab/kegg at a pinned revision, in exactly the row format bio-mirage uses.

bio-mirage's `kegg_pairs.load_rows` calls `load_dataset(dataset)` without a revision; we pin the
revision (configs/models.yaml) but keep the row dict schema and the `kegg_{split}_{index}` ids
identical so its `build_strata` / `build_donors` can be reused unchanged.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

from dnaharness import config

SPLITS = ("train", "test", "val")


def pair_key(ref: str, var: str) -> str:
    """Identical to bio-mirage prepare.py `variant_key` (bootstrap cluster unit)."""
    return hashlib.sha1((ref + "|" + var).encode()).hexdigest()[:16]


def text_key(question: str) -> str:
    return hashlib.sha1(question.encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def load_rows() -> tuple:
    """All 1,449 rows as dicts: example_id, split, index, question, answer, reference_sequence,
    variant_sequence. Returned as a tuple (cached; treat as read-only)."""
    from datasets import load_dataset

    ds_cfg = config.models()["dataset"]
    ds = load_dataset(ds_cfg["repo"], revision=ds_cfg["revision"],
                      cache_dir=str(config.path("hf_home")) + "/datasets")
    rows: List[dict] = []
    for split in SPLITS:
        if split not in ds:
            continue
        for i, r in enumerate(ds[split]):
            rows.append({
                "example_id": f"kegg_{split}_{i}", "split": split, "index": i,
                "question": r["question"], "answer": r["answer"].strip(),
                "reference_sequence": r["reference_sequence"],
                "variant_sequence": r["variant_sequence"],
            })
    return tuple(rows)


def label_vocabulary(rows=None) -> List[str]:
    """Closed label set = every gold answer in the dataset (bio-mirage score_bioreason.py; 37)."""
    rows = rows if rows is not None else load_rows()
    return sorted({r["answer"] for r in rows} - {""})
