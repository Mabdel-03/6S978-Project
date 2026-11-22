"""Input conditions for one KEGG row, built on bio-mirage's own windowing and perturbations.

Every condition starts from the same variant-centred window: bio-mirage `prepare.truncate_around`
(per_side=1024, i.e. 2,049 bp centred on the first differing base), then clipped to the first 2,048
bases -- exactly what the Evo2 tokenizer keeps (`max_length_dna=2048`) -- so the raw-string and the
native-embedding interfaces carry *identical* bases.

Genome perturbations reuse `interventions.make_conditions` with bio-mirage's per-row seed
(20260812 + row index), so `scramble` is byte-identical to the paper's arm.

Conditions (question text is unchanged unless stated):
  wt            [ref, var]
  no_dna        []                     (text only)
  scramble      [shuffle(ref), shuffle(var)]   composition-preserving, both blocks
  revert_ref    [ref, ref]             (reference-only / locus information without the variant)
  swap_pair     [donor_ref, donor_var] complete donor pair (bio-mirage donor: different label and
                                       genome). NOT bio-mirage's `swap_variant_donor`, which keeps
                                       our reference and swaps only the variant -> artificial diff.
  text_conflict [donor_ref, donor_var] donor restricted to the SAME gene (same locus), different
                                       label, different pair, different text group; skipped when no
                                       such donor exists. Source-preference diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from dnaharness import config
from dnaharness.data.kegg import load_rows

PER_SIDE = 1024
MAX_LEN_DNA = 2048
BM_SEED = 20260812

CORE_CONDITIONS = ("wt", "no_dna", "scramble")
EXTENDED_CONDITIONS = ("revert_ref", "swap_pair", "text_conflict")
ALL_CONDITIONS = CORE_CONDITIONS + EXTENDED_CONDITIONS


@dataclass
class ConditionInput:
    row_id: str
    condition: str
    question: str
    dna: List[str]                      # [] or [ref_window, var_window]
    donor_row_id: Optional[str] = None
    donor_label: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)


def _bm():
    config.ensure_bio_mirage_on_path()
    from input_use.core import conditions as C
    from input_use.modalities.dna.interventions import make_conditions, stem_gene
    from input_use.modalities.dna.prepare import truncate_around
    return C, make_conditions, truncate_around, stem_gene


def window(ref: str, var: str) -> tuple[str, str]:
    _, _, truncate_around, _ = _bm()
    r, v = truncate_around(ref, var, PER_SIDE)
    return r[:MAX_LEN_DNA], v[:MAX_LEN_DNA]


@lru_cache(maxsize=1)
def _rows_by_id() -> Dict[str, dict]:
    return {r["example_id"]: r for r in load_rows()}


def _same_gene_donor(row: dict, manifest) -> Optional[str]:
    m = manifest.set_index("row_id")
    me = m.loc[row["example_id"]]
    cand = m[(m.gene == me.gene) & (m.label != me.label) & (m.pair_id != me.pair_id)
             & (m.text_group_id != me.text_group_id)]
    if cand.empty:
        return None
    # deterministic: prefer cohort rows, then lexicographic row id
    return sorted(cand.index, key=lambda rid: (not bool(cand.loc[rid, "in_cohort"]), rid))[0]


def build(row_id: str, conditions=CORE_CONDITIONS, manifest=None) -> Dict[str, ConditionInput]:
    """{condition: ConditionInput} for one row. `manifest` (from cohort.build_manifest) is needed
    only for swap_pair / text_conflict donors."""
    C, make_conditions, truncate_around, _ = _bm()
    rows = _rows_by_id()
    r = rows[row_id]
    ref_w, var_w = truncate_around(r["reference_sequence"], r["variant_sequence"], PER_SIDE)
    bm = make_conditions(ref_w, var_w, r["question"], seed=BM_SEED + r["index"])

    def clip(seqs):
        return [s[:MAX_LEN_DNA] for s in seqs]

    out: Dict[str, ConditionInput] = {}
    q = r["question"]
    for cond in conditions:
        if cond == "wt":
            out[cond] = ConditionInput(row_id, cond, q, clip(bm[C.WT]["dna_sequences"]))
        elif cond == "no_dna":
            out[cond] = ConditionInput(row_id, cond, q, [])
        elif cond == "scramble":
            out[cond] = ConditionInput(row_id, cond, q, clip(bm[C.SCRAMBLE]["dna_sequences"]))
        elif cond == "revert_ref":
            out[cond] = ConditionInput(row_id, cond, q, clip(bm[C.REVERT_VARIANT_BLOCK]["dna_sequences"]))
        elif cond in ("swap_pair", "text_conflict"):
            if manifest is None:
                raise ValueError(f"{cond} needs the manifest for donor selection")
            if cond == "swap_pair":
                donor_id = manifest.set_index("row_id").loc[row_id, "bm_donor_row_id"]
            else:
                donor_id = _same_gene_donor(r, manifest)
            if donor_id is None or (isinstance(donor_id, float)):
                continue
            d = rows[donor_id]
            dr, dv = window(d["reference_sequence"], d["variant_sequence"])
            out[cond] = ConditionInput(row_id, cond, q, [dr, dv], donor_row_id=donor_id,
                                       donor_label=d["answer"])
        else:
            raise ValueError(f"unknown condition {cond!r}")
    # invariants (bio-mirage prepare.py): genome arms never change the text
    for ci in out.values():
        assert ci.question == q
        assert all(len(s) <= MAX_LEN_DNA for s in ci.dna)
    return out
