import pytest

from dnaharness.data import cohort
from dnaharness.data import conditions as C


@pytest.mark.network
def test_cohort_counts_match_paper(manifest):
    s = cohort.cohort_summary(manifest)
    cohort.check_expected(s)                       # 165 / 63 / 14 / 12 / 6, 126/19/20, 98 PRNP
    assert s["label_vocabulary"] == 37
    assert s["dev_rows"] == cohort.DEV_SIZE


@pytest.mark.network
def test_dev_and_train_pool_disjoint_from_cohort(manifest):
    c = manifest[manifest.in_cohort]
    for role in ("dev", "train_pool"):
        r = manifest[manifest.role == role]
        assert not set(r.pair_id) & set(c.pair_id), role
        assert not set(r.text_group_id) & set(c.text_group_id), role


@pytest.mark.network
def test_conditions_invariants_and_biomirage_parity(manifest):
    from dnaharness import config
    config.ensure_bio_mirage_on_path()
    from input_use.modalities.dna.interventions import make_conditions
    from input_use.modalities.dna.prepare import truncate_around
    from dnaharness.data.kegg import load_rows

    rows = {r["example_id"]: r for r in load_rows()}
    for rid in sorted(manifest[manifest.in_cohort].row_id)[:12]:
        cis = C.build(rid, conditions=C.ALL_CONDITIONS, manifest=manifest)
        r = rows[rid]
        ref_w, var_w = truncate_around(r["reference_sequence"], r["variant_sequence"], 1024)
        bm = make_conditions(ref_w, var_w, r["question"], seed=20260812 + r["index"])
        # identical bases to bio-mirage, clipped to what Evo2 keeps (2048)
        assert cis["wt"].dna == [ref_w[:2048], var_w[:2048]]
        assert cis["scramble"].dna == [s[:2048] for s in bm["scramble"]["dna_sequences"]]
        assert cis["revert_ref"].dna == [ref_w[:2048]] * 2
        assert cis["no_dna"].dna == []
        for ci in cis.values():
            assert ci.question == r["question"]
            assert all(len(s) <= 2048 for s in ci.dna)
        if "swap_pair" in cis:                       # complete donor pair, not variant-only
            d = rows[cis["swap_pair"].donor_row_id]
            dr, dv = truncate_around(d["reference_sequence"], d["variant_sequence"], 1024)
            assert cis["swap_pair"].dna == [dr[:2048], dv[:2048]]
            assert cis["swap_pair"].donor_label != r["answer"]
        if "text_conflict" in cis:
            m = manifest.set_index("row_id")
            assert m.loc[cis["text_conflict"].donor_row_id, "gene"] == m.loc[rid, "gene"]
            assert cis["text_conflict"].donor_label != r["answer"]
