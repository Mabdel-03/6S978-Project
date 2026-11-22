import random

import pytest

from dnaharness.adapters.harness_tasks import diff_answer, mutate, rand_seq
from dnaharness.runtime import scoring as S
from dnaharness.runtime import seqtools as T

LABELS = ["parkinson disease", "amyotrophic lateral sclerosis", "prion disease", "creutzfeldt-jakob disease",
          "alzheimer disease", "huntington disease", "colorectal cancer", "melanoma"]
GENS = [
    "creutzfeldt-jakob disease", "Answer: Parkinson disease", "<think>maybe ALS</think>\n\nAnswer: amyotrophic lateral sclerosis",
    "I think it is a prion disease (Creutzfeldt-Jakob disease).", "no idea", "", "Answer: melanoma; skin cancer",
    "Answer:  Alzheimer's disease", "colorectal  CANCER!",
]


def test_sandbox_scoring_copy_matches_biomirage():
    from dnaharness import config
    config.ensure_bio_mirage_on_path()
    from input_use.metrics import dna
    for g in GENS:
        assert S.extract_predicted_label(g, LABELS) == dna.extract_predicted_label(g, LABELS), g
        assert S.normalize_label_text(g) == dna.normalize_label_text(g)
    for gold in LABELS + ["prion disease; something"]:
        assert S.canonical_gold(gold) == dna.canonical_gold(gold)


def test_host_and_sandbox_score_agree():
    from dnaharness.data.labels import score_answer
    for g in GENS:
        for gold in ("prion disease", "melanoma", "parkinson disease"):
            h = score_answer(g, gold, LABELS)
            s = S.score(g, gold, LABELS)
            assert (h.predicted_label, h.correct, h.mapped) == (s["predicted_label"], s["correct"], s["mapped"])


def test_longest_label_wins():
    # "creutzfeldt-jakob disease" must beat the shorter contained label
    assert S.extract_predicted_label("creutzfeldt-jakob disease", LABELS)[0] == "creutzfeldt-jakob disease"


@pytest.mark.parametrize("seed", range(40))
def test_find_differences_and_diff_answer_agree(seed):
    rng = random.Random(seed)
    ref = rand_seq(rng, 400)
    var, kind, off, rs, vs = mutate(rng, ref)
    d = T.find_differences(ref, var)
    assert d["start"] == off
    assert d["type"] == kind
    assert diff_answer(ref, var) == f"{d['start']}:{d['ref_segment'] or '-'}>{d['var_segment'] or '-'}"
    assert (d["ref_segment"], d["var_segment"]) == (rs, vs)


def test_seqtools_basics():
    assert T.reverse_complement("AACGT") == "ACGTT"
    assert abs(T.gc_content("GGCA") - 0.75) < 1e-9
    assert T.translate("ATGGCC") == "MA"
    assert T.find_motif("AAAA", "AA") == [0, 1, 2]
    assert T.find_differences("ACGT", "ACGT")["type"] == "identical"
    assert "ref" in T.show_alignment("ACGTACGT", "ACGAACGT")


def test_harness_json_mode_accepts_python_dict_repr_only_with_equal_values():
    """Teacher smoke: answer["content"] = {...} reaches answer.txt as a Python repr (single quotes)."""
    from dnaharness.analysis.collect import score_harness
    g = {"mode": "json", "answer": {"gc_ref": 0.505, "variant_type": "deletion"}}
    assert score_harness("{'gc_ref': 0.505, 'variant_type': 'deletion'}", g)
    assert score_harness('{"gc_ref": 0.5051, "variant_type": "deletion"}', g)
    assert not score_harness("{'gc_ref': 0.486, 'variant_type': 'deletion'}", g)
    assert not score_harness("[1, 2]", g) and not score_harness("__import__('os')", g) and not score_harness("", g)
