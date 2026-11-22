"""Handle policy + think splitting (pure Python; no torch)."""
import pytest

from dnaharness.serving import native as N

H1, H2, H3 = "<DNA:0123456789abcdef>", "<DNA:fedcba9876543210>", "<DNA:aaaaaaaaaaaaaaaa>"
HMAP = {H1: "sha_ref", H2: "sha_var"}


def _msgs():
    return [
        {"role": "system", "content": f"system mentions {H1}"},
        {"role": "user", "content": f"Answer the following: Reference DNA: {H1}\nVariant DNA: {H2}\n\nQ"},
        {"role": "assistant", "content": f"print(context['dna_handles'])  # {H1}"},
        {"role": "user", "content": f"REPL output: {{'reference': '{H1}'}}"},
    ]


def test_root_resolves_only_first_user_message():
    r = N.apply_handle_policy(_msgs(), "root", native=True, handle_map=HMAP)
    assert r.dna_shas == ["sha_ref", "sha_var"]
    assert r.n_resolved == 2 and r.n_neutralized == 3
    first_user = r.messages[1]["parts"]
    assert [k for k, _ in first_user].count("dna") == 2
    for i in (0, 2, 3):
        assert all(k == "text" for k, _ in r.messages[i]["parts"])
        assert N.NEUTRAL in "".join(t for _, t in r.messages[i]["parts"])


def test_child_resolves_its_prompt():
    r = N.apply_handle_policy([{"role": "user", "content": f"Describe {H1} vs {H2}"}], "child", True, HMAP)
    assert r.dna_shas == ["sha_ref", "sha_var"]


def test_non_native_model_never_resolves():
    r = N.apply_handle_policy(_msgs(), "child", native=False, handle_map=HMAP)
    assert r.dna_shas == [] and r.n_neutralized == 5


def test_unknown_handle_rejected():
    with pytest.raises(N.HandleError):
        N.apply_handle_policy([{"role": "user", "content": f"x {H3}"}], "child", True, HMAP)


def test_cap_on_handles_per_call():
    with pytest.raises(N.HandleError):
        N.apply_handle_policy([{"role": "user", "content": " ".join([H1] * (N.MAX_HANDLES_PER_CALL + 1))}],
                              "child", True, HMAP)


def test_canonical_prompt_is_adjacent_dna_then_question():
    from dnaharness.runtime.prompts import canonical_native_prompt
    p = canonical_native_prompt("  Q text \n", {"reference": H1, "variant": H2})
    r = N.apply_handle_policy([{"role": "user", "content": p}], "direct", True, HMAP)
    assert [k for k, _ in r.messages[0]["parts"]] == ["dna", "dna", "text"]
    assert r.messages[0]["parts"][2][1] == "Q text"


@pytest.mark.parametrize("text,exp", [
    ("<think>\nabc\n</think>\n\nAnswer: x", ("abc", "Answer: x", False)),
    ("Answer: x", ("", "Answer: x", False)),
    ("<think>\nlong thought never closed", ("long thought never closed", "", True)),
])
def test_split_thinking(text, exp):
    assert N.split_thinking(text) == exp
