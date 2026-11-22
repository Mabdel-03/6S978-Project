import difflib

from dnaharness.runtime import prompts as P


def test_coding_prompt_is_rlm_prompt_minus_subcall_sections():
    for native in (False, True):
        for task in ("kegg", "generic"):
            rlm = P.system_prompt(subcalls=True, native=native, task=task).format(custom_tools_section="")
            cod = P.system_prompt(subcalls=False, native=native, task=task).format(custom_tools_section="")
            assert "llm_query" in rlm and "llm_query" not in cod
            removed = [l for l in difflib.ndiff(rlm.splitlines(), cod.splitlines()) if l.startswith("- ")]
            added = [l for l in difflib.ndiff(rlm.splitlines(), cod.splitlines()) if l.startswith("+ ")]
            assert not added                                    # coding prompt only removes lines
            # ...and what it removes is exactly the sub-call section(s), nothing else
            expected = P._SUBCALLS.splitlines() + (P._NATIVE_SUBCALL.splitlines() if native else [])
            assert [l[2:] for l in removed] == [l for l in expected if l.strip()]


def test_root_prompt_handles_only_when_root_sees_dna():
    h = {"reference": "<DNA:" + "a" * 16 + ">", "variant": "<DNA:" + "b" * 16 + ">"}
    assert h["reference"] in P.root_prompt("Q", h, True)
    assert "<DNA:" not in P.root_prompt("Q", h, False)
    assert P.root_prompt(" Q ", None, True) == "Q"


def test_direct_raw_prompt_same_for_both_checkpoints():
    assert P.direct_raw_prompt("Q", "AC", "AG").count("Answer: <disease name>") == 1
    assert "Reference DNA" not in P.direct_raw_prompt("Q", None, None)


def test_label_list_is_the_dataset_vocabulary_and_in_every_noncanonical_prompt():
    from dnaharness.data import kegg
    assert list(P.KEGG_LABELS) == kegg.label_vocabulary()
    h = {"reference": "<DNA:" + "a" * 16 + ">", "variant": "<DNA:" + "b" * 16 + ">"}
    assert P.LABEL_LIST in P.direct_raw_prompt("Q", "ACGT", "ACTT")
    assert P.LABEL_LIST in P.direct_raw_prompt("Q", None, None)
    for sub in (False, True):
        for native in (False, True):
            assert P.LABEL_LIST in P.system_prompt(subcalls=sub, native=native).format(custom_tools_section="")
    assert P.LABEL_LIST in P.direct_harness_messages("Q", h, True)[0]["content"]
    # BioReason's canonical (training-format) prompt and the generic pilot prompt carry no list
    assert P.canonical_native_prompt("Q", h) == f"{h['reference']}{h['variant']}Q"
    assert P.LABEL_LIST not in P.system_prompt(subcalls=True, native=False, task="generic")
