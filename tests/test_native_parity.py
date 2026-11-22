"""Native BioReason prompt_embeds parity (serve env + downloaded checkpoints; CPU is fine).

Reference = BioReason/bio-mirage construction, written out step by step from
bio-mirage `runner.format_item` + `build_prompt_embeds` and BioReason `DLProcessor.__call__`:
  render CHAT_TEMPLATE on [dna, dna, text(question.strip())] with add_generation_prompt
  -> replace each `<|dna_pad|>` in the TEXT by N copies (N = Evo2 rows) -> tokenize
  -> embed_tokens (bf16) -> overwrite pad positions with dna_projection(rows) (bf16).
The gateway instead tokenizes first and expands pad ids afterwards; this test asserts equality.
Evo2 rows are random stand-ins (the projection/embedding path is what is under test here; the
Evo2 cache itself is validated end-to-end by the greedy reproduction run).
"""
import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dnaharness import config  # noqa: E402

REAL_CKPTS = None


def _ckpt(name):
    import re
    from pathlib import Path
    real_root = re.search(r"^root:\s*(\S+)", (config.CONFIG_DIR / "paths.yaml").read_text(), re.M).group(1)
    p = Path(real_root) / "ckpts" / name
    if not (p / "config.json").exists():
        pytest.skip(f"checkpoint {name} not prepared (python -m dnaharness.serving.prepare_ckpt {name})")
    return p


@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    from dnaharness.serving.native import BioReasonRenderer
    cache = tmp_path_factory.mktemp("evo2")
    r = BioReasonRenderer(str(_ckpt("bioreason-1.7b")), str(cache), device="cpu")
    r._test_cache = cache
    return r


def _fake_rows(cache, seq, seed):
    sha = hashlib.sha256(seq.encode()).hexdigest()
    g = torch.Generator().manual_seed(seed)
    torch.save(torch.randn(min(len(seq), 2048), 1920, generator=g).to(torch.bfloat16), cache / f"{sha}.pt")
    return sha


def _reference(renderer, question, rows_list):
    """bio-mirage / DLProcessor construction."""
    item = [{"role": "user", "content": [*({"type": "dna", "text": None} for _ in rows_list),
                                         {"type": "text", "text": question.strip()}]}]
    text = renderer.tok.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
    for rows in rows_list:   # DLProcessor: replace each pad with N placeholders, in order
        text = text.replace("<|dna_pad|>", "<|placeholder|>" * rows.shape[0], 1)
    text = text.replace("<|placeholder|>", "<|dna_pad|>")
    ids = torch.tensor(renderer.tok(text, add_special_tokens=False)["input_ids"])
    emb = renderer.embed[ids].clone()
    mask = ids == renderer.dna_pad_id
    if rows_list:
        emb[mask] = torch.cat([renderer.proj(r.to(torch.bfloat16)) for r in rows_list]).to(emb.dtype)
    return ids, emb, text


@pytest.mark.gpu
def test_dna_pad_id_and_projection_shape(renderer):
    assert renderer.dna_pad_id == 151670
    assert renderer.proj.weight.shape == (2048, 1920)


@pytest.mark.gpu
@pytest.mark.parametrize("lens", [(2048, 2048), (2048, 2045), (1500, 1503)])   # SNV, deletion, insertion-like
def test_gateway_embeds_equal_reference(renderer, lens):
    from dnaharness.runtime.prompts import canonical_native_prompt
    from dnaharness.serving import native as N
    import random
    rng = random.Random(sum(lens))
    seqs = ["".join(rng.choice("ACGT") for _ in range(n)) for n in lens]
    shas = [_fake_rows(renderer._test_cache, s, i) for i, s in enumerate(seqs)]
    handles = {"reference": "<DNA:" + "1" * 16 + ">", "variant": "<DNA:" + "2" * 16 + ">"}
    hmap = {handles["reference"]: shas[0], handles["variant"]: shas[1]}
    q = "Chromosome Number: 20\nNetwork Definition of the pathway: PRNP*\nWhat disease?  "
    res = N.apply_handle_policy([{"role": "user", "content": canonical_native_prompt(q, handles)}], "direct", True, hmap)
    built = renderer.build(res, enable_thinking=True)
    rows = [torch.load(renderer._test_cache / f"{s}.pt") for s in shas]
    ref_ids, ref_emb, ref_text = _reference(renderer, q, rows)
    assert built["n_dna"] == sum(r.shape[0] for r in rows)
    assert built["prompt_embeds"].shape == ref_emb.shape
    assert torch.equal(built["prompt_embeds"], ref_emb)
    assert ref_text.startswith("<|im_start|>user\n<|dna_start|><|dna_pad|>")
    assert ref_text.endswith("What disease?<|im_end|>\n<|im_start|>assistant\n")


@pytest.mark.gpu
def test_changing_variant_changes_only_variant_span(renderer):
    from dnaharness.serving import native as N
    a = "ACGT" * 400
    b = a[:800] + ("C" if a[800] != "C" else "G") + a[801:]
    s_ref = _fake_rows(renderer._test_cache, a, 11)
    s_v1 = _fake_rows(renderer._test_cache, a + "T", 12)
    s_v2 = _fake_rows(renderer._test_cache, b + "T", 13)
    H = ["<DNA:" + c * 16 + ">" for c in "abc"]
    hmap = {H[0]: s_ref, H[1]: s_v1, H[2]: s_v2}
    e1 = renderer.build(N.apply_handle_policy([{"role": "user", "content": f"{H[0]}{H[1]}Q"}], "direct", True, hmap))["prompt_embeds"]
    e2 = renderer.build(N.apply_handle_policy([{"role": "user", "content": f"{H[0]}{H[2]}Q"}], "direct", True, hmap))["prompt_embeds"]
    diff = (e1 != e2).any(dim=1).nonzero().flatten()
    ids = renderer.tok(renderer.tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "dna"}, {"type": "dna"}, {"type": "text", "text": "Q"}]}],
        tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    p0 = ids.index(renderer.dna_pad_id)            # first DNA slot (before expansion)
    var_start = p0 + len(a) + 2                    # ref rows, then <|dna_end|><|dna_start|>
    var_len = len(a) + 1
    # exactly the variant block changed (random stand-in rows differ at every position)
    assert diff.tolist() == list(range(var_start, var_start + var_len))


@pytest.mark.gpu
def test_text_only_bioreason_template_equals_qwen3_template(renderer):
    """train_lora renders text-only conversations with the official Qwen3 template; the gateway serves
    BioReason students with CHAT_TEMPLATE. They must agree for text-only conversations."""
    from transformers import AutoTokenizer
    from dnaharness.serving import native as N
    qwen = AutoTokenizer.from_pretrained(str(_ckpt("qwen3-1.7b")))
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Answer the following: Q"},
            {"role": "user", "content": "Turn 1/12:"}, {"role": "assistant", "content": "```repl\nprint(1)\n```"},
            {"role": "user", "content": "Code executed:\n1"}, {"role": "user", "content": "Turn 2/12:"}]
    res = N.apply_handle_policy(msgs, "root", True, {})
    a = renderer.tok.apply_chat_template(renderer.to_hf_messages(res), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
    b = qwen.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert a == b
