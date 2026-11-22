"""Verifiable, NON-biological harness tasks: the competence pilot (eval) and the Stage-3 SFT training
distribution. The two splits use disjoint seeds, templates and size ranges.

Task types (all automatically checkable; none needs disease knowledge):
  inspect_dict      read a dict holding two DNA strings; report a property
  diff_substring    return '<offset>:<ref_segment>><var_segment>' for an SNV / indel (window offsets)
  child_call        make one llm_query and use its returned value (rlm only; gateway log confirms the call)
  combine_json      combine two computed observations into a JSON object
  stop_budget       trivial; must submit within 3 turns (checked from run_meta in analysis)
  native_passthrough  (native interface) pass a DNA handle to llm_query and submit the reply
  -- train only --
  delegate_batch    many short texts needing a semantic judgment each (delegation helps)
  tool_recovery     a provided parser raises on malformed records; must catch and continue

    python -m dnaharness.adapters.harness_tasks --split pilot
    python -m dnaharness.adapters.harness_tasks --split train --n-per-type 300
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from dnaharness import config
from dnaharness.adapters.common import (hmac_hex, reset_dir, seq_sha, sif_path, write_solution,
                                        write_task_toml, write_verifier)
from dnaharness.adapters.kegg_dx.adapter import make_handle

SPLITS = {
    "pilot": {"seed": 1000, "len": (300, 900), "n_per_type": 8,
              "types": ["inspect_dict", "diff_substring", "child_call", "combine_json", "stop_budget",
                        "native_passthrough"]},
    "train": {"seed": 5000, "len": (1000, 2048), "n_per_type": 300,
              "types": ["inspect_dict", "diff_substring", "child_call", "combine_json", "stop_budget",
                        "delegate_batch", "tool_recovery"]},
}
# Stamped into task.toml only when a task dir is written.
SPLIT_TASK_META = {
    "pilot": {"access": "eval", "role": "pilot"},
    "train": {"access": "train", "role": "train"},
}

SCORE_PY = r'''"""Pilot/train verifier: exact / numeric / JSON comparison against /tests/gold.json."""
import ast, json, os
gold = json.load(open("/tests/gold.json"))
p = "/logs/agent/answer.txt"
ans = open(p).read().strip() if os.path.exists(p) else ""
ok = False
try:
    if gold["mode"] == "exact":
        ok = ans.strip().strip('"').strip("'") == gold["answer"]
    elif gold["mode"] == "exact_ci":
        ok = ans.strip().strip('"').strip("'").lower() == gold["answer"].lower()
    elif gold["mode"] == "int":
        ok = int(ans.strip().strip('"')) == int(gold["answer"])
    elif gold["mode"] == "json":
        # a dict assigned to answer["content"] arrives as its Python repr (single quotes): accept that too
        try:
            a = json.loads(ans)
        except ValueError:
            a = ast.literal_eval(ans)
        assert isinstance(a, dict)
        g = gold["answer"]
        ok = set(a) == set(g) and all(
            abs(float(a[k]) - float(v)) <= 1e-3 if isinstance(v, (int, float)) and not isinstance(v, bool)
            else str(a[k]) == str(v) for k, v in g.items())
    elif gold["mode"] == "nonempty":
        ok = bool(ans)
except Exception:
    ok = False
os.makedirs("/logs/verifier", exist_ok=True)
json.dump({"correct": int(ok), "valid_output": int(bool(ans))}, open("/logs/verifier/reward.json", "w"))
json.dump({"answer": ans, "gold": gold}, open("/logs/verifier/prediction.json", "w"))
'''

INSTRUCTION = """# {title}

The inputs are in the JSON file `/task/inputs.json` (keys: {keys}). The task statement is in its `question` field.

When you are done, write ONLY the final answer to `/logs/agent/answer.txt` (for example with `echo`), then finish.
"""


def rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def mutate(rng: random.Random, ref: str) -> Tuple[str, str, int, str, str]:
    """-> (var, kind, offset, ref_seg, var_seg) with a single SNV / insertion / deletion near the middle."""
    k = rng.choice(["SNV", "insertion", "deletion"])
    off = len(ref) // 2 + rng.randint(-20, 20)
    if k == "SNV":
        b = rng.choice([c for c in "ACGT" if c != ref[off]])
        return ref[:off] + b + ref[off + 1:], k, off, ref[off], b
    if k == "insertion":
        ins = rand_seq(rng, rng.randint(1, 4))
        while ins[0] == ref[off]:          # keep the first-differing offset unambiguous
            ins = rand_seq(rng, len(ins))
        return ref[:off] + ins + ref[off:], k, off, "", ins
    ln = rng.randint(1, 4)
    while ref[off + ln] == ref[off]:
        off += 1
    return ref[:off] + ref[off + ln:], k, off, ref[off:off + ln], ""


def diff_answer(ref: str, var: str) -> str:
    """Canonical '<offset>:<ref_segment>><var_segment>' (indel-safe trimming, '-' for empty)."""
    n = min(len(ref), len(var))
    i = 0
    while i < n and ref[i] == var[i]:
        i += 1
    j = 0
    while j < n - i and ref[-1 - j] == var[-1 - j]:
        j += 1
    rs, vs = ref[i:len(ref) - j], var[i:len(var) - j]
    return f"{i}:{rs or '-'}>{vs or '-'}"


# --------------------------------------------------------------------------------------- types

def t_inspect_dict(rng, L, split):
    ref = rand_seq(rng, rng.randint(*L))
    var, *_ = mutate(rng, ref)
    prop = rng.choice(["length of the variant sequence", "number of 'G' characters in the reference sequence",
                       "number of 'AT' occurrences (non-overlapping, str.count) in the variant sequence"])
    val = len(var) if prop.startswith("length") else ref.count("G") if "'G'" in prop else var.count("AT")
    q = f"Report the {prop}. Answer with the integer only."
    return {"question": q, "reference_sequence": ref, "variant_sequence": var}, {"mode": "int", "answer": val}


def t_diff_substring(rng, L, split):
    ref = rand_seq(rng, rng.randint(*L))
    var, *_ = mutate(rng, ref)
    q = ("The variant sequence differs from the reference at one place. Answer in the form "
         "'<offset>:<reference segment>><variant segment>' where offset is the 0-based position of the first "
         "differing character and '-' denotes an empty segment (e.g. '512:A>G' for a substitution, "
         "'300:->TT' for an insertion of TT, '300:CA>-' for a deletion of CA).")
    return {"question": q, "reference_sequence": ref, "variant_sequence": var}, {"mode": "exact", "answer": diff_answer(ref, var)}


_CHILD_Q = [("What is the capital of France?", "PARIS"), ("What is the chemical symbol for gold?", "AU"),
            ("How many legs does a spider have? Answer with a word.", "EIGHT"),
            ("What color is the sky on a clear day?", "BLUE"), ("What is the opposite of 'cold'?", "HOT"),
            ("What planet is known as the Red Planet?", "MARS")]
_CHILD_Q_TRAIN = [("What is the largest ocean on Earth? One word.", "PACIFIC"),
                  ("What is the freezing point of water in Celsius? Answer with a number word.", "ZERO"),
                  ("Which animal is known as the king of the jungle?", "LION"),
                  ("What is the primary language spoken in Brazil?", "PORTUGUESE"),
                  ("What gas do plants absorb from the air? Give its common short name.", "CO2")]


def t_child_call(rng, L, split):
    q_, a_ = rng.choice(_CHILD_Q if split == "pilot" else _CHILD_Q_TRAIN)
    q = (f"Use llm_query to ask the helper model this question, asking for a one-word reply: \"{q_}\". "
         "Submit the helper's one-word answer converted to UPPERCASE with no punctuation.")
    return {"question": q, "helper_question": q_}, {"mode": "exact_ci", "answer": a_, "requires_child_call": True}


def t_combine_json(rng, L, split):
    ref = rand_seq(rng, rng.randint(*L))
    var, kind, *_ = mutate(rng, ref)
    gc = round((ref.count("G") + ref.count("C")) / len(ref), 3)
    q = ('Submit a JSON object with exactly two keys: "gc_ref" = GC fraction of the reference sequence rounded '
         'to 3 decimals, and "variant_type" = one of "SNV", "insertion", "deletion".')
    return ({"question": q, "reference_sequence": ref, "variant_sequence": var},
            {"mode": "json", "answer": {"gc_ref": gc, "variant_type": kind}})


def t_stop_budget(rng, L, split):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    q = f"Compute {a} * {b}. Submit within your first 3 turns. Answer with the integer only."
    return {"question": q}, {"mode": "int", "answer": a * b, "max_turns": 3}


def t_native_passthrough(rng, L, split):
    ref = rand_seq(rng, rng.randint(*L))
    var, *_ = mutate(rng, ref)
    q = ("Call llm_query exactly once with a prompt that contains context['dna_handles']['reference'] verbatim and "
         "asks the helper to describe that DNA sequence in one short sentence. Submit the helper's reply.")
    return ({"question": q, "_native": [ref, var]}, {"mode": "nonempty", "answer": "", "requires_child_dna": True})


_POS = ["Loved it, works perfectly", "Excellent quality and fast shipping", "Would buy again, very happy",
        "Great value, exceeded expectations", "Fantastic, my kids adore it"]
_NEG = ["Broke after one day", "Terrible, does not work at all", "Waste of money, very disappointed",
        "Arrived damaged and support ignored me", "Awful quality, returning it"]
_FILL = ["I ordered this last month.", "The box was blue.", "My neighbour has one too.", "It took a week to arrive.",
         "I use it mostly on weekends.", "The manual was in three languages."]


def t_delegate_batch(rng, L, split):
    n = rng.randint(25, 45)
    items, pos = [], 0
    for _ in range(n):
        p = rng.random() < 0.5
        pos += p
        core = rng.choice(_POS if p else _NEG)
        filler = " ".join(rng.sample(_FILL, rng.randint(2, 4)))
        items.append(f"{filler} {core}. {rng.choice(_FILL)}")
    q = ("context['reviews'] is a list of customer reviews. Count how many reviews are positive. "
         "Answer with the integer only.")
    return {"question": q, "reviews": items}, {"mode": "int", "answer": pos}


def t_tool_recovery(rng, L, split):
    n = rng.randint(15, 30)
    recs, total = [], 0
    for _ in range(n):
        if rng.random() < 0.25:
            recs.append(rng.choice(["id=;value=", "garbage", "id=7;value=abc", ""]))
        else:
            v = rng.randint(1, 100)
            total += v
            recs.append(f"id={rng.randint(1, 999)};value={v}")
    q = ("context['records'] holds 'id=<int>;value=<int>' strings, some malformed. Parse each with "
         "int() on the value field, skipping records that fail to parse, and submit the sum of all valid values "
         "as an integer only.")
    return {"question": q, "records": recs}, {"mode": "int", "answer": total}


TYPES: Dict[str, Callable] = {
    "inspect_dict": t_inspect_dict, "diff_substring": t_diff_substring, "child_call": t_child_call,
    "combine_json": t_combine_json, "stop_budget": t_stop_budget, "native_passthrough": t_native_passthrough,
    "delegate_batch": t_delegate_batch, "tool_recovery": t_tool_recovery,
}


def generate(split: str, out_dir: Path, n_per_type: int | None = None, overwrite: bool = False,
             sif: str | None = None) -> dict:
    spec = SPLITS[split]
    n = n_per_type or spec["n_per_type"]
    key = config.load_secrets(create=True)["handle_hmac_key"]
    sif = sif or sif_path()
    ds = out_dir / split
    meta = out_dir / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    index, handles, seqs = [], {}, {}
    for t in spec["types"]:
        rng = random.Random(f"{split}|{t}|{spec['seed']}")
        for i in range(n):
            inputs, gold = TYPES[t](rng, spec["len"], split)
            episode_id = hmac_hex(key, f"{split}|{t}|{i}", 12)
            task_name = f"hx-{episode_id}"
            if "_native" in inputs:
                ref, var = inputs.pop("_native")
                hs = {}
                for role, s in (("reference", ref), ("variant", var)):
                    sha = seq_sha(s)
                    h = make_handle(key, episode_id, role, sha)
                    handles[h], seqs[sha] = sha, s
                    hs[role] = h
                inputs["dna_handles"] = hs
            inputs["task_kind"] = "generic"
            index.append({"task_name": task_name, "episode_id": episode_id, "split": split, "type": t,
                          "gold": gold, "interface": "native" if "dna_handles" in inputs else "raw"})
            tdir = ds / task_name
            if not reset_dir(tdir, overwrite):
                continue
            (tdir / "environment").mkdir()
            (tdir / "environment" / "inputs.json").write_text(json.dumps(inputs))
            keys = ", ".join(f"`{k}`" for k in inputs if k != "task_kind")
            (tdir / "instruction.md").write_text(INSTRUCTION.format(title=f"Harness task ({split})", keys=keys))
            write_task_toml(tdir, name=task_name, description=f"harness {split} task",
                            metadata={"episode_id": episode_id, "benchmark": f"harness-{split}",
                                      **SPLIT_TASK_META[split]}, sif=sif,
                            agent_timeout_sec=1200.0)
            write_verifier(tdir, SCORE_PY, gold)
            ans = gold["answer"]
            write_solution(tdir, json.dumps(ans) if isinstance(ans, dict) else str(ans or "oracle"))
    (meta / f"index_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    old = json.loads((meta / "handles.json").read_text()) if (meta / "handles.json").exists() else {}
    old.update(handles)
    (meta / "handles.json").write_text(json.dumps(old, sort_keys=True))
    # merge with sequences already recorded (other splits share _meta), one line per sha
    seq_p = meta / "sequences.jsonl"
    allseq = {}
    if seq_p.exists():
        for line in seq_p.read_text().splitlines():
            r = json.loads(line)
            allseq[r["sha"]] = r["seq"]
    allseq.update(seqs)
    seq_p.write_text("".join(json.dumps({"sha": k, "seq": v}) + "\n" for k, v in allseq.items()))
    return {"split": split, "tasks": len(index), "out": str(ds), "native_handles": len(handles)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--n-per-type", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.output_dir) if a.output_dir else config.path("datasets") / f"harness-{a.split}"
    print(json.dumps(generate(a.split, out, a.n_per_type, a.overwrite), indent=2))


if __name__ == "__main__":
    main()
