"""Harbor adapter: KEGG variant->disease rows -> Harbor task directories.

One local Harbor dataset per (interface, condition):  <out>/<interface>-<condition>/<task dirs>
Host-only metadata goes to <out>/_meta/ (never inside a task dir):
  index.jsonl      task_name, episode_id, row_id, condition, interface, gold label, donor, ...
  handles.json     DNA handle token -> sha256(sequence)       (gateway resolves handles with it)
  sequences.jsonl  sha256 -> sequence                         (Evo2 cache builder input)

Interfaces (what the sandbox sees in /task/inputs.json; whitelisted keys only):
  raw         question, reference_sequence, variant_sequence
  native      question, dna_handles {reference, variant}       (no raw sequence anywhere)
  native_raw  question, reference_sequence, variant_sequence, dna_handles
  (no_dna drops every DNA key, for every interface)

    python -m dnaharness.adapters.kegg_dx.adapter --interfaces raw native --conditions wt no_dna scramble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from dnaharness import config
from dnaharness.adapters.common import (hmac_hex, reset_dir, seq_sha, sif_path, write_solution,
                                        write_task_toml, write_verifier)
from dnaharness.data import conditions as C
from dnaharness.data.kegg import label_vocabulary

TEMPLATES = Path(__file__).parent / "templates"
INTERFACES = ("raw", "native", "native_raw")
# Stamped into task.toml only when a task dir is written. Existing dirs are skipped by reset_dir.
ROLE_ACCESS = {"cohort": "eval", "dev": "dev", "train_pool": "train"}
ALLOWED_KEYS = {
    "raw": {"question", "reference_sequence", "variant_sequence"},
    "native": {"question", "dna_handles"},
    "native_raw": {"question", "reference_sequence", "variant_sequence", "dna_handles"},
}


def make_handle(key: str, episode_id: str, role: str, sha: str) -> str:
    """Unguessable, self-identifying: the gateway maps it to sha via _meta/handles.json."""
    return f"<DNA:{hmac_hex(key, f'{episode_id}|{role}|{sha}', 16)}>"


def build_inputs(ci: C.ConditionInput, interface: str, episode_id: str, key: str,
                 handles_out: Dict[str, str], seqs_out: Dict[str, str]) -> dict:
    inputs: dict = {"question": ci.question}
    if ci.dna:
        ref, var = ci.dna
        if interface in ("raw", "native_raw"):
            inputs["reference_sequence"] = ref
            inputs["variant_sequence"] = var
        if interface in ("native", "native_raw"):
            hs = {}
            for role, s in (("reference", ref), ("variant", var)):
                sha = seq_sha(s)
                h = make_handle(key, episode_id, role, sha)
                handles_out[h] = sha
                seqs_out[sha] = s
                hs[role] = h
            inputs["dna_handles"] = hs
    assert set(inputs) <= ALLOWED_KEYS[interface], (interface, set(inputs))
    return inputs


def render_instruction(inputs: dict) -> str:
    lines, tools = [], ""
    if "reference_sequence" in inputs:
        lines.append("- `reference_sequence`, `variant_sequence`: reference and variant DNA "
                     "(the same 2,048-base window, centred on the variant).")
        tools = ("\nPython helpers for examining sequences are importable as "
                 "`from dnah_runtime import seqtools` (find_differences, show_alignment, subsequence, "
                 "reverse_complement, gc_content, translate, find_motif).\n")
    if "dna_handles" in inputs:
        lines.append("- `dna_handles`: opaque handles standing for the reference and variant DNA.")
    tmpl = (TEMPLATES / "instruction.md").read_text()
    from dnaharness.runtime.prompts import LABEL_LIST
    return tmpl.format(input_lines="\n".join(lines) + ("\n" if lines else ""), tool_lines=tools,
                       label_lines=LABEL_LIST)


def run(out_dir: Path, interfaces: Iterable[str], conditions: Iterable[str], role: str = "cohort",
        row_ids: Optional[List[str]] = None, limit: int = 0, overwrite: bool = False,
        sif: Optional[str] = None) -> dict:
    out_dir = Path(out_dir)
    secrets = config.load_secrets(create=True)
    key = secrets["handle_hmac_key"]
    manifest = pd.read_parquet(config.path("manifest"))
    sel = manifest[manifest.role == role] if row_ids is None else manifest[manifest.row_id.isin(row_ids)]
    ids = sorted(sel.row_id)
    if limit:
        ids = ids[:limit]
    labels = label_vocabulary()
    gold_of = dict(zip(manifest.row_id, manifest.label))
    score_py = (TEMPLATES / "score.py").read_text()
    sif = sif or sif_path()

    meta = out_dir / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    handles: Dict[str, str] = json.loads((meta / "handles.json").read_text()) if (meta / "handles.json").exists() else {}
    seqs: Dict[str, str] = {}
    if (meta / "sequences.jsonl").exists():
        for line in (meta / "sequences.jsonl").read_text().splitlines():
            d = json.loads(line)
            seqs[d["sha"]] = d["seq"]
    index: Dict[str, dict] = {}
    if (meta / "index.jsonl").exists():
        for line in (meta / "index.jsonl").read_text().splitlines():
            d = json.loads(line)
            index[d["task_name"]] = d

    n_written = 0
    for rid in ids:
        cis = C.build(rid, conditions=tuple(conditions), manifest=manifest)
        for cond, ci in cis.items():
            for itf in interfaces:
                episode_id = hmac_hex(key, f"{rid}|{cond}|{itf}", 12)
                task_name = f"kdx-{episode_id}"
                tdir = out_dir / f"{itf}-{cond}" / task_name
                inputs = build_inputs(ci, itf, episode_id, key, handles, seqs)
                index[task_name] = {
                    "task_name": task_name, "episode_id": episode_id, "row_id": rid,
                    "condition": cond, "interface": itf, "dataset": f"{itf}-{cond}",
                    "label": gold_of[rid], "donor_row_id": ci.donor_row_id,
                    "donor_label": ci.donor_label, "has_dna": bool(ci.dna),
                    "dna_shas": [seq_sha(s) for s in ci.dna],
                    "handles": inputs.get("dna_handles"),
                }
                if not reset_dir(tdir, overwrite):
                    continue
                env = tdir / "environment"
                env.mkdir()
                (env / "inputs.json").write_text(json.dumps(inputs))
                (tdir / "instruction.md").write_text(render_instruction(inputs))
                write_task_toml(tdir, name=task_name, description="KEGG variant->disease (opaque episode)",
                                metadata={"episode_id": episode_id, "interface": itf,
                                          "benchmark": "kegg-dx", "access": ROLE_ACCESS[role],
                                          "role": role}, sif=sif)
                write_verifier(tdir, score_py, {"label": gold_of[rid], "labels": labels})
                write_solution(tdir, gold_of[rid])
                n_written += 1

    (meta / "handles.json").write_text(json.dumps(handles, indent=0, sort_keys=True))
    with open(meta / "sequences.jsonl", "w") as fh:
        for sha in sorted(seqs):
            fh.write(json.dumps({"sha": sha, "seq": seqs[sha]}) + "\n")
    with open(meta / "index.jsonl", "w") as fh:
        for tn in sorted(index):
            fh.write(json.dumps(index[tn]) + "\n")
    return {"tasks_written": n_written, "index_size": len(index), "handles": len(handles),
            "sequences": len(seqs), "out": str(out_dir)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="default: <datasets>/kegg-dx (or kegg-dx-<role>)")
    ap.add_argument("--interfaces", nargs="+", default=["raw"], choices=INTERFACES)
    ap.add_argument("--conditions", nargs="+", default=list(C.CORE_CONDITIONS), choices=C.ALL_CONDITIONS)
    ap.add_argument("--role", default="cohort", choices=["cohort", "dev", "train_pool"])
    ap.add_argument("--task-ids", nargs="*", default=None, help="explicit KEGG row ids (kegg_<split>_<i>)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sif", default=None)
    a = ap.parse_args(argv)
    out = Path(a.output_dir) if a.output_dir else config.path("datasets") / (
        "kegg-dx" if a.role == "cohort" else f"kegg-dx-{a.role}")
    print(json.dumps(run(out, a.interfaces, a.conditions, a.role, a.task_ids, a.limit, a.overwrite, a.sif), indent=2))


if __name__ == "__main__":
    main()
