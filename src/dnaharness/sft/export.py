"""Teacher Harbor jobs -> matched SFT datasets (coding-trajectory, RLM-trajectory, task-answer control).

Pipeline (RLM paper v3 recipe, adapted):
  1. success filter: correct (incl. protocol requirement, e.g. a real child call), status ok, >1 turn
  2. keep only tasks solved by BOTH teachers (removes the selection bias of the RLM teacher with a weak
     child succeeding mostly on easy tasks); one trajectory (lowest attempt) per task and arm
  3. one SFT sample per ROOT turn: prompt = the exact request messages the root saw (history already
     without earlier thinking, as at inference), completion = "<think>{reasoning}</think>\\n\\n{content}"
     of that turn; samples whose prompt exceeds --max-prompt-chars are dropped
  4. match the coding and RLM sets on unique tasks (by construction) and on total completion tokens
     (random drop from the larger set, seeded, to within --token-tol)
  5. task-answer control on the SAME tasks: question + serialized inputs -> final answer only
Requires the teacher jobs to be run with `serve.log_messages: true`.

    python -m dnaharness.sft.export --coding <jobs>/sft_teacher_coding --rlm <jobs>/sft_teacher_rlm \
        --student bioreason-1.7b --out <root>/sft_data/bioreason-1.7b
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dnaharness import config
from dnaharness.analysis.collect import collect_job, load_gateway_calls


def _calls_by_trial(job: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for c in load_gateway_calls(job):
        if c.get("route") == "root" and c.get("status") == 200:
            out[c["trial"]].append(c)
    for v in out.values():
        v.sort(key=lambda c: c["call_idx"])
    return out


def successes(df: pd.DataFrame) -> pd.DataFrame:
    ok = df["correct"].fillna(False).astype(bool)
    if "correct_with_protocol" in df:
        ok &= df["correct_with_protocol"].fillna(df["correct"]).fillna(False).astype(bool)
    ok &= df["status"].eq("ok") & (pd.to_numeric(df["n_iterations"], errors="coerce").fillna(0) > 1)
    return df[ok].sort_values(["task_name", "attempt"]).drop_duplicates("task_name")


def trajectory_samples(job: Path, rows: pd.DataFrame, max_prompt_chars: int) -> List[dict]:
    calls = _calls_by_trial(job)
    out = []
    for _, r in rows.iterrows():
        usage_p = job / r["trial_name"] / "agent" / "gateway_usage.json"
        if not usage_p.exists():
            continue
        key = json.loads(usage_p.read_text())["key"]
        for c in calls.get(key, []):
            msgs = c.get("messages")
            if not msgs:
                raise RuntimeError("teacher job ran without serve.log_messages: true")
            if len(json.dumps(msgs)) > max_prompt_chars:
                continue
            target = f"<think>\n{(c.get('reasoning') or '').strip()}\n</think>\n\n{c.get('content') or ''}"
            out.append({"prompt": msgs, "completion": [{"role": "assistant", "content": target}],
                        "task_name": r["task_name"], "type": r.get("type"), "turn": c["call_idx"]})
    return out


def _ntok(s: List[dict], tok) -> int:
    return sum(len(tok(x["completion"][0]["content"], add_special_tokens=False)["input_ids"]) for x in s)


def _trim(s: List[dict], target: int, tok, tol: float, rng: random.Random) -> List[dict]:
    s = s[:]
    rng.shuffle(s)
    total = _ntok(s, tok)
    while s and total > target * (1 + tol):
        total -= _ntok([s.pop()], tok)
    return s


def match_tokens(a: List[dict], b: List[dict], tok, tol: float, seed: int) -> tuple:
    """Drop random samples from whichever set has more completion tokens until within tol."""
    rng = random.Random(seed)
    ta, tb = _ntok(a, tok), _ntok(b, tok)
    if ta > tb:
        a = _trim(a, tb, tok, tol, rng)
    elif tb > ta:
        b = _trim(b, ta, tok, tol, rng)
    return a, b


def answer_samples(task_rows: pd.DataFrame) -> List[dict]:
    out = []
    for _, r in task_rows.iterrows():
        tdir = Path(r["task_path"])
        inputs = json.loads((tdir / "environment" / "inputs.json").read_text())
        gold = json.loads((tdir / "tests" / "gold.json").read_text())
        ans = gold.get("answer")
        ans = json.dumps(ans) if isinstance(ans, dict) else str(ans)
        q = inputs.pop("question")
        inputs.pop("task_kind", None)
        user = f"{q}\n\nInputs (JSON):\n{json.dumps(inputs)}\n\nReply with the final answer only."
        out.append({"prompt": [{"role": "user", "content": user}],
                    "completion": [{"role": "assistant", "content": f"<think>\n\n</think>\n\n{ans}"}],
                    "task_name": r["task_name"]})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coding", required=True, nargs="+", help="coding-teacher job dir(s) (chunks)")
    ap.add_argument("--rlm", required=True, nargs="+", help="RLM-teacher job dir(s) (chunks)")
    ap.add_argument("--student", required=True, help="served name of the student (tokenizer for token matching)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompt-chars", type=int, default=100_000)
    ap.add_argument("--token-tol", type=float, default=0.02)
    ap.add_argument("--no-intersection", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(config.path("ckpts") / "qwen3-1.7b"))

    jcs, jrs = [Path(x) for x in a.coding], [Path(x) for x in a.rlm]

    def _succ(jobs):   # successes over all chunks, one trajectory per task (lowest attempt), with its job dir
        df = pd.concat([collect_job(j).assign(_job=str(j)) for j in jobs], ignore_index=True)
        return successes(df)

    dc, dr = _succ(jcs), _succ(jrs)
    keep = set(dc.task_name) | set(dr.task_name) if a.no_intersection else set(dc.task_name) & set(dr.task_name)
    dc, dr = dc[dc.task_name.isin(keep)], dr[dr.task_name.isin(keep)]
    sc = [x for j, g in dc.groupby("_job") for x in trajectory_samples(Path(j), g, a.max_prompt_chars)]
    sr = [x for j, g in dr.groupby("_job") for x in trajectory_samples(Path(j), g, a.max_prompt_chars)]
    sc, sr = match_tokens(sc, sr, tok, a.token_tol, a.seed)

    # task paths for the answer control (from the teacher jobs' trial configs)
    paths = {}
    for rp in [rp for j in jcs + jrs for rp in j.glob("*/result.json")]:
        r = json.loads(rp.read_text())
        p = Path((r.get("config") or {}).get("task", {}).get("path", ""))
        paths[p.name] = str(p)
    ans_rows = pd.DataFrame({"task_name": sorted(keep)})
    ans_rows["task_path"] = ans_rows.task_name.map(paths)
    sa = answer_samples(ans_rows.dropna())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, s in (("coding", sc), ("rlm", sr), ("answer", sa)):
        with open(out / f"{name}.jsonl", "w") as fh:
            for x in s:
                fh.write(json.dumps(x) + "\n")
        stats[name] = {"samples": len(s), "tasks": len({x["task_name"] for x in s}),
                       "completion_tokens": _ntok(s, tok)}
    stats["kept_tasks"] = len(keep)
    stats["intersection"] = not a.no_intersection
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
