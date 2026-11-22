"""Harbor job dirs -> one tidy parquet row per trial, with AUTHORITATIVE host-side re-scoring.

Joins: trial result.json (+ config) | agent/answer.txt (re-scored here with bio-mirage's metric
against host-held gold; the in-sandbox verifier is only a convenience) | agent/run_meta.json |
agent/gateway_usage.json or the gateway call log (mini-swe sessions) | dataset _meta/index | manifest.

    python -m dnaharness.analysis.collect <jobs>/exp_a_shared [...] --out results/exp_a.parquet
    python -m dnaharness.analysis.collect --gate-audit <jobs>/audit      # exit 1 unless audit passed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from dnaharness import config


def _load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


@lru_cache(maxsize=None)
def _index_for(meta_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for f in Path(meta_dir).glob("index*.jsonl"):
        for r in _load_jsonl(f):
            out[r["task_name"]] = r
    return out


@lru_cache(maxsize=1)
def _labels():
    from dnaharness.data.kegg import label_vocabulary
    return tuple(label_vocabulary())


def arm_label(agent_cfg: dict) -> str:
    kw = agent_cfg.get("kwargs") or {}
    ip = agent_cfg.get("import_path") or ""
    name = agent_cfg.get("name") or ""
    if ip.endswith(":RlmAgent"):
        return kw.get("arm", "rlm") + ("_tool" if kw.get("bioreason_tool") else "")
    if ip.endswith(":VoteAgent"):
        return f"vote@{kw.get('k', 5)}"
    if ip.endswith(":DirectAgent"):
        v = kw.get("variant", "auto")
        return "direct" if v in ("auto", "canonical", "raw") else f"direct_{v}"
    if name == "mini-swe-agent" or ip.endswith(":MiniSweGatewayAgent"):
        return "mini_swe"
    return name or ip


def score_harness(answer: Optional[str], gold: dict) -> bool:
    """Host copy of the pilot/train verifier logic (adapters.harness_tasks.SCORE_PY)."""
    ans = (answer or "").strip()
    try:
        m = gold["mode"]
        if m == "exact":
            return ans.strip('"').strip("'") == gold["answer"]
        if m == "exact_ci":
            return ans.strip('"').strip("'").lower() == str(gold["answer"]).lower()
        if m == "int":
            return int(ans.strip('"')) == int(gold["answer"])
        if m == "json":
            try:
                a = json.loads(ans)
            except ValueError:          # Python repr of a dict (answer["content"] = {...}); same as SCORE_PY
                import ast
                a = ast.literal_eval(ans)
            if not isinstance(a, dict):
                return False
            g = gold["answer"]
            return set(a) == set(g) and all(
                abs(float(a[k]) - float(v)) <= 1e-3 if isinstance(v, (int, float)) and not isinstance(v, bool)
                else str(a[k]) == str(v) for k, v in g.items())
        if m == "nonempty":
            return bool(ans)
    except Exception:
        return False
    return False


def _session_usage(gw_calls: List[dict]) -> Dict[str, dict]:
    agg: Dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for c in gw_calls:
        if not str(c.get("trial", "")).startswith("session:"):
            continue
        a = agg[c["trial"]]
        a["n_calls"] += 1
        if c.get("status") == 200:
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "dna_positions"):
                a[k] += c.get(k, 0) or 0
            a["truncated_in_think"] += int(bool(c.get("truncated_in_think")))
        elif c.get("status") == 429:
            a["budget_exhausted"] = 1
    return agg


def load_gateway_calls(job_dir: Path) -> List[dict]:
    """Every gateway call logged for this job, de-duplicated by (trial, call_idx, ts).

    Authoritative: <root>/logs/gateway/<job>-<slurm id>/calls.jsonl for every allocation that ran the job
    (`harbor jobs resume` deletes non-trial subdirectories of the job dir, incl. copied gateway_logs/),
    plus allocations whose jobs.txt lists the job (multi-job allocations log under the FIRST job's
    name) and any copies inside the job dir (older runs)."""
    job_dir = Path(job_dir)
    gw_root = config.path("root") / "logs" / "gateway"
    files = [*job_dir.glob("gateway_logs/calls.jsonl"), *job_dir.glob("gateway_logs/*/calls.jsonl"),
             *gw_root.glob(f"{job_dir.name}-*/calls.jsonl")]
    for jt in gw_root.glob("*/jobs.txt"):
        if job_dir.name in jt.read_text().split():
            files.append(jt.parent / "calls.jsonl")
    seen, out = set(), []
    for f in sorted({f for f in files if f.exists()}):
        for c in _load_jsonl(f):
            k = (c.get("trial"), c.get("call_idx"), c.get("ts"))
            if k not in seen:
                seen.add(k)
                out.append(c)
    return out


def collect_job(job_dir: Path) -> pd.DataFrame:
    job_dir = Path(job_dir)
    gw_calls = load_gateway_calls(job_dir)
    sessions = _session_usage(gw_calls)
    child_dna: Dict[str, int] = defaultdict(int)
    for c in gw_calls:
        if c.get("route") == "child" and c.get("status") == 200 and c.get("handles_resolved", 0) > 0:
            child_dna[c["trial"]] += 1
    rows = []
    for rp in sorted(job_dir.glob("*/result.json")):
        tdir = rp.parent
        try:
            res = json.loads(rp.read_text())
        except Exception:
            continue
        cfg = res.get("config") or {}
        agent_cfg = cfg.get("agent") or {}
        task_cfg = cfg.get("task") or {}
        task_path = Path(task_cfg.get("path") or "")
        task_name = task_path.name or res.get("task_name", "").split("/")[-1]
        meta_dir = str(task_path.parent.parent / "_meta")
        idx = _index_for(meta_dir).get(task_name, {})
        answer_p = tdir / "agent" / "answer.txt"
        answer = answer_p.read_text().strip() if answer_p.exists() else None
        run_meta = json.loads((tdir / "agent" / "run_meta.json").read_text()) if (tdir / "agent" / "run_meta.json").exists() else {}
        usage = json.loads((tdir / "agent" / "gateway_usage.json").read_text()) if (tdir / "agent" / "gateway_usage.json").exists() else None
        if usage is None:
            # mini-swe-agent: the session id sent as X-Session-ID contains the trial name
            hits = [v for k, v in sessions.items() if res.get("trial_name", "\0") in k]
            usage = dict(hits[0]) if hits else {}
            usage.setdefault("n_calls", {})
        rewards = ((res.get("verifier_result") or {}).get("rewards")) or {}
        rec = {
            "job": job_dir.name, "trial_name": res.get("trial_name"), "task_name": task_name,
            "arm": arm_label(agent_cfg), "model": (agent_cfg.get("model_name") or "").split("/")[-1] or None,
            "agent_kwargs": json.dumps(agent_cfg.get("kwargs") or {}, sort_keys=True),
            "answer": answer, "valid_output": bool(answer),
            "verifier_correct": rewards.get("correct"),
            "exception": (res.get("exception_info") or {}).get("exception_type"),
            "status": run_meta.get("status"), "n_iterations": run_meta.get("n_iterations"),
            # REPL arms: answer came from rlms' out-of-turns fallback call (never submitted in-protocol)
            "fallback_answer": run_meta.get("fallback_answer"), "n_code_blocks": run_meta.get("n_code_blocks"),
            "prompt_version": run_meta.get("prompt_version"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"), "dna_positions": usage.get("dna_positions"),
            "child_calls": (usage.get("n_calls") or {}).get("child") if isinstance(usage.get("n_calls"), dict) else None,
            "tool_calls": (usage.get("n_calls") or {}).get("tool") if isinstance(usage.get("n_calls"), dict) else None,
            "blocked_child_calls": usage.get("blocked_child_calls"),
            "child_calls_with_dna": child_dna.get(usage.get("key", ""), 0),
            "truncated_in_think": usage.get("truncated_in_think"),
            "budget_exhausted": bool(usage.get("budget_exhausted")) or run_meta.get("status") == "budget_exhausted",
            "started_at": res.get("started_at"), "finished_at": res.get("finished_at"),
            **{k: idx.get(k) for k in ("episode_id", "row_id", "condition", "interface", "dataset",
                                       "label", "donor_row_id", "donor_label", "type", "split")},
        }
        if idx.get("label") is not None:                    # KEGG task: bio-mirage metric
            from dnaharness.data.labels import score_answer
            s = score_answer(answer, idx["label"], _labels()) if answer else None
            rec.update(predicted_label=s.predicted_label if s else None, correct=bool(s and s.correct),
                       mapped=bool(s and s.mapped))
            if idx.get("donor_label"):
                from dnaharness.data.labels import normalize, canonical_gold
                p = normalize(rec["predicted_label"] or "")
                rec["follows_donor"] = bool(p) and p == canonical_gold(idx["donor_label"])
                rec["follows_original"] = bool(p) and p == canonical_gold(idx["label"])
        elif idx.get("gold") is not None:                   # harness pilot / train task
            g = idx["gold"]
            rec["correct"] = score_harness(answer, g)
            if g.get("requires_child_call"):
                rec["correct_with_protocol"] = rec["correct"] and bool(rec["child_calls"])
            if g.get("max_turns"):
                rec["correct_with_protocol"] = rec["correct"] and (rec["n_iterations"] or 99) <= g["max_turns"]
            if g.get("requires_child_dna"):
                rec["correct_with_protocol"] = rec["correct"] and rec["child_calls_with_dna"] > 0
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty and "row_id" in df and df["row_id"].notna().any():
        man = pd.read_parquet(config.path("manifest"))[[
            "row_id", "split", "text_group_id", "pair_id", "gene", "group_majority_label",
            "is_minority_label", "in_cohort"]].rename(columns={"split": "orig_split"})
        df = df.merge(man, on="row_id", how="left")
    # attempt index within (task, arm, model) by start time -> seed id
    if not df.empty:
        df = df.sort_values("started_at")
        df["attempt"] = df.groupby(["task_name", "arm", "model"]).cumcount()
    return df


def gate_audit(job_dir: Path) -> bool:
    df = collect_job(job_dir)
    for rp in Path(job_dir).glob("*/verifier/prediction.json"):
        print(rp.read_text())
    ok = (not df.empty) and bool(df["verifier_correct"].fillna(0).astype(int).eq(1).all())
    print(f"[audit] {'PASSED' if ok else 'FAILED'}")
    return ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-audit", default=None)
    a = ap.parse_args(argv)
    if a.gate_audit:
        sys.exit(0 if gate_audit(Path(a.gate_audit)) else 1)
    df = pd.concat([collect_job(Path(j)) for j in a.jobs], ignore_index=True)
    out = Path(a.out or (config.path("root") / "results" / "trials.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} trials -> {out}")
    if not df.empty:
        print(df.groupby(["job", "arm", "model"], dropna=False)[["correct", "valid_output"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
