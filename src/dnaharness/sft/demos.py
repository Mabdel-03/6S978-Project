"""In-context demonstrations condition (Stage 3, no weight updates): k successful teacher trajectories
per harness rendered as compact transcripts, passed to RlmAgent via `demos_file`.

    python -m dnaharness.sft.demos --job <jobs>/sft_teacher_rlm --k 3 --out <sft_data>/demos_rlm.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnaharness.analysis.collect import collect_job
from dnaharness.sft.export import successes

MAX_BLOCK = 1500


def render(trial_dir: Path, question: str) -> str:
    logs = sorted((trial_dir / "agent").glob("rlm_*.jsonl"))
    parts = [f"### Task\n{question}\n"]
    for f in logs:
        for line in f.read_text().splitlines():
            it = json.loads(line)
            if it.get("type") != "iteration":
                continue
            parts.append(f"### Turn {it['iteration']}\nAssistant:\n{(it.get('response') or '')[:MAX_BLOCK]}")
            for cb in it.get("code_blocks") or []:
                out = ((cb.get("result") or {}).get("stdout") or "")[:MAX_BLOCK]
                parts.append(f"REPL output:\n{out}")
    return "\n\n".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    job = Path(a.job)
    ok = successes(collect_job(job))
    ok = ok.sort_values(["type", "n_iterations"]).drop_duplicates("type")      # diverse task types, short
    chosen = ok.head(a.k)
    blocks = []
    for _, r in chosen.iterrows():
        cfg = json.loads((job / r["trial_name"] / "result.json").read_text())["config"]
        inputs = json.loads((Path(cfg["task"]["path"]) / "environment" / "inputs.json").read_text())
        blocks.append(render(job / r["trial_name"], inputs["question"]))
    Path(a.out).write_text("\n\n---\n\n".join(blocks))
    print(f"{len(blocks)} demos -> {a.out}")


if __name__ == "__main__":
    main()
