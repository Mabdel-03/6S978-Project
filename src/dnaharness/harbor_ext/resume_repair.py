"""Make jobs rendered before 2026-11-22 resumable.

Harbor scrubs the agent's env values (OPENAI_API_KEY=sandbox for mini-swe) from every file it saves for
a trial, including the trial's config.json. Jobs rendered before the fix in jobs.py also carried that
same dummy value in the mini-swe native config (model.model_kwargs.api_key), so each stored mini-swe
trial config holds "[REDACTED]" where the job plan has "sandbox", and `harbor jobs resume` refuses:
"Existing trial config does not match planned job config."

This restores, in place, every "[REDACTED]" leaf of a trial's agent config from the same path in the
job's own config.json (also the agent env values, which a RESUMED run stores scrubbed) -- nothing else. The value is a dummy (the gateway ignores it), not a secret.
Idempotent; serve_and_run.sbatch runs it before every resume. New renders don't need it.

    python -m dnaharness.harbor_ext.resume_repair <job dir> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REDACTED = "[REDACTED]"


def _restore(trial_node, job_node) -> int:
    """Copy job-config values into `trial_node` wherever the trial holds the literal "[REDACTED]"."""
    n = 0
    if isinstance(trial_node, dict) and isinstance(job_node, dict):
        for k, v in trial_node.items():
            if v == REDACTED and k in job_node and job_node[k] != REDACTED:
                trial_node[k] = job_node[k]
                n += 1
            else:
                n += _restore(v, job_node.get(k))
    return n


def repair(job_dir: Path) -> int:
    """Number of trial configs changed. Only "[REDACTED]" leaves of a trial's `agent` are touched, and
    only where the job's own agent config has a value at the same path (env values, the old mini-swe
    model_kwargs.api_key)."""
    job_dir = Path(job_dir)
    cfg_p = job_dir / "config.json"
    if not cfg_p.exists():
        return 0
    agents = {(a.get("import_path"), a.get("name")): a for a in json.loads(cfg_p.read_text()).get("agents", [])}
    n = 0
    for tp in job_dir.glob("*/config.json"):
        text = tp.read_text()
        if REDACTED not in text:
            continue
        t = json.loads(text)
        a = t.get("agent") or {}
        ja = agents.get((a.get("import_path"), a.get("name")))
        if ja is not None and _restore(a, ja):
            tp.write_text(json.dumps(t, indent=4))
            n += 1
    return n


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(f"[resume-repair] {Path(d).name}: restored {repair(Path(d))} trial configs")
