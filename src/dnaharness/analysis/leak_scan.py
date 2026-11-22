"""Scan agent trajectories for label-leakage / isolation-bypass attempts.

Flags (per trial): URLs, pip/uv installs, huggingface / datasets / kegg mentions, reads of /staging,
/tests or /logs/verifier, and localhost ports other than the gateway. Scans everything the model
produced or executed that Harbor saved under <trial>/agent/ (rlms JSONL logs, mini-swe trajectory,
driver.log, responses).

    python -m dnaharness.analysis.leak_scan <jobs>/exp_a_shared [...]     # exit 1 if anything flagged
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dnaharness import config

GW_PORT = str(config.paths()["gateway_port"])
PATTERNS = {
    "url": re.compile(r"https?://(?!127\.0\.0\.1|localhost)[^\s'\"<>]+", re.I),
    "pip_install": re.compile(r"\b(pip3?|uv)\s+(install|add)\b", re.I),
    "hf": re.compile(r"huggingface|hf_hub|load_dataset|snapshot_download|wanglab", re.I),
    "kegg_dataset": re.compile(r"\bkegg\b.{0,40}(dataset|parquet|download)", re.I),
    "staging": re.compile(r"/staging\b"),
    "tests_dir": re.compile(r"(?<![\w/])/tests\b|/logs/verifier"),
    "other_local_port": re.compile(r"(?:127\.0\.0\.1|localhost):(\d{2,5})"),
    "raw_socket": re.compile(r"\bsocket\.(socket|create_connection)\b|ProxyHandler\(\s*\{\s*\}\s*\)|proxies\s*=\s*\{\s*\}"),
}


def scan_text(text: str) -> dict:
    hits = {}
    for k, rx in PATTERNS.items():
        found = rx.findall(text)
        if k == "other_local_port":
            found = [p for p in found if p != GW_PORT]
        if found:
            hits[k] = sorted(set(map(str, found)))[:10]
    return hits


def model_text(tdir: Path) -> str:
    """Only model-authored content (responses + executed code), not our own prompts/instructions."""
    chunks = []
    for f in (tdir / "agent").glob("rlm_*.jsonl"):
        for line in f.read_text(errors="ignore").splitlines():
            try:
                it = json.loads(line)
            except Exception:
                continue
            chunks.append(str(it.get("response", "")))
            for cb in it.get("code_blocks", []) or []:
                chunks.append(str(cb.get("code", "")))
    traj = tdir / "agent" / "mini-swe-agent.trajectory.json"
    if traj.exists():
        try:
            for m in json.loads(traj.read_text()).get("messages", []):
                if m.get("role") == "assistant":
                    chunks.append(str(m.get("content", "")))
        except Exception:
            pass
    return "\n".join(chunks)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    flagged = {}
    n = 0
    for j in a.jobs:
        for rp in Path(j).glob("*/result.json"):
            n += 1
            hits = scan_text(model_text(rp.parent))
            if hits:
                flagged[str(rp.parent)] = hits
    print(f"[leak-scan] {len(flagged)} / {n} trials flagged")
    for k, v in list(flagged.items())[:50]:
        print(" ", k, json.dumps(v))
    if a.out:
        Path(a.out).write_text(json.dumps(flagged, indent=2))
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
