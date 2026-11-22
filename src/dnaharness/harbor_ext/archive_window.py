"""Discard (archive) every trial of a job that FINISHED inside a time window, so the next
`harbor jobs resume` re-runs those trial configs.

Used for the 2026-11-21 sandbox-leak incident: between 23:30 UTC and the cancellation (~00:21 UTC)
the leaked sandboxes collapsed node throughput (~450 trials / 30 min vs ~3,000) and agent timeouts
rose from 0.1% to 4-13%. Trials are selected by finish time ONLY -- whatever their outcome -- so
the re-run does not give failed trials a second chance while keeping successes.

Trial dirs are MOVED (never deleted) to <root>/jobs/_archive/<tag>/<job>/ -- outside the job dir,
because `harbor jobs resume` deletes unknown subdirectories of a job dir -- with a manifest.jsonl.
Do not run on a job whose harbor process is still running.

    python -m dnaharness.harbor_ext.archive_window --start 2026-11-21T23:30 --end 2026-11-22T00:25 \
        --tag degraded-20261121 [--dry-run] <job dir> [...]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dnaharness import config


def archive(job_dir: Path, start: str, end: str, tag: str, dry_run: bool = False) -> int:
    job_dir = Path(job_dir)
    dest = config.path("root") / "jobs" / "_archive" / tag / job_dir.name
    moved = []
    for rp in sorted(job_dir.glob("*/result.json")):
        try:
            r = json.loads(rp.read_text())
        except (OSError, ValueError):
            continue
        t = (r.get("finished_at") or "")[:19]
        if t and start <= t < end:
            exc = (r.get("exception_info") or {}).get("exception_type")
            moved.append({"trial": rp.parent.name, "finished_at": t, "exception": exc})
            if not dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rp.parent), str(dest / rp.parent.name))
    if moved and not dry_run:
        with open(dest / "manifest.jsonl", "a") as fh:
            for m in moved:
                fh.write(json.dumps({**m, "window": [start, end], "job": job_dir.name}) + "\n")
    return len(moved)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--start", required=True, help="UTC, ISO (inclusive)")
    ap.add_argument("--end", required=True, help="UTC, ISO (exclusive)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    for j in a.jobs:
        n = archive(Path(j), a.start, a.end, a.tag, a.dry_run)
        print(f"[archive] {Path(j).name}: {'would move' if a.dry_run else 'moved'} {n} trials")


if __name__ == "__main__":
    main()
