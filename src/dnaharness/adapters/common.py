"""Shared helpers for writing Harbor task directories."""
from __future__ import annotations

import hashlib
import hmac
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from dnaharness import config

TASK_ORG = "dnaharness"

SANDBOX_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    # Dead proxy: blocks outbound HTTP from model-generated code (pip, HF downloads, ...). The
    # gateway on 127.0.0.1 stays reachable via NO_PROXY.
    "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
    "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9",
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    "PYTHONPATH": "/opt/dnah",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1",
}


def hmac_hex(key: str, msg: str, n: int = 16) -> str:
    return hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()[:n]


def seq_sha(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string literals are valid TOML basic strings


def write_task_toml(task_dir: Path, *, name: str, description: str, metadata: Dict[str, object],
                    sif: str, agent_timeout_sec: float = 1800.0, verifier_timeout_sec: float = 120.0,
                    memory_mb: int = 3072, cpus: int = 1, extra_env: Optional[Dict[str, str]] = None) -> None:
    env = {**SANDBOX_ENV, **(extra_env or {})}
    lines = [
        'schema_version = "1.4"', "",
        "[task]", f"name = {toml_str(f'{TASK_ORG}/{name}')}", f"description = {toml_str(description)}", "",
        "[metadata]",
        *[f"{k} = {toml_str(str(v))}" for k, v in metadata.items()], "",
        "[agent]", f"timeout_sec = {float(agent_timeout_sec)}", "",
        "[verifier]", f"timeout_sec = {float(verifier_timeout_sec)}", 'environment_mode = "shared"', "",
        "[environment]", f"docker_image = {toml_str(sif)}", 'workdir = "/task"',
        f"memory_mb = {int(memory_mb)}", f"cpus = {int(cpus)}", 'network_mode = "public"', "",
        "[environment.env]",
        *[f"{k} = {toml_str(v)}" for k, v in env.items()], "",
    ]
    (task_dir / "task.toml").write_text("\n".join(lines))


TEST_SH = """#!/bin/bash
# Convenience verifier. Authoritative scores are recomputed on the host (dnaharness.analysis.collect).
exec /usr/bin/python3 /tests/score.py
"""


def write_verifier(task_dir: Path, score_py: str, gold: Dict[str, object]) -> None:
    tests = task_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test.sh").write_text(TEST_SH)
    (tests / "test.sh").chmod(0o755)
    (tests / "score.py").write_text(score_py)
    (tests / "gold.json").write_text(json.dumps(gold))


def write_solution(task_dir: Path, answer: str) -> None:
    sol = task_dir / "solution"
    sol.mkdir(parents=True, exist_ok=True)
    (sol / "solve.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/agent\ncat > /logs/agent/answer.txt <<'EOF_ANSWER'\n"
        f"{answer}\nEOF_ANSWER\n")
    (sol / "solve.sh").chmod(0o755)


def reset_dir(p: Path, overwrite: bool) -> bool:
    """Returns False when the dir exists and overwrite is off (skip)."""
    if p.exists():
        if not overwrite:
            return False
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return True


def sif_path() -> str:
    return str(config.path("sandbox_sif"))
