"""Repo paths, configs/paths.yaml + configs/models.yaml loading, and secrets.

`${key}` references inside paths.yaml are expanded against the same file; every top-level key can
be overridden with an environment variable DNAH_<KEY_UPPER> (e.g. DNAH_ROOT=/elsewhere).
"""
from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
THIRD_PARTY = REPO_ROOT / "third_party"
BIO_MIRAGE = THIRD_PARTY / "bio-mirage"
BIO_POSTTRAIN_BIOREASON = THIRD_PARTY / "bio-posttrain" / "BioReason"

_REF = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


def _expand(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    for k in list(out):
        env = os.environ.get(f"DNAH_{k.upper()}")
        if env is not None:
            out[k] = env
    for _ in range(5):  # resolve nested references
        changed = False
        for k, v in out.items():
            if isinstance(v, str) and _REF.search(v):
                nv = _REF.sub(lambda m: str(out[m.group(1)]), v)
                changed |= nv != v
                out[k] = nv
        if not changed:
            break
    return out


def port_offset() -> int:
    """Per-allocation port offset (slurm/serve_and_run.sbatch claims a free node-local slot), so several
    allocations on one node never share a gateway / vLLM port -- sandboxes use the host network."""
    return int(os.environ.get("DNAH_PORT_OFFSET", "0") or 0)


@lru_cache(maxsize=None)
def paths() -> dict[str, Any]:
    p = _expand(yaml.safe_load((CONFIG_DIR / "paths.yaml").read_text()))
    off = port_offset()
    p["gateway_port"] = int(p["gateway_port"]) + off
    p["vllm_ports"] = {k: int(v) + off for k, v in p["vllm_ports"].items()}
    return p


def path(key: str) -> Path:
    return Path(paths()[key])


@lru_cache(maxsize=None)
def models() -> dict[str, Any]:
    """configs/models.yaml, extended with locally produced checkpoints (<ckpts>/registry.yaml,
    written by dnaharness.sft.merge for Stage-3 SFT models)."""
    m = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())
    reg = path("ckpts") / "registry.yaml"
    if reg.exists():
        extra = yaml.safe_load(reg.read_text()) or {}
        m["models"].update(extra.get("models", {}))
    return m


def model_port(served_name: str) -> int:
    cfg = model_cfg(served_name)
    if cfg.get("port"):
        return int(cfg["port"]) + port_offset()
    return int(paths()["vllm_ports"][cfg["vllm_port_key"]])


def model_cfg(served_name: str) -> dict[str, Any]:
    try:
        return models()["models"][served_name]
    except KeyError as e:
        raise KeyError(f"unknown served model {served_name!r}; see configs/models.yaml") from e


def harness_cfg(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "harness" / f"{name}.yaml").read_text())


def ensure_bio_mirage_on_path() -> None:
    """bio-mirage is vendored as a submodule and imported for cohort/perturbation/scoring logic.

    We import individual pure modules rather than installing the package (its pins are heavy).
    """
    p = str(BIO_MIRAGE)
    if not (BIO_MIRAGE / "input_use").is_dir():
        raise RuntimeError(f"bio-mirage submodule missing at {BIO_MIRAGE}; run `git submodule update --init`")
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_bioreason_on_path() -> None:
    p = str(BIO_POSTTRAIN_BIOREASON)
    if not (BIO_POSTTRAIN_BIOREASON / "bioreason").is_dir():
        raise RuntimeError(f"bio-posttrain submodule missing at {BIO_POSTTRAIN_BIOREASON}")
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------------------- secrets


def load_secrets(create: bool = False) -> dict[str, str]:
    """HMAC secret for DNA handles + API keys. Lives under the storage root, never in git."""
    p = path("secrets")
    if p.exists():
        return json.loads(p.read_text())
    if not create:
        raise FileNotFoundError(f"{p} missing; run `python -m dnaharness.cli secrets`")
    data = {
        "handle_hmac_key": _secrets.token_hex(32),
        "vllm_api_key": _secrets.token_hex(24),
        "gateway_admin_key": _secrets.token_hex(24),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    os.chmod(p, 0o600)
    return data
