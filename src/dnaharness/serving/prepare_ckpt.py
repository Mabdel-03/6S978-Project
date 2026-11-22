"""Download pinned checkpoints into <ckpts>/<served_name> (patched where needed).

BioReason: the released config.json keeps rope_theta only under `rope_parameters` (transformers
5.1 layout); vLLM would silently fall back to rope_theta=10000 (bio-mirage `_assert_dna_checkpoint`).
We write a local dir with weights symlinked and config.json patched (top-level rope_theta).

    python -m dnaharness.serving.prepare_ckpt qwen3-1.7b bioreason-1.7b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dnaharness import config


def prepare(served: str) -> Path:
    from huggingface_hub import snapshot_download

    cfg = config.model_cfg(served)
    src = Path(snapshot_download(cfg["repo"], revision=cfg["revision"],
                                 cache_dir=str(config.path("hf_home") / "hub")))
    dst = config.path("ckpts") / served
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if f.name == "config.json" and cfg.get("patch_rope_theta"):
            c = json.loads(f.read_text())
            rp = c.get("rope_parameters") or {}
            if rp.get("rope_theta") is not None:
                c["rope_theta"] = rp["rope_theta"]
            if target.is_symlink() or target.exists():
                target.unlink()
            target.write_text(json.dumps(c, indent=2))
            continue
        if not target.exists():
            os.symlink(f.resolve(), target)
    (dst / "PROVENANCE.json").write_text(json.dumps(
        {"repo": cfg["repo"], "revision": cfg["revision"], "patched_rope_theta": bool(cfg.get("patch_rope_theta"))},
        indent=2))
    return dst


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args(argv)
    for m in a.models:
        print(m, "->", prepare(m))


if __name__ == "__main__":
    main()
