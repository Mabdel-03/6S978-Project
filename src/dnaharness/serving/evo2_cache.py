"""Precompute frozen Evo2-1B embeddings (layer blocks.20.mlp.l3, pre-projection) for every distinct
DNA window referenced by generated tasks. Runs ONCE inside evo2.sif on a GPU node.

Faithful to BioReason's own encoder path (bio-posttrain BioReason):
  * BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1 + import hook blocks Transformer Engine, so vortex uses
    PyTorch Linear and FP8 input projections are disabled -- the setting the trained projection
    saw (bio-mirage refuses to run without it: "the run is silently invalid").
  * `bioreason.utils.evo2_compat.load_evo2`, `Evo2Tokenizer` (char-level, truncation to 2048,
    no special tokens) and `run_evo2_embeddings` (autocast disabled), model cast to bf16.
  * one sequence per forward pass (bio-mirage DNARowCache), avoiding the batched left-pad bug.

Output: <evo2_cache>/<sha256(seq)>.pt  bf16 tensor [L, 1920] (L = min(len(seq), 2048));
        <evo2_cache>/_stats.json        counts + GPU-seconds (compute accounting)

    python -m dnaharness.serving.evo2_cache --meta-dirs <datasets>/kegg-dx/_meta
"""
from __future__ import annotations

import os

os.environ.setdefault("BIOREASON_USE_VORTEX_PYTORCH_LINEAR", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


def iter_sequences(meta_dirs):
    seen = set()
    for d in meta_dirs:
        p = Path(d) / "sequences.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            if rec["sha"] not in seen:
                seen.add(rec["sha"])
                assert hashlib.sha256(rec["seq"].encode()).hexdigest() == rec["sha"]
                yield rec["sha"], rec["seq"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dirs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bioreason-repo", default=None, help="path to bio-posttrain/BioReason")
    ap.add_argument("--layer", default="blocks.20.mlp.l3")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--model-name", default="evo2_1b_base")
    a = ap.parse_args(argv)

    repo = a.bioreason_repo or str(Path(__file__).resolve().parents[3] / "third_party" / "bio-posttrain" / "BioReason")
    sys.path.insert(0, repo)
    from bioreason.utils.force_vortex_pytorch_linear import install_import_hook
    install_import_hook()                               # must precede `import torch`

    import torch
    if not torch.cuda.is_available():   # fail before vortex's import-time Triton autotuners do
        raise SystemExit("[evo2-cache] torch sees no CUDA device (run with --nv on a GPU node; CUDA error 803 "
                         "means a container compat libcuda is shadowing the host driver)")
    from bioreason.models.evo2_tokenizer import Evo2Tokenizer
    from bioreason.utils.evo2_compat import load_evo2
    from bioreason.utils.evo2_inference import run_evo2_embeddings

    if a.out is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from dnaharness import config
        out = config.path("evo2_cache")
    else:
        out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    todo = [(sha, s) for sha, s in iter_sequences(a.meta_dirs) if not (out / f"{sha}.pt").exists()]
    print(f"[evo2-cache] {len(todo)} sequences to encode -> {out}", flush=True)
    if not todo:
        return
    device = torch.device("cuda")
    evo2 = load_evo2(a.model_name)
    evo2.model = evo2.model.to(device, dtype=torch.bfloat16).eval()
    tok = Evo2Tokenizer(evo2.tokenizer)
    torch.cuda.synchronize()
    t0 = time.time()
    for k, (sha, seq) in enumerate(todo):
        enc = tok([seq], padding=True, truncation=True, max_length=a.max_length,
                  return_tensors="pt", return_attention_mask=True, add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        valid = int(enc["attention_mask"][0].sum())
        with torch.no_grad():
            emb = run_evo2_embeddings(evo2, ids, [a.layer])[a.layer].squeeze(0)   # (L, 1920)
        rows = emb[:valid].to(torch.bfloat16).cpu().contiguous()
        assert rows.shape == (min(len(seq), a.max_length), 1920), rows.shape
        torch.save(rows, out / f"{sha}.pt")
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(todo)}", flush=True)
    torch.cuda.synchronize()
    gpu_s = time.time() - t0
    stats_p = out / "_stats.json"
    stats = json.loads(stats_p.read_text()) if stats_p.exists() else {"runs": []}
    stats["runs"].append({"n": len(todo), "gpu_seconds": round(gpu_s, 2), "gpu": torch.cuda.get_device_name(),
                          "layer": a.layer, "vortex_pytorch_linear": os.environ["BIOREASON_USE_VORTEX_PYTORCH_LINEAR"],
                          "ts": time.time()})
    stats_p.write_text(json.dumps(stats, indent=2))
    print(f"[evo2-cache] done: {len(todo)} in {gpu_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
