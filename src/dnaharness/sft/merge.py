"""Merge a LoRA adapter into its base checkpoint -> <ckpts>/<name>, registered for serving.

Carries over everything the native DNA interface needs UNCHANGED (dna_projection.pt,
dna_model_config.json, tokenizer incl. DNA special tokens, chat template) and the rope_theta patch,
then asserts the merged input-embedding matrix is bit-identical to the base's (LoRA never touched it),
so the gateway's native prompt_embeds construction is unchanged for SFT students.

    python -m dnaharness.sft.merge --base bioreason-1.7b --adapter <ckpts>/adapters/bioreason-rlm \
        --name bioreason-1.7b-sft-rlm --port 8011
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from dnaharness import config

CARRY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
         "generation_config.json", "dna_projection.pt", "dna_model_config.json", "special_tokens_map.json",
         "added_tokens.json"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_dir = config.path("ckpts") / a.base
    out = config.path("ckpts") / a.name
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16)
    emb_before = model.get_input_embeddings().weight.detach().clone()
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    assert torch.equal(emb_before, model.get_input_embeddings().weight), "LoRA changed input embeddings"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    for f in CARRY:
        src = base_dir / f
        if src.exists():
            shutil.copy2(src.resolve(), out / f)
    base_cfg = json.loads((base_dir / "config.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    for k in ("rope_theta", "rope_parameters", "rope_scaling"):
        if k in base_cfg:
            cfg[k] = base_cfg[k]
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    base_entry = config.model_cfg(a.base)
    adapter_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(a.adapter).glob("*.safetensors")))).hexdigest()[:16]
    entry = {k: v for k, v in base_entry.items() if k in ("native_dna", "template", "dna_projection", "dna_pad_id")}
    entry.update({"repo": f"local:{a.adapter}", "revision": adapter_hash, "base": a.base, "port": a.port,
                  "vllm_port_key": "extra"})
    reg_p = config.path("ckpts") / "registry.yaml"
    reg = yaml.safe_load(reg_p.read_text()) if reg_p.exists() else {"models": {}}
    reg["models"][a.name] = entry
    reg_p.write_text(yaml.safe_dump(reg, sort_keys=True))
    print(f"merged -> {out}; registered {a.name} on port {a.port}")


if __name__ == "__main__":
    main()
