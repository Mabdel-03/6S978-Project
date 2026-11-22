"""LoRA SFT of the ROOT model on exported samples (prompt/completion conversational format; loss on
the completion only). Identical hyperparameters and optimizer steps for every Stage-3 condition.

LoRA targets attention + MLP projections only: embeddings / lm_head (and therefore the DNA special-
token rows) are untouched, and Evo2 + dna_projection.pt are not part of this model at all -- the
native DNA interface stays frozen.

Text-only rendering uses the official Qwen3 chat template for both students (BioReason's
CHAT_TEMPLATE renders text-only conversations identically; tests/test_templates.py checks this).

    python -m dnaharness.sft.train_lora --base bioreason-1.7b --data <sft_data>/rlm.jsonl --out <ckpts>/adapters/bioreason-rlm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnaharness import config

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="served name under <ckpts>/ (student starting checkpoint)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_dir = config.path("ckpts") / a.base
    tok = AutoTokenizer.from_pretrained(str(config.path("ckpts") / "qwen3-1.7b"))   # official template
    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                                 attn_implementation="sdpa")
    ds = load_dataset("json", data_files=a.data, split="train").select_columns(["prompt", "completion"])
    peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=a.out, max_steps=a.max_steps, per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * a.max_steps)), bf16=use_bf16, gradient_checkpointing=True, logging_steps=5, save_strategy="no",
        max_length=a.max_length, completion_only_loss=True, report_to="none", seed=a.seed,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.out)
    Path(a.out, "dnah_train.json").write_text(json.dumps({**vars(a), "lora_targets": LORA_TARGETS,
                                                          "n_samples": len(ds)}, indent=2))


if __name__ == "__main__":
    main()
