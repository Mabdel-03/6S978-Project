"""In-sandbox copy of bio-mirage `input_use/metrics/dna.py` label extraction (verbatim logic).

Used only by the convenience verifier (tests/score.py). Authoritative scores are recomputed on the
host from /logs/agent/answer.txt with the imported bio-mirage module (dnaharness.data.labels);
tests/test_scoring.py asserts this copy and bio-mirage agree.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple


def normalize_label_text(text) -> str:
    if text is None:
        return ""
    n = text.strip().lower().replace("\n", " ")
    n = re.sub(r"[^\w\s\-'/]", " ", n)
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def extract_predicted_label(generation: str, labels: Iterable[str]) -> Tuple[Optional[str], str]:
    full = generation or ""
    seg = full
    if "<|im_start|>assistant" in seg:
        seg = seg.rsplit("<|im_start|>assistant", 1)[-1]
    if "Answer:" in seg:
        seg = seg.rsplit("Answer:", 1)[-1]
    if "<|im_end|>" in seg:
        seg = seg.split("<|im_end|>", 1)[0]
    seg = seg.strip()
    na, nf = normalize_label_text(seg), normalize_label_text(full)
    norm = sorted(((normalize_label_text(l), l) for l in labels if normalize_label_text(l)),
                  key=lambda x: len(x[0]), reverse=True)
    for n, raw in norm:
        if n in na:
            return raw, seg
    for n, raw in norm:
        if n in nf:
            return raw, seg
    return None, seg


def canonical_gold(gold: str) -> str:
    return normalize_label_text((gold or "").split(";")[0].strip())


def score(answer_text: Optional[str], gold: str, labels) -> dict:
    text = answer_text or ""
    gen = text if "Answer:" in text else f"Answer: {text}"
    pred, seg = extract_predicted_label(gen, labels)
    correct = pred is not None and normalize_label_text(pred) == canonical_gold(gold)
    return {"predicted_label": pred, "correct": bool(correct), "mapped": pred is not None,
            "segment": seg}
