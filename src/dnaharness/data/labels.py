"""Authoritative (host-side) scoring: bio-mirage `input_use/metrics/dna.py`, imported unchanged.

A free-text answer is mapped onto the closed 37-label vocabulary (text after the last `Answer:`,
normalized, longest contained label first; falls back to the whole generation) and compared with the
canonical gold label. Unmapped answers count as wrong and are reported via `mapped=False`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from dnaharness import config


def _dna_metrics():
    config.ensure_bio_mirage_on_path()
    from input_use.metrics import dna
    return dna


@dataclass
class Score:
    predicted_label: Optional[str]
    correct: bool
    mapped: bool
    segment: str


def score_answer(answer_text: Optional[str], gold: str, labels: Iterable[str]) -> Score:
    dna = _dna_metrics()
    labels = list(labels)
    text = answer_text or ""
    # Agents return just the final answer; prefix "Answer:" when absent so extraction targets it
    # exactly as for native BioReason generations (which end "... Answer: <label>").
    gen = text if "Answer:" in text else f"Answer: {text}"
    pred, seg = dna.extract_predicted_label(gen, labels)
    correct = pred is not None and dna.normalize_label_text(pred) == dna.canonical_gold(gold)
    return Score(predicted_label=pred, correct=bool(correct), mapped=pred is not None, segment=seg)


def normalize(text: str) -> str:
    return _dna_metrics().normalize_label_text(text)


def canonical_gold(gold: str) -> str:
    return _dna_metrics().canonical_gold(gold)
