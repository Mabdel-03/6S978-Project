"""Deterministic sequence-inspection tools exposed to agents (identical across checkpoints/harnesses).

These only transform the supplied evidence; none of them encodes disease knowledge or a learned
predictor (that would change the claim from "the LM used the DNA" to "the LM orchestrated a
classifier"). Positions are 0-based offsets *within the supplied window*, not genomic coordinates.
"""
from __future__ import annotations

import difflib
from typing import Dict, List

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_CODON = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        ((x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    )
}


def find_differences(reference: str, variant: str) -> Dict[str, object]:
    """Locate where `variant` differs from `reference` (indel-safe).

    Returns start (first differing offset), ref_end/var_end (exclusive ends of the differing
    segment in each sequence), the differing segments, a coarse type (identical, SNV, MNV,
    insertion, deletion, complex) and the substitution offsets when lengths are equal.
    """
    if reference == variant:
        return {"type": "identical", "start": None, "ref_segment": "", "var_segment": "",
                "ref_end": None, "var_end": None, "substitution_offsets": []}
    n = min(len(reference), len(variant))
    i = 0
    while i < n and reference[i] == variant[i]:
        i += 1
    j = 0
    while j < n - i and reference[len(reference) - 1 - j] == variant[len(variant) - 1 - j]:
        j += 1
    ref_end, var_end = len(reference) - j, len(variant) - j
    rs, vs = reference[i:ref_end], variant[i:var_end]
    if len(reference) == len(variant):
        subs = [k for k in range(len(reference)) if reference[k] != variant[k]]
        kind = "SNV" if len(subs) == 1 else "MNV"
    else:
        subs = []
        kind = "insertion" if not rs else "deletion" if not vs else "complex"
    return {"type": kind, "start": i, "ref_end": ref_end, "var_end": var_end,
            "ref_segment": rs, "var_segment": vs, "substitution_offsets": subs[:50],
            "n_substitutions": len(subs), "ref_length": len(reference), "var_length": len(variant)}


def show_alignment(reference: str, variant: str, flank: int = 20) -> str:
    """Human-readable alignment of the differing region with `flank` bases of context."""
    d = find_differences(reference, variant)
    if d["type"] == "identical":
        return "sequences are identical"
    lo = max(0, d["start"] - flank)
    r = reference[lo:d["ref_end"] + flank]
    v = variant[lo:d["var_end"] + flank]
    sm = difflib.SequenceMatcher(a=r, b=v, autojunk=False)
    top, mid, bot = [], [], []
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        a, b = r[a0:a1], v[b0:b1]
        w = max(len(a), len(b))
        top.append(a.ljust(w, "-"))
        bot.append(b.ljust(w, "-"))
        mid.append(("|" * w) if tag == "equal" else ("*" * w))
    return (f"window offset {lo}..\nref {''.join(top)}\n    {''.join(mid)}\nvar {''.join(bot)}")


def subsequence(seq: str, start: int, end: int) -> str:
    """seq[start:end] with bounds clipped."""
    return seq[max(0, start):max(0, end)]


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def gc_content(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(1, len(s))


def translate(seq: str, frame: int = 0) -> str:
    """Standard-code translation of seq[frame:] ('X' for codons with non-ACGT bases)."""
    s = seq.upper()[frame:]
    return "".join(_CODON.get(s[k:k + 3], "X") for k in range(0, len(s) - 2, 3))


def find_motif(seq: str, motif: str, max_hits: int = 100) -> List[int]:
    """Offsets of (possibly overlapping) exact occurrences of motif."""
    hits, k = [], seq.find(motif)
    while k != -1 and len(hits) < max_hits:
        hits.append(k)
        k = seq.find(motif, k + 1)
    return hits


TOOLS = {
    "find_differences": find_differences,
    "show_alignment": show_alignment,
    "subsequence": subsequence,
    "reverse_complement": reverse_complement,
    "gc_content": gc_content,
    "translate": translate,
    "find_motif": find_motif,
}

DESCRIPTIONS = {
    "find_differences": "find_differences(reference, variant) -> dict: where/how the variant differs (indel-safe; 0-based offsets within the window)",
    "show_alignment": "show_alignment(reference, variant, flank=20) -> str: printable alignment of the differing region",
    "subsequence": "subsequence(seq, start, end) -> str: clipped slice",
    "reverse_complement": "reverse_complement(seq) -> str",
    "gc_content": "gc_content(seq) -> float",
    "translate": "translate(seq, frame=0) -> str: standard genetic code (reading frame unknown!)",
    "find_motif": "find_motif(seq, motif) -> list[int]: offsets of exact matches",
}
