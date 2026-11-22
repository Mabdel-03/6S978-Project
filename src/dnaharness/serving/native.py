"""Prompt rendering for the gateway: chat messages -> token ids, or -> prompt_embeds for native DNA.

All models are rendered here (not by vLLM's chat endpoint) so every arm shares one code path for
templating, think-splitting and token accounting.

Native BioReason construction (mirrors bio-mirage `runner.build_prompt_embeds` / BioReason
`DLProcessor`): render with BioReason's CHAT_TEMPLATE where each DNA part becomes
`<|dna_start|><|dna_pad|><|dna_end|>`; tokenize without special tokens; expand each `<|dna_pad|>` to
one slot per Evo2 row; embed text tokens with the checkpoint's `model.embed_tokens.weight` (bf16);
overwrite pad slots with `dna_projection(evo2_rows)` computed in bf16, one sequence at a time.

Handle policy: DNA handles are resolved only in (a) the first user message of a root call and
(b) every message of child/tool/direct calls; anywhere else they are replaced by an inert
"[DNA handle]" so DNA is never re-injected through REPL output echoes.
"""
from __future__ import annotations

import base64
import io
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HANDLE_RE = re.compile(r"<DNA:[0-9a-f]{16}>")
NEUTRAL = "[DNA handle]"
MAX_HANDLES_PER_CALL = 4


class HandleError(ValueError):
    pass


# --------------------------------------------------------------------------- message utilities

def flatten_content(content: Any) -> str:
    """OpenAI content (str or list of parts) -> plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
            parts.append(p.get("text", ""))
        elif isinstance(p, str):
            parts.append(p)
    return "".join(parts)


def first_user_index(messages: List[dict]) -> Optional[int]:
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            return i
    return None


@dataclass
class Resolved:
    """Messages after handle policy: each message content is a list of ("text", str) | ("dna", sha)."""
    messages: List[dict]
    dna_shas: List[str] = field(default_factory=list)
    n_resolved: int = 0
    n_neutralized: int = 0


def apply_handle_policy(messages: List[dict], route: str, native: bool, handle_map: Dict[str, str]) -> Resolved:
    allowed_idx = set()
    if native:
        if route == "root":
            fu = first_user_index(messages)
            if fu is not None:
                allowed_idx.add(fu)
        else:  # child / tool / direct: the whole (short) prompt is under our control
            allowed_idx = {i for i, m in enumerate(messages) if m.get("role") != "assistant"}
    out, shas, n_res, n_neu = [], [], 0, 0
    for i, m in enumerate(messages):
        text = flatten_content(m.get("content"))
        parts: List[Tuple[str, str]] = []
        pos = 0
        for mt in HANDLE_RE.finditer(text):
            tok = mt.group(0)
            if i in allowed_idx:
                sha = handle_map.get(tok)
                if sha is None:
                    raise HandleError(f"unknown DNA handle {tok}")
                if mt.start() > pos:
                    parts.append(("text", text[pos:mt.start()]))
                parts.append(("dna", sha))
                shas.append(sha)
                n_res += 1
            else:
                if mt.start() > pos:
                    parts.append(("text", text[pos:mt.start()]))
                parts.append(("text", NEUTRAL))
                n_neu += 1
            pos = mt.end()
        if pos < len(text):
            parts.append(("text", text[pos:]))
        out.append({"role": m.get("role", "user"), "parts": parts})
    if n_res > MAX_HANDLES_PER_CALL:
        raise HandleError(f"{n_res} DNA handles in one call (max {MAX_HANDLES_PER_CALL})")
    return Resolved(out, shas, n_res, n_neu)


def split_thinking(text: str) -> Tuple[str, str, bool]:
    """-> (reasoning, content, truncated_in_think). Qwen3 emits '<think>..</think>\\n\\ncontent'."""
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        return pre.replace("<think>", "", 1).strip("\n"), post.lstrip("\n"), False
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip("\n"), "", True
    return "", text, False


def tensor_to_b64(t) -> str:
    import torch
    buf = io.BytesIO()
    torch.save(t.contiguous().cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- renderers

class TextRenderer:
    """Any HF chat model (Qwen3 official template)."""

    native = False

    def __init__(self, ckpt_dir: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir)
        self.stop_token_ids = [i for i in {self.tok.convert_tokens_to_ids("<|im_end|>"),
                                           self.tok.convert_tokens_to_ids("<|endoftext|>")} if i is not None]

    def to_hf_messages(self, res: Resolved) -> List[dict]:
        return [{"role": m["role"], "content": "".join(t for k, t in m["parts"] if k == "text")}
                for m in res.messages]

    def render_ids(self, res: Resolved, enable_thinking: bool = True) -> List[int]:
        text = self.tok.apply_chat_template(self.to_hf_messages(res), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=enable_thinking)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def build(self, res: Resolved, enable_thinking: bool = True) -> dict:
        if res.dna_shas:
            raise HandleError("this model has no native DNA interface")
        ids = self.render_ids(res, enable_thinking)
        return {"prompt": ids, "n_prompt": len(ids), "n_dna": 0}

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if text else 0


class BioReasonRenderer(TextRenderer):
    """BioReason: CHAT_TEMPLATE + Evo2->projection prompt_embeds."""

    native = True

    def __init__(self, ckpt_dir: str, evo2_cache_dir: str, device: str = "cuda", cache_items: int = 512):
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer

        from dnaharness import config
        config.ensure_bioreason_on_path()
        from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir)
        self.tok.chat_template = CHAT_TEMPLATE
        self.tok.add_special_tokens({"additional_special_tokens": ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]})
        self.dna_pad_id = self.tok.convert_tokens_to_ids("<|dna_pad|>")
        self.stop_token_ids = [i for i in {self.tok.convert_tokens_to_ids("<|im_end|>"),
                                           self.tok.convert_tokens_to_ids("<|endoftext|>")} if i is not None]
        ck = Path(ckpt_dir)
        files = [ck / "model.safetensors"] if (ck / "model.safetensors").exists() else sorted(ck.glob("model-*.safetensors"))
        w = None
        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as fh:
                if "model.embed_tokens.weight" in fh.keys():
                    w = fh.get_tensor("model.embed_tokens.weight")
                    break
        if w is None:
            raise RuntimeError(f"model.embed_tokens.weight not found in {ckpt_dir}")
        self.embed = w.to(torch.bfloat16)                       # CPU; index-select is exact
        state = torch.load(ck / "dna_projection.pt", map_location="cpu")
        din = state["weight"].shape[1]
        dout = state["weight"].shape[0]
        self.proj = torch.nn.Linear(din, dout)
        self.proj.load_state_dict(state, strict=True)
        self.proj = self.proj.to(self.device, dtype=torch.bfloat16).eval()   # as BioReason: bf16
        self.evo2_dir = Path(evo2_cache_dir)
        self._rows: "OrderedDict[str, Any]" = OrderedDict()
        self._cap = cache_items

    # -- DNA rows ---------------------------------------------------------------------------
    def projected_rows(self, sha: str):
        if sha in self._rows:
            self._rows.move_to_end(sha)
            return self._rows[sha]
        p = self.evo2_dir / f"{sha}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Evo2 embedding missing from cache: {p} (run serving.evo2_cache)")
        h = self.torch.load(p, map_location="cpu")
        with self.torch.no_grad():
            rows = self.proj(h.to(self.device, dtype=self.torch.bfloat16)).to("cpu")
        self._rows[sha] = rows
        if len(self._rows) > self._cap:
            self._rows.popitem(last=False)
        return rows

    # -- rendering --------------------------------------------------------------------------
    def to_hf_messages(self, res: Resolved) -> List[dict]:
        msgs = []
        for m in res.messages:
            content = []
            for kind, val in m["parts"]:
                if kind == "dna":
                    content.append({"type": "dna", "text": None})
                elif val:
                    content.append({"type": "text", "text": val})
            if m["role"] == "assistant" or any(c["type"] == "dna" for c in content):
                # CHAT_TEMPLATE reads assistant content as message.content[0].text -> must be a list
                msgs.append({"role": m["role"], "content": content or [{"type": "text", "text": ""}]})
            else:
                msgs.append({"role": m["role"], "content": "".join(c["text"] for c in content)})
        return msgs

    def build(self, res: Resolved, enable_thinking: bool = True) -> dict:
        torch = self.torch
        text = self.tok.apply_chat_template(self.to_hf_messages(res), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=enable_thinking)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        n_pads = sum(1 for i in ids if i == self.dna_pad_id)
        if n_pads != len(res.dna_shas):
            raise HandleError(f"template rendered {n_pads} DNA slots for {len(res.dna_shas)} sequences")
        if not res.dna_shas:
            return {"prompt": ids, "n_prompt": len(ids), "n_dna": 0}
        rows = [self.projected_rows(s) for s in res.dna_shas]
        expanded: List[int] = []
        k = 0
        for i in ids:
            if i == self.dna_pad_id:
                expanded.extend([self.dna_pad_id] * rows[k].shape[0])
                k += 1
            else:
                expanded.append(i)
        ids_t = torch.tensor(expanded, dtype=torch.long)
        emb = self.embed[ids_t].clone()
        mask = ids_t == self.dna_pad_id
        emb[mask] = torch.cat(rows, dim=0).to(emb.dtype)
        return {"prompt_embeds": emb, "n_prompt": int(emb.shape[0]), "n_dna": int(mask.sum()),
                "text_ids": [i for i in expanded if i != self.dna_pad_id]}


def load_handle_map(meta_dirs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in meta_dirs:
        p = Path(d) / "handles.json"
        if p.exists():
            out.update(json.loads(p.read_text()))
    return out
