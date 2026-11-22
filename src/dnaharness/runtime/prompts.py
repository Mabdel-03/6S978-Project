"""Versioned prompts. The coding arm's system prompt is the RLM prompt with ONLY the sub-call
sections removed, so the controlled comparison differs in recursion alone.

Prompts are frozen after the dev-set prompt-development phase; bump PROMPT_VERSION on any change
(it is logged with every trial).
"""
from __future__ import annotations

from typing import Dict, Optional

PROMPT_VERSION = "v3"   # v2: fenced ```repl examples (v1 showed code unfenced; 1.7B models copied it verbatim)
                        # v3: closed label list in every non-canonical KEGG prompt (dev: only 20-25% of Qwen's
                        #     free-text answers mapped to a label). BioReason's canonical prompt is unchanged.

# The closed answer set of the KEGG task (bio-mirage metrics/dna.py vocabulary; identical for every task, so
# it reveals nothing about any one gold label). tests/test_prompts.py checks it against the dataset.
KEGG_LABELS = (
    "acute myeloid leukemia", "adenine phosphoribosyltransferase deficiency", "alzheimer's disease",
    "amyotrophic lateral sclerosis", "basal cell carcinoma", "chronic myeloid leukemia",
    "clear cell renal cell carcinoma", "colorectal cancer", "creutzfeldt-jakob disease", "cushing syndrome",
    "gastric cancer", "gaucher disease", "glioblastoma multiforme", "hepatocellular carcinoma",
    "huntington's disease", "lesch-nyhan syndrome", "melanoma", "methylmalonic aciduria and homocystinuria",
    "multiple endocrine neoplasia type 1", "n-acetylglutamate synthase deficiency", "non-small cell lung cancer",
    "pancreatic ductal adenocarcinoma", "papillary renal cell carcinoma", "papillary thyroid carcinoma",
    "parkinson's disease", "pituitary adenoma", "primary aldosteronism", "prion disease", "prostate cancer",
    "renal cell carcinoma", "robinow syndrome", "sphingolipidoses", "spinocerebellar ataxia", "thyroid cancer",
    "thyroid dyshormonogenesis", "urothelial carcinoma", "von hippel-lindau syndrome",
)
LABEL_LIST = "; ".join(KEGG_LABELS)

ANSWER_RULE = (
    'answer["content"] must be exactly ONE disease name from this list, copied verbatim, with no explanation: '
    + LABEL_LIST + "."
)

GENERIC_ANSWER_RULE = 'answer["content"] must be exactly the final answer the task asks for, in the requested format, with no explanation.'

# NOTE: these strings go through str.format(custom_tools_section=...) inside rlms -> braces doubled.
_TASK_KEGG = "The question describes a gene, its pathway context and (usually) a human DNA variant, and asks which disease the variant contributes to."
_TASK_GENERIC = "The task statement says exactly what to compute and in which format to answer."
_HEAD = """You are answering a question by working in a persistent Python REPL. {task_line} You will be queried turn by turn until you submit an answer.

Write Python code inside a fenced block that starts with ```repl and ends with ```. Code outside such a block is NOT executed. The REPL persists across turns. Only print(...) output is shown back to you (long outputs are truncated), so wrap inspections in print(...). For example:
```repl
print(context.keys())
```

Available in the REPL:
- `context`: a dict with the inputs for this question. Start by printing its keys and sizes.
"""

_SUBCALLS = """- `llm_query(prompt: str) -> str`: one call to a helper language model (the same model as you, without a REPL). It sees ONLY the prompt you pass, so include everything it needs.
- `llm_query_batched(prompts: list[str]) -> list[str]`: several helper calls run concurrently.
  For example:
```repl
reply = llm_query("Here is some evidence: ... Question: ...")
print(reply)
```
"""

_TAIL = """- `SHOW_VARS()`: list the variables you have created.
- `answer`: a dict initialized to {{"content": "", "ready": False}}. To submit, run this in a ```repl block:
```repl
answer["content"] = "<final answer>"
answer["ready"] = True
```
{custom_tools_section}

__ANSWER_RULE__

Inspect the inputs first, use the REPL to examine the evidence, __STRATEGY__ then submit. You have a limited number of turns; submit your best answer before running out.
"""

_NATIVE_ROOT = """
DNA handles: `context["dna_handles"]` maps "reference"/"variant" to handle strings such as "<DNA:0f3a...>". A handle stands for a DNA sequence that you read natively: the sequences behind the handles were shown to you in the first message. Printing a handle only shows the handle text.
"""

_NATIVE_SUBCALL = """When you put a handle verbatim inside an `llm_query` prompt, the helper model receives that DNA sequence natively (exactly as you did), so you can ask it questions about the sequences.
"""

_BIOREASON_TOOL = """- `bioreason_answer() -> str`: asks the model, in its trained question format (reference DNA, variant DNA, question), for its answer; returns that answer text. It takes no arguments.
"""


def _esc(x: str) -> str:
    return x.replace("{", "{{").replace("}", "}}")


def system_prompt(*, subcalls: bool, native: bool, bioreason_tool: bool = False, task: str = "kegg") -> str:
    """task='kegg' for the disease questions, 'generic' for pilot / SFT harness tasks."""
    kegg = task == "kegg"
    s = _HEAD.replace("{task_line}", _TASK_KEGG if kegg else _TASK_GENERIC)
    if subcalls:
        s += _SUBCALLS
    if bioreason_tool:
        s += _BIOREASON_TOOL
    s += (_TAIL.replace("__ANSWER_RULE__", _esc(ANSWER_RULE if kegg else GENERIC_ANSWER_RULE))
               .replace("__STRATEGY__", "reason about the gene, pathway and variant," if kegg else "compute what is asked,"))
    if native:
        s += _NATIVE_ROOT
        if subcalls:
            s += _NATIVE_SUBCALL
    return s


def root_prompt(question: str, handles: Optional[Dict[str, str]], root_sees_dna: bool) -> str:
    """Shown to the root model as 'Answer the following: ...' (rlms first user message). The
    gateway resolves handles ONLY in that first root user message (and in child prompts)."""
    if handles and root_sees_dna:
        return (f"Reference DNA: {handles['reference']}\nVariant DNA: {handles['variant']}\n\n"
                f"{question.strip()}")
    return question.strip()


# ------------------------------------------------------------------ direct (single-call) prompts

DIRECT_FORMAT = "Think it through, then end your response with a final line of the form 'Answer: <disease name>'."
LABEL_MENU = "Choose your answer from these disease names (copy one verbatim): " + LABEL_LIST + "."


def direct_raw_prompt(question: str, ref: Optional[str], var: Optional[str]) -> str:
    """Exp A direct arm (raw strings inline). Same text for Qwen and BioReason."""
    if ref is None:
        return f"{question.strip()}\n\n{LABEL_MENU}\n\n{DIRECT_FORMAT}"
    return (f"Reference DNA sequence:\n{ref}\n\nVariant DNA sequence:\n{var}\n\n"
            f"{question.strip()}\n\n{LABEL_MENU}\n\n{DIRECT_FORMAT}")


def canonical_native_prompt(question: str, handles: Optional[Dict[str, str]]) -> str:
    """Exactly BioReason's training format: [dna ref][dna var][question], no system prompt, no
    instruction. Adjacent handles render as <|dna_start|>..<|dna_end|><|dna_start|>..<|dna_end|>."""
    if not handles:
        return question.strip()
    return f"{handles['reference']}{handles['variant']}{question.strip()}"


def direct_harness_messages(question: str, handles: Optional[Dict[str, str]], native: bool) -> list:
    """Exp B control: ONE call carrying the harness system prompt + handles (no REPL loop)."""
    sys_p = system_prompt(subcalls=False, native=native).format(custom_tools_section="")
    user = (f"Answer the following: {root_prompt(question, handles, True)}\n\n"
            "There is no REPL for this question: reply directly, ending with 'Answer: <disease name>'.")
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
