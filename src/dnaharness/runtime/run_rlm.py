"""In-sandbox driver for the `rlm` and `coding_repl` arms (official `rlms` RLM loop).

Both arms run the SAME loop, REPL, tools, prompt (modulo sub-call sections) and budgets. The only
difference is whether model-generated code can call the model:
  rlm          depth-1 `llm_query` / `llm_query_batched` -> gateway /child route (same checkpoint)
  coding_repl  sub-call names are removed from the REPL namespace (and re-removed after every cell,
               since rlms restores its scaffold), AND the child route still points at the gateway's
               /child endpoint, which returns 403 for coding trials -- two independent guards.

Writes /logs/agent/answer.txt (final answer only), /logs/agent/run_meta.json and rlms JSONL logs.

    python -m dnah_runtime.run_rlm --inputs /task/inputs.json --arm rlm --model qwen3-1.7b \
        --gateway http://127.0.0.1:8000 --token <trial token> --out /logs/agent
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from . import prompts, seqtools

SUBCALL_NAMES = ("llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched")


def _install_nosubcall_repl():
    """Patch rlms so the 'local' environment is a LocalREPL without sub-call functions."""
    import rlm.core.rlm as rlm_core
    from rlm.environments.local_repl import LocalREPL

    class NoSubcallREPL(LocalREPL):
        def setup(self):
            super().setup()
            for n in SUBCALL_NAMES:
                self.globals.pop(n, None)

        def _restore_scaffold(self):
            super()._restore_scaffold()
            for n in SUBCALL_NAMES:
                self.globals.pop(n, None)

    orig = rlm_core.get_environment

    def get_environment(environment, environment_kwargs):
        if environment == "local":
            return NoSubcallREPL(**environment_kwargs)
        return orig(environment, environment_kwargs)

    rlm_core.get_environment = get_environment


def _make_bioreason_tool(gateway: str, token: str, model: str, question: str, handles: dict):
    """Non-recursive Exp-B control tool: one fixed canonical-format call (BioReason training format)."""
    import openai

    client = openai.OpenAI(base_url=f"{gateway}/t/{token}/tool/v1", api_key="sandbox", max_retries=0,
                           timeout=600)
    canonical = prompts.canonical_native_prompt(question, handles)

    def bioreason_answer() -> str:
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": canonical}])
            return r.choices[0].message.content or ""
        except Exception as e:  # surfaced to the model as text, like rlms does for llm_query
            return f"Error: {e}"

    return bioreason_answer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/task/inputs.json")
    ap.add_argument("--arm", choices=["rlm", "coding_repl"], required=True)
    ap.add_argument("--model", required=True, help="served model name at the gateway (root and child)")
    ap.add_argument("--child-model", default=None, help="defaults to --model (same checkpoint)")
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--max-timeout", type=float, default=1500.0)
    ap.add_argument("--root-sees-dna", action="store_true")
    ap.add_argument("--bioreason-tool", action="store_true")
    ap.add_argument("--demos-file", default=None, help="in-context demonstrations appended to the system prompt")
    ap.add_argument("--out", default="/logs/agent")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    meta = {"arm": a.arm, "model": a.model, "prompt_version": prompts.PROMPT_VERSION,
            "max_iterations": a.max_iterations, "status": "started", "t_start": time.time()}

    def finish(answer, status, **extra):
        meta.update(status=status, t_end=time.time(), **extra)
        with open(os.path.join(a.out, "answer.txt"), "w") as fh:
            fh.write("" if answer is None else str(answer))
        with open(os.path.join(a.out, "run_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    try:
        from rlm import RLM
        from rlm.logger import RLMLogger

        inputs = json.load(open(a.inputs))
        question = inputs["question"]
        handles = inputs.get("dna_handles")
        task_kind = inputs.get("task_kind", "kegg")       # "generic" for pilot / SFT harness tasks
        has_raw = "reference_sequence" in inputs
        native = handles is not None

        subcalls = a.arm == "rlm"
        if not subcalls:
            _install_nosubcall_repl()

        tools = {}
        if has_raw:
            tools.update({n: {"tool": f, "description": seqtools.DESCRIPTIONS[n]}
                          for n, f in seqtools.TOOLS.items()})
        if a.bioreason_tool:
            # Without DNA (no_dna condition) the tool sends the canonical prompt with the question only --
            # exactly what the native `direct` arm sends in that condition.
            tools["bioreason_answer"] = {
                "tool": _make_bioreason_tool(a.gateway, a.token, a.model, question, handles),
                "description": "bioreason_answer() -> str: the model's answer in its trained format"}

        child = a.child_model or a.model
        # Pilot (2026-11-21): runaway 16k-token generations queued behind 24 concurrent trials took up to
        # ~1460 s, so a 900 s client timeout turned slow-but-valid calls into errors. 1500 s stays under the
        # Harbor agent timeout (1700 s), which remains the hard per-trial limit.
        common = {"api_key": "sandbox", "max_retries": 0, "timeout": 1500}
        backend_kwargs = {"model_name": a.model, "base_url": f"{a.gateway}/t/{a.token}/root/v1", **common}
        child_kwargs = {"model_name": child, "base_url": f"{a.gateway}/t/{a.token}/child/v1", **common}

        sys_prompt = prompts.system_prompt(subcalls=subcalls, native=native,
                                           bioreason_tool=a.bioreason_tool, task=task_kind)
        if a.demos_file:
            demos = open(a.demos_file).read()
            sys_prompt += "\n\nWorked examples of operating this REPL on other tasks:\n\n" + demos.replace("{", "{{").replace("}", "}}")
            meta["demos"] = True
        logger = RLMLogger(log_dir=a.out, file_name="rlm")
        rlm = RLM(
            backend="vllm", backend_kwargs=backend_kwargs,
            other_backends=["vllm"], other_backend_kwargs=[child_kwargs],
            environment="local",
            max_depth=1, max_iterations=a.max_iterations, max_timeout=a.max_timeout,
            custom_system_prompt=sys_prompt,
            custom_tools=tools or None,
            orchestrator=False,          # the long-context orchestrator addendum is off for both arms
            logger=logger, verbose=False,
            max_concurrent_subcalls=4,
        )
        context = {k: v for k, v in inputs.items() if k != "task_kind"}
        root = prompts.root_prompt(question, handles, a.root_sees_dna)
        try:
            res = rlm.completion(context, root_prompt=root)
        except Exception as e:  # budget exhaustion (HTTP 429 from gateway), timeouts, ...
            partial = getattr(rlm, "_best_partial_answer", None)
            status = "budget_exhausted" if "429" in str(e) or "budget" in str(e).lower() else "error"
            finish(None, status, error=f"{type(e).__name__}: {e}", best_partial=partial)
            return 0
        usage = getattr(res, "usage_summary", None)
        iters = list(getattr(logger, "_iterations", []) or [])
        if not iters and isinstance(res.metadata, dict):
            iters = res.metadata.get("iterations", [])
        # rlms falls back to one extra "give your final answer" call when turns run out, and LOGS it as
        # iteration max_iterations+1 with final_answer set -- that is not a protocol submission.
        in_budget = [it for it in iters if isinstance(it, dict) and int(it.get("iteration") or 0) <= a.max_iterations]
        submitted = any(it.get("final_answer") is not None for it in in_budget)
        finish(res.response, "ok" if submitted else "max_iterations",
               fallback_answer=not submitted,
               n_code_blocks=sum(len(it.get("code_blocks") or []) for it in in_budget),
               execution_time=res.execution_time, n_iterations=len(in_budget),
               usage=usage.to_dict() if usage is not None and hasattr(usage, "to_dict") else None)
        return 0
    except SystemExit:
        raise
    except Exception as e:
        finish(None, "error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
