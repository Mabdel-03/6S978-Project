"""Harbor agents for the experiment arms. Load with `import_path: dnaharness.harbor_ext.agents:<Class>`.

  RlmAgent      arm = rlm | coding_repl. Host side registers a trial token with the gateway, then
                runs the official `rlms` loop INSIDE the sandbox (dnah_runtime.run_rlm), whose LM
                calls go to the gateway on 127.0.0.1 with that token.
  DirectAgent   one gateway call (no REPL): canonical (BioReason training format), raw-inline, or
                harness-prompt variants. Runs on the host; writes /logs/agent/answer.txt.
  VoteAgent     compute-matched control: k direct calls, majority vote over mapped labels.

Model names are given as `gateway/<served_name>` (Harbor parses `provider/name`).
All answers land in <trial>/agent/answer.txt (== /logs/agent/answer.txt in the sandbox).
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shlex
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

import httpx
from pydantic import Field

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage

from dnaharness import config
from dnaharness.runtime import prompts

DEFAULT_GATEWAY = f"http://127.0.0.1:{config.paths()['gateway_port']}"


def gateway_url(configured: str) -> str:
    """The allocation's gateway: DNAH_GATEWAY_URL (exported by serve_and_run.sbatch, which picks a
    per-allocation port) wins over the URL baked into the job config at render time, so a job also
    resumes correctly in a later allocation that got a different port."""
    return (os.environ.get("DNAH_GATEWAY_URL") or configured).rstrip("/")


class _GatewayMixin:
    @property
    def _gw(self) -> str:
        return gateway_url(self.options.gateway_url)

    """Trial registration + usage readback against the LM gateway (admin API, host only)."""

    def _served_model(self) -> str:
        return (self.model_name or "").split("/")[-1]

    def _admin(self) -> dict:
        return {"X-Admin-Key": config.load_secrets()["gateway_admin_key"]}

    async def _register(self, *, arm: str, subcalls: bool, tool: bool, episode_id: Optional[str]) -> str:
        token = uuid.uuid4().hex
        body = {"token": token, "model": self._served_model(), "arm": arm, "subcalls": subcalls,
                "tool": tool, "episode_id": episode_id,
                "max_generated_tokens": self.options.max_generated_tokens,
                "max_child_calls": self.options.max_child_calls, "policy": self.options.policy}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self._gw}/admin/trials", json=body, headers=self._admin())
            r.raise_for_status()
        self._token = token
        return token

    async def _usage(self) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{self._gw}/admin/trials/{self._token}", headers=self._admin())
            r.raise_for_status()
            return r.json()

    def _fill_context(self, context: AgentContext, usage: dict, extra: Optional[dict] = None) -> None:
        context.n_input_tokens = usage["prompt_tokens"]
        context.n_output_tokens = usage["completion_tokens"]
        context.n_cache_tokens = 0
        context.cost_usd = 0.0
        context.model_usage = {self._served_model(): ModelUsage(
            n_input_tokens=usage["prompt_tokens"], n_output_tokens=usage["completion_tokens"])}
        context.metadata = {**(context.metadata or {}), "gateway": usage, **(extra or {})}
        (self.logs_dir / "gateway_usage.json").write_text(json.dumps(usage, indent=2))

    async def _read_inputs(self, environment: BaseEnvironment) -> dict:
        res = await environment.exec("cat /task/inputs.json", timeout_sec=60)
        if res.return_code != 0:
            raise RuntimeError(f"cannot read /task/inputs.json: {res.stderr}")
        return json.loads(res.stdout)

    async def _episode_id(self, environment: BaseEnvironment) -> Optional[str]:
        # task names are opaque `kdx-<episode_id>`; session_id is `<trial_name>__env`
        sid = getattr(environment, "session_id", "") or ""
        return sid.split("__")[0] or None


class _CommonOptions(AgentOptions):
    gateway_url: str = Field(default=DEFAULT_GATEWAY)
    max_generated_tokens: int = Field(default=48000, description="per-trial generated-token budget (root+child+tool)")
    max_child_calls: int = Field(default=8)
    policy: Optional[str] = Field(default=None, description="override decoding policy (sampled|greedy)")


class RlmOptions(_CommonOptions):
    arm: str = Field(default="rlm", description="rlm | coding_repl")
    max_iterations: int = Field(default=12)
    agent_timeout_sec: int = Field(default=1700)
    root_sees_dna: Optional[bool] = Field(default=None, description="default: True when DNA handles are present")
    bioreason_tool: bool = Field(default=False)
    child_model: Optional[str] = Field(default=None, description="served name for depth-1 calls (default: same as root)")
    demos_file: Optional[str] = Field(default=None, description="host path of in-context demonstrations (Stage 3)")


class RlmAgent(_GatewayMixin, BaseAgent):
    options_model = RlmOptions

    @staticmethod
    def name() -> str:
        return "dnah-rlm"

    def version(self) -> str:
        return f"rlms-0.1.3/{prompts.PROMPT_VERSION}"

    async def setup(self, environment: BaseEnvironment) -> None:
        r = await environment.exec("test -f /opt/dnah/dnah_runtime/run_rlm.py && python3 -c 'import rlm'",
                                   timeout_sec=120)
        if r.return_code != 0:
            raise RuntimeError("bio-sandbox.sif lacks dnah_runtime/rlms (rebuild containers/bio-sandbox.def)")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        o: RlmOptions = self.options
        if o.arm not in ("rlm", "coding_repl"):
            raise ValueError(f"arm must be rlm|coding_repl, got {o.arm}")
        inputs = await self._read_inputs(environment)
        has_handles = "dna_handles" in inputs
        root_sees = has_handles if o.root_sees_dna is None else (o.root_sees_dna and has_handles)
        token = await self._register(arm=o.arm if not o.bioreason_tool else f"{o.arm}+bioreason_tool",
                                     subcalls=o.arm == "rlm", tool=o.bioreason_tool,
                                     episode_id=await self._episode_id(environment))
        cmd = ["python3", "-m", "dnah_runtime.run_rlm", "--inputs", "/task/inputs.json", "--arm", o.arm,
               "--model", self._served_model(), "--gateway", self._gw, "--token", token,
               "--max-iterations", str(o.max_iterations), "--max-timeout", str(o.agent_timeout_sec - 60),
               "--out", "/logs/agent"]
        if root_sees:
            cmd.append("--root-sees-dna")
        if o.bioreason_tool:
            cmd.append("--bioreason-tool")
        if o.child_model:
            cmd += ["--child-model", o.child_model]
        if o.demos_file:
            await environment.upload_file(o.demos_file, "/task/demos.txt")
            cmd += ["--demos-file", "/task/demos.txt"]
        res = await environment.exec(" ".join(shlex.quote(c) for c in cmd) + " > /logs/agent/driver.log 2>&1",
                                     timeout_sec=o.agent_timeout_sec)
        usage = await self._usage()
        meta = {}
        mp = self.logs_dir / "run_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        self._fill_context(context, usage, {"driver_return_code": res.return_code, "run_meta": meta,
                                            "root_sees_dna": root_sees})


class DirectOptions(_CommonOptions):
    variant: str = Field(default="auto", description="auto | canonical | raw | harness")


class DirectAgent(_GatewayMixin, BaseAgent):
    """Single call, no loop. `auto`: canonical BioReason format when DNA handles are present (native),
    raw-inline format otherwise."""

    options_model = DirectOptions

    @staticmethod
    def name() -> str:
        return "dnah-direct"

    def version(self) -> str:
        return prompts.PROMPT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _messages(self, inputs: dict, variant: str) -> list:
        q = inputs["question"]
        handles = inputs.get("dna_handles")
        if variant == "auto":
            variant = "canonical" if handles else "raw"
        if variant == "canonical":
            return [{"role": "user", "content": prompts.canonical_native_prompt(q, handles)}]
        if variant == "raw":
            return [{"role": "user", "content": prompts.direct_raw_prompt(
                q, inputs.get("reference_sequence"), inputs.get("variant_sequence"))}]
        if variant == "harness":
            return prompts.direct_harness_messages(q, handles, native=handles is not None)
        raise ValueError(variant)

    async def _call(self, messages: list) -> dict:
        async with httpx.AsyncClient(timeout=1800) as c:
            r = await c.post(f"{self._gw}/t/{self._token}/direct/v1/chat/completions",
                             json={"model": self._served_model(), "messages": messages})
        if r.status_code != 200:
            return {"error": r.text[:500], "content": None}
        return r.json()["choices"][0]["message"]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        inputs = await self._read_inputs(environment)
        await self._register(arm=f"direct:{self.options.variant}", subcalls=False, tool=False,
                             episode_id=await self._episode_id(environment))
        msg = await self._call(self._messages(inputs, self.options.variant))
        content = msg.get("content")
        (self.logs_dir / "answer.txt").write_text(content or "")
        (self.logs_dir / "response.json").write_text(json.dumps(msg, indent=2))
        self._fill_context(context, await self._usage(), {"variant": self.options.variant})


class VoteOptions(DirectOptions):
    k: int = Field(default=5, description="number of direct samples (set to match RLM mean token spend)")


class VoteAgent(DirectAgent):
    """Compute-matched control: k independent direct samples; majority over mapped labels."""

    options_model = VoteOptions

    @staticmethod
    def name() -> str:
        return "dnah-vote"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        from dnaharness.data.kegg import label_vocabulary
        from dnaharness.data.labels import score_answer

        inputs = await self._read_inputs(environment)
        await self._register(arm=f"vote@{self.options.k}:{self.options.variant}", subcalls=False, tool=False,
                             episode_id=await self._episode_id(environment))
        labels = label_vocabulary()
        msgs = self._messages(inputs, self.options.variant)
        samples = list(await asyncio.gather(*[self._call(msgs) for _ in range(self.options.k)]))
        mapped = [score_answer(s.get("content"), "", labels).predicted_label for s in samples]
        votes = Counter(m for m in mapped if m)
        winner = votes.most_common(1)[0][0] if votes else (samples[0].get("content") or "")
        (self.logs_dir / "answer.txt").write_text(winner)
        (self.logs_dir / "response.json").write_text(json.dumps({"samples": samples, "mapped": mapped}, indent=2))
        self._fill_context(context, await self._usage(), {"k": self.options.k, "votes": dict(votes)})


from harbor.agents.installed.mini_swe_agent import MiniSweAgent as _HarborMiniSweAgent  # noqa: E402


class MiniSweGatewayAgent(_HarborMiniSweAgent):
    """Harbor's built-in mini-swe-agent, unchanged except that the LM endpoint (the native config's
    model.model_kwargs.api_base) is resolved at run time with `gateway_url` -- the same rule our own
    agents use -- instead of the port baked in at render time."""

    def __init__(self, *args, config: Optional[dict] = None, **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            mk = config.setdefault("model", {}).setdefault("model_kwargs", {})
            if mk.get("api_base"):
                mk["api_base"] = gateway_url(mk["api_base"].rsplit("/v1", 1)[0]) + "/v1"
        super().__init__(*args, config=config, **kwargs)
