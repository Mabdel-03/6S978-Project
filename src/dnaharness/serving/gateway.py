"""LM gateway: the ONLY model endpoint sandboxes and agents use.

  POST /t/{token}/{route}/v1/chat/completions   route in root | child | tool | direct
  POST /v1/chat/completions                     (X-Session-ID header; mini-swe-agent)
  GET  /t/{token}/{route}/v1/models, /v1/models, /health
  POST /admin/trials            register a trial token (host-side agents; X-Admin-Key)
  GET  /admin/trials/{key}      usage summary for a trial (token or "session:<id>")

Responsibilities
  * one decoding policy for every arm (temperature/top_p/top_k/max_tokens/seed are overwritten)
  * per-trial budgets: generated tokens (root+child+tool, thinking included), child-call count;
    exhaustion -> HTTP 429. Child route on a coding-arm trial -> HTTP 403 (counted).
  * renders prompts itself (serving.native) and calls vLLM /v1/completions with token ids or
    native-DNA prompt_embeds; splits <think> into `reasoning_content`
  * JSONL call log with full accounting (text tokens, DNA positions, reasoning tokens, latency)

    python -m dnaharness.serving.gateway --models qwen3-1.7b bioreason-1.7b --policy sampled \
        --meta-dirs <datasets>/kegg-dx/_meta --log-dir <gateway_logs>/<job>
"""
import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from dnaharness import config
from dnaharness.serving import native as N

ROUTES = ("root", "child", "tool", "direct")


@dataclass
class TrialState:
    key: str
    model: str
    arm: str
    subcalls: bool = False
    tool: bool = False
    max_generated_tokens: int = 48000
    max_child_calls: int = 8
    episode_id: Optional[str] = None
    policy: Optional[str] = None                 # override the job-wide policy (e.g. greedy repro)
    n_calls: Dict[str, int] = field(default_factory=lambda: {r: 0 for r in ROUTES})
    blocked_child_calls: int = 0
    prompt_tokens: int = 0
    dna_positions: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    truncated_in_think: int = 0
    budget_exhausted: bool = False
    errors: int = 0
    next_call: int = 0
    t_first: Optional[float] = None
    t_last: Optional[float] = None


class Gateway:
    # must equal vLLM's --max-model-len (serve_and_run.sbatch exports DNAH_MAX_MODEL_LEN for both)
    max_model_len: int = int(os.environ.get("DNAH_MAX_MODEL_LEN", "32768"))

    def __init__(self, models: List[str], policy: str, meta_dirs: List[str], log_dir: str,
                 log_messages: bool = False, session_defaults: Optional[dict] = None):
        self.secrets = config.load_secrets()
        self.policy_name = policy
        self.policies = config.models()["decoding"]
        self.handle_map = N.load_handle_map(meta_dirs)
        self.trials: Dict[str, TrialState] = {}
        self.session_defaults = session_defaults or {"arm": "mini_swe"}
        self.log_messages = log_messages
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / "calls.jsonl"
        self._log_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0))
        self.backends: Dict[str, dict] = {}
        for m in models:
            cfg = config.model_cfg(m)
            ckpt = str(config.path("ckpts") / m)
            renderer = (N.BioReasonRenderer(ckpt, str(config.path("evo2_cache")))
                        if cfg.get("native_dna") else N.TextRenderer(ckpt))
            self.backends[m] = {"cfg": cfg, "renderer": renderer,
                                "url": f"http://127.0.0.1:{config.model_port(m)}/v1/completions"}

    # ------------------------------------------------------------------ trial bookkeeping
    def register(self, body: dict) -> TrialState:
        st = TrialState(key=body["token"], model=body["model"], arm=body["arm"],
                        subcalls=bool(body.get("subcalls", False)), tool=bool(body.get("tool", False)),
                        max_generated_tokens=int(body.get("max_generated_tokens", 48000)),
                        max_child_calls=int(body.get("max_child_calls", 8)),
                        episode_id=body.get("episode_id"), policy=body.get("policy"))
        self.trials[st.key] = st
        return st

    def session_trial(self, session_id: str, model: str) -> TrialState:
        key = f"session:{session_id}"
        if key not in self.trials:
            self.trials[key] = TrialState(key=key, model=model, **self.session_defaults)
        return self.trials[key]

    async def log(self, rec: dict) -> None:
        async with self._log_lock:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ main path
    async def chat(self, st: TrialState, route: str, body: dict) -> dict:
        t0 = time.time()
        st.t_first = st.t_first or t0
        model = body.get("model") or st.model
        if model not in self.backends:
            # agents may send provider-prefixed names (hosted_vllm/qwen3-1.7b)
            model = model.split("/")[-1]
        if model not in self.backends:
            raise GatewayError(404, f"model {model!r} not served")
        call_idx = st.next_call          # allocated before any await -> unique per call (seeds)
        st.next_call += 1
        rec = {"ts": t0, "trial": st.key, "arm": st.arm, "model": model, "route": route,
               "call_idx": call_idx, "episode_id": st.episode_id}

        if route == "child" and not st.subcalls:
            st.blocked_child_calls += 1
            await self.log({**rec, "status": 403})
            raise GatewayError(403, "sub-LLM calls are not available in this harness")
        if route == "tool" and not st.tool:
            await self.log({**rec, "status": 403})
            raise GatewayError(403, "tool calls are not enabled for this trial")
        remaining = st.max_generated_tokens - st.completion_tokens
        if remaining <= 0 or (route == "child" and st.n_calls["child"] >= st.max_child_calls):
            st.budget_exhausted = True
            await self.log({**rec, "status": 429})
            raise GatewayError(429, "budget exhausted for this trial")

        be = self.backends[model]
        rnd: N.TextRenderer = be["renderer"]
        pol = dict(self.policies[st.policy or self.policy_name])
        try:
            res = N.apply_handle_policy(body.get("messages", []), route, rnd.native, self.handle_map)
            built = rnd.build(res, enable_thinking=pol.get("enable_thinking", True))
        except N.HandleError as e:
            st.errors += 1
            await self.log({**rec, "status": 400, "error": str(e)})
            raise GatewayError(400, str(e))

        # Never ask vLLM for more than fits in the context window (it would reject the request);
        # a full window ends the trial exactly like an exhausted token budget.
        room = self.max_model_len - int(built["n_prompt"]) - 8
        if room < 64:
            st.budget_exhausted = True
            await self.log({**rec, "status": 429, "error": "context window full",
                            "prompt_tokens": built["n_prompt"]})
            raise GatewayError(429, "context window full for this trial")
        seed = int(hashlib.sha256(f"{st.key}|{rec['call_idx']}".encode()).hexdigest()[:8], 16)
        req = {"model": model, "max_tokens": int(min(pol["max_tokens"], remaining, room)),
               "temperature": pol["temperature"], "top_p": pol["top_p"], "top_k": pol["top_k"],
               "seed": seed, "stop_token_ids": rnd.stop_token_ids, "skip_special_tokens": True}
        if "prompt_embeds" in built:
            req["prompt_embeds"] = N.tensor_to_b64(built["prompt_embeds"])
        else:
            req["prompt"] = built["prompt"]
        headers = {"Authorization": f"Bearer {self.secrets['vllm_api_key']}"}
        r = await self.client.post(be["url"], json=req, headers=headers)
        if r.status_code != 200:
            st.errors += 1
            await self.log({**rec, "status": 502, "error": r.text[:500]})
            raise GatewayError(502, f"backend error: {r.text[:300]}")
        out = r.json()
        choice = out["choices"][0]
        text = choice.get("text", "")
        reasoning, content, trunc = N.split_thinking(text)
        n_completion = int((out.get("usage") or {}).get("completion_tokens", rnd.count(text)))
        n_reason = rnd.count(reasoning)

        st.n_calls[route] += 1
        st.prompt_tokens += built["n_prompt"]
        st.dna_positions += built["n_dna"]
        st.completion_tokens += n_completion
        st.reasoning_tokens += n_reason
        st.truncated_in_think += int(trunc)
        st.t_last = time.time()
        log = {**rec, "status": 200, "seed": seed, "policy": st.policy or self.policy_name,
               "prompt_tokens": built["n_prompt"], "dna_positions": built["n_dna"],
               "text_prompt_tokens": built["n_prompt"] - built["n_dna"],
               "handles_resolved": res.n_resolved, "handles_neutralized": res.n_neutralized,
               "completion_tokens": n_completion, "reasoning_tokens": n_reason,
               "finish_reason": choice.get("finish_reason"), "truncated_in_think": trunc,
               "latency_s": round(st.t_last - t0, 3), "content": content, "reasoning": reasoning}
        if self.log_messages:
            log["messages"] = body.get("messages")
        await self.log(log)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion",
            "created": int(t0), "model": model,
            "choices": [{"index": 0, "finish_reason": choice.get("finish_reason") or "stop",
                         "message": {"role": "assistant", "content": content,
                                     "reasoning_content": reasoning or None}}],
            "usage": {"prompt_tokens": built["n_prompt"], "completion_tokens": n_completion,
                      "total_tokens": built["n_prompt"] + n_completion},
        }


class GatewayError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


def create_app(gw: "Gateway"):
    app = FastAPI(title="dnaharness LM gateway")

    def err(e: GatewayError):
        return JSONResponse(status_code=e.status, content={"error": {"message": e.msg, "code": e.status}})

    def check_admin(key: Optional[str]):
        if key != gw.secrets["gateway_admin_key"]:
            raise HTTPException(status_code=401, detail="bad admin key")

    @app.get("/health")
    async def health():
        return {"ok": True, "models": list(gw.backends), "policy": gw.policy_name,
                "trials": len(gw.trials), "handles": len(gw.handle_map)}

    @app.post("/admin/trials")
    async def register(request: Request, x_admin_key: Optional[str] = Header(None)):
        check_admin(x_admin_key)
        st = gw.register(await request.json())
        return asdict(st)

    @app.get("/admin/trials/{key:path}")
    async def trial(key: str, x_admin_key: Optional[str] = Header(None)):
        check_admin(x_admin_key)
        st = gw.trials.get(key)
        if st is None:
            raise HTTPException(status_code=404, detail="unknown trial")
        return asdict(st)

    @app.post("/admin/reload_handles")
    async def reload_handles(request: Request, x_admin_key: Optional[str] = Header(None)):
        check_admin(x_admin_key)
        body = await request.json()
        gw.handle_map.update(N.load_handle_map(body.get("meta_dirs", [])))
        return {"handles": len(gw.handle_map)}

    def models_payload():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "dnaharness"}
                                          for m in gw.backends]}

    @app.get("/v1/models")
    async def models_root():
        return models_payload()

    @app.get("/t/{token}/{route}/v1/models")
    async def models_t(token: str, route: str):
        return models_payload()

    @app.post("/t/{token}/{route}/v1/chat/completions")
    async def chat_t(token: str, route: str, request: Request):
        if route not in ROUTES:
            return err(GatewayError(404, f"unknown route {route}"))
        st = gw.trials.get(token)
        if st is None:
            return err(GatewayError(401, "unregistered trial token"))
        try:
            return await gw.chat(st, route, await request.json())
        except GatewayError as e:
            return err(e)

    @app.post("/v1/chat/completions")
    async def chat_session(request: Request, x_session_id: Optional[str] = Header(None)):
        if not x_session_id:
            return err(GatewayError(401, "X-Session-ID header required"))
        body = await request.json()
        st = gw.session_trial(x_session_id, (body.get("model") or "").split("/")[-1])
        try:
            return await gw.chat(st, "root", body)
        except GatewayError as e:
            return err(e)

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--policy", default="sampled", choices=["sampled", "greedy", "teacher"])
    ap.add_argument("--meta-dirs", nargs="*", default=[])
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--log-messages", action="store_true", help="store request messages (teacher/SFT jobs)")
    ap.add_argument("--session-max-generated-tokens", type=int, default=48000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(config.paths()["gateway_port"]))
    a = ap.parse_args(argv)
    import uvicorn
    gw = Gateway(a.models, a.policy, a.meta_dirs, a.log_dir, a.log_messages,
                 session_defaults={"arm": "mini_swe", "max_generated_tokens": a.session_max_generated_tokens})
    uvicorn.run(create_app(gw), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
